"""The agent itself. Read this file first.

Each node is one step: it reads the shared state and returns what changed. The edges say what runs
next, and four of them are decisions taken at run time from what the agent has just observed:

  rule_labels   Who should label these comments?     All to the LLM, part to the LLM with a student model, or rules alone.
  check_labels  Too many comments fit no topic?      Go back and find more topics.
  investigate   Is there more worth checking?        Run another check, or move on to writing.
  verify        Did the draft fail the checker?      Send the reasons back to the writer.

If the LLM fails at any point, the run carries on with rules instead of crashing, and says so.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from . import prompts, report
from .analyse import build_analysis, run_check
from .classify import Labeller, apply_rules, label_budgeted, label_summary, labelling_room, llm_targets
from .clean import clean_comments, unique_texts
from .config import Settings
from .ingest import MAX_TEXT, load_comments
from .insights import check_key, choose_next_check, finalise, template_insights, verify, write_insights
from .llm import LLM, LLMError, LLMFatal
from .runlog import RunLog
from .schemas import TopicNames
from .topics import GENERAL, OTHER, TopicModel, cluster_examples, discover

MAX_OTHER_SHARE = 0.15  # above this, the topic list is not good enough
MAX_TOPIC_ROUNDS = 2
MAX_CHECKS = 6
MAX_WRITES = 3
MAX_DRAFT_CHARS = 4000


class State(TypedDict, total=False):
    comments: pd.DataFrame     # every row of the input, with its status
    info: dict                 # what the file looked like
    unique: pd.DataFrame       # one row per distinct usable text
    labelled: pd.DataFrame     # the same, with labels
    model: TopicModel
    taxonomy: list[str]
    topic_round: int
    relabel_ids: list[str]
    need_more_topics: bool
    stats: dict
    analysis: Any
    checks_done: list[str]
    enough_checks: bool
    insights: Any
    problems: list[dict]
    write_round: int
    removed_claims: int
    outputs: dict


class Agent:
    def __init__(self, settings: Settings, input_path: str, out_dir: str, title: str = "", text_col: str | None = None, quiet: bool = False, limit: int | None = None):
        self.s = settings
        self.limit = limit
        self.input_path, self.out_dir, self.title, self.text_col = input_path, Path(out_dir), title, text_col
        self.log = RunLog(self.out_dir, quiet=quiet)
        self.llm: LLM | None = None       # set to None if the LLM stops working during the run
        self.llm_used: LLM | None = None  # kept for the usage figures in the report

    # ---------- nodes ----------

    def ingest(self, state: State) -> dict:
        df, info = load_comments(self.input_path, self.text_col)
        found = ", ".join(f"{k.replace('_col', '')}: {v}" for k, v in info.items() if k.endswith("_col") and v)
        self.log.decision("ingest", f"Read {info['rows']:,} rows. Columns detected, {found}.")
        repairs = []
        if info["repaired_rows"]:
            repairs.append(f"{info['repaired_rows']:,} rows had unquoted separators in the text and were put back together")
        if info["duplicate_ids"]:
            repairs.append(f"{info['duplicate_ids']:,} repeated ids were made unique")
        if info["cut_long_comments"]:
            repairs.append(f"{info['cut_long_comments']:,} very long comments were cut to {MAX_TEXT:,} characters")
        if info["dates_parsed"] is not None and info["dates_parsed"] < 1:
            repairs.append(f"{1 - info['dates_parsed']:.1%} of dates could not be read")
        if repairs:
            self.log.decision("ingest", "Repairs and limits: " + "; ".join(repairs) + ".")
        if self.limit and self.limit < len(df):
            info["sampled_from"] = len(df)
            df = df.sample(self.limit, random_state=0).reset_index(drop=True)
            self.log.decision("ingest", f"Using a random sample of {self.limit:,} of the {info['sampled_from']:,} rows, as asked.")
        return {"comments": df, "info": info}

    def clean(self, state: State) -> dict:
        comments = clean_comments(state["comments"])
        unique = unique_texts(comments)
        counts = comments["status"].value_counts().to_dict()
        langs = comments.loc[comments["status"] == "ok", "lang"].value_counts().to_dict()
        held = int(comments.loc[comments["status"] == "ok", "hold_from_llm"].sum())
        kept = f" {held:,} kept for their safety wording but held back from the LLM because they also look like prompt injection." if held else ""
        self.log.decision("clean", f"Status of rows: {counts}. {len(unique):,} distinct texts remain, so repeats are labelled once. Languages: {langs}.{kept}")
        if unique.empty:
            raise ValueError("No usable comments are left after cleaning.")
        return {"comments": comments, "unique": unique}

    def plan(self, state: State) -> dict:
        n = len(state["unique"])
        if not self.s.uses_llm:
            self.log.decision("plan", "No LLM configured, so this run uses rules only. Sentiment will be approximate.")
            return {}
        llm = LLM(self.s, self.log)
        try:
            llm.ping()
        except LLMError as e:
            # The user still gets a report rather than a crash. High availability matters in this kind of system.
            self.log.decision("plan", f"LLM '{self.s.model}' is not usable ({str(e)[:120]}). Falling back to rules so the run still finishes.")
            return {}
        self.llm = self.llm_used = llm
        calls = math.ceil(n / self.s.batch_size)
        limit = "There is no cap on calls, so labelling time is the only cost. Use --limit or --max-llm-calls to shorten it." if self.s.max_calls is None else f"The budget is {self.s.max_calls:,} calls."
        self.log.decision("plan", f"LLM '{self.s.model}' responded. {n:,} distinct comments at {self.s.batch_size} per call need about {calls:,} calls. {limit}")
        return {}

    def discover_topics(self, state: State) -> dict:
        round_no = state.get("topic_round", 0) + 1
        if round_no == 1:
            unique = state["unique"]
            model = discover(unique["clean_text"].tolist())
            names = self._name_topics(model, unique.loc[~unique["hold_from_llm"].astype(bool), "clean_text"].tolist())
            taxonomy = _dedupe(names) + [GENERAL, OTHER]
            self.log.decision("discover", f"Found {len(_dedupe(names))} topics in the data: {', '.join(_dedupe(names))}.")
            return {"model": model, "taxonomy": taxonomy, "topic_round": round_no}
        # Later rounds look only at the comments that fit no topic.
        others = state["labelled"][state["labelled"]["topic"] == OTHER]
        texts = others["clean_text"].tolist()
        model = discover(texts, k=min(5, max(2, len(texts) // 40)))
        names = self._name_topics(model, others.loc[~others["hold_from_llm"].astype(bool), "clean_text"].tolist())
        old = [t for t in state["taxonomy"] if t not in (GENERAL, OTHER)]
        new = [n for n in _dedupe(names) if n not in old]
        self.log.decision("discover", f"Round {round_no}: looked at {len(texts):,} unmatched comments and added topics: {', '.join(new) or 'none'}.")
        return {"model": model, "taxonomy": old + new + [GENERAL, OTHER], "topic_round": round_no, "relabel_ids": others.index.tolist()}

    def _name_topics(self, model: TopicModel, texts: list[str]) -> list[str]:
        if not model.names or self.llm is None:
            return model.names
        try:
            reply = self.llm.ask(prompts.TOPIC_SYSTEM, prompts.topic_user(cluster_examples(model, texts)), TopicNames, purpose="name_topics", max_tokens=60 * len(model.names) + 60)
            for item in reply.topics:
                if 0 <= item.cluster < len(model.names) and item.name.strip():
                    model.names[item.cluster] = item.name.strip()[:40]
        except LLMError as e:
            self._llm_failed(e)
            self.log.event("discover", naming_failed=str(e)[:200])
        return model.names

    def rule_labels(self, state: State) -> dict:
        relabel = state.get("relabel_ids")
        if relabel is None:
            return {"labelled": apply_rules(state["unique"], state["model"])}
        df = state["labelled"]  # a later round: only unmatched comments change
        new_topics = state["model"].assign(df.loc[relabel, "clean_text"].tolist(), [0.0] * len(relabel))
        df.loc[relabel, "topic"] = new_topics
        df.loc[relabel, "topic_rule"] = new_topics
        return {"labelled": df}

    def llm_all(self, state: State) -> dict:
        ids = llm_targets(state["labelled"], state.get("relabel_ids"))
        limit = "There is no cap on calls" if self.s.max_calls is None else "The budget allows it"
        self.log.decision("classify", f"{limit}, so the LLM labels all {len(ids):,} comments in this step.")

        def work(df: pd.DataFrame, labeller: Labeller) -> dict:
            labeller.label(df, ids, state["taxonomy"], note="labelling")
            return {}
        return self._labelling(state, work)

    def llm_budgeted(self, state: State) -> dict:
        ids = llm_targets(state["labelled"], state.get("relabel_ids"))
        room = labelling_room(self.llm, self.s.batch_size)
        return self._labelling(state, lambda df, labeller: label_budgeted(df, ids, state["taxonomy"], labeller, room, self.log))

    def _labelling(self, state: State, work) -> dict:
        """Run the labelling. If the LLM dies midway, the labels so far stay and rules cover the rest."""
        df = state["labelled"]
        labeller = Labeller(self.llm, self.log, self.s.batch_size)
        stats = dict(state.get("stats", {}))
        try:
            stats.update(work(df, labeller))
        except LLMFatal as e:
            self.log.decision("classify", f"LLM stopped working ({str(e)[:160]}). Keeping what was labelled and using rules for the rest.")
            self.llm = None
        stats["grounding_failures"] = stats.get("grounding_failures", 0) + labeller.grounding_failures
        stats.update(label_summary(df))
        return {"labelled": df, "stats": stats}

    def check_labels(self, state: State) -> dict:
        df = state["labelled"]
        share = df.loc[df["topic"] == OTHER, "repeats"].sum() / df["repeats"].sum()
        again = share > MAX_OTHER_SHARE and state["topic_round"] < MAX_TOPIC_ROUNDS and (df["topic"] == OTHER).sum() >= 20
        if again:
            self.log.decision("check", f"{share:.1%} of comments fit no topic, above the {MAX_OTHER_SHARE:.0%} limit. Going back to find more topics.")
        else:
            self.log.decision("check", f"{share:.1%} of comments fit no topic. Accepting the topic list.")
        stats = {**state.get("stats", {}), "source_counts": df["source"].value_counts().to_dict()}
        return {"need_more_topics": bool(again), "stats": stats}

    def analyse(self, state: State) -> dict:
        a = build_analysis(state["comments"], state["labelled"])
        alerts = "; ".join(f"{x['label']}: {x['count']}" for x in a.alerts) or "none"
        spikes = "; ".join(f"{s['topic']} from {s['date']}" for s in a.spikes) or "none"
        self.log.decision("analyse", f"Computed {len(a.book.facts)} facts in code. The LLM labelled {math.floor(a.coverage * 1000) / 10:g}% of analysed comments. Alerts: {alerts}. Spikes: {spikes}.")
        return {"analysis": a}

    def investigate(self, state: State) -> dict:
        a = state["analysis"]
        done = state.get("checks_done", [])
        check, chooser = choose_next_check(a, self.llm, done)
        if check.action == "stop":
            self.log.decision("investigate", f"{chooser} decided there is enough evidence: {check.why or 'stopping'}.")
            return {"enough_checks": True}
        result = run_check(a, check.action, check.topic, check.sentiment, check.query)
        subject = f"'{check.query}'" if check.action == "search" else f"'{check.topic}'"
        self.log.decision("investigate", f"{chooser} chose {check.action} on {subject} ({check.why}). Found: {result[:170]}")
        done = done + [check_key(check)]
        return {"checks_done": done, "enough_checks": len(done) >= MAX_CHECKS}

    def write(self, state: State) -> dict:
        a = state["analysis"]
        round_no = state.get("write_round", 0) + 1
        problems = [p["problem"] for p in state.get("problems", [])]
        if self.llm is not None and self.llm.calls_left > 0:
            draft = state["insights"].model_dump_json(indent=None)[:MAX_DRAFT_CHARS] if problems else None
            try:
                ins = write_insights(a, self.llm, problems or None, draft)
                self.log.decision("write", f"LLM wrote draft {round_no}: {len(ins.points)} points, {len(ins.actions)} actions.")
                return {"insights": ins, "write_round": round_no}
            except LLMError as e:
                self._llm_failed(e)
                self.log.decision("write", f"The LLM writer failed ({str(e)[:100]}). Using the fixed-sentence writer instead.")
        ins = template_insights(a)
        self.log.decision("write", f"Fixed-sentence writer produced {len(ins.points)} points and {len(ins.actions)} actions from the facts.")
        return {"insights": ins, "write_round": round_no}

    def verify_draft(self, state: State) -> dict:
        problems = verify(state["insights"], state["analysis"])
        if problems:
            self.log.decision("verify", f"Checker rejected draft {state['write_round']} with {len(problems)} problem{'s' if len(problems) != 1 else ''}. First: {problems[0]['problem']}")
        else:
            self.log.decision("verify", f"Checker passed draft {state['write_round']}: every number, date and quote traces back to the facts it cites.")
        return {"problems": problems}

    def finish(self, state: State) -> dict:
        a = state["analysis"]
        problems = state.get("problems", [])
        ins, removed, used_template = finalise(state["insights"], problems, a)
        if used_template:
            self.log.decision("verify", f"Too little passed the checker, with {removed} statements removed, so the fixed-sentence writer produced the findings.")
        elif removed:
            self.log.decision("verify", f"Rewrites ran out. Removed {removed} statements that still failed the checker. The report says so.")
        usage = self.llm_used.usage() if self.llm_used is not None else {"provider": "none", "model": "", "calls": 0}
        outputs = report.write_all(self.out_dir, self.title or Path(self.input_path).stem, a, ins, state["labelled"], state["comments"], self.log, usage,
                                   state.get("stats", {}), removed, state["info"], self.graph_mermaid())
        self.log.decision("report", f"Report written to {outputs['html']}")
        return {"insights": ins, "removed_claims": removed, "outputs": outputs}

    def _llm_failed(self, error: Exception) -> None:
        """A server that is gone will not come back mid-run, so stop waiting on it."""
        if isinstance(error, LLMFatal):
            self.log.decision("llm", f"LLM stopped working ({str(error)[:160]}). Carrying on without it.")
            self.llm = None

    # ---------- decisions (conditional edges) ----------

    def route_labelling(self, state: State) -> str:
        if self.llm is None:
            return "check_labels"
        ids = llm_targets(state["labelled"], state.get("relabel_ids"))
        room = labelling_room(self.llm, self.s.batch_size)
        if ids and room == 0:
            self.log.decision("classify", "No LLM budget is left for labelling, so these comments keep their rule labels.")
        if not ids or room == 0:
            return "check_labels"
        return "llm_all" if room >= len(ids) else "llm_budgeted"

    def after_check(self, state: State) -> str:
        return "discover_topics" if state["need_more_topics"] else "analyse"

    def after_investigate(self, state: State) -> str:
        return "write" if state.get("enough_checks") else "investigate"

    def after_verify(self, state: State) -> str:
        can_rewrite = self.llm is not None and self.llm.calls_left > 0 and state["write_round"] < MAX_WRITES
        return "write" if state["problems"] and can_rewrite else "finish"

    # ---------- wiring ----------

    def build(self):
        g = StateGraph(State)
        g.add_node("ingest", self.ingest)
        g.add_node("clean", self.clean)
        g.add_node("plan", self.plan)
        g.add_node("discover_topics", self.discover_topics)
        g.add_node("rule_labels", self.rule_labels)
        g.add_node("llm_all", self.llm_all)
        g.add_node("llm_budgeted", self.llm_budgeted)
        g.add_node("check_labels", self.check_labels)
        g.add_node("analyse", self.analyse)
        g.add_node("investigate", self.investigate)
        g.add_node("write", self.write)
        g.add_node("verify", self.verify_draft)
        g.add_node("finish", self.finish)

        g.add_edge(START, "ingest")
        g.add_edge("ingest", "clean")
        g.add_edge("clean", "plan")
        g.add_edge("plan", "discover_topics")
        g.add_edge("discover_topics", "rule_labels")
        g.add_conditional_edges("rule_labels", self.route_labelling, ["llm_all", "llm_budgeted", "check_labels"])
        g.add_edge("llm_all", "check_labels")
        g.add_edge("llm_budgeted", "check_labels")
        g.add_conditional_edges("check_labels", self.after_check, ["discover_topics", "analyse"])
        g.add_edge("analyse", "investigate")
        g.add_conditional_edges("investigate", self.after_investigate, ["investigate", "write"])
        g.add_edge("write", "verify")
        g.add_conditional_edges("verify", self.after_verify, ["write", "finish"])
        g.add_edge("finish", END)
        return g.compile()

    def graph_mermaid(self) -> str:
        return self.build().get_graph().draw_mermaid()

    def run(self) -> State:
        return self.build().invoke({}, {"recursion_limit": 60})


def _dedupe(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))
