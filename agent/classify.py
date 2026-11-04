"""Step 5: label every distinct comment with sentiment, reason, intent, topic and urgency.

The labels come from three workers. The graph chooses who labels what, from the budget and from measured quality:

  LLM      best quality, but slow or limited. Reads sarcasm and Indian languages.
  Student  a tiny classifier trained on the LLM's own labels. Instant, and runs on a CPU. This is
           knowledge distillation in its simplest form: a big model teaches a small one.
  Rules    free and offline, the fallback when no LLM is available.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from . import clean, prompts, rules
from .llm import LLM, BudgetExceeded, LLMError, LLMFatal
from .runlog import RunLog
from .schemas import LabelBatch
from .topics import TopicModel, match_topic

SCRIPT_LANGS_NOT_READ_BY_RULES = {"Bengali", "Gurmukhi", "Gujarati", "Tamil", "Telugu", "Kannada", "Malayalam", "Urdu/Arabic"}
MIN_LABELS_FOR_STUDENT = 300  # below this, a test of the student on held-out labels means little
CALLS_KEPT_FOR_LATER_STEPS = 12  # investigating and writing still need the LLM after labelling
MAX_FAILED_BATCHES = 5  # this many failures in a row means the model is not coping, so stop spending time on it
RETRY_BATCH = 5
# A long quote may have one small slip in it, but a missing negation would reverse its meaning.
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "without", "isn", "wasn", "aren", "weren", "don", "doesn", "didn", "won", "can", "cant", "couldn", "wouldn", "shouldn",
                  "nahi", "nahin", "na", "mat", "नहीं", "न", "मत", "ନାହିଁ", "ନୁହେଁ"}


def apply_rules(unique: pd.DataFrame, model: TopicModel) -> pd.DataFrame:
    """Give every comment a first label from rules. Free, instant, and later improved where possible."""
    out = unique.copy()
    texts = out["clean_text"].tolist()
    scores = [rules.score(t) for t in texts]
    out["sentiment"] = ["positive" if s >= 0.05 else "negative" if s <= -0.05 else "neutral" for s in scores]
    out["intent"] = [rules.intent(t, s) for t, s in zip(texts, out["sentiment"])]
    out["reason"] = [rules.reason_snippet(t, s) for t, s in zip(texts, out["sentiment"])]
    out["evidence"] = out["reason"]
    out["topic"] = model.assign(texts, scores)
    out["risk"] = [rules.risk_type(t) for t in texts]
    out["llm_doubted"] = False
    out["source"] = "rules"
    # Rules cannot read most Indian scripts, and a comment held back from the LLM has only these labels. A person should check both.
    out["needs_review"] = out["lang"].isin(SCRIPT_LANGS_NOT_READ_BY_RULES) | out["hold_from_llm"].astype(bool)
    for col in ("sentiment", "intent", "topic"):
        out[f"{col}_rule"] = out[col]
    return out.set_index("id", drop=False)


def evidence_is_real(evidence: str, text: str) -> bool:
    """The quote must really be in the comment. This one check stops invented reasons at the comment level."""
    quote, comment = clean.words(evidence), clean.words(text)
    if not quote:
        return False
    n = len(quote)
    if any(comment[i:i + n] == quote for i in range(len(comment) - n + 1)):
        return True
    if n < 6:  # a short quote must match exactly, because one wrong word changes it too much
        return False
    present = set(comment)
    missing = [w for w in quote if w not in present]
    return len(missing) <= 1 and not (set(missing) & NEGATION_WORDS)


def label_tokens(provider: str, n: int) -> int:
    """Room for the reply. Indian scripts use many more tokens per word, and a reply that is cut off is wasted."""
    return 110 * n + 100 if provider == "ollama" else 150 * n + 100


def llm_targets(df: pd.DataFrame, only_ids: list[str] | None = None) -> list[str]:
    """Which comments may go to the LLM. A comment that looks like prompt injection stays out, even if it is kept for its safety words."""
    ids = list(df.index) if only_ids is None else list(only_ids)
    return [i for i in ids if not df.at[i, "hold_from_llm"]]


def labelling_room(llm: LLM, batch_size: int) -> int:
    """How many comments the remaining calls can label, keeping some calls for later steps and for retries."""
    return max(0, int((llm.calls_left - CALLS_KEPT_FOR_LATER_STEPS) * 0.9)) * batch_size


def _time_left(started: float, sent: int, total: int) -> str:
    if not sent or sent >= total:
        return ""
    seconds = (time.time() - started) / sent * (total - sent)
    return f", about {seconds / 60:.0f} minutes left" if seconds >= 90 else f", about {seconds:.0f} seconds left"


class Labeller:
    def __init__(self, llm: LLM, log: RunLog, batch_size: int):
        self.llm, self.log, self.batch_size = llm, log, batch_size
        self.grounding_failures = 0
        self.failed_in_a_row = 0

    def _no_calls_left(self) -> bool:
        return self.llm.calls_left <= CALLS_KEPT_FOR_LATER_STEPS

    def label(self, df: pd.DataFrame, ids: list[str], taxonomy: list[str], note: str = "") -> set[str]:
        """Send comments to the LLM in batches and check every result. Returns the ids that were labelled."""
        done: set[str] = set()
        second_chance: list[str] = []
        total = len(ids)
        started = time.time()
        for number, i in enumerate(range(0, total, self.batch_size), start=1):
            if self._no_calls_left():
                break
            ok, retry = self._one_batch(df, ids[i:i + self.batch_size], taxonomy, last_try=False)
            done |= ok
            second_chance += retry
            sent = min(i + self.batch_size, total)
            if number <= 3 or number % 5 == 0 or sent == total:
                self.log.progress(f"{note}: {sent:,}/{total:,} comments, {self.llm.calls:,} LLM calls so far{_time_left(started, sent, total)}")
        # Missing ids or quotes not found in the comment get one more try in a small batch.
        for i in range(0, len(second_chance), RETRY_BATCH):
            if self._no_calls_left():
                break
            ok, _ = self._one_batch(df, second_chance[i:i + RETRY_BATCH], taxonomy, last_try=True)
            done |= ok
        return done

    def _one_batch(self, df, batch, taxonomy, last_try) -> tuple[set[str], list[str]]:
        # Short ids for this batch only: small models copy c1 reliably, and long platform ids waste tokens.
        real_id = {f"c{n}": cid for n, cid in enumerate(batch, start=1)}
        items = [(short, df.at[cid, "clean_text"]) for short, cid in real_id.items()]
        error = "the reply had no usable labels"
        got = {}
        try:
            reply = self.llm.ask(prompts.LABEL_SYSTEM, prompts.label_user(taxonomy, items, last_try), LabelBatch, purpose="label", max_tokens=label_tokens(self.llm.s.provider, len(batch)))
            got = {real_id[r.id]: r for r in reply.results if r.id in real_id}
        except BudgetExceeded:
            return set(), []
        except LLMFatal:
            raise
        except LLMError as e:
            error = str(e)
        if not got:
            self.log.event("classify", batch_failed=error[:200], ids=batch)
            self.failed_in_a_row += 1
            if self.failed_in_a_row >= MAX_FAILED_BATCHES:
                raise LLMFatal(f"{MAX_FAILED_BATCHES} batches failed in a row. Last problem: {error[:150]}")
            return set(), [] if last_try else list(batch)
        self.failed_in_a_row = 0

        ok, retry = set(), []
        for cid in batch:
            r = got.get(cid)
            if r is None:
                retry.append(cid)
                continue
            text = df.at[cid, "clean_text"]
            real = evidence_is_real(r.evidence, text)
            if not real and not last_try:
                retry.append(cid)
                continue
            if not real:  # still no real quote on the second try: keep the label but flag it for a person
                self.grounding_failures += 1
            df.at[cid, "sentiment"] = r.sentiment
            df.at[cid, "intent"] = r.intent
            df.at[cid, "topic"] = match_topic(r.topic, taxonomy, df.at[cid, "topic_rule"])
            df.at[cid, "reason"] = r.reason.strip()[:120] or rules.reason_snippet(text, r.sentiment)
            df.at[cid, "evidence"] = r.evidence.strip()[:160] if real else ""
            df.at[cid, "source"] = "llm"
            df.at[cid, "needs_review"] = not real
            _apply_urgency(df, cid, r.urgent)
            ok.add(cid)
        return ok, ([] if last_try else retry)


def _apply_urgency(df: pd.DataFrame, cid: str, urgent: bool) -> None:
    """The LLM can add to the alert list, but it cannot take anything off. A rule hit it doubts stays, with the doubt noted for the person who reads it."""
    risk = df.at[cid, "risk"]
    if urgent:
        df.at[cid, "risk"] = risk or "llm_flagged"
        df.at[cid, "llm_doubted"] = False
    elif risk in ("safety", "legal", "privacy"):
        df.at[cid, "llm_doubted"] = True


def _student():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000),
                         LogisticRegression(max_iter=1000, class_weight="balanced", C=4))


def student_vs_rules(df: pd.DataFrame, labelled_random: list[str], column: str) -> tuple[bool, float, float]:
    """Should the student replace rules for this column? Test both on LLM labels the student has not seen."""
    y = df.loc[labelled_random, column]
    if len(y) < MIN_LABELS_FOR_STUDENT or y.nunique() < 2 or y.value_counts().min() < 2:
        return False, 0.0, 0.0
    train_ids, test_ids = train_test_split(labelled_random, test_size=0.3, random_state=0, stratify=y)
    model = _student().fit(df.loc[train_ids, "clean_text"], df.loc[train_ids, column])
    student_f1 = f1_score(df.loc[test_ids, column], model.predict(df.loc[test_ids, "clean_text"]), average="macro", zero_division=0)
    rules_f1 = f1_score(df.loc[test_ids, column], df.loc[test_ids, f"{column}_rule"], average="macro", zero_division=0)
    return student_f1 > rules_f1, float(student_f1), float(rules_f1)


def label_budgeted(df: pd.DataFrame, ids: list[str], taxonomy: list[str], labeller: Labeller, room: int, log: RunLog) -> dict:
    """There are too few calls for every comment, so spend them where they help most."""
    hard = [i for i in ids if df.at[i, "lang"] != "English" or df.at[i, "risk"]]
    hard_set = set(hard)
    easy = [i for i in ids if i not in hard_set]
    first = hard[:int(room * 0.3)]
    n_random = min(max(int(room * 0.4), MIN_LABELS_FOR_STUDENT), room - len(first), len(easy))
    random_ids = np.random.default_rng(0).permutation(easy)[:n_random].tolist() if easy else []
    log.decision("classify", f"The budget covers about {room:,} of {len(ids):,} comments. Plan: {len(first):,} hard ones first (Indian languages and risk words), "
                             f"{len(random_ids):,} at random to train and test a small student model, then whichever comments the student is least sure about.")
    labeller.label(df, first, taxonomy, note="hard comments")
    done_random = labeller.label(df, random_ids, taxonomy, note="random sample")
    return _fill_with_student(df, ids, sorted(done_random), labeller, taxonomy, log)


def _fill_with_student(df, ids, random_ids, labeller, taxonomy, log) -> dict:
    labelled = [i for i in ids if df.at[i, "source"] == "llm"]
    rest = [i for i in ids if df.at[i, "source"] != "llm"]
    spare = (labeller.llm.calls_left - CALLS_KEPT_FOR_LATER_STEPS) * labeller.batch_size
    can_teach = len(random_ids) >= MIN_LABELS_FOR_STUDENT and df.loc[labelled, "sentiment"].nunique() >= 2
    if rest and spare > 0:
        if can_teach:  # spend what is left on the comments the student finds hardest. This is active learning.
            model = _student().fit(df.loc[labelled, "clean_text"], df.loc[labelled, "sentiment"])
            probs = np.sort(model.predict_proba(df.loc[rest, "clean_text"]), axis=1)
            margin = probs[:, -1] - probs[:, -2]
            unsure = [rest[i] for i in margin.argsort()[:spare]]
            log.decision("classify", f"Student trained on {len(labelled):,} LLM labels. Sending the {len(unsure):,} comments it is least sure about to the LLM.")
            labeller.label(df, unsure, taxonomy, note="unsure comments")
        else:
            log.decision("classify", f"{len(random_ids):,} random labels are too few to test a student (it needs {MIN_LABELS_FOR_STUDENT}), so the remaining budget labels more comments directly.")
            labeller.label(df, rest[:spare], taxonomy, note="more comments")

    # Replace rules with the student only where the student measurably beats rules on held-out LLM labels.
    labelled = [i for i in ids if df.at[i, "source"] == "llm"]
    rest = [i for i in ids if df.at[i, "source"] != "llm"]
    students, preds, sure = {}, {}, None
    for column in ("sentiment", "intent", "topic"):
        wins, s_f1, r_f1 = student_vs_rules(df, random_ids, column)
        students[column] = {"student_macro_f1": round(s_f1, 3), "rules_macro_f1": round(r_f1, 3), "used": bool(wins)}
        if s_f1 or r_f1:
            verdict = "use the student" if wins else "keep the rules"
            log.decision("classify", f"{column}: student macro-F1 {s_f1:.2f} vs rules {r_f1:.2f} on held-out LLM labels, so {verdict} for the remaining {len(rest):,}.")
        if wins and rest and df.loc[labelled, column].nunique() >= 2:
            model = _student().fit(df.loc[labelled, "clean_text"], df.loc[labelled, column])
            preds[column] = model.predict(df.loc[rest, "clean_text"])
            if column == "sentiment":
                sure = model.predict_proba(df.loc[rest, "clean_text"]).max(axis=1)
    if rest:
        _apply_student(df, rest, preds, sure)
    return {"students": students}


def _apply_student(df: pd.DataFrame, rest: list[str], preds: dict, sure) -> None:
    """Write the student's answers, and keep each comment's other fields consistent with them."""
    if "topic" in preds:
        df.loc[rest, "topic"] = preds["topic"]
    texts = df.loc[rest, "clean_text"].tolist()
    if "sentiment" in preds:
        df.loc[rest, "sentiment"] = preds["sentiment"]
        df.loc[rest, "source"] = "student"
        df.loc[rest, "needs_review"] = df.loc[rest, "needs_review"].astype(bool).to_numpy() | (sure < 0.5)
        # The reason and evidence must fit the new sentiment. They are copied from the comment, so they cannot be invented.
        snippets = [rules.reason_snippet(t, s) for t, s in zip(texts, preds["sentiment"])]
        df.loc[rest, "reason"] = snippets
        df.loc[rest, "evidence"] = snippets
    if "intent" in preds:
        df.loc[rest, "intent"] = preds["intent"]
    elif "sentiment" in preds:
        df.loc[rest, "intent"] = [rules.intent(t, s) for t, s in zip(texts, preds["sentiment"])]
    sentiment, intent = df.loc[rest, "sentiment"], df.loc[rest, "intent"]
    df.loc[rest, "intent"] = intent.mask((sentiment == "positive") & (intent == "complaint"), "praise").mask((sentiment == "negative") & (intent == "praise"), "complaint")


def label_summary(df: pd.DataFrame) -> dict:
    """A quality signal that needs no human labels: how often do rules agree with the LLM on what it labelled?"""
    llm_rows = df[df["source"] == "llm"]
    if llm_rows.empty:
        return {}
    return {"rules_llm_agreement": float((llm_rows["sentiment"] == llm_rows["sentiment_rule"]).mean())}
