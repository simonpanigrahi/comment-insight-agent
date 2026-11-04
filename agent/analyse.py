"""Step 6: compute everything the report will claim. All counting is done here, in code.

The LLM never calculates a number. It only sees facts produced here, each with an id, and may only
repeat those facts. This is what makes the verifier in insights.py possible.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from .clean import TOPIC_TOKEN_PATTERN
from .rules import RISK_LABEL
from .topics import GENERAL, OTHER, STOP_WORDS

SURGE_MIN_TOPICS = 4
MAX_SPIKES_SHOWN = 3
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUM_RE = re.compile(r"(?<![\w.,])\d[\d,]*(?:\.\d+)?(?!\w)")


def numbers_in(text: str) -> list[float]:
    text = DATE_RE.sub(" ", text)
    out = []
    for m in NUM_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def dates_in(text: str) -> list[str]:
    return DATE_RE.findall(text)


@dataclass
class Fact:
    id: str
    kind: str
    text: str
    topic: str = ""
    alert: bool = False
    comment_ids: list[str] = field(default_factory=list)


class FactBook:
    def __init__(self):
        self.facts: list[Fact] = []

    def add(self, kind: str, text: str, topic: str = "", alert: bool = False, comment_ids=None) -> Fact:
        fact = Fact(f"F{len(self.facts) + 1}", kind, text, topic, alert, list(comment_ids or [])[:20])
        self.facts.append(fact)
        return fact

    def get(self, fact_id: str) -> Fact | None:
        return next((f for f in self.facts if f.id == fact_id), None)

    def of(self, kind: str, topic: str | None = None) -> list[Fact]:
        return [f for f in self.facts if f.kind == kind and (topic is None or f.topic == topic)]

    def as_prompt(self, max_chars: int = 6000) -> str:
        """The facts as numbered lines, most important kinds first, cut to fit the model's context window."""
        order = ["alert", "overall", "surge", "spike", "driver", "trend", "platform_topic", "search", "quote", "reasons_detail", "topic", "reason", "question", "request", "intent", "platform", "data"]
        lines, used = [], 0
        for fact in sorted(self.facts, key=lambda f: order.index(f.kind) if f.kind in order else len(order)):
            line = f"{fact.id}{' ALERT' if fact.alert else ''}: {fact.text}"
            if used + len(line) > max_chars and not fact.alert:
                continue
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)


def pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def quoted(text: str, limit: int = 140) -> str:
    """Customer words inside a fact. Their own quotation marks are swapped, so the checker can tell where the quote ends."""
    return '"' + re.sub(r'["\u201c\u201d]', "'", str(text)[:limit]) + '"'


def group_phrases(rows: pd.DataFrame, column: str, threshold: float = 0.75, max_unique: int = 800) -> list[dict]:
    """Group near-identical phrases, for example 'battery drains fast' and 'battery drain too fast'.

    Character pieces are compared, so small differences in spelling or word form do not matter. Returns
    groups sorted by how many comments they cover, each with a label, a count and example comment ids.
    """
    rows = rows[rows[column].astype(str).str.strip() != ""]
    if rows.empty:
        return []
    norm = rows[column].astype(str).str.lower().str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    counts = norm.value_counts()
    uniques = counts.index[:max_unique].tolist()
    labels = np.arange(len(uniques))  # each phrase starts as its own group
    try:
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(uniques)
    except ValueError:  # the phrases are too short to compare
        X = None
    if X is not None:
        usable = np.flatnonzero(X.getnnz(axis=1) > 0)  # cosine distance is undefined for an empty vector
        if len(usable) >= 2:
            labels[usable] = len(uniques) + AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average").fit_predict(X[usable].toarray())
    group_of = dict(zip(uniques, labels))
    assigned = norm.map(group_of)
    groups = []
    for g in sorted(set(labels)):
        mask = assigned == g
        members = rows[mask]
        keys = norm[mask]
        top_key = keys.value_counts().idxmax()
        label = members.loc[keys == top_key, column].astype(str).iloc[0]
        groups.append({"label": label, "count": int(mask.sum()), "comment_ids": members["id"].astype(str).tolist()[:20],
                       "engagement": float(members["engagement"].sum())})
    return sorted(groups, key=lambda g: -g["count"])


@dataclass
class Analysis:
    rows: pd.DataFrame  # every usable comment, with labels copied to repeats
    book: FactBook
    received: int = 0
    removed: dict = field(default_factory=dict)
    distinct: int = 0
    overall: dict = field(default_factory=dict)
    topics: pd.DataFrame = None
    intents: dict = field(default_factory=dict)
    platforms: pd.DataFrame = None
    languages: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    coverage: float = 0.0  # the share of analysed comments that the LLM labelled
    spikes: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    requests: list = field(default_factory=list)
    reasons: dict = field(default_factory=dict)
    daily: pd.DataFrame = None
    has_time: bool = False
    has_engagement: bool = False


def merge_labels(comments: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    """Copy each distinct text's labels to all comments that share it."""
    ok = comments[comments["status"] == "ok"].copy()
    cols = ["key", "sentiment", "intent", "topic", "reason", "evidence", "risk", "llm_doubted", "source", "needs_review"]
    return ok.merge(labelled[cols].reset_index(drop=True), on="key", how="left")


def build_analysis(comments: pd.DataFrame, labelled: pd.DataFrame) -> Analysis:
    rows = merge_labels(comments, labelled)
    book = FactBook()
    a = Analysis(rows=rows, book=book, received=len(comments), distinct=len(labelled))
    a.removed = comments["status"].value_counts().drop("ok", errors="ignore").to_dict()
    a.has_time = rows["created_at"].notna().mean() > 0.5
    a.has_engagement = rows["engagement"].sum() > 0
    a.coverage = float((rows["source"] == "llm").mean()) if len(rows) else 0.0

    total = len(rows)
    counts = rows["sentiment"].value_counts()
    a.overall = {s: int(counts.get(s, 0)) for s in ("positive", "neutral", "negative")}
    a.intents = rows["intent"].value_counts().to_dict()
    a.languages = rows["lang"].value_counts().to_dict()

    removed_text = ", ".join(f"{v:,} {k.replace('_', ' ')}" for k, v in a.removed.items()) or "nothing"
    labelled_by = f"The LLM labelled {math.floor(a.coverage * 1000) / 10:g}% of the analysed comments." if a.coverage else "Labels come from rules, not from an LLM."
    book.add("data", f"Received {a.received:,} comments. Removed {removed_text}. Analysed {total:,} comments, of which {a.distinct:,} are distinct texts. {labelled_by}")
    book.add("overall", f"Sentiment across {total:,} comments: {pct(a.overall['negative'], total)}% negative ({a.overall['negative']:,}), "
                        f"{pct(a.overall['positive'], total)}% positive ({a.overall['positive']:,}), {pct(a.overall['neutral'], total)}% neutral ({a.overall['neutral']:,}).")
    intents_text = ", ".join(f"{k.replace('_', ' ')} {pct(v, total)}%" for k, v in a.intents.items())
    book.add("intent", f"What people are doing in their comments: {intents_text}.")

    _alerts(a)
    _topics(a)
    _spikes(a)
    _reasons(a)
    _questions_and_requests(a)
    _platforms(a)
    return a


def _alerts(a: Analysis) -> None:
    rows = a.rows[a.rows["risk"].astype(str) != ""]
    for risk, group in rows.groupby("risk"):
        label = RISK_LABEL.get(risk, "Flagged by the LLM as urgent")
        doubted = int(group["llm_doubted"].astype(bool).sum())
        if (group["source"] == "llm").all() and not doubted:  # an LLM has read all of these and agreed
            label = label.replace("Possible s", "S").replace("Possible l", "L").replace("Possible p", "P")
        top = group.sort_values("engagement", ascending=False).drop_duplicates("key")
        first = group["created_at"].min()
        when = f" First seen {first.date()}." if pd.notna(first) else ""
        reach = f" Total engagement {int(group['engagement'].sum()):,}." if a.has_engagement else ""
        doubt = f" The LLM judged {doubted:,} of them false alarms, but they stay on this list for a person to check." if doubted else ""
        # The fact holds counts only. Example comments go on the report's alert card, and not into the writer's prompt.
        fact = a.book.add("alert", f"{label} reports: {len(group):,} comment{'s' if len(group) != 1 else ''}.{reach}{when}{doubt}",
                          alert=risk in ("safety", "legal", "privacy", "llm_flagged"), comment_ids=top["id"].astype(str).tolist())
        a.alerts.append({"risk": risk, "label": label, "count": len(group), "doubted": doubted, "engagement": float(group["engagement"].sum()), "examples": top[["id", "clean_text", "engagement"]].head(4).to_dict("records"), "fact": fact.id})
    order = {"safety": 0, "legal": 1, "privacy": 2, "llm_flagged": 3, "reputation": 4}
    a.alerts.sort(key=lambda x: order.get(x["risk"], 9))


def _topics(a: Analysis) -> None:
    rows = a.rows
    total = len(rows)
    table = []
    for topic, g in rows.groupby("topic"):
        neg = int((g["sentiment"] == "negative").sum())
        pos = int((g["sentiment"] == "positive").sum())
        mean_eng = float(g.loc[g["sentiment"] == "negative", "engagement"].mean()) if neg else 0.0
        # Impact: how many unhappy comments, scaled up when they travel further. A simple, explainable heuristic, not a validated measure.
        impact = neg * (1 + math.log10(1 + mean_eng))
        table.append(dict(topic=topic, count=len(g), share=pct(len(g), total), negative=neg, positive=pos, neutral=len(g) - neg - pos,
                          neg_share=pct(neg, len(g)), pos_share=pct(pos, len(g)), impact=round(impact, 1), trend=_trend(g) if a.has_time else ""))
    a.topics = pd.DataFrame(table).sort_values("impact", ascending=False).reset_index(drop=True)
    for r in a.topics.itertuples():
        if r.topic == OTHER:
            continue
        trend = f" Negative comments are {r.trend}." if r.trend else ""
        a.book.add("topic", f"{r.topic}: {r.count:,} comments ({r.share}% of all), {r.neg_share}% negative, {r.pos_share}% positive, impact score {r.impact:,}.{trend}", topic=r.topic)


def _trend(g: pd.DataFrame) -> str:
    neg = g[(g["sentiment"] == "negative") & g["created_at"].notna()]
    if len(neg) < 20:
        return ""
    days = neg["created_at"].dt.floor("D")
    mid = days.min() + (days.max() - days.min()) / 2
    early, late = (days <= mid).sum(), (days > mid).sum()
    if late >= early * 1.3:
        return "rising"
    if early >= late * 1.3:
        return "falling"
    return "steady"


def _daily_negative(rows: pd.DataFrame, topic: str, all_days) -> pd.Series:
    g = rows[(rows["topic"] == topic) & (rows["sentiment"] == "negative") & rows["created_at"].notna()]
    return g.groupby(g["created_at"].dt.floor("D")).size().reindex(all_days, fill_value=0)


def _spikes(a: Analysis) -> None:
    """A spike: a day at least 3 times the usual level so far, with at least 10 negative comments, that stays raised."""
    if not a.has_time:
        return
    rows = a.rows
    dated = rows[rows["created_at"].notna()]
    all_days = pd.date_range(dated["created_at"].min().floor("D"), dated["created_at"].max().floor("D"))
    if len(all_days) < 6:
        return
    daily = {}
    for topic in a.topics["topic"]:
        series = _daily_negative(rows, topic, all_days)
        daily[topic] = series
        if topic in (OTHER, GENERAL):
            continue
        for i in range(3, len(series)):
            usual = float(series.iloc[:i].median())
            if series.iloc[i] >= max(10, 3 * max(usual, 1.0)) and series.iloc[i:i + 3].mean() >= max(6, 2 * max(usual, 1.0)):
                while i > 3 and series.iloc[i - 1] >= 2 * max(usual, 1.0):  # the rise may have begun a day earlier
                    i -= 1
                before, after = float(series.iloc[:i].mean()), float(series.iloc[i:].mean())
                day = series.index[i].date()
                ratio = round(after / max(before, 0.1), 1)
                if ratio < 1.5:  # one busy day that did not last is not a spike
                    break
                fact = a.book.add("spike", f"Spike in {topic}: negative comments rose from about {before:.1f} per day to about {after:.1f} per day from {day} ({ratio} times).", topic=topic)
                a.spikes.append({"topic": topic, "date": str(day), "before": before, "after": after, "ratio": ratio, "fact": fact.id})
                break
    a.daily = pd.DataFrame(daily)
    # Many topics rising together points to one wider event, so say that instead of listing every topic.
    if len(a.spikes) >= SURGE_MIN_TOPICS:
        dates = sorted(s["date"] for s in a.spikes)
        a.book.add("surge", f"Negative comments rose in {len(a.spikes)} topics at about the same time, starting around {dates[len(dates) // 2]}. A wider event may be behind this, rather than separate problems in each topic.")
    a.spikes = sorted(a.spikes, key=lambda s: -(s["after"] - s["before"]))[:MAX_SPIKES_SHOWN]


def _prompt_safe(rows: pd.DataFrame) -> pd.DataFrame:
    """Facts reach the writer's prompt, so a comment held back from the LLM must stay out of them."""
    return rows[~rows["hold_from_llm"].astype(bool)] if "hold_from_llm" in rows else rows


def _reasons(a: Analysis) -> None:
    safe = _prompt_safe(a.rows)
    for topic in a.topics["topic"].head(10):
        g = safe[safe["topic"] == topic]
        entry = {}
        for sentiment in ("negative", "positive"):
            groups = group_phrases(g[g["sentiment"] == sentiment], "reason")[:5]
            entry[sentiment] = groups
            if topic != OTHER:
                for grp in groups[:1 if sentiment == "positive" else 2]:
                    if grp["count"] >= 3:
                        a.book.add("reason", f"Top {sentiment} reason in {topic}: {quoted(grp['label'])} ({grp['count']:,} comments).", topic=topic, comment_ids=grp["comment_ids"])
        a.reasons[topic] = entry


def _questions_and_requests(a: Analysis) -> None:
    rows = _prompt_safe(a.rows)
    a.questions = [g for g in group_phrases(rows[rows["intent"] == "question"], "clean_text", threshold=0.8) if g["count"] >= 2][:12]
    a.requests = [g for g in group_phrases(rows[rows["intent"] == "suggestion"], "clean_text", threshold=0.8) if g["count"] >= 2][:10]
    for g in a.questions[:6]:
        a.book.add("question", f"Common question ({g['count']:,} comments): {quoted(g['label'], 120)}", comment_ids=g["comment_ids"])
    for g in a.requests[:4]:
        a.book.add("request", f"Feature request ({g['count']:,} comments): {quoted(g['label'], 120)}", comment_ids=g["comment_ids"])


def _platforms(a: Analysis) -> None:
    rows = a.rows[a.rows["platform"] != "unknown"]
    if rows["platform"].nunique() < 2:
        a.platforms = pd.DataFrame()
        return
    t = rows.groupby("platform").agg(count=("id", "count"), negative=("sentiment", lambda s: (s == "negative").sum()), positive=("sentiment", lambda s: (s == "positive").sum())).reset_index()
    t["neg_share"] = (100 * t["negative"] / t["count"]).round(1)
    t["pos_share"] = (100 * t["positive"] / t["count"]).round(1)
    a.platforms = t.sort_values("count", ascending=False).reset_index(drop=True)
    text = ", ".join(f"{r.platform} {r.neg_share}% (of {r.count:,})" for r in a.platforms.itertuples())
    a.book.add("platform", f"Negative share by platform: {text}.")


# ---- Tools the investigator can call. Each one adds new facts, and each reads only real data. ----

def tool_spike_drivers(a: Analysis, topic: str) -> str:
    spike = next((s for s in a.spikes if s["topic"] == topic), None)
    if not spike:
        return f"No spike was found in {topic}."
    g = a.rows[(a.rows["topic"] == topic) & (a.rows["sentiment"] == "negative") & a.rows["created_at"].notna()]
    start = pd.Timestamp(spike["date"])
    before, after = g[g["created_at"] < start], g[g["created_at"] >= start]
    if len(before) < 5 or len(after) < 5:
        return f"Too few comments around the {topic} spike to compare."
    stop = set(STOP_WORDS)
    tokens = lambda s: {w for w in re.findall(TOPIC_TOKEN_PATTERN, s.lower()) if w not in stop}
    b, c = Counter(), Counter()
    for t in before["clean_text"]:
        b.update(tokens(t))
    for t in after["clean_text"]:
        c.update(tokens(t))
    # Rank words by how much their share of comments grew, and ignore words that only look big because they were rare.
    scored = []
    for w, n in c.items():
        share_after, share_before = n / len(after), b[w] / len(before)
        if n >= 5 and share_after >= 2 * share_before:
            scored.append((w, share_after - share_before))
    top = [w for w, _ in sorted(scored, key=lambda x: (-x[1], x[0]))][:6]
    if not top:
        return f"No words stand out in the {topic} spike."
    ids = after[after["clean_text"].str.lower().str.contains(re.escape(top[0]))]["id"].astype(str).tolist()
    fact = a.book.add("driver", f"Words far more common in negative {topic} comments from {spike['date']} than before it: {', '.join(top)}. This shows they happened together, not that one caused the other.", topic=topic, comment_ids=ids)
    spike["drivers"] = top
    return fact.text


def tool_platform_split(a: Analysis, topic: str) -> str:
    g = a.rows[(a.rows["topic"] == topic) & (a.rows["platform"] != "unknown")]
    if g["platform"].nunique() < 2:
        return f"Only one platform has comments on {topic}."
    t = g.groupby("platform")["sentiment"].agg(n="count", neg=lambda s: pct((s == "negative").sum(), len(s))).sort_values("neg", ascending=False)
    text = ", ".join(f"{p} {r.neg}% (of {int(r.n):,})" for p, r in t.iterrows())
    return a.book.add("platform_topic", f"Negative share for {topic} by platform: {text}.", topic=topic).text


def tool_top_quotes(a: Analysis, topic: str, sentiment: str = "negative") -> str:
    rows = _prompt_safe(a.rows)
    g = rows[(rows["topic"] == topic) & (rows["sentiment"] == sentiment)].sort_values("engagement", ascending=False).drop_duplicates("key").head(3)
    if g.empty:
        return f"No {sentiment} comments on {topic}."
    parts = [f"{quoted(r.clean_text)} (engagement {int(r.engagement):,})" for r in g.itertuples()]
    return a.book.add("quote", f"Most engaged {sentiment} comments on {topic}: " + "; ".join(parts), topic=topic, comment_ids=g["id"].astype(str).tolist()).text


def tool_more_reasons(a: Analysis, topic: str, sentiment: str = "negative") -> str:
    safe = _prompt_safe(a.rows)
    groups = a.reasons.get(topic, {}).get(sentiment) or group_phrases(safe[(safe["topic"] == topic) & (safe["sentiment"] == sentiment)], "reason")[:5]
    if not groups:
        return f"No reasons found for {sentiment} comments on {topic}."
    text = "; ".join(f"{quoted(g['label'])} ({g['count']:,})" for g in groups[:5])
    ids = [i for g in groups[:5] for i in g["comment_ids"][:4]]
    return a.book.add("reasons_detail", f"Reasons for {sentiment} comments on {topic}: {text}.", topic=topic, comment_ids=ids).text


def tool_time_trend(a: Analysis, topic: str) -> str:
    if a.daily is None or topic not in a.daily:
        return "No dates in this data."
    s = a.daily[topic]
    third = max(1, len(s) // 3)
    first, last = float(s.iloc[:third].mean()), float(s.iloc[-third:].mean())
    peak_day, peak = s.idxmax().date(), int(s.max())
    return a.book.add("trend", f"{topic} negative comments per day: {first:.1f} in the first third of the period, {last:.1f} in the last third, peak {peak:,} on {peak_day}.", topic=topic).text


def tool_search(a: Analysis, query: str) -> str:
    """Count the comments that contain a word or short phrase, and show two of them."""
    query = " ".join(str(query).split())[:60]
    if len(query) < 2:
        return "The search needs a word or a short phrase."
    found = _prompt_safe(a.rows)
    found = found[found["clean_text"].str.contains(re.escape(query), case=False, regex=True)]
    if found.empty:
        return f"No comment contains {quoted(query)}."
    negative = pct((found["sentiment"] == "negative").sum(), len(found))
    top = found.sort_values("engagement", ascending=False).drop_duplicates("key").head(2)
    examples = "; ".join(quoted(t) for t in top["clean_text"])
    return a.book.add("search", f"Comments containing {quoted(query)}: {len(found):,} ({pct(len(found), len(a.rows))}% of all), {negative}% negative. Examples: {examples}", comment_ids=top["id"].astype(str).tolist()).text


TOOLS = {"spike_drivers": tool_spike_drivers, "platform_split": tool_platform_split, "top_quotes": tool_top_quotes, "more_reasons": tool_more_reasons, "time_trend": tool_time_trend}


def run_check(a: Analysis, action: str, topic: str = "", sentiment: str = "negative", query: str = "") -> str:
    """Run the check the investigator chose. Every tool reads only real data."""
    if action == "search":
        return tool_search(a, query)
    if action in ("top_quotes", "more_reasons"):
        return TOOLS[action](a, topic, sentiment)
    return TOOLS[action](a, topic)
