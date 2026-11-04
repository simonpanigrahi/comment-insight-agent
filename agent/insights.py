"""Steps 7 to 9: decide what to investigate, write the findings, and check them.

The writer may be an LLM, but the checker is plain code. A check that can hallucinate is not a check.
The checker confirms a claim traces back to computed facts. It cannot tell that a fact was misread.
"""
from __future__ import annotations

import re

from . import prompts
from .analyse import Analysis, Fact, dates_in, numbers_in
from .llm import LLM, LLMError
from .schemas import OWNERS, Action, Insights, NextCheck, Point
from .topics import GENERAL, OTHER, match_topic

OWNER_WORDS = [("deliver", "Logistics"), ("order", "Logistics"), ("courier", "Logistics"), ("service", "Customer Support"), ("support", "Customer Support"),
               ("care", "Customer Support"), ("price", "Marketing"), ("offer", "Marketing"), ("worth", "Marketing"), ("value", "Marketing")]
SERIOUS_RISKS = ("safety", "legal", "privacy", "llm_flagged")
CALLS_KEPT_FOR_WRITING = 4
ROUGH_BELOW = 0.9  # if the LLM read less than this share of comments, overall sentiment figures are only a rough estimate

# Each pattern below catches a way of stating a quantity or a cause that a plain number check would miss.
MULTIPLIER_RE = re.compile(r"\b\d[\d,.]*\s*(?:k|m|mn|bn|x|lakhs?|crores?|thousand|million|billion|times|fold)\b|\b\d[\d,.]*-fold\b", re.I)
MONEY_RE = re.compile(r"(?:\brs\.?|\binr\b|₹|\$|€|£)\s*\d[\d,.]*", re.I)
WORD_QUANTITY_RE = re.compile(r"\b(?:doubl(?:ed?|ing)|tripl(?:ed?|ing)|quadrupled?|halved?|twice|thrice|half of|a third|a quarter|two thirds|three quarters|"
                              r"(?:one|two|three|four|five|six|seven|eight|nine)(?: out)? (?:in|of) (?:two|three|four|five|ten)|hundreds of|thousands of|millions of)\b", re.I)
CAUSE_RE = re.compile(r"\b(?:caused?|causing|due to|led to|leads? to|leading to|results? (?:in|from)|resulted (?:in|from)|driven by|owing to|triggered|because of)\b", re.I)
HEDGE_RE = re.compile(r"\b(?:may|might|possibly|perhaps|likely|could|appears?|seems?|suggests?|not|no evidence)\b", re.I)
QUOTE_RE = re.compile(r"[\"“]([^\"“”]+)[\"”]|(?<!\w)['‘]([^'‘’]+)['’](?!\w)")
PAIR_RE = re.compile(r"(\d[\d,.]*)\s*%\s*(?:of\s+[^.,;:]{0,40}?\s+(?:are|were|is|was|being)\s+)?(negative|positive|neutral)\b", re.I)
_GAP = r"(?:about\s+|around\s+|roughly\s+|nearly\s+)?"
MOVE_RE = re.compile(rf"\b(rose|rise|risen|rising|increased?|increasing|grew|grown|growing|climbed?|climbing|jumped?|surged?|went up|fell|fall|fallen|falling|dropped?|dropping|declined?|declining|decreased?|decreasing|went down)\b"
                     rf"[^.;]{{0,60}}?\bfrom\s+{_GAP}(\d[\d,.]*)[^.;\d]{{0,30}}?\bto\s+{_GAP}(\d[\d,.]*)", re.I)
UP_WORDS = ("ros", "ris", "incr", "gr", "climb", "jump", "surg", "went up")


def check_key(check: NextCheck) -> str:
    return f"{check.action}:{check.query}" if check.action == "search" else f"{check.action}:{check.topic}:{check.sentiment}"


def choose_next_check(a: Analysis, llm: LLM | None, done: list[str]) -> tuple[NextCheck, str]:
    """Choose what to look at next, and say who chose. An LLM decides if it can. Otherwise a fixed order of priorities applies."""
    note = ""
    if llm is not None and llm.calls_left > CALLS_KEPT_FOR_WRITING:
        topics = [t for t in a.topics["topic"] if t != OTHER]
        try:
            choice = llm.ask(prompts.INVESTIGATE_SYSTEM, prompts.investigate_user(a.book.as_prompt(_fact_chars(llm) // 2), topics, done), NextCheck, purpose="investigate", max_tokens=200)
            if choice.action == "stop":
                unchecked = [s for s in a.spikes if f"spike_drivers:{s['topic']}:negative" not in done]
                if not unchecked:  # looking into a spike costs nothing, and a report without it would miss the main story
                    return choice, "LLM"
                note = "the LLM chose to stop, but a spike has not been looked into yet"
            elif choice.action == "search":
                if len(choice.query.strip()) >= 2 and check_key(choice) not in done:
                    return choice, "LLM"
                note = "the LLM's search had no new query"
            else:
                topic = match_topic(choice.topic, topics, "")
                choice = choice.model_copy(update={"topic": topic or choice.topic})
                if topic and check_key(choice) not in done:
                    return choice, "LLM"
                note = f"the LLM chose {choice.action} on '{choice.topic[:40]}', which is an unknown topic or was already done"
        except LLMError as e:
            note = f"the LLM could not choose: {str(e)[:80]}"
    return _default_next_check(a, done), "Fixed priority order" + (f" ({note})" if note else "")


def _default_next_check(a: Analysis, done: list[str]) -> NextCheck:
    wanted: list[NextCheck] = []
    for s in a.spikes[:2]:
        wanted.append(NextCheck(action="spike_drivers", topic=s["topic"], why="find out what changed when the spike began"))
    named = [t for t in a.topics["topic"] if t not in (OTHER, GENERAL)]
    for topic in named[:2]:
        wanted.append(NextCheck(action="top_quotes", topic=topic, why="see the comments with the most reach in a high-impact topic"))
    if named:
        wanted.append(NextCheck(action="platform_split", topic=named[0], why="see whether the top problem is tied to one platform"))
        wanted.append(NextCheck(action="time_trend", topic=named[0], why="see whether the top problem is growing"))
    for check in wanted:
        if check_key(check) not in done:
            return check
    return NextCheck(action="stop", why="the priority checks are done")


def _fact_chars(llm: LLM) -> int:
    """A local model has a small context window, and the prompt plus the reply must both fit inside it."""
    return 5500 if llm.s.provider == "ollama" else 14000


def write_insights(a: Analysis, llm: LLM, problems: list[str] | None = None, draft: str | None = None) -> Insights:
    return llm.ask(prompts.WRITE_SYSTEM, prompts.write_user(a.book.as_prompt(_fact_chars(llm)), problems, draft), Insights, purpose="write", max_tokens=1500)


def _headline(a: Analysis, rough: bool) -> tuple[str, list[str]]:
    book = a.book
    total = len(a.rows)
    overall = book.of("overall")[0]
    neg, pos = a.overall["negative"], a.overall["positive"]
    mood = "more negative than positive" if neg > pos else "more positive than negative" if pos > neg else "evenly split"
    mood_text = f"{mood} overall ({neg:,} negative, {pos:,} positive of {total:,})"
    alert = next((x for x in a.alerts if x["risk"] == "safety"), None) or next((x for x in a.alerts if x["risk"] in SERIOUS_RISKS), None)
    urgent = f"{alert['count']:,} {alert['label'].lower()} report{'s need' if alert['count'] != 1 else ' needs'} a person to read {'them' if alert['count'] != 1 else 'it'} today" if alert else ""
    if not rough:
        if alert:
            return f"Comments are {mood_text}, and {urgent}.", [overall.id, alert["fact"]]
        return f"Comments are {mood_text}.", [overall.id]
    # Most of these labels are rough, so lead with what does not depend on overall sentiment.
    if alert:
        return f"{urgent[0].upper()}{urgent[1:]}. Overall sentiment figures are a rough rules estimate.", [alert["fact"]]
    if a.spikes:
        s = a.spikes[0]
        return f"Complaints about {s['topic']} rose about {s['ratio']} times from {s['date']}. Overall sentiment figures are a rough rules estimate.", [s["fact"]]
    return f"Rough rules estimate: comments look {mood_text}.", [overall.id]


def template_insights(a: Analysis) -> Insights:
    """The offline writer. It builds the same structure from the same facts, using fixed sentences."""
    book, points, actions = a.book, [], []
    overall = book.of("overall")[0]
    rough = a.coverage < ROUGH_BELOW
    overall_point = Point(text=overall.text + (" This is a rough rules estimate." if rough else ""), fact_ids=[overall.id])
    if not rough:
        points.append(overall_point)

    for alert in [x for x in a.alerts if x["risk"] in SERIOUS_RISKS]:
        f = book.get(alert["fact"])
        count = f"{alert['count']:,} comment{'s' if alert['count'] != 1 else ''}"
        points.append(Point(text=f"{alert['label']} reports: {count}, few by volume but serious by nature.", fact_ids=[f.id]))
        owner, what = {"safety": ("Engineering", "Have a person read these safety reports now, escalate any that are real, and contact those customers."),
                       "legal": ("Legal", "Have a person read these legal and fraud threats now and contact those customers."),
                       "privacy": ("Engineering", "Have a person read these security and privacy claims now and investigate any that are real.")}.get(alert["risk"], ("Leadership", "Have a person read the comments the LLM flagged as urgent."))
        actions.append(Action(action=what, why=f"{count} report this, and the risk does not depend on volume.", owner=owner, urgency="today", fact_ids=[f.id]))

    surge = next(iter(book.of("surge")), None)
    if surge:
        points.append(Point(text=surge.text, fact_ids=[surge.id]))

    for spike in a.spikes[:2]:
        f = book.get(spike["fact"])
        driver = next(iter(book.of("driver", spike["topic"])), None)
        ids = [f.id] + ([driver.id] if driver else [])
        linked = f" Words that became more common: {', '.join(spike.get('drivers', [])[:4])}." if spike.get("drivers") else ""
        points.append(Point(text=f"{f.text}{linked}", fact_ids=ids))
        actions.append(Action(action=f"Find out what changed on {spike['date']} for {spike['topic']} and tell customers what is being done.",
                              why=f"Negative comments rose about {spike['ratio']} times from that date.", owner="Engineering", urgency="today", fact_ids=ids))
    if rough:
        points.append(overall_point)

    named = a.topics[~a.topics["topic"].isin([OTHER, GENERAL])]
    spiked = {s["topic"] for s in a.spikes}
    for r in named.head(3).itertuples():
        f = next(iter(book.of("topic", r.topic)))
        reason = next(iter([x for x in book.of("reason", r.topic) if "negative" in x.text]), None)
        ids = [f.id] + ([reason.id] if reason else [])
        if r.topic not in spiked:
            why = f"{r.neg_share}% of its {r.count:,} comments are negative."
            what = f"Assign an owner to fix the main complaint in {r.topic}" + (f": {reason.text.split(':', 1)[1].split(' (')[0].strip()}." if reason else ".")
            owner = next((o for w, o in OWNER_WORDS if w in r.topic.lower()), "Product")
            actions.append(Action(action=what, why=why, owner=owner, urgency="this_week", fact_ids=ids))
        points.append(Point(text=f"{r.topic} has the {'highest' if r.Index == named.index[0] else 'next highest'} impact score: {r.neg_share}% of its {r.count:,} comments are negative.", fact_ids=ids))

    # The most liked subject: most positive comments, among topics that are mostly positive or close to it.
    liked = a.topics[(~a.topics["topic"].isin([OTHER, GENERAL])) & (a.topics["pos_share"] >= 40)].sort_values("positive", ascending=False).head(1)
    for r in liked.itertuples():
        f = next(iter(book.of("topic", r.topic)))
        points.append(Point(text=f"{r.topic} is the most liked subject: {r.pos_share}% of its {r.count:,} comments are positive.", fact_ids=[f.id]))

    qs = book.of("question")
    if qs:
        actions.append(Action(action="Publish clear answers to the most common questions, starting with the ones listed in this report.", why="These questions come up repeatedly.",
                              owner="Customer Support", urgency="this_week", fact_ids=[q.id for q in qs[:3]]))
    if not actions:
        actions.append(Action(action="Have a person read a sample of the comments to confirm these findings.", why="No alert, spike or repeated question stood out in this data.", owner="Product", urgency="this_month", fact_ids=[overall.id]))
    if len(points) < 2:
        points.append(Point(text=book.of("intent")[0].text, fact_ids=[book.of("intent")[0].id]))
    headline, headline_ids = _headline(a, rough)
    return Insights(headline=headline, headline_fact_ids=headline_ids, points=points[:7], actions=actions[:8])


def _number(text: str) -> float:
    return float(text.replace(",", "").rstrip("."))


def _matches(value: float, allowed: list[float]) -> bool:
    """A number is fine if it equals a cited number, or is that number sensibly rounded (70.5 may be written as 71)."""
    for x in allowed:
        if value == x:
            return True
        if x != int(x) and abs(value - x) <= (0.5 if value == int(value) else 0.05):
            return True
    return False


def _flat(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _spaced(text: str) -> str:
    return " ".join(text.lower().split())


def _outside_quotes(text: str) -> str:
    return QUOTE_RE.sub(" ", text)


def _quotes_in(text: str) -> list[str]:
    return [a or b for a, b in QUOTE_RE.findall(text)]


def _sentiment_pairs(facts: list[Fact]) -> dict[str, list[float]]:
    """Which percentages the cited facts tie to which sentiment."""
    pairs: dict[str, list[float]] = {"negative": [], "positive": [], "neutral": []}
    for fact in facts:
        text = _outside_quotes(fact.text)
        for number, word in PAIR_RE.findall(text):
            pairs[word.lower()].append(_number(number))
        if "negative share" in text.lower():  # "Negative share by platform: X 38.3% (of 120)"
            pairs["negative"] += [_number(n) for n in re.findall(r"(\d[\d,.]*)%", text)]
    return pairs


def _check_text(where: str, text: str, cited: list[Fact], ids: list[str], topics: list[str], because_counts: bool) -> list[dict]:
    """Everything that can be checked in one piece of writing against the facts it cites."""
    problems: list[dict] = []

    def problem(message: str) -> None:
        problems.append({"where": where, "problem": message})

    cited_ids = ", ".join(ids)
    source = " ".join(f.text for f in cited)
    source_plain = _outside_quotes(source)  # a number inside a customer's quote is not a measured figure
    plain = _outside_quotes(text)
    allowed = numbers_in(source_plain)
    for n in numbers_in(plain):
        if not _matches(n, allowed):
            problem(f"{where} says {n:g}, but that number is not in the facts it cites ({cited_ids}).")
    for d in dates_in(text):
        if d not in source:
            problem(f"{where} says {d}, but that date is not in the facts it cites.")
    for quote in _quotes_in(text):
        if len(quote.strip()) >= 4 and _flat(quote) not in _flat(source) and _flat(quote) != "may be linked":
            problem(f"{where} quotes \"{quote[:60]}\", which is not in the facts it cites.")
    # Forms such as 5k, 3x and 20.5 million read as a smaller number, so they must appear exactly as written.
    for pattern, kind in ((MULTIPLIER_RE, "amount"), (MONEY_RE, "amount of money"), (WORD_QUANTITY_RE, "quantity")):
        for m in pattern.finditer(plain):
            if _spaced(m.group(0)) not in _spaced(source_plain):
                problem(f"{where} says \"{m.group(0)}\", but that {kind} is not in the facts it cites. Use the exact figures from the facts.")
    pairs = _sentiment_pairs(cited)
    for number, word in PAIR_RE.findall(plain):
        if not _matches(_number(number), pairs[word.lower()]):
            problem(f"{where} says {number}% {word.lower()}, but the facts it cites do not give that share for {word.lower()}.")
    for word, a, b in MOVE_RE.findall(plain):
        went_up = word.lower().startswith(UP_WORDS)
        if (went_up and _number(a) > _number(b)) or (not went_up and _number(a) < _number(b)):
            problem(f"{where} says {word} from {a} to {b}, which is the wrong direction.")
    for sentence in re.split(r"(?<=[.!?])\s+", plain):
        cause = CAUSE_RE.search(sentence) or (because_counts and re.search(r"\bbecause\b", sentence, re.I))
        if cause and not HEDGE_RE.search(sentence):
            problem(f"{where} states a cause (\"{cause.group(0)}\") as proven. The facts only show things happening together, so write \"may be linked\".")
    # A claim about a topic must cite a fact about that topic.
    for topic in topics:
        if re.search(rf"\b{re.escape(topic)}\b", text, re.I) and not any(topic.lower() in f.text.lower() for f in cited):
            problem(f"{where} talks about {topic} but cites no fact about {topic}.")
    return problems


def verify(ins: Insights, a: Analysis) -> list[dict]:
    """Check the draft against the facts. Returns a list of problems, each saying where it is and what is wrong."""
    problems: list[dict] = []
    book = a.book
    topics = [t for t in a.topics["topic"] if t not in (OTHER, GENERAL)]

    def check_item(where: str, text: str, ids: list[str], because_counts: bool = True) -> None:
        if not ids:
            problems.append({"where": where, "problem": f"{where} cites no facts. Add the ids of the facts that support it."})
            return
        cited: list[Fact] = []
        for fid in ids:
            fact = book.get(fid)
            if fact is None:
                problems.append({"where": where, "problem": f"{where} cites {fid}, which does not exist."})
            else:
                cited.append(fact)
        problems.extend(_check_text(where, text, cited, ids, topics, because_counts))

    if not ins.headline.strip():
        problems.append({"where": "headline", "problem": "The headline is empty."})
    else:
        check_item("headline", ins.headline, ins.headline_fact_ids)
    for i, p in enumerate(ins.points):
        check_item(f"point {i + 1}", p.text, p.fact_ids)
    for i, x in enumerate(ins.actions):
        # In an action, "because" gives the reason for acting. It is not a claim about what caused what.
        check_item(f"action {i + 1}", f"{x.action} {x.why}", x.fact_ids, because_counts=False)
        if x.owner not in OWNERS:
            problems.append({"where": f"action {i + 1}", "problem": f"action {i + 1} has owner '{x.owner}'. Choose from: {', '.join(OWNERS)}."})
    if len(ins.points) < 2:
        problems.append({"where": "content", "problem": "The draft needs at least 2 points."})
    if not ins.actions:
        problems.append({"where": "content", "problem": "The draft needs at least 1 action."})

    # Every serious alert must lead to an action today. A serious report must not be dropped.
    for f in book.facts:
        if f.alert and not any(f.id in x.fact_ids and x.urgency == "today" for x in ins.actions):
            problems.append({"where": "coverage", "problem": f"{f.id} is an ALERT but no action with urgency today cites it. Add one."})
    return problems


def remove_failed(ins: Insights, problems: list[dict]) -> tuple[Insights, int]:
    """Last resort after the rewrites run out: drop whatever still fails, and say so in the report."""
    bad_points = {int(p["where"].split()[1]) - 1 for p in problems if p["where"].startswith("point ")}
    bad_actions = {int(p["where"].split()[1]) - 1 for p in problems if p["where"].startswith("action ")}
    points = [p for i, p in enumerate(ins.points) if i not in bad_points]
    actions = [x for i, x in enumerate(ins.actions) if i not in bad_actions]
    removed = len(ins.points) - len(points) + len(ins.actions) - len(actions)
    return ins.model_copy(update={"points": points, "actions": actions}), removed


def finalise(ins: Insights, problems: list[dict], a: Analysis) -> tuple[Insights, int, bool]:
    """Apply the guarantees that must hold whatever the writer did. Returns the findings, how many were removed, and whether the fixed writer had to step in."""
    ins, removed = remove_failed(ins, problems)
    fallback = template_insights(a)
    # Too little survived the checker, so the fixed-sentence writer takes over rather than show a near-empty report.
    used_template = len(ins.points) < 2 or not ins.actions
    if used_template:
        return fallback, removed, True
    # A headline that failed its check is replaced, because it is the first thing a reader sees.
    if any(p["where"] == "headline" for p in problems):
        ins = ins.model_copy(update={"headline": fallback.headline, "headline_fact_ids": fallback.headline_fact_ids})
    # A serious alert always gets an action, even if the writer ignored it.
    for f in a.book.facts:
        if f.alert and not any(f.id in x.fact_ids and x.urgency == "today" for x in ins.actions):
            match = next((x for x in fallback.actions if f.id in x.fact_ids), None)
            if match:
                ins.actions.insert(0, match)
    return ins, removed, False
