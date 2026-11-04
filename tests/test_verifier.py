import pandas as pd
import pytest

from agent.analyse import FactBook, numbers_in
from agent.insights import finalise, remove_failed, verify
from agent.report import csv_safe, md
from agent.schemas import Action, Insights, Point


class FakeAnalysis:
    def __init__(self):
        self.book = FactBook()
        self.topics = pd.DataFrame({"topic": ["Battery", "Camera", "Other"], "neg_share": [70.5, 10.0, 0.0], "pos_share": [10.0, 80.0, 0.0], "count": [400, 300, 300], "positive": [40, 240, 0]})
        self.alerts = []
        self.spikes = []
        self.coverage = 1.0
        self.rows = pd.DataFrame({"x": range(1000)})
        self.overall = {"negative": 441, "positive": 300, "neutral": 259}
        self.book.add("overall", "Sentiment across 1,000 comments: 44.1% negative (441), 30.0% positive (300), 25.9% neutral (259).")
        self.book.add("topic", "Battery: 400 comments (40.0% of all), 70.5% negative, 10.0% positive.", topic="Battery")
        self.book.add("alert", "Safety reports: 12 comments. First seen 2026-09-20.", alert=True)
        self.book.add("spike", "Spike in Battery: negative comments rose from about 6.0 per day to about 22.8 per day from 2026-09-23 (3.8 times).", topic="Battery")
        self.book.add("quote", 'Most engaged negative comments on Battery: "refund me 5000 now, it burned my hand" (engagement 42)', topic="Battery")
        self.book.add("platform", "Negative share by platform: X 38.3% (of 120), Reddit 12.5% (of 80).")
        self.book.add("intent", "What people are doing in their comments: complaint 40.0%, praise 30.0%.")


def good() -> Insights:
    return Insights(headline="Battery is the main complaint across 1,000 comments.", headline_fact_ids=["F1", "F2"],
                    points=[Point(text="Battery: 70.5% of 400 comments are negative.", fact_ids=["F2"]),
                            Point(text="Overall, 44.1% of comments are negative.", fact_ids=["F1"])],
                    actions=[Action(action="Investigate the safety reports.", why="12 comments report harm.", owner="Engineering", urgency="today", fact_ids=["F3"])])


def problems_after(change) -> list[str]:
    d = good()
    change(d)
    return [p["problem"] for p in verify(d, FakeAnalysis())]


def test_a_faithful_draft_passes():
    assert verify(good(), FakeAnalysis()) == []


@pytest.mark.parametrize("text,ids", [
    ("Battery: about 71% of comments are negative.", ["F2"]),  # sensible rounding
    ("Negative comments on Battery rose from about 6.0 per day to about 22.8 per day, 3.8 times, and this may be linked to the update.", ["F4"]),
    ('A customer wrote "it burned my hand" about Battery.', ["F5"]),
    ("On X, 38.3% of comments are negative.", ["F6"]),
    ("Complaints may be caused by the update to Battery.", ["F2"]),  # a hedged cause
    ("This shows they happened together, not that the update caused the Battery complaints.", ["F2"]),  # a cause that is denied
    ('The update and the Battery complaints "may be linked".', ["F2"]),  # the writer was told to use these words
])
def test_faithful_wording_passes(text, ids):
    def change(d):
        d.points[0].text, d.points[0].fact_ids = text, ids
    assert problems_after(change) == []


@pytest.mark.parametrize("change,fragment", [
    (lambda d: setattr(d.points[0], "text", "Battery: 92.5% of comments are negative."), "92.5"),
    (lambda d: setattr(d.points[0], "fact_ids", ["F99"]), "does not exist"),
    (lambda d: setattr(d.points[0], "fact_ids", []), "cites no facts"),
    (lambda d: setattr(d.points[0], "text", 'People said "this phone ruined my whole life" about Battery.'), "quotes"),
    (lambda d: setattr(d.points[0], "text", "People said “this phone ruined my whole life” about Battery."), "quotes"),
    (lambda d: setattr(d.points[0], "text", "People said 'this phone ruined my whole life' about Battery."), "quotes"),
    (lambda d: setattr(d.points[0], "text", "Camera is loved by everyone."), "Camera"),
    (lambda d: setattr(d.points[0], "text", "Battery got worse on 2026-10-01."), "2026-10-01"),
    (lambda d: setattr(d, "headline", "Sales fell 37%."), "37"),
    (lambda d: setattr(d, "headline_fact_ids", []), "cites no facts"),
    (lambda d: setattr(d, "headline", " "), "empty"),
    (lambda d: setattr(d.actions[0], "owner", "Nobody"), "owner"),
    (lambda d: setattr(d.actions[0], "urgency", "this_month"), "ALERT"),
    (lambda d: setattr(d, "points", d.points[:1]), "at least 2 points"),
    (lambda d: setattr(d, "actions", []), "at least 1 action"),
    # A figure dressed up so that a plain number check would read a smaller number
    (lambda d: setattr(d.points[0], "text", "Battery has about 400k comments."), "400k"),
    (lambda d: setattr(d.points[0], "text", "Battery has 70.5 million comments."), "70.5 million"),
    (lambda d: setattr(d.points[0], "text", "Battery complaints are 40.0x higher."), "40.0x"),
    (lambda d: setattr(d.points[0], "text", "Battery cost Rs 400 per comment."), "Rs 400"),
    (lambda d: setattr(d.points[0], "text", "Battery cost ₹400 per comment."), "₹400"),
    # Words that state a quantity
    (lambda d: setattr(d.points[0], "text", "Battery complaints doubled."), "doubled"),
    (lambda d: setattr(d.points[0], "text", "Nine in ten Battery comments are bad."), "Nine in ten"),
    # A real number attached to the wrong thing
    (lambda d: setattr(d.points[0], "text", "Battery: 10.0% of comments are negative."), "negative"),
    (lambda d: setattr(d.points[0], "text", "Battery comments fell from about 6.0 per day to about 22.8 per day."), "wrong direction"),
    # A cause stated as proven
    (lambda d: setattr(d.points[0], "text", "The Battery spike was caused by the update."), "cause"),
    (lambda d: setattr(d.points[0], "text", "Battery comments are negative due to the update."), "cause"),
    (lambda d: setattr(d.points[0], "text", "Battery comments are negative because the update broke it."), "cause"),
])
def test_each_kind_of_invention_is_caught(change, fragment):
    problems = problems_after(change)
    assert any(fragment in p for p in problems), problems


def test_a_number_inside_a_customer_quote_is_not_a_measured_figure():
    def change(d):
        d.points[0].text, d.points[0].fact_ids = "Battery has 5000 complaints.", ["F5"]
    assert any("5000" in p for p in problems_after(change))


def test_because_in_an_action_gives_a_reason_and_is_allowed():
    def change(d):
        d.actions[0].why = "Act because 12 comments report harm."
    assert problems_after(change) == []


def test_numbers_in_ignores_versions_and_dates():
    assert numbers_in("v1.2 on 2026-09-23 had 1,234 comments, 45.5% bad") == [1234.0, 45.5]


def test_failed_statements_are_removed_and_counted():
    d = good()
    d.points.append(Point(text="Made up 99.9% figure.", fact_ids=["F1"]))
    a = FakeAnalysis()
    problems = verify(d, a)
    cleaned, removed = remove_failed(d, problems)
    assert removed == 1 and len(cleaned.points) == 2


def test_finalise_puts_back_a_dropped_alert_action(monkeypatch):
    a = FakeAnalysis()
    d = good()
    d.actions = [Action(action="Do something useful about Battery.", why="It is the main complaint.", owner="Product", urgency="this_week", fact_ids=["F2"])]
    fallback = good()
    monkeypatch.setattr("agent.insights.template_insights", lambda _: fallback)
    ins, _, used_template = finalise(d, verify(d, a), a)
    assert not used_template
    assert ins.actions[0].urgency == "today" and "F3" in ins.actions[0].fact_ids


def test_finalise_uses_the_fixed_writer_when_too_little_survives(monkeypatch):
    a = FakeAnalysis()
    d = good()
    d.points = [Point(text="Made up 99.9% figure.", fact_ids=["F1"]), Point(text="Also made up 88.8% figure.", fact_ids=["F1"])]
    fallback = good()
    monkeypatch.setattr("agent.insights.template_insights", lambda _: fallback)
    ins, removed, used_template = finalise(d, verify(d, a), a)
    assert ins is fallback and used_template and removed == 2


def test_finalise_replaces_a_failed_headline(monkeypatch):
    a = FakeAnalysis()
    d = good()
    d.headline = "Sales fell 37%."
    fallback = good()
    monkeypatch.setattr("agent.insights.template_insights", lambda _: fallback)
    ins, _, _ = finalise(d, verify(d, a), a)
    assert ins.headline == fallback.headline


def test_markdown_cells_cannot_break_the_table_or_carry_html():
    assert md("a|b <script>\nnext") == "a\\|b &lt;script> next"


@pytest.mark.parametrize("cell,expected", [("=1+1", "'=1+1"), ("+91 call", "'+91 call"), ("-5 stars", "'-5 stars"), ("@cmd", "'@cmd"), ("fine", "fine"), ("", "")])
def test_spreadsheet_formulas_are_defused(cell, expected):
    assert csv_safe(pd.Series([cell])).iloc[0] == expected
