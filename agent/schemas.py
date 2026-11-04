"""The exact shape of every LLM reply. If a reply does not fit, it is rejected and we ask again.

Real models make small slips, so the validators below repair what is safe to repair: "Negative" for
negative, null for an empty reason, "f2" for F2. One bad comment in a batch is dropped and retried
alone, so it does not cost the whole batch.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

Sentiment = Literal["positive", "neutral", "negative"]
Intent = Literal["complaint", "question", "praise", "suggestion", "purchase_intent", "other"]
_INTENTS = {"complaint", "question", "praise", "suggestion", "purchase_intent", "other"}
_SENTIMENT_WORDS = {"positive": "positive", "pos": "positive", "negative": "negative", "neg": "negative", "neutral": "neutral", "neu": "neutral", "mixed": "neutral"}
OWNERS = ["Engineering", "Customer Support", "Logistics", "Marketing", "Legal", "Product", "Leadership"]


class CommentLabel(BaseModel):
    # All fields are listed as required, so a local model's schema-guided output includes them. Validation is more forgiving.
    model_config = ConfigDict(json_schema_extra={"required": ["id", "sentiment", "reason", "evidence", "intent", "topic", "urgent"]})

    id: str
    sentiment: Sentiment
    reason: str = ""
    evidence: str = ""
    intent: Intent = "other"
    topic: str = ""
    urgent: bool = False

    @field_validator("id", mode="before")
    @classmethod
    def _id_as_text(cls, v):
        v = str(v).strip().strip("[]").lower()
        return f"c{v}" if v.isdigit() else v

    @field_validator("sentiment", mode="before")
    @classmethod
    def _known_sentiment(cls, v):
        return _SENTIMENT_WORDS.get(str(v).strip().lower(), str(v))

    @field_validator("intent", mode="before")
    @classmethod
    def _known_intent(cls, v):
        v = str(v).strip().lower().replace(" ", "_")
        return v if v in _INTENTS else "other"

    @field_validator("reason", "evidence", "topic", mode="before")
    @classmethod
    def _text_or_empty(cls, v):
        return "" if v is None else str(v)

    @field_validator("urgent", mode="before")
    @classmethod
    def _bool_or_false(cls, v):
        return False if v is None else v


class LabelBatch(BaseModel):
    results: list[CommentLabel]

    @field_validator("results", mode="before")
    @classmethod
    def _keep_the_valid_ones(cls, items):
        good = []
        for item in items if isinstance(items, list) else []:
            try:
                good.append(CommentLabel.model_validate(item))
            except ValidationError:
                pass  # treated as missing, so it gets another try on its own
        return good


class TopicName(BaseModel):
    cluster: int
    name: str


class TopicNames(BaseModel):
    topics: list[TopicName]


class NextCheck(BaseModel):
    """The investigator's choice of what to look at next."""
    action: Literal["spike_drivers", "platform_split", "top_quotes", "more_reasons", "time_trend", "search", "stop"]
    topic: str = ""
    sentiment: Literal["positive", "negative"] = "negative"
    query: str = ""
    why: str = ""

    @field_validator("topic", "query", "why", mode="before")
    @classmethod
    def _text_or_empty(cls, v):
        return "" if v is None else str(v)

    @field_validator("sentiment", mode="before")
    @classmethod
    def _default_sentiment(cls, v):
        return "positive" if str(v).strip().lower() == "positive" else "negative"


def _fact_id(x) -> str:
    """Models write F1, f1, "Fact 1" or a bare 1. All of them mean the same fact."""
    number = re.search(r"\d+", str(x))
    return f"F{int(number.group())}" if number else str(x).strip().upper()


def _fact_ids(v) -> list[str]:
    return [_fact_id(x) for x in (v if isinstance(v, list) else [v] if v else [])]


class Point(BaseModel):
    text: str
    fact_ids: list[str]

    @field_validator("fact_ids", mode="before")
    @classmethod
    def _clean_ids(cls, v):
        return _fact_ids(v)


class Action(BaseModel):
    action: str
    why: str
    owner: str
    urgency: Literal["today", "this_week", "this_month"]
    fact_ids: list[str]

    @field_validator("fact_ids", mode="before")
    @classmethod
    def _clean_ids(cls, v):
        return _fact_ids(v)

    @field_validator("urgency", mode="before")
    @classmethod
    def _known_urgency(cls, v):
        v = str(v).strip().lower().replace(" ", "_")
        return v if v in {"today", "this_week", "this_month"} else "this_week"

    @field_validator("owner", mode="before")
    @classmethod
    def _known_owner(cls, v):
        return next((o for o in OWNERS if o.lower() == str(v).strip().lower()), str(v))


class Insights(BaseModel):
    # Listed as required, so a local model's schema-guided output includes the headline's citations.
    model_config = ConfigDict(json_schema_extra={"required": ["headline", "headline_fact_ids", "points", "actions"]})

    headline: str
    headline_fact_ids: list[str] = []
    points: list[Point]
    actions: list[Action]

    @field_validator("headline_fact_ids", mode="before")
    @classmethod
    def _clean_ids(cls, v):
        return _fact_ids(v)


class Answer(BaseModel):
    answer: str
    comment_ids: list[str]

    @field_validator("comment_ids", mode="before")
    @classmethod
    def _ids_as_text(cls, v):
        return [str(x).strip().strip("[]") for x in (v or [])]
