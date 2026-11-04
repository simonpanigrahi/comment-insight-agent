"""Rule-based labels. Free, instant and offline, but only a fallback.

Measured on real labelled tweets (see README), rules alone reach about 61 percent accuracy on airline
tweets and 53 percent on product tweets. They read very little sarcasm and only a little Hindi. The
safety rules in risk_type are different: they always run, even with an LLM, because a safety report
must not depend on one model noticing it.
"""
from __future__ import annotations

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_vader = SentimentIntensityAnalyzer()
# Words common in complaints that VADER's general lexicon misses, plus common Hindi-English words.
# Measured gain on airline tweets: accuracy 0.55 to 0.59. No loss on product tweets.
_vader.lexicon.update({
    "delayed": -1.6, "delay": -1.4, "delays": -1.5, "cancelled": -1.8, "canceled": -1.8, "refund": -1.2, "stuck": -1.6,
    "waiting": -0.9, "wait": -0.6, "broken": -2.0, "crash": -1.8, "crashes": -1.8, "crashed": -1.8, "crashing": -1.8,
    "lag": -1.5, "lags": -1.5, "laggy": -1.8, "overheating": -2.2, "overheats": -2.2, "drains": -1.6, "drain": -1.4,
    "dies": -1.6, "useless": -2.2, "nobody": -1.0, "never": -0.8, "unacceptable": -2.4, "rude": -2.2, "lost": -1.5,
    "missing": -1.3, "overpriced": -2.0, "expensive": -1.0, "scam": -2.6, "worst": -3.1,
    "bakwas": -2.5, "bakwaas": -2.5, "bekar": -2.2, "bekaar": -2.2, "kharab": -2.2, "ghatiya": -2.6, "faltu": -2.0, "ganda": -2.0,
    "barbaad": -2.2, "dhoka": -2.6, "pareshan": -1.8, "mehnga": -1.2, "garam": -1.0,
    # In customer service text these words name a department or are mere politeness, so they are not praise.
    # Measured gain on airline tweets: accuracy 0.59 to 0.61. No loss on product tweets.
    "care": 0.0, "support": 0.0, "help": 0.0, "please": 0.0, "thanks": 0.4, "thank": 0.4,
    "mast": 2.2, "badhiya": 2.2, "badiya": 2.2, "zabardast": 2.6, "shandaar": 2.6, "shandar": 2.6, "kamaal": 2.4, "accha": 1.8, "acha": 1.8, "achha": 1.8,
})
_PHRASES = [
    (r"\b(no|zero) (reply|response|answer|help|update)\b", -1.2),
    (r"\b(not|isn't|doesn't|don't|won't|didn't) (work|working|respond|help|load|charge|turn on|connect)\b", -1.5),
    (r"\bstopped working\b", -1.8), (r"\bstill (waiting|no|not)\b", -1.2), (r"\bon hold\b", -1.0), (r"\bthanks for nothing\b", -2.5),
    (r"\b(acha|accha|achha|badhiya|sahi|theek|worth) nahi\b", -1.8), (r"\bnahi (aayi|aaya|mil|chal|uthate)\w*\b", -1.5), (r"\bpaisa vasool\b", 2.0),
]
_HINDI_NEG = ["खराब", "बेकार", "बर्बाद", "जल्दी खत्म", "नहीं आई", "नहीं देता", "गर्म"]
_HINDI_POS = ["शानदार", "अच्छा", "बढ़िया", "बहुत बढ़िया"]
_ODIA_POS = ["ସୁନ୍ଦର", "ଭଲ"]
_ODIA_NEG = ["ଖରାପ", "ସରିଯାଉଛି", "ଆସିନାହିଁ"]

QUESTION_START = re.compile(r"^\s*(?:\w+[,:]\s*)?(?:what(?! an? )|when|where|which|who|how|why|is|are|does|do|did|can|could|will|would|should|kya|kab|kahan|kaise|kitne|kitna)\b", re.I)
SUGGESTION_RE = re.compile(r"\b(please add|pls add|plz add|should add|should include|add an?\b|bring back|wish (it|there|you)|would be nice|need a proper|give us|let us|please (launch|offer|open|include)|feature request)\b", re.I)
PURCHASE_RE = re.compile(r"\b(take my money|pre-?book(ed)?|pre-?order(ed)?|ordering|added to cart|planning to buy|going to buy|will (buy|upgrade)|buying this|want to buy|just bought)\b", re.I)

# Tested on 23,000 real tweets, where looser patterns fired on "Burning Man", "money to burn" and "my head will explode".
# These word lists are narrow. They catch common wordings, not every way a person might describe harm.
RISK_PATTERNS = {
    "safety": re.compile(
        r"\bburn(ed|t|s)? (my|his|her|their|our) (\w+ )?(hand|hands|finger|fingers|thumb|palm|skin|arm|leg|thigh|lap|face|ear|pocket|table|desk|bed|sofa|charger|port|cable)\b|\bburn(ed|t|s)? (a|the) (hole|mark|skin|port|charger|cable)\b|\bleft a burn\b|\bburn marks?\b|\bburning (smell|plastic)\b|\bsmell\w* (of )?burn|"
        r"\b(caught|catch|catches|catching) fire\b|\bfire hazard\b|\bin flames\b|\b(saw|see|seeing) smoke\b|\bsmoke (from|came|comes|coming|out)\b|"
        r"\b(phone|battery|charger|device|port|unit)( is| was| started| began)? smoking\b|\bstarted smoking\b|\bexplod(ed|es)\b|\bmelted\b|\b(started|starting|began) to melt\b|"
        r"\bswollen\b|\bbulging\b|\belectric shock\b|\bgot a (small |big )?shock\b|\b(i|me|my \w+) (got|was|am|were) injured\b|\btoo hot to hold\b|"
        r"\bhaath jal|\bjalne ki\b|\bblast ho\b|\baag lag|\bdhuan\b|\bphat gay|\bcurrent lag|"
        r"जल गया|जल गई|जलने की|फट गई|फट गया|आग लग|धुआं|धुआँ|करंट लग|ଜଳିଗଲା|ନିଆଁ ଲାଗି|ଧୂଆଁ", re.I),
    "legal": re.compile(
        r"\bconsumer (court|forum)\b|\blegal notice\b|\blawsuit\b|\bfile a case\b|\bchargeback\b|\bsue (you|them|the)\b|"
        r"\bcheated by\b|\b(this is|is a|it'?s a) (a )?(fraud|scam)\b|\bfraud(ulent)? (company|store|seller|charge)\b|\bdhoka\b|धोखा|ठग लिया", re.I),
    "privacy": re.compile(
        r"\b(account|phone|data|id) (got |was |has been )?hacked\b|\bhacked my\b|\bdata leak\w*|\bprivacy (nightmare|breach|issue)\b|\bstolen data\b|\bspyware\b|\bsending my data\b", re.I),
    "reputation": re.compile(r"\bboycott\b|\bclass action\b|\brecall this\b", re.I),
}
RISK_LABEL = {"safety": "Possible safety", "legal": "Possible legal or fraud", "privacy": "Possible privacy or security", "reputation": "Reputation"}


def score(text: str) -> float:
    """A sentiment score from -1 (negative) to +1 (positive)."""
    value = _vader.polarity_scores(text)["compound"]
    for pattern, weight in _PHRASES:
        if re.search(pattern, text, re.I):
            value = (min(value, 0.0) if weight < 0 else max(value, 0.0)) + weight / 3
    if any(w in text for w in _HINDI_NEG + _ODIA_NEG):
        value -= 0.6
    elif any(w in text for w in _HINDI_POS + _ODIA_POS):
        value += 0.6
    # A plain question with no strong feeling is neutral.
    if text.rstrip().endswith("?") and abs(value) < 0.4:
        value = 0.0
    return max(-1.0, min(1.0, value))


def sentiment(text: str) -> str:
    s = score(text)
    return "positive" if s >= 0.05 else "negative" if s <= -0.05 else "neutral"


def intent(text: str, label: str) -> str:
    # A question ends in "?", or opens like one and is a single plain sentence of a few words.
    opens_like_one = bool(QUESTION_START.match(text)) and len(text.split()) >= 4 and not re.search(r"[.!]", text)
    is_question = text.rstrip().endswith("?") or opens_like_one
    # An angry "why does it keep dying??" is a complaint in the shape of a question.
    angry = label == "negative" and (re.match(r"\s*why\b", text, re.I) or score(text) <= -0.4)
    if is_question and not angry:
        return "question"
    if SUGGESTION_RE.search(text):
        return "suggestion"
    if PURCHASE_RE.search(text):
        return "purchase_intent"
    return {"negative": "complaint", "positive": "praise"}.get(label, "other")


def risk_type(text: str) -> str:
    """Returns safety, legal, privacy or reputation if the text matches, otherwise an empty string."""
    for name, pattern in RISK_PATTERNS.items():
        if pattern.search(text):
            return name
    return ""


def reason_snippet(text: str, label: str) -> str:
    """The clause that carries the feeling, copied word for word, so it can never be invented."""
    # Split at sentence marks, but not inside something like "v1.2".
    clauses = [c.strip(" ,.-:") for c in re.split(r"[.!?;,](?:\s|$)|\bbut\b|\bhowever\b|\band\b", text) if c.strip(" ,.-:")]
    clauses = [c for c in clauses if len(c.split()) >= 3] or clauses
    if not clauses:
        return ""
    direction = {"positive": 1, "negative": -1}.get(label, 0)
    best = max(clauses, key=lambda c: score(c) * direction) if direction else clauses[0]
    return " ".join(best.split()[:12])
