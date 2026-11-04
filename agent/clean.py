"""Step 2: make the text safe and usable before any model sees it.

Masks personal data, removes links and handles, flags spam and prompt injection, tags the language,
and groups exact repeats so that each distinct text is labelled once and the label is copied to its repeats.
"""
from __future__ import annotations

import html
import re

import pandas as pd

from . import rules

URL_RE = re.compile(r"https?://\S+|www\.\S+|\{link\}|\b[\w-]+\.(?:ly|me|gg|link|xyz)/\S*", re.I)
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"^\s*RT\b:?\s*", re.I)
EMAIL_RE = re.compile(r"[\w.+-]{1,64}@[\w-]{1,64}(?:\.[\w-]{1,64})+")
# Any run of 9 or more digits, with common separators: phone, Aadhaar and card numbers. Dates are left alone.
LONG_NUMBER_RE = re.compile(r"(?<![\w.])(?!\d{4}-\d{2}-\d{2})\+?\(?\d(?:[\s().-]{0,2}\d){8,18}(?!\w)")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

SPAM_RE = re.compile(
    r"(?:win a free|free followers|\bfollowers today\b|earn \S* ?(?:daily|per day|from home)|guaranteed profit|crypto signals|"
    r"click here (?:to|for|now)\b|\breplica (?:watch|bag|phone|shoe)|link in bio|dm me now|use (?:promo )?code\b|\bt\.me/)", re.I)
# Comments are untrusted input. These are the obvious attempts to give orders to whichever model reads them. A filter like this
# can be beaten, so it is only the first line of defence. Tested on 23,000 real tweets to avoid catching real complaints.
INJECTION_RE = re.compile(
    r"(?:ignore|disregard|forget)\s+(?:\w+\s+){0,4}(?:instructions?|prompts?|rules?|everything (?:said|above)|the above)\b"
    r"|\bsystem\s*(?:prompt|message)?\s*:\s*(?:you|ignore|mark|output|rate|classify)\b"
    r"|\byou are now (?:in |an? |the )?(?:admin|developer|root|assistant|ai|model)\b|\badmin mode\b|<\|im_start\|>|\bnew task\s*:"
    r"|\bact as\b.{0,20}\b(?:model|assistant|ai)\b|निर्देश.{0,20}(?:अनदेखा|भूल)|(?:ignore|bhool).{0,20}nirdesh|nirdesh.{0,20}(?:ignore|bhool)", re.I)

# The same account posting the same text this many times looks like a bot, not a person.
BOT_REPEAT_MIN = 3
PLACEHOLDER_AUTHORS = {"", "anonymous", "anon", "unknown", "user", "guest", "na", "n/a", "none", "null", "nan"}

SCRIPTS = [("Hindi/Devanagari", "ऀ", "ॿ"), ("Bengali", "ঀ", "৿"), ("Gurmukhi", "਀", "੿"),
           ("Gujarati", "઀", "૿"), ("Odia", "଀", "୿"), ("Tamil", "஀", "௿"),
           ("Telugu", "ఀ", "౿"), ("Kannada", "ಀ", "೿"), ("Malayalam", "ഀ", "ൿ"), ("Urdu/Arabic", "؀", "ۿ")]
HINGLISH_WORDS = set("hai hain nahi nahin bahut kya yaar bhai mera meri mere kyu kyun kab kaise kahan raha rahi rahe gaya gayi hota hoti "
                     "accha acha achha bekar bekaar bakwas kharab mast badhiya paisa paise lekin aur abhi tak wale walo mein ke liye se ko "
                     "mehnga garam jaldi ekdum zabardast shandaar kamaal ghante din kitne".split())
# Words too ambiguous to count alone (they are also English words or names).
HINGLISH_WEAK = {"se", "ko", "ke", "aur", "tak", "din", "mein", "raha", "kya"}
# Python's \w leaves out the vowel signs of Indian scripts, which breaks words into pieces. This pattern keeps a word whole.
WORD_CHARS = r"\wऀ-ॣ०-ൿ"
WORD_RE = re.compile(rf"[{WORD_CHARS}]+")
TOPIC_TOKEN_PATTERN = rf"(?u)[^\W\d_।॥][{WORD_CHARS}.-]*[{WORD_CHARS}]"


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def mask_pii(text: str) -> str:
    text = EMAIL_RE.sub("[email]", text)
    text = PAN_RE.sub("[id]", text)
    return LONG_NUMBER_RE.sub("[number]", text)


def normalise(text: str) -> str:
    """The version of the text that models see: personal data masked, links and handles removed, tidy spacing."""
    text = html.unescape(str(text))
    text = mask_pii(text)
    text = URL_RE.sub(" ", text)
    text = RT_RE.sub("", text)
    text = MENTION_RE.sub(" ", text)
    text = re.sub(r"^\s*:\s*", "", text.strip())
    text = text.replace("#", " ")
    return re.sub(r"\s+", " ", text).strip()


def dedupe_key(clean_text: str) -> str:
    """Same words and same emoji count as a repeat. Case and punctuation do not matter."""
    return re.sub(r"\s+", " ", re.sub(r"[!-/:-@\[-`{-~_‘-”…]+", " ", clean_text.lower())).strip()


def has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def detect_language(text: str) -> str:
    for name, lo, hi in SCRIPTS:
        if any(lo <= ch <= hi for ch in text):
            return name
    found = re.findall(r"[a-z]+", text.lower())
    strong = sum(w in HINGLISH_WORDS and w not in HINGLISH_WEAK for w in found)
    weak = sum(w in HINGLISH_WEAK for w in found)
    return "Hinglish" if strong >= 2 or (strong >= 1 and weak >= 1) else "English"


def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Adds clean_text, key, lang, risk, status and hold_from_llm. Status is one of ok, spam, injection, no_content."""
    out = df.copy()
    out["clean_text"] = out["text"].map(normalise)
    out["key"] = out["clean_text"].map(dedupe_key)
    out["lang"] = out["clean_text"].map(detect_language)
    raw = out["text"].astype(str)

    spam = raw.str.contains(SPAM_RE)
    injection = raw.str.contains(INJECTION_RE)
    # Repeats by different people are real volume, such as retweets, so they stay. A repeat by one real account does not.
    authors = out["author"].astype(str).str.strip()
    is_person = ~authors.str.lower().isin(PLACEHOLDER_AUTHORS) & (authors.map(authors.value_counts()) <= 0.5 * len(out))
    bot = (out.groupby(["key", "author"])["id"].transform("count") >= BOT_REPEAT_MIN) & is_person

    # A filter must never throw away a possible safety or legal report, so those stay in.
    protected = out["clean_text"].map(rules.risk_type) != ""
    status = pd.Series("ok", index=out.index)
    status[~out["clean_text"].map(has_letters)] = "no_content"
    status[(spam | bot) & ~protected] = "spam"
    status[injection & ~protected] = "injection"
    out["status"] = status
    out["hold_from_llm"] = injection  # kept rows that look like injection get rule labels only
    return out


def unique_texts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct usable text, with how many comments share it and their total engagement."""
    ok = df[df["status"] == "ok"]
    grouped = ok.groupby("key", sort=False).agg(
        id=("id", "first"), clean_text=("clean_text", "first"), lang=("lang", "first"),
        repeats=("id", "count"), engagement=("engagement", "sum"), hold_from_llm=("hold_from_llm", "max"))
    return grouped.reset_index()
