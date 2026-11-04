"""Find the topics in the data instead of assuming them, because we do not know the product in advance.

TF-IDF turns comments into numbers, weighting words that are distinctive. NMF then splits that table
into a small number of word groups, and each group is one topic. An LLM, if available, gives each
group a readable name. No LLM is needed to find the groups, so this works offline and at any size.
"""
from __future__ import annotations

import difflib
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .clean import TOPIC_TOKEN_PATTERN

GENERAL = "General opinion"
OTHER = "Other"

# Feeling words such as "love" or "terrible" say how people feel, not what about, so they are not topic words.
# The list below keeps the feeling-like words that do name a subject.
_KEEP = {"support", "help", "care", "service", "charge", "charging", "charged", "delay", "delayed", "cancelled", "refund", "free", "offer", "pay",
         "problem", "problems", "issue", "issues", "bug", "bugs", "crash", "crashes", "crashed", "lag", "hang", "hot", "warm", "burn", "burned", "damage", "damaged",
         "broken", "defect", "warranty", "fix", "fixed", "scratch", "scratches", "wait", "waiting", "stuck", "lost", "miss", "missed", "price", "cost", "worth", "value", "smooth", "bright", "sharp", "clean", "share"}
_FEELING = {w for w, v in SentimentIntensityAnalyzer().lexicon.items() if abs(v) >= 1.2 and w.isalpha()} - _KEEP
_FILLER = {"rt", "amp", "just", "really", "like", "get", "got", "im", "ive", "dont", "doesnt", "didnt", "cant", "wont", "thanks", "thank", "please", "pls", "plz", "ok", "okay", "yes", "lol", "new",
           "hai", "hain", "nahi", "bahut", "kya", "yaar", "bhai", "mera", "meri", "mere", "ke", "ki", "ka", "ko", "se", "mein", "aur", "ho", "raha", "rahi", "gaya", "gayi", "ye", "isme", "liye", "wale", "tak", "abhi", "hota", "hoti", "jata", "jati",
           "accha", "acha", "achha", "mast", "badhiya", "bekar", "bekaar", "bakwas", "kharab", "highly", "recommended", "recommend", "absolutely", "honestly", "seriously", "owned", "purchase", "bought", "buy", "buying", "feels", "far",
           "one", "two", "still", "even", "also", "would", "could", "want", "need", "use", "using", "used", "know", "think", "make", "made", "go", "going", "see", "say", "said", "ve", "don", "didn", "doesn", "ll", "re", "way", "much", "time"}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | _FEELING | _FILLER)


@dataclass
class TopicModel:
    names: list[str] = field(default_factory=list)
    keywords: list[list[str]] = field(default_factory=list)
    vectorizer: object = None
    nmf: object = None
    group_of_component: list[int] = field(default_factory=list)  # after merging, which group each NMF component belongs to
    cutoff: float = 0.0

    def weights(self, texts: list[str]) -> np.ndarray:
        """One row per comment, one column per merged topic."""
        raw = self.nmf.transform(self.vectorizer.transform(texts))
        out = np.zeros((len(texts), len(self.names)))
        for comp, group in enumerate(self.group_of_component):
            out[:, group] += raw[:, comp]
        return out

    def assign(self, texts: list[str], scores: list[float]) -> list[str]:
        """Pick each comment's strongest topic. A comment with no topic words is general if it shows feeling, else Other."""
        if not self.names:
            return [GENERAL if abs(s) >= 0.3 else OTHER for s in scores]
        w = self.weights(texts)
        best, strength = w.argmax(axis=1), w.max(axis=1)
        return [self.names[b] if st > self.cutoff else (GENERAL if abs(sc) >= 0.3 else OTHER) for b, st, sc in zip(best, strength, scores)]


def choose_k(n: int) -> int:
    """More comments can support more topics. Tested on the sample data: 14 groups for about 2,400 comments gave the best fit."""
    return int(min(16, max(3, round(np.sqrt(n) / 3.5))))


def discover(texts: list[str], k: int | None = None, merge_at: float = 0.2) -> TopicModel:
    """Find topic groups. Returns an empty model if there is too little text."""
    if len(texts) < 10:
        return TopicModel()
    k = k or choose_k(len(texts))
    min_df = 2 if len(texts) < 500 else 3
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=0.4, stop_words=STOP_WORDS, sublinear_tf=True, max_features=30000,
                                 token_pattern=TOPIC_TOKEN_PATTERN)  # words must start with a letter, so bare numbers are skipped
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:  # nothing left after removing stop words
        return TopicModel()
    k = min(k, X.shape[1], X.shape[0] - 1)
    if k < 1:
        return TopicModel()
    nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=800)
    with warnings.catch_warnings():  # a slightly unconverged topic model is still useful here
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)  # on very little text NMF can report a harmless rounding problem
        nmf.fit(X)
    terms = np.array(vectorizer.get_feature_names_out())

    # Merge components that use nearly the same words, for example two spellings of one issue.
    sim = cosine_similarity(nmf.components_)
    group = list(range(k))
    for a in range(k):
        for b in range(a + 1, k):
            if sim[a, b] >= merge_at:
                old, new = max(group[a], group[b]), min(group[a], group[b])
                group = [new if g == old else g for g in group]
    ids = sorted(set(group))
    remap = {g: i for i, g in enumerate(ids)}
    group = [remap[g] for g in group]

    keywords, names = [], []
    for g in range(len(ids)):
        comp = nmf.components_[[i for i, x in enumerate(group) if x == g]].sum(axis=0)
        top = [terms[i] for i in comp.argsort()[::-1][:8]]
        keywords.append(top)
        names.append(_offline_name(top))
    model = TopicModel(names=_unique(names), keywords=keywords, vectorizer=vectorizer, nmf=nmf, group_of_component=group)
    weights = model.weights(texts).max(axis=1)
    positive = weights[weights > 0]
    model.cutoff = float(np.median(positive) * 0.1) if len(positive) else 0.0
    return model


def match_topic(name: str, taxonomy: list[str], fallback: str) -> str:
    """Map the name a model wrote onto the topic list. An explicit Other stays Other. A name that fits nothing gets the fallback."""
    key = name.strip().lower()
    lowered = {t.lower(): t for t in taxonomy}
    if key in lowered:
        return lowered[key]
    named = [t for t in lowered if t not in (OTHER.lower(), GENERAL.lower())]
    within = [t for t in named if key and (key in t or t in key)]
    if len(within) == 1:
        return lowered[within[0]]
    close = difflib.get_close_matches(key, named, n=1, cutoff=0.8)
    return lowered[close[0]] if close else fallback


def _offline_name(top_terms: list[str]) -> str:
    """Without an LLM, a topic is named by its top words."""
    picked: list[str] = []
    for term in top_terms:
        words = term.split()
        if any(w in " ".join(picked).split() for w in words):
            continue
        picked.append(term)
        if len(picked) == 3:
            break
    return ", ".join(picked).capitalize()


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


def cluster_examples(model: TopicModel, texts: list[str], per_group: int = 5) -> list[dict]:
    """The comments that best represent each group, to show the LLM when it names them."""
    if not model.names:
        return []
    w = model.weights(texts)
    out = []
    for g in range(len(model.names)):
        order = w[:, g].argsort()[::-1][:per_group]
        out.append({"cluster": g, "keywords": model.keywords[g], "examples": [texts[i] for i in order]})
    return out
