"""Ask questions about a finished run. This is a small retrieval-augmented (RAG) chatbot over the comments.

It retrieves the closest comments, answers only from them, cites them, and refuses when they do not answer
the question. It is the same grounding idea a support chatbot needs. TF-IDF retrieval keeps it fully offline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import prompts
from .clean import WORD_CHARS
from .config import Settings
from .llm import LLM, LLMError
from .schemas import Answer

MIN_SIMILARITY = 0.08
TOP_K = 12
NO_ANSWER = "The comments do not answer this."


def retrieve(df: pd.DataFrame, question: str, k: int = TOP_K) -> tuple[pd.DataFrame, float]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", token_pattern=rf"(?u)[{WORD_CHARS}]{{2,}}")
    X = vectorizer.fit_transform(df["clean_text"].astype(str))
    sims = cosine_similarity(vectorizer.transform([question]), X).ravel()
    order = sims.argsort()[::-1][:k]
    return df.iloc[order].assign(similarity=sims[order]), float(sims[order[0]]) if len(order) else 0.0


def answer(run_dir: str, question: str, settings: Settings) -> str:
    path = Path(run_dir) / "labelled_comments.csv"
    if not path.exists():
        return f"No finished run found at {run_dir}. Run the analyse command first."
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # A comment that looks like prompt injection stays away from the LLM here too.
    df = df[(df["status"] == "ok") & (df["hold_from_llm"] != "True")].drop_duplicates("clean_text")
    df = df.assign(clean_text=df["clean_text"].str.replace(r"^'(?=[=+\-@])", "", regex=True))  # undo the spreadsheet safety prefix
    if df.empty:
        return NO_ANSWER
    try:
        hits, best = retrieve(df, question)
    except ValueError:  # nothing usable to compare
        return NO_ANSWER
    if best < MIN_SIMILARITY:
        return NO_ANSWER + " (No comment is close to this question.)"
    sentiments = hits["sentiment"].value_counts().to_dict()
    listing = "\n".join(f"  [{r.id}] ({r.sentiment}) {r.clean_text[:160]}" for r in hits.itertuples())
    if not settings.uses_llm:
        return f"Closest comments, with no LLM configured to summarise them. Sentiment among them: {sentiments}\n{listing}"
    settings.max_calls = 4
    llm = LLM(settings)
    items = list(zip(hits["id"], hits["clean_text"]))
    valid = set(hits["id"])
    try:
        reply = llm.ask(prompts.ASK_SYSTEM, prompts.ask_user(question, items), Answer, purpose="ask", max_tokens=400)
    except LLMError as e:
        return f"The LLM is not available ({e}). Closest comments:\n{listing}"
    cited = [c for c in reply.comment_ids if c in valid]
    if not cited and NO_ANSWER not in reply.answer:
        return f"The answer cited no retrieved comment, so it was rejected. Closest comments:\n{listing}"
    return f"{reply.answer}\n\nSources: {', '.join(cited) or 'none'}"
