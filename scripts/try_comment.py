#!/usr/bin/env python3
"""Type a comment and see how it is labelled, using the same cleaning, rules and LLM prompt as a full run.

  python scripts/try_comment.py "the battery dies by noon"                          rules only, no key needed
  python scripts/try_comment.py "the battery dies by noon" --provider gemini --model gemini-3.5-flash-lite
  python scripts/try_comment.py --provider ollama                                   asks for comments one by one

This tests how one comment is read. It does not run the agent: there is no report, and a full run finds its
topics from the data, while this one uses the list in TOPICS unless you pass --topics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from agent import prompts, rules  # noqa: E402
from agent.classify import evidence_is_real, label_tokens  # noqa: E402
from agent.clean import clean_comments  # noqa: E402
from agent.config import NEEDS_KEY, load_settings  # noqa: E402
from agent.llm import LLM, LLMError  # noqa: E402
from agent.schemas import LabelBatch  # noqa: E402

TOPICS = ["Battery", "Camera", "Display", "Performance", "Software", "Price", "Delivery", "Customer service", "Other"]


def show(text: str, llm: LLM | None, topics: list[str]) -> None:
    frame = pd.DataFrame({"id": ["c1"], "text": [text], "author": ["you"], "engagement": [0.0], "platform": ["typed"], "created_at": [pd.NaT]})
    row = clean_comments(frame).iloc[0]
    clean = row["clean_text"]
    print(f"\nComment:   {text}")
    if clean != text:
        print(f"Cleaned:   {clean}")
    print(f"Language:  {row['lang']}")
    if row["status"] != "ok":
        print(f"Filter:    a full run would remove this comment, as {row['status']}.")
    if not clean:
        return
    sentiment = rules.sentiment(clean)
    print(f"Rules:     {sentiment}, {rules.intent(clean, sentiment)}, alert: {rules.risk_type(clean) or 'none'}")
    if llm is None:
        print(f"Reason:    {rules.reason_snippet(clean, sentiment)} (rules copy this from the comment)")
        print("LLM:       not used, so this is a rough result. Add --provider gemini or --provider ollama for the LLM's label, reason and quote.")
        return
    if row["hold_from_llm"] or row["status"] == "injection":
        print("LLM:       held back, because this looks like prompt injection.")
        return
    try:
        reply = llm.ask(prompts.LABEL_SYSTEM, prompts.label_user(topics, [("c1", clean)]), LabelBatch, purpose="try", max_tokens=label_tokens(llm.s.provider, 1), use_cache=False)
    except LLMError as e:
        print(f"LLM:       failed. {e}")
        return
    if not reply.results:
        print("LLM:       returned no label.")
        return
    r = reply.results[0]
    print(f"LLM:       {r.sentiment}, {r.intent}, topic: {r.topic}, urgent: {r.urgent}")
    print(f"Reason:    {r.reason}")
    print(f"Evidence:  \"{r.evidence}\" ({'found in the comment' if evidence_is_real(r.evidence, clean) else 'NOT found in the comment'})")


def main() -> int:
    ap = argparse.ArgumentParser(description="See how one comment is labelled.")
    ap.add_argument("comments", nargs="*", help="one or more comments, each in quotes. With none, it asks for them one by one.")
    ap.add_argument("--provider", help="ollama, gemini or groq. Without it, only the rules run.")
    ap.add_argument("--model")
    ap.add_argument("--topics", help="comma-separated topic names to choose from")
    args = ap.parse_args()

    llm = None
    if args.provider:
        s = load_settings(args.provider, args.model)
        if s.provider in NEEDS_KEY and not s.api_key:
            print(f"No API key found. Put {s.provider.upper()}_API_KEY in .env first.")
            return 1
        s.cache_dir = str(ROOT / ".cache" / "try")
        llm = LLM(s)
    topics = [t.strip() for t in args.topics.split(",") if t.strip()] if args.topics else TOPICS
    if "Other" not in topics:
        topics = topics + ["Other"]

    for text in args.comments:
        show(text, llm, topics)
    if args.comments:
        return 0
    try:
        while True:
            text = input("\nType a comment (or press Enter to stop): ").strip()
            if not text:
                return 0
            show(text, llm, topics)
    except (EOFError, KeyboardInterrupt):
        return 0


if __name__ == "__main__":
    sys.exit(main())
