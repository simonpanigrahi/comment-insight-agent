#!/usr/bin/env python3
"""Command line for the Comment Insight Agent.

  python run.py analyse data/sample_comments.csv                     rules only, no key needed
  python run.py analyse data/sample_comments.csv --provider ollama   with a local LLM
  python run.py doctor --provider ollama                             test your LLM setup
  python run.py ask output/sample_comments "why is battery bad?"     ask questions about a finished run
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from agent.config import NEEDS_KEY, load_settings
from agent.ingest import InputError
from agent.llm import LLM, LLMError, list_models, ollama_placement


def cmd_analyse(args) -> int:
    from agent.graph import Agent

    settings = load_settings(args.provider, args.model, args.max_llm_calls, args.batch_size)
    if settings.provider in NEEDS_KEY and not settings.api_key:
        print(f"Warning: no {settings.provider.upper()}_API_KEY found, so the LLM will fail and this run will use rules. Copy .env.example to .env and add the key.", file=sys.stderr)
    out_dir = args.out or str(Path("output") / Path(args.input).stem)
    agent = Agent(settings, args.input, out_dir, title=args.title, text_col=args.text_col, quiet=args.quiet, limit=args.limit)
    started = time.time()
    try:
        final = agent.run()
    except (InputError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    print(f"\nDone in {time.time() - started:.0f} seconds. Open {final['outputs']['html']}")
    return 0


def cmd_doctor(args) -> int:
    s = load_settings(args.provider, args.model)
    if not s.uses_llm:
        print("Provider is 'none'. Pass --provider ollama, gemini or groq, or set LLM_PROVIDER in .env.")
        return 1
    print(f"Provider: {s.provider}\nModel:    {s.model}\nServer:   {s.base_url}")
    if s.provider in NEEDS_KEY and not s.api_key:
        print(f"\nFAILED: no API key. Copy .env.example to .env in this folder and set {s.provider.upper()}_API_KEY.")
        return 1
    s.max_calls = 5
    s.cache_dir = ".cache/doctor"
    llm = LLM(s)
    started = time.time()
    try:
        llm.ping()
    except LLMError as e:
        print(f"\nFAILED: {e}")
        models = list_models(s)
        if models:
            print("Models this server offers:\n  " + "\n  ".join(models[:40]))
        return 1
    print(f"\nOK: valid JSON came back in {time.time() - started:.1f} seconds. The first call also loads the model.")
    from agent import prompts
    from agent.classify import label_tokens
    from agent.schemas import LabelBatch

    items = [(cid, text) for cid, text, *_ in prompts.TEST_COMMENTS]
    tokens_before = llm.tokens_out
    started = time.time()
    try:
        reply = llm.ask(prompts.LABEL_SYSTEM, prompts.label_user(prompts.TEST_TOPICS, items), LabelBatch, purpose="doctor", max_tokens=label_tokens(s.provider, len(items)), use_cache=False)
    except LLMError as e:
        print(f"Labelling test FAILED: {e}")
        return 1
    elapsed = time.time() - started
    print(f"Labelling test took {elapsed:.1f} seconds for {len(items)} comments ({llm.tokens_out - tokens_before} tokens out):")
    got = {r.id: r for r in reply.results}
    right = 0
    for cid, _, sentiment, urgent, tests in prompts.TEST_COMMENTS:
        r = got.get(cid)
        if r is None:
            print(f"  {cid}: missing from the reply ({tests})")
            continue
        fine = r.sentiment == sentiment and r.urgent == urgent
        right += fine
        print(f"  {cid}: {'right' if fine else 'WRONG'} on {tests}. Said {r.sentiment}, {r.intent}, topic {r.topic}, urgent {r.urgent}. Reason: {r.reason} | Evidence: {r.evidence}")
    print(f"Right on {right} of {len(items)}. Four comments only show the model works, so measure its real accuracy with scripts/evaluate.py.")
    print(f"Rough speed: {elapsed / len(items):.1f} seconds per comment, so the sample's 2,533 distinct comments would take about {elapsed / len(items) * 2533 / 60:.0f} minutes. Bigger batches are usually a little faster per comment.")
    where = ollama_placement(s) if s.provider == "ollama" else None
    if where:
        print(f"Model memory: {where['gb']:.1f} GB, {where['gpu_share']:.0f}% of it on the GPU.")
    if s.provider == "ollama" and elapsed / len(items) > 5:
        print("That is slow. Run python scripts/speed_test.py to see why, and what to do about it.")
    return 0


def cmd_ask(args) -> int:
    from agent.ask import answer

    s = load_settings(args.provider, args.model)
    print(answer(args.run_dir, args.question, s))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Turn social media comments into an insight report that traces back to its data.")
    parser.add_argument("--debug", action="store_true", help="show the full error trace if something goes wrong")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyse", help="analyse a CSV or JSON file of comments")
    a.add_argument("input", help="path to a .csv, .tsv, .json or .jsonl file")
    a.add_argument("--provider", choices=["none", "ollama", "gemini", "groq", "openai_compatible"], help="which LLM to use (default: LLM_PROVIDER in .env, else none)")
    a.add_argument("--model", help="model name, overriding the default")
    a.add_argument("--max-llm-calls", type=int, help="the most LLM calls this run may make (default: no cap for a local model)")
    a.add_argument("--batch-size", type=int, help="comments per LLM call")
    a.add_argument("--text-col", help="name of the column holding the comment text, if it is not detected")
    a.add_argument("--limit", type=int, help="analyse only a random sample of this many rows, for a quick demo")
    a.add_argument("--title", default="", help="title for the report")
    a.add_argument("--out", help="output folder (default: output/<file name>)")
    a.add_argument("--quiet", action="store_true", help="print less")
    a.set_defaults(func=cmd_analyse)

    d = sub.add_parser("doctor", help="check that your LLM setup works")
    d.add_argument("--provider")
    d.add_argument("--model")
    d.set_defaults(func=cmd_doctor)

    q = sub.add_parser("ask", help="ask a question about the comments of a finished run")
    q.add_argument("run_dir", help="the output folder of a finished run")
    q.add_argument("question")
    q.add_argument("--provider")
    q.add_argument("--model")
    q.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):  # an older Windows console cannot print Indian scripts, and that must not stop a run
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nStopped. Finished LLM calls are saved in the cache, so the same command continues from there.", file=sys.stderr)
        return 130
    except Exception as e:  # a person should see one clear line, and --debug shows the rest
        if args.debug:
            traceback.print_exc()
        print(f"Error: {type(e).__name__}: {e}. Run with --debug before the command for the full trace.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
