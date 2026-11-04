#!/usr/bin/env python3
"""Put a finished run's labels next to labels written by people, and read where they differ.

  python scripts/compare_labels.py output_sample/gemini_airline_1000 benchmarks/data/airline.csv --label airline_sentiment --reason negativereason

Works on any CSV or JSON you analysed, if it has a column of human labels. Labels other than positive, negative
and neutral need --map, for example --map "1=negative,3=neutral,5=positive" or --map "Irrelevant=skip".
Text written by the model and by people is shown as it is, so read it yourself: that is the point of this check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

from agent.ingest import load_comments, read_table  # noqa: E402

CLASSES = ["negative", "neutral", "positive"]


def human_labels(path: str, label_col: str, text_col: str | None, mapping: dict[str, str], reason_col: str | None) -> pd.DataFrame:
    """The same ids a run gives each row, next to that row's human label."""
    raw = read_table(path)
    for col in (label_col, reason_col):
        if col and col not in raw.columns:
            raise SystemExit(f"Column '{col}' is not in {path}. Columns found: {list(raw.columns)}")
    loaded, _ = load_comments(path, text_col)
    out = pd.DataFrame({"id": loaded["id"].astype(str).values, "human": raw[label_col].astype(str).str.strip().values})
    out["human"] = out["human"].map(lambda v: mapping.get(v, mapping.get(v.lower(), v.lower())))
    if reason_col:
        out["human_reason"] = raw[reason_col].fillna("").astype(str).values
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare a run's labels with labels written by people.")
    ap.add_argument("run_dir", help="the output folder of a finished run")
    ap.add_argument("data_file", help="the file that run analysed, with its human labels")
    ap.add_argument("--label", required=True, help="the column holding the human sentiment label")
    ap.add_argument("--reason", help="a column holding a human reason, if there is one")
    ap.add_argument("--text-col", help="the same --text-col you gave the run, if you gave one")
    ap.add_argument("--map", default="", help='how to read other labels, for example "1=negative,3=neutral,5=positive,Irrelevant=skip"')
    ap.add_argument("--show", type=int, default=15, help="how many wrong rows to print, and half as many right ones")
    ap.add_argument("--seed", type=int, default=0, help="change this to read a different set of rows")
    args = ap.parse_args()

    mapping = {k.strip(): v.strip().lower() for k, v in (pair.split("=", 1) for pair in args.map.split(",") if "=" in pair)}
    mapping.update({k.lower(): v for k, v in mapping.items()})
    run = pd.read_csv(Path(args.run_dir) / "labelled_comments.csv", dtype=str, keep_default_na=False)
    human = human_labels(args.data_file, args.label, args.text_col, mapping, args.reason)
    d = run[run["status"] == "ok"].merge(human, on="id")
    if d.empty:
        raise SystemExit("No rows matched. Is this the file that run analysed?")
    other = ~d["human"].isin(CLASSES)
    if other.any():
        print(f"Left out {int(other.sum()):,} rows whose human label is not positive, negative or neutral: {d.loc[other, 'human'].value_counts().head(6).to_dict()}")
    d = d[~other]
    if d.empty:
        raise SystemExit("No usable human labels. Use --map to say how to read them.")

    acc = accuracy_score(d["human"], d["sentiment"])
    macro = f1_score(d["human"], d["sentiment"], average="macro", labels=CLASSES, zero_division=0)
    print(f"\n{len(d):,} comments compared. Labels by source: {d['source'].value_counts().to_dict()}")
    print(f"accuracy {acc:.3f}   macro-F1 {macro:.3f}   (always guessing the commonest class: accuracy {d['human'].value_counts(normalize=True).iloc[0]:.3f})\n")
    print(pd.DataFrame(confusion_matrix(d["human"], d["sentiment"], labels=CLASSES), index=[f"people said {c}" for c in CLASSES], columns=[f"agent said {c}" for c in CLASSES]).to_string())

    def rows(title: str, part: pd.DataFrame, n: int) -> None:
        print(f"\n{title} ({len(part):,} in all)")
        for r in part.sample(min(n, len(part)), random_state=args.seed).itertuples():
            print(f"\n  {r.clean_text[:280]}")
            print(f"    people: {r.human}" + (f" ({r.human_reason})" if args.reason and r.human_reason else "") + f"   agent: {r.sentiment}, {r.intent}, topic {r.topic}")
            print(f"    agent's reason: {r.reason}   evidence: \"{r.evidence}\"")
    rows("WHERE PEOPLE AND THE AGENT DISAGREE", d[d["human"] != d["sentiment"]], args.show)
    rows("WHERE THEY AGREE", d[d["human"] == d["sentiment"]], max(1, args.show // 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
