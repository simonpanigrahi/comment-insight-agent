"""Measure accuracy, so the claims in the README can be checked.

On real tweets labelled by humans:
  python scripts/get_benchmarks.py
  python scripts/evaluate.py --dataset airline --mode rules
  python scripts/evaluate.py --dataset airline --mode llm --provider ollama --n 300
  python scripts/evaluate.py --dataset airline --mode student --labels 600

On a finished run of the synthetic sample, by hard slice (sarcasm, Hinglish, Indian scripts, safety):
  python scripts/evaluate.py --mode gold --run output/sample_comments

rules     the offline fallback.
vader     plain VADER with nothing added, on the raw text and on the cleaned text.
llm       a random sample labelled by your LLM. This is the number to report for your model.
student   the small classifier, trained on human labels as a stand-in for a perfect LLM teacher, tested on tweets it has
          not seen. This is an upper limit for what the student can do at that number of labels.
transfer  the same classifier trained on all of one dataset and tested on the other.
gold      a finished run compared with the labels the sample generator planted. The author wrote that data, so this
          shows which slices a mode can read. It is not a measure of real-world accuracy.

Accuracy alone misleads on imbalanced data: on airline tweets, always guessing "negative" scores 63 percent.
So macro-F1 is reported too, which gives each class equal weight.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: E402

from agent import prompts, rules  # noqa: E402
from agent.classify import _student, label_tokens  # noqa: E402
from agent.clean import normalise  # noqa: E402
from agent.config import load_settings  # noqa: E402
from agent.llm import LLM, LLMError, LLMFatal  # noqa: E402
from agent.schemas import LabelBatch  # noqa: E402

DATA = ROOT / "benchmarks" / "data"
RESULTS = ROOT / "benchmarks"
CLASSES = ["negative", "neutral", "positive"]


def load(dataset: str) -> pd.DataFrame:
    if dataset == "airline":
        d = pd.read_csv(DATA / "airline.csv")
        out = pd.DataFrame({"text": d["text"].astype(str), "y": d["airline_sentiment"]})
    elif dataset == "sxsw":
        d = pd.read_csv(DATA / "sxsw.csv").dropna(subset=["tweet_text"])
        # Note: "No emotion toward brand" is not quite the same thing as neutral, so this mapping is approximate.
        mapping = {"Positive emotion": "positive", "Negative emotion": "negative", "No emotion toward brand or product": "neutral"}
        out = pd.DataFrame({"text": d["tweet_text"].astype(str), "y": d.iloc[:, 2].map(mapping)}).dropna()
    else:
        raise SystemExit("Unknown dataset")
    out["clean"] = out["text"].map(normalise)
    return out.reset_index(drop=True)


def llm_labels(rows: pd.DataFrame, provider, model, batch) -> tuple[list, dict]:
    s = load_settings(provider, model, batch_size=batch)
    if not s.uses_llm:
        raise SystemExit("Choose an LLM with --provider, or set LLM_PROVIDER in .env.")
    s.max_calls = None
    llm = LLM(s)
    preds = {}
    ids = rows.index.tolist()
    for i in range(0, len(ids), s.batch_size):
        chunk = ids[i:i + s.batch_size]
        short = {f"c{n}": row for n, row in enumerate(chunk, start=1)}
        items = [(k, rows.at[row, "clean"]) for k, row in short.items()]
        try:
            reply = llm.ask(prompts.LABEL_SYSTEM, prompts.label_user(["Other"], items), LabelBatch, purpose="evaluate", max_tokens=label_tokens(s.provider, len(chunk)))
            preds.update({short[r.id]: r.sentiment for r in reply.results if r.id in short})
        except LLMFatal as e:
            print(f"The LLM stopped working: {e}")
            break
        except LLMError as e:
            print(f"Batch failed: {e}")
        print(f"  {min(i + s.batch_size, len(ids))}/{len(ids)}", flush=True)
    return [preds.get(row) for row in ids], llm.usage()


def report(name: str, y, p, extra: dict | None = None) -> None:
    y, p = pd.Series(list(y)).reset_index(drop=True), pd.Series(list(p)).reset_index(drop=True)
    answered = p.notna()
    if not answered.any():
        raise SystemExit(f"{name}: no labels came back, so there is nothing to score.")
    acc, macro = accuracy_score(y[answered], p[answered]), f1_score(y[answered], p[answered], average="macro", labels=CLASSES, zero_division=0)
    base = y.value_counts(normalize=True).iloc[0]
    print(f"\n{name}\n  rows {int(answered.sum()):,} of {len(y):,}\n  accuracy {acc:.3f}   macro-F1 {macro:.3f}   (always guessing the commonest class: accuracy {base:.3f})")
    print(classification_report(y[answered], p[answered], labels=CLASSES, digits=3, zero_division=0))
    print(pd.DataFrame(confusion_matrix(y[answered], p[answered], labels=CLASSES), index=[f"true {c}" for c in CLASSES], columns=[f"said {c}" for c in CLASSES]))
    RESULTS.mkdir(exist_ok=True)
    out = {"name": name, "rows": int(answered.sum()), "accuracy": round(float(acc), 3), "macro_f1": round(float(macro), 3), "majority_baseline": round(float(base), 3), **(extra or {})}
    (RESULTS / f"{name.replace(' ', '_').replace(':', '')}.json").write_text(json.dumps(out, indent=1))


def plain_vader(texts) -> list[str]:
    analyzer = SentimentIntensityAnalyzer()  # a fresh copy, without the words this project adds
    scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
    return ["positive" if s >= 0.05 else "negative" if s <= -0.05 else "neutral" for s in scores]


def gold_report(run_dir: str) -> None:
    table = pd.read_csv(Path(run_dir) / "labelled_comments.csv", dtype=str, keep_default_na=False)
    gold = pd.read_csv(ROOT / "data" / "sample_gold.csv", dtype=str, keep_default_na=False)
    d = table.merge(gold, left_on="id", right_on="comment_id")
    ok = d[(d["status"] == "ok") & d["gold_sentiment"].isin(CLASSES)]
    print(f"{run_dir}: {len(ok):,} usable comments. Labels by source: {ok['source'].value_counts().to_dict()}")
    print("The author wrote this data, so this shows which slices a mode can read. It is not real-world accuracy.\n")

    def line(name: str, rows: pd.DataFrame) -> dict:
        acc = accuracy_score(rows["gold_sentiment"], rows["sentiment"]) if len(rows) else float("nan")
        print(f"  {name:<22} sentiment accuracy {acc:.3f}  ({len(rows):,} comments)")
        return {"accuracy": round(float(acc), 3), "rows": int(len(rows))}

    results = {"all": line("all", ok)}
    for slice_name in ("sarcasm", "hinglish", "indic", "update_spike"):
        results[slice_name] = line(slice_name, ok[ok["planted"] == slice_name])
    macro = f1_score(ok["gold_sentiment"], ok["sentiment"], average="macro", labels=CLASSES, zero_division=0)
    intent = accuracy_score(ok["gold_intent"], ok["intent"])
    print(f"\n  sentiment macro-F1 {macro:.3f}\n  intent accuracy    {intent:.3f}")
    results.update(macro_f1=round(float(macro), 3), intent_accuracy=round(float(intent), 3))

    for risk in ("safety", "legal", "privacy"):
        planted = d[d["planted"] == risk]
        found = planted[planted["risk"] == risk]
        doubted = found[found["llm_doubted"] == "True"]
        false_hits = d[(d["risk"] == risk) & (d["planted"] != risk)]
        print(f"  {risk:<8} reports: {len(found)} of {len(planted)} on the alert list, {len(doubted)} of them doubted by the LLM, {len(false_hits)} false hits")
        results[risk] = {"found": len(found), "planted": len(planted), "doubted_by_llm": len(doubted), "false_hits": len(false_hits)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"gold_{Path(run_dir).name}.json").write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["airline", "sxsw"], default="airline")
    ap.add_argument("--mode", choices=["rules", "vader", "llm", "student", "transfer", "gold"], default="rules")
    ap.add_argument("--n", type=int, default=300, help="sample size for llm mode")
    ap.add_argument("--labels", type=int, default=600, help="training labels for student mode")
    ap.add_argument("--run", help="output folder of a finished run, for gold mode")
    ap.add_argument("--provider")
    ap.add_argument("--model")
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args()
    if args.mode == "gold":
        if not args.run:
            raise SystemExit("gold mode needs --run <output folder of a finished run of data/sample_comments.csv>")
        return gold_report(args.run)
    d = load(args.dataset)
    if args.mode == "rules":
        report(f"{args.dataset} rules", d["y"], d["clean"].map(rules.sentiment))
    elif args.mode == "vader":
        report(f"{args.dataset} vader raw", d["y"], plain_vader(d["text"]))
        report(f"{args.dataset} vader cleaned", d["y"], plain_vader(d["clean"]))
    elif args.mode == "student":
        train, test = train_test_split(d, test_size=0.3, stratify=d["y"], random_state=0)
        train = train.head(args.labels)
        model = _student().fit(train["clean"], train["y"])
        report(f"{args.dataset} student {args.labels} labels", test["y"], model.predict(test["clean"]), {"train_labels": len(train)})
    elif args.mode == "transfer":
        other = "sxsw" if args.dataset == "airline" else "airline"
        target = load(other)
        model = _student().fit(d["clean"], d["y"])
        report(f"train {args.dataset} test {other}", target["y"], model.predict(target["clean"]), {"train_labels": len(d)})
    else:
        sample = d.sample(min(args.n, len(d)), random_state=0).reset_index(drop=True)
        preds, usage = llm_labels(sample, args.provider, args.model, args.batch_size)
        report(f"{args.dataset} llm {usage['model'].replace('/', '-').replace(':', '-')}", sample["y"], preds, usage)
        print("\nLLM use:", usage)


if __name__ == "__main__":
    main()
