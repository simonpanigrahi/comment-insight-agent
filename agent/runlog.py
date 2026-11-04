"""The audit trail. Every decision the agent makes is printed and saved to run_log.jsonl."""
from __future__ import annotations

import json
import time
from pathlib import Path


class RunLog:
    def __init__(self, out_dir: Path | str, quiet: bool = False):
        self.path = Path(out_dir) / "run_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.quiet = quiet
        self.start = time.time()
        self.decisions: list[dict] = []

    def _write(self, record: dict) -> None:
        record = {"t": round(time.time() - self.start, 2), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def decision(self, step: str, text: str) -> None:
        """Something the reader of the report should know the agent decided or found."""
        self.decisions.append({"step": step, "text": text})
        self._write({"type": "decision", "step": step, "text": text})
        if not self.quiet:
            print(f"[{step:<11}] {text}", flush=True)

    def progress(self, text: str) -> None:
        """Screen only, so a long LLM run does not look frozen."""
        if not self.quiet:
            print(f"{'':14}{text}", flush=True)

    def event(self, step: str, **fields) -> None:
        """Technical detail, saved to the log file only."""
        self._write({"type": "event", "step": step, **fields})
