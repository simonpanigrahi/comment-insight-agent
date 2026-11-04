"""Find out why a local model is slow, and what to do about it.

  python scripts/speed_test.py
  python scripts/speed_test.py --model qwen3:1.7b

It prints the state of the machine, where Ollama loaded the model, and how fast the model reads and writes
under each kind of output limit. It talks only to Ollama on this machine.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from agent import prompts  # noqa: E402
from agent.classify import label_tokens  # noqa: E402
from agent.config import load_settings  # noqa: E402
from agent.llm import ollama_placement, parse_reply  # noqa: E402
from agent.schemas import LabelBatch  # noqa: E402

ITEMS = [(cid, text) for cid, text, *_ in prompts.TEST_COMMENTS]
SAMPLE_COMMENTS = 2533  # distinct comments in data/sample_comments.csv
LOG_LINES = re.compile(r"no compatible GPU|inference compute|offload|unable to load cuda|gpu discovery|failure during GPU", re.I)


def shell(*command: str) -> str:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return done.stdout.strip() if done.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def machine_report(url: str) -> dict:
    gpu = shell("nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,pstate,power.limit,temperature.gpu", "--format=csv,noheader")
    print(f"NVIDIA GPU: {gpu or 'nvidia-smi is missing or failed, so the driver is not usable'}")
    plugged = None
    for path in glob.glob("/sys/class/power_supply/*/online"):
        try:
            plugged = Path(path).read_text().strip() == "1"
        except OSError:
            pass
    if plugged is not None:
        print(f"Power: {'plugged in' if plugged else 'ON BATTERY, which limits both the GPU and the CPU'}")
    profile = shell("powerprofilesctl", "get")
    if profile:
        print(f"Power profile: {profile}")
    try:
        memory = dict(re.findall(r"(\w+):\s+(\d+) kB", Path("/proc/meminfo").read_text()))
        print(f"RAM: {int(memory['MemAvailable']) / 1e6:.1f} GB free of {int(memory['MemTotal']) / 1e6:.1f} GB. CPU threads: {os.cpu_count()}")
    except (OSError, KeyError):
        print(f"CPU threads: {os.cpu_count()}")
    try:
        print(f"Ollama version: {requests.get(f'{url}/api/version', timeout=10).json().get('version')}")
    except (requests.RequestException, ValueError, AttributeError):
        print(f"Could not reach Ollama at {url}. Is it running?")
    return {"gpu": bool(gpu), "plugged": plugged}


def one_test(url: str, model: str, name: str, limit, num_ctx: int) -> dict | None:
    body = {"model": model, "stream": False, "think": False,
            "messages": [{"role": "system", "content": prompts.LABEL_SYSTEM}, {"role": "user", "content": prompts.label_user(prompts.TEST_TOPICS, ITEMS)}],
            "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": label_tokens("ollama", len(ITEMS))}}
    if limit:
        body["format"] = limit
    try:
        data = requests.post(f"{url}/api/chat", json=body, timeout=900).json()
    except (requests.RequestException, ValueError) as e:
        print(f"  {name:<14} failed: {str(e)[:150]}")
        return None
    if data.get("error"):
        print(f"  {name:<14} failed: {str(data['error'])[:150]}")
        return None
    tokens, seconds = data.get("eval_count", 0), data.get("eval_duration", 0) / 1e9
    read_tokens, read_seconds = data.get("prompt_eval_count", 0), data.get("prompt_eval_duration", 0) / 1e9
    write_rate = tokens / seconds if seconds else 0.0
    read_rate = read_tokens / read_seconds if read_seconds else 0.0
    try:
        usable = len(parse_reply((data.get("message") or {}).get("content") or "", LabelBatch).results)
    except ValueError:
        usable = 0
    print(f"  {name:<14} writes {write_rate:6.1f} tokens/s, reads {read_rate:7.1f} tokens/s, {tokens} tokens, {data.get('total_duration', 0) / 1e9:5.1f} s in all, "
          f"load {data.get('load_duration', 0) / 1e9:4.1f} s, usable labels {usable} of {len(ITEMS)}")
    return {"name": name, "write_rate": write_rate, "tokens": tokens, "usable": usable}


def advice(found: dict, placement: dict | None, results: list[dict]) -> None:
    print("\nWhat this means:")
    schema = next((r for r in results if r["name"] == "JSON schema"), None)
    usable = [r for r in results if r["usable"] == len(ITEMS)]
    fastest = max(usable or results, key=lambda r: r["write_rate"], default=None)
    plain = next((r for r in usable if r["name"] == "plain JSON"), None)
    if plain and fastest is not plain and fastest["write_rate"] < 1.2 * plain["write_rate"]:
        fastest = plain  # a small gain is not worth losing the JSON limit
    share = placement["gpu_share"] if placement else None
    if share is not None and share >= 99:
        print("- The model is fully on the GPU.")
    elif share is not None and share >= 1:
        print(f"- Only {share:.0f}% of the model fits on the GPU. A split runs far slower than a full fit. Close anything using the GPU (see nvidia-smi), or add OLLAMA_NUM_CTX=3072 to .env, or try --model qwen3:1.7b.")
    elif not found["gpu"]:
        print("- The NVIDIA driver is not working, so the model can only use the CPU. Fix the driver first (sudo ubuntu-drivers autoinstall, then reboot).")
    elif share is not None:
        print("- The model is entirely on the CPU, even though the driver works. Ollama has not found the GPU, which often happens after the laptop sleeps. Try:")
        print("    sudo systemctl stop ollama && sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm && sudo systemctl start ollama")
        print("  Then run this test again. If it is still on the CPU, reinstall Ollama with its install script, because that is when it looks for CUDA.")
    if found["plugged"] is False:
        print("- Plug the laptop in. On battery, both the GPU and the CPU are held back.")
    if schema and fastest and fastest in usable and fastest is not schema and fastest["write_rate"] > 1.3 * schema["write_rate"]:
        print(f"- The JSON schema is costing speed. Adding OLLAMA_FORMAT={'json' if fastest['name'] == 'plain JSON' else 'none'} to .env writes {fastest['write_rate'] / max(schema['write_rate'], 0.1):.1f} times faster, and replies are still validated in code.")
    chosen = fastest if fastest in usable else schema
    if chosen and chosen["write_rate"] and chosen["tokens"]:
        minutes = (chosen["tokens"] / len(ITEMS)) * SAMPLE_COMMENTS / chosen["write_rate"] / 60
        print(f"- At {chosen['write_rate']:.1f} tokens/s, labelling all {SAMPLE_COMMENTS:,} distinct sample comments takes about {minutes:.0f} minutes.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find out why a local model is slow.")
    ap.add_argument("--model", help="model to test (default: the Ollama default in this project)")
    ap.add_argument("--url", help="Ollama address (default: this machine)")
    args = ap.parse_args(argv)
    s = load_settings("ollama", args.model)
    url = (args.url or s.base_url).rstrip("/")
    s.base_url = url
    print(f"Testing {s.model} at {url}\n")
    found = machine_report(url)

    print(f"\nThe model labels {len(ITEMS)} comments under each kind of output limit:")
    results = []
    placement = None
    for name, limit in (("JSON schema", LabelBatch.model_json_schema()), ("plain JSON", "json"), ("no limit", None)):
        result = one_test(url, s.model, name, limit, s.num_ctx)
        if result:
            results.append(result)
        placement = placement or ollama_placement(s)

    if placement:
        print(f"\nOllama loaded {placement['name']}: {placement['gb']:.1f} GB in memory, {placement['gpu_share']:.0f}% of it on the GPU, context {placement['context']}.")
    else:
        print("\nOllama did not report where the model is loaded.")
    usage = shell("nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader")
    if usage:
        print(f"GPU memory now: {usage}")
    log = [line for line in shell("journalctl", "-u", "ollama", "-n", "400", "--no-pager").splitlines() if LOG_LINES.search(line)]
    if log:
        print("Recent Ollama log lines about the GPU:")
        for line in log[-4:]:
            print("  " + line[line.find("msg=") if "msg=" in line else 0:][:210])
    advice(found, placement, results)
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
