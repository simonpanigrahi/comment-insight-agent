"""A pretend LLM server for tests, speaking both the Ollama and the OpenAI-compatible formats.

It lets the tests check everything around the model (HTTP calls, retries, validation, the checker loop)
with no network and no GPU. It says nothing about how good a real model is.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent import rules


class Behaviour:
    def __init__(self):
        self.requests = 0
        self.prompts: list[str] = []  # every prompt the server received
        self.rate_limit_first = 0     # answer 429 this many times first
        self.invalid_first = 0        # answer with broken JSON this many times first
        self.truncate_first = 0       # say the reply was cut off by the token limit, this many times first
        self.labels_always_invalid = False  # labelling replies are never valid JSON
        self.drop_id_once = False     # leave one comment out of a batch reply, once
        self.fake_evidence_once = False  # invent a quote, once
        self.bad_item_once = False    # one item in a batch has an unknown sentiment, once
        self.sloppy_labels = False    # right meaning, wrong form: "Negative", ids without the c, null reasons
        self.teacher = "rules"        # "rules", or "keywords": a teacher that differs from rules, so a student can beat them
        self.hallucinate_writes = 1   # first drafts that contain a made-up number
        self.empty_writes = False     # the writer returns a blank headline and no findings
        self.all_other = False        # put every comment in the topic "Other"
        self.deny_urgent = False      # treat every rule hit as a false alarm
        self.urgent_word = ""         # also call any comment with this word urgent
        self.investigate_script: list[dict] = []  # what the investigator chooses, in order, before it stops
        self.investigate_calls = 0
        self.write_calls = 0
        self.last_write_prompt = ""
        self.down_after = None        # start refusing after this many requests
        self.vram_bytes = 3000000000  # how much of the model the server says is on the GPU


def _keyword_teacher(text: str) -> str:
    return "positive" if re.search(r"camera|photo|display|screen|design|colou?r", text, re.I) else "negative"


def label_reply(b: Behaviour, user: str) -> dict:
    topics = [t.strip() for t in re.search(r"Topic list: (.*)", user).group(1).split("|")]
    rows = [json.loads(line) for line in user.split("Comments:\n", 1)[1].splitlines() if line.startswith("{")]
    results = []
    for row in rows:
        cid, text = row["id"], row["text"]
        sentiment = _keyword_teacher(text) if b.teacher == "keywords" else rules.sentiment(text)
        words = text.split()
        topic = "Other" if b.all_other else next((t for t in topics if t.lower().split(",")[0] in text.lower()), topics[0])
        results.append({"id": cid, "sentiment": sentiment, "reason": "mock reason about " + " ".join(words[:3]), "evidence": " ".join(words[:4]),
                        "intent": rules.intent(text, sentiment), "topic": topic, "urgent": (bool(rules.risk_type(text)) and not b.deny_urgent) or (b.urgent_word != "" and b.urgent_word in text.lower())})
    if b.drop_id_once and results:
        b.drop_id_once = False
        results = results[:-1]
    if b.fake_evidence_once and results:
        b.fake_evidence_once = False
        results[0]["evidence"] = "words that are not in the comment at all"
    if b.bad_item_once and len(results) > 1:
        b.bad_item_once = False
        results[0]["sentiment"] = "happy"
    if b.sloppy_labels:
        for r in results:
            r["id"] = r["id"].lstrip("c")
            r["sentiment"] = r["sentiment"].capitalize()
            r["intent"] = r["intent"].upper()
            r["reason"] = None
            r["topic"] = r["topic"].upper()
    return {"results": results}


def write_reply(b: Behaviour, user: str) -> dict:
    b.write_calls += 1
    b.last_write_prompt = user
    if b.empty_writes:
        return {"headline": " ", "headline_fact_ids": [], "points": [], "actions": []}
    facts = dict(re.findall(r"^(F\d+)(?: ALERT)?: (.*)$", user, flags=re.M))
    alerts = re.findall(r"^(F\d+) ALERT:", user, flags=re.M)
    overall = next((k for k, v in facts.items() if v.startswith("Sentiment across")), "F1")
    intent = next((k for k, v in facts.items() if v.startswith("What people are doing")), overall)
    points = [{"text": facts[overall], "fact_ids": [overall]}, {"text": facts[intent], "fact_ids": [intent]}]
    actions = [{"action": "Investigate this report and contact the customers.", "why": "Serious reports need a person now.", "owner": "Engineering", "urgency": "today", "fact_ids": [a]} for a in alerts]
    actions.append({"action": "Assign an owner to the largest complaint.", "why": "It leads the list.", "owner": "Product", "urgency": "this_week", "fact_ids": [overall]})
    if b.write_calls <= b.hallucinate_writes:
        points.append({"text": "Exactly 87.5% of buyers plan to return the product.", "fact_ids": [overall]})
    return {"headline": "Serious reports need action first.", "headline_fact_ids": [overall], "points": points, "actions": actions}


class Handler(BaseHTTPRequestHandler):
    behaviour: Behaviour

    def log_message(self, *args):  # keep test output quiet
        pass

    def _send(self, code: int, payload, headers=None):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/api/tags"):
            self._send(200, {"models": [{"name": "fake-model"}]})
        elif self.path.endswith("/api/ps"):
            self._send(200, {"models": [{"name": "fake-model", "size": 3000000000, "size_vram": self.behaviour.vram_bytes, "context_length": 4096}]})
        elif self.path.endswith("/api/version"):
            self._send(200, {"version": "0.0-fake"})
        else:
            self._send(200, {"data": [{"id": "fake-model"}]})

    def do_POST(self):
        b = self.behaviour
        b.requests += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        b.prompts += [m["content"] for m in body["messages"] if m["role"] == "user"]
        if b.down_after is not None and b.requests > b.down_after:
            return self._send(404, {"error": "model not found"})
        if b.rate_limit_first > 0:
            b.rate_limit_first -= 1
            return self._send(429, {"error": "slow down"}, {"Retry-After": "0"})
        text = self._reply_text(body)
        if b.invalid_first > 0:
            b.invalid_first -= 1
            text = "Sure! Here is your answer: {not valid json"
        cut_off = b.truncate_first > 0
        if cut_off:
            b.truncate_first -= 1
        usage_in, usage_out = 100, 50
        if self.path.endswith("/api/chat"):
            slow = 4 if "format" in body and isinstance(body["format"], dict) else 1  # a schema costs speed in this pretend server
            self._send(200, {"message": {"role": "assistant", "content": text}, "done": True, "done_reason": "length" if cut_off else "stop", "prompt_eval_count": usage_in, "eval_count": usage_out,
                             "prompt_eval_duration": 10**9, "eval_duration": slow * 10**9, "total_duration": (slow + 1) * 10**9, "load_duration": 0})
        else:
            self._send(200, {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "length" if cut_off else "stop"}], "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out}})

    def _reply_text(self, body) -> str:
        b = self.behaviour
        system = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
        user = next((m["content"] for m in body["messages"] if m["role"] == "user"), "")
        if "You label social media comments" in system:
            if b.labels_always_invalid:
                return "I cannot do that."
            return json.dumps(label_reply(b, user))
        if "You name groups" in system:
            groups = re.findall(r"^Group (\d+)\nKeywords: ([^\n]*)", user, flags=re.M)
            return "<think>hmm, let me think</think>" + json.dumps({"topics": [{"cluster": int(c), "name": k.split(",")[0].strip().capitalize()} for c, k in groups]})
        if "deciding what to check next" in system:
            b.investigate_calls += 1
            topic = re.search(r"Topics you can choose: ([^|\n]*)", user).group(1).strip()
            if b.investigate_script:
                return json.dumps(b.investigate_script.pop(0))
            if b.investigate_calls == 1:
                return json.dumps({"action": "top_quotes", "topic": topic, "sentiment": "negative", "why": "see the loudest complaints"})
            return json.dumps({"action": "stop", "why": "enough evidence"})
        if "You write the findings" in system:
            return json.dumps(write_reply(b, user))
        if "You answer questions" in system:
            ids = re.findall(r'^\{"id": "([^"]+)"', user, flags=re.M)
            return json.dumps({"answer": "People mention this.", "comment_ids": ids[:2]})
        return json.dumps({"ok": True})


def start_server() -> tuple[HTTPServer, Behaviour, str]:
    behaviour = Behaviour()
    handler = type("BoundHandler", (Handler,), {"behaviour": behaviour})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, behaviour, f"http://127.0.0.1:{server.server_port}"
