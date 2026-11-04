"""The LLM client. One small class that talks to Ollama or any OpenAI-compatible server.

It uses plain HTTP so there is no vendor lock-in: the same code runs on a laptop GPU, an
on-premise server or a cloud API. It adds what a real run needs: a call budget, rate limiting,
retries, a disk cache, and validation of every reply against a schema.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import requests
from pydantic import BaseModel

from .config import Settings
from .runlog import RunLog

THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", flags=re.S)


class LLMError(Exception):
    """A call failed even after retries."""


class LLMFatal(LLMError):
    """Retrying will not help: wrong key, unknown model, server not running."""


class BudgetExceeded(LLMError):
    """The call budget for this run is used up."""


def json_candidates(text: str) -> list:
    """Every complete JSON value in a reply. Models may add notes, thinking tags or code fences around the JSON."""
    text = THINK_RE.sub("", text)
    decoder = json.JSONDecoder()
    found, i = [], 0
    while i < len(text):
        if text[i] in "{[":
            try:
                value, end = decoder.raw_decode(text, i)
                found.append(value)
                i = end
                continue
            except ValueError:
                pass
        i += 1
    return found


def parse_reply(text: str, schema: type[BaseModel]) -> BaseModel:
    """Return the first JSON value that fits the schema. A bare list is accepted where the schema wraps one in results."""
    last_error = None
    for value in json_candidates(text):
        if isinstance(value, list) and "results" in schema.model_fields:
            value = {"results": value}
        try:
            return schema.model_validate(value)
        except ValueError as e:  # pydantic's ValidationError is a kind of ValueError
            last_error = e
    raise ValueError(str(last_error)[:400] if last_error else "no JSON object in the reply")


class _Ping(BaseModel):
    ok: bool


class LLM:
    def __init__(self, settings: Settings, log: RunLog | None = None):
        self.s = settings
        self.log = log
        self.calls = 0
        self.cached = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.seconds = 0.0
        self._last_call = 0.0
        self._use_json_mode = True
        self.cache_dir = Path(settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def calls_left(self) -> int:
        return 10**9 if self.s.max_calls is None else max(0, self.s.max_calls - self.calls)

    def ask(self, system: str, user: str, schema: type[BaseModel], purpose: str = "", max_tokens: int = 1500, use_cache: bool = True) -> BaseModel:
        """Send a prompt and return a validated object. An invalid reply gets the error sent back, up to two more tries."""
        key = hashlib.sha256(f"{self.s.model}\n{schema.__name__}\n{system}\n{user}".encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if use_cache and path.exists():  # a crashed or repeated run does not spend quota again
            try:
                result = schema.model_validate_json(path.read_text(encoding="utf-8"))
                self.cached += 1
                return result
            except (ValueError, OSError):
                pass  # a damaged cache file is simply a miss

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error = ""
        for _ in range(3):
            if self.calls_left <= 0:
                raise BudgetExceeded(f"budget of {self.s.max_calls} calls is used up")
            reply, complete = self._post(messages, schema, max_tokens, purpose)
            if not complete:  # asking again would be cut off at the same place
                raise LLMError("the reply was cut off by the token limit")
            try:
                result = parse_reply(reply, schema)
                self._save(path, result)
                return result
            except ValueError as e:
                last_error = str(e)[:400]
                if self.log:
                    self.log.event("llm", purpose=purpose, invalid_reply=last_error)
                messages = messages[:2] + [
                    {"role": "assistant", "content": reply[:3000]},
                    {"role": "user", "content": f"That reply was not valid: {last_error}\nReturn only valid JSON in the requested shape."},
                ]
        raise LLMError(f"no valid reply after 3 tries: {last_error}")

    def ping(self) -> None:
        """A small real call that skips the cache, to confirm the server, model and key all work."""
        reply, _ = self._post([{"role": "user", "content": 'Return this JSON only: {"ok": true}'}], _Ping, 300, "ping", attempts=2, timeout=180)
        try:
            parse_reply(reply, _Ping)
        except ValueError as e:
            raise LLMError(f"the model did not return valid JSON: {str(e)[:200]}")

    def _save(self, path: Path, result: BaseModel) -> None:
        """Write to a temporary file first, so a crash cannot leave a half-written cache entry."""
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json())
        os.replace(tmp, path)

    def _post(self, messages: list[dict], schema: type[BaseModel], max_tokens: int, purpose: str, attempts: int | None = None, timeout: int | None = None) -> tuple[str, bool]:
        """One logical call, with waiting and retrying on rate limits and server errors. Returns the text and whether it was complete."""
        self._respect_rate_limit()
        self.calls += 1
        started = time.time()
        first_call = self.calls <= 1  # if the very first call cannot connect, the server is simply not running
        delay = 2.0 if first_call else 5.0
        tries = 0
        while True:
            tries += 1
            try:
                if self.s.provider == "ollama":
                    text, complete, tin, tout = self._post_ollama(messages, schema, max_tokens, timeout)
                else:
                    text, complete, tin, tout = self._post_openai(messages, max_tokens, timeout)
                elapsed = time.time() - started
                self.seconds += elapsed
                self.tokens_in += tin
                self.tokens_out += tout
                if self.log:
                    self.log.event("llm", purpose=purpose, seconds=round(elapsed, 2), tokens_in=tin, tokens_out=tout, call=self.calls, complete=complete)
                return text, complete
            except _Retry as e:
                limit = attempts or (5 if e.rate_limited or (e.connection and not first_call) else 3)
                if tries >= limit:
                    if e.connection:
                        raise LLMFatal(str(e))
                    raise LLMError(f"gave up after {tries} attempts: {e}")
                wait = e.wait if e.wait is not None else delay
                if self.log:
                    self.log.event("llm", purpose=purpose, retry=str(e), wait=wait)
                time.sleep(wait)
                delay = min(delay * 2, 60)

    def _respect_rate_limit(self) -> None:
        if self.s.rpm:
            gap = 60.0 / self.s.rpm - (time.time() - self._last_call)
            if gap > 0:
                time.sleep(gap)
        self._last_call = time.time()

    def _post_ollama(self, messages, schema, max_tokens, timeout):
        # The prompt and the reply must both fit in the context window, and a long prompt would otherwise be cut silently.
        estimate = sum(len(m["content"]) for m in messages) // 3 + max_tokens + 100
        body = {
            "model": self.s.model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "num_ctx": max(self.s.num_ctx, min(estimate, 8192)), "num_predict": max_tokens},
        }
        # With a schema the server can only produce JSON of this shape, at some cost in speed. Replies are validated here either way.
        limit = {"schema": schema.model_json_schema(), "json": "json"}.get(self.s.ollama_format)
        if limit:
            body["format"] = limit
        data = self._http(f"{self.s.base_url.rstrip('/')}/api/chat", body, {}, timeout)
        if data.get("error"):
            raise LLMError(str(data["error"])[:300])
        message = data.get("message") or {}
        return message.get("content") or "", data.get("done_reason") != "length", data.get("prompt_eval_count", 0), data.get("eval_count", 0)

    def _post_openai(self, messages, max_tokens, timeout):
        body = {"model": self.s.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if self._use_json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.s.api_key}"} if self.s.api_key else {}
        url = f"{self.s.base_url.rstrip('/')}/chat/completions"
        try:
            data = self._http(url, body, headers, timeout)
        except _BadRequest as e:
            if not self._use_json_mode or "json_validate_failed" in str(e):
                raise LLMError(str(e))  # this one call failed, not the whole server
            body.pop("response_format")  # the server may not support JSON mode, so try once without it
            data = self._http(url, body, headers, timeout)
            self._use_json_mode = False
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"the reply had no choices: {str(data)[:200]}")
        finish = choices[0].get("finish_reason")
        if finish == "content_filter":
            raise LLMError("the server's content filter blocked this reply")
        usage = data.get("usage") or {}
        content = (choices[0].get("message") or {}).get("content") or ""
        return content, finish != "length", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def _http(self, url: str, body: dict, headers: dict, timeout: int | None) -> dict:
        try:
            r = requests.post(url, json=body, headers=headers, timeout=timeout or self.s.timeout or 300)
        except requests.ConnectionError as e:
            raise _Retry(f"cannot reach {url}. Is the server running? ({type(e).__name__})", connection=True)
        except requests.Timeout:
            raise _Retry(f"timed out after {timeout or self.s.timeout} seconds")
        except requests.RequestException as e:
            raise _Retry(str(e)[:200])
        if r.status_code == 429:
            raise _Retry("rate limited (429)", wait=_retry_after(r), rate_limited=True)
        if r.status_code >= 500:
            raise _Retry(f"server error {r.status_code}")
        if r.status_code == 400:
            raise _BadRequest(f"400: {r.text[:300]}")
        if not r.ok:  # 401, 403, 404 and so on: wrong key or unknown model
            raise LLMFatal(f"{r.status_code}: {r.text[:300]}")
        try:
            data = r.json()
        except ValueError:
            raise _Retry("the server returned something that is not JSON")
        if not isinstance(data, dict):
            raise _Retry("the server returned an unexpected shape")
        return data

    def usage(self) -> dict:
        return {"provider": self.s.provider, "model": self.s.model, "calls": self.calls, "cached_calls": self.cached, "budget": self.s.max_calls,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out, "seconds_in_llm": round(self.seconds, 1)}


class _Retry(Exception):
    def __init__(self, message: str, wait: float | None = None, rate_limited: bool = False, connection: bool = False):
        super().__init__(message)
        self.wait, self.rate_limited, self.connection = wait, rate_limited, connection


class _BadRequest(LLMError):
    pass


def _retry_after(response) -> float | None:
    try:
        return float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None


def ollama_placement(s: Settings) -> dict | None:
    """Where Ollama has loaded the model. One that is split across the GPU and the CPU runs far slower than one fully on the GPU."""
    try:
        models = requests.get(f"{s.base_url.rstrip('/')}/api/ps", timeout=15).json().get("models", [])
    except (requests.RequestException, ValueError, AttributeError):
        return None
    for m in models:
        if m.get("name") == s.model or len(models) == 1:
            size = m.get("size") or 0
            return {"name": m.get("name", s.model), "gb": size / 1e9, "gpu_share": 100.0 * (m.get("size_vram") or 0) / size if size else 0.0, "context": m.get("context_length")}
    return None


def list_models(s: Settings) -> list[str]:
    """Used by the doctor command, to show what is available when a model name is wrong."""
    try:
        if s.provider == "ollama":
            r = requests.get(f"{s.base_url.rstrip('/')}/api/tags", timeout=15)
            return [m["name"] for m in r.json().get("models", [])]
        headers = {"Authorization": f"Bearer {s.api_key}"} if s.api_key else {}
        r = requests.get(f"{s.base_url.rstrip('/')}/models", headers=headers, timeout=15)
        return sorted(m["id"] for m in r.json().get("data", []))
    except Exception:
        return []
