"""Settings: which LLM to use and how much of it. Read from .env, overridable by command line flags."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# rpm = calls per minute we allow ourselves (0 means no limit).
# max_calls = None means no cap, which suits a local model: its only cost is time.
# Cloud free tiers have daily limits, so their presets stay inside them.
PRESETS = {
    "ollama": dict(base_url="http://localhost:11434", model="qwen3:4b", batch_size=12, max_calls=None, rpm=0, timeout=300),
    "gemini": dict(base_url="https://generativelanguage.googleapis.com/v1beta/openai", model="gemini-flash-lite-latest",
                   batch_size=25, max_calls=200, rpm=8, timeout=120),
    "groq": dict(base_url="https://api.groq.com/openai/v1", model="openai/gpt-oss-120b", batch_size=20, max_calls=80, rpm=3, timeout=120),
    "openai_compatible": dict(base_url="http://localhost:8000/v1", model="", batch_size=20, max_calls=None, rpm=0, timeout=300),
    "none": dict(base_url="", model="", batch_size=0, max_calls=0, rpm=0, timeout=0),
}
NEEDS_KEY = {"gemini", "groq"}


@dataclass
class Settings:
    provider: str = "none"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    batch_size: int = 0
    max_calls: int | None = 0
    rpm: int = 0
    timeout: int = 0
    num_ctx: int = 4096  # Ollama context window; a small one fits a 4 GB GPU
    ollama_format: str = "schema"  # how tightly Ollama limits a reply: schema, json or none. Tighter is safer, but can be slower.
    cache_dir: str = ".cache/llm"

    @property
    def uses_llm(self) -> bool:
        return self.provider != "none"


def load_settings(provider=None, model=None, max_calls=None, batch_size=None) -> Settings:
    """Flags win over .env, and .env wins over the preset defaults."""
    load_dotenv()
    env_provider = (os.getenv("LLM_PROVIDER") or "").lower()
    name = (provider or env_provider or "none").lower()
    if name not in PRESETS:
        raise SystemExit(f"Unknown provider '{name}'. Choose one of: {', '.join(PRESETS)}")
    p = PRESETS[name]
    # A model or server named in .env belongs to the provider named in .env, so ignore them if a flag chose another provider.
    same = name == env_provider
    return Settings(
        provider=name,
        model=model or (os.getenv("LLM_MODEL") if same else None) or p["model"],
        base_url=(os.getenv("LLM_BASE_URL") if same else None) or p["base_url"],
        api_key=os.getenv(f"{name.upper()}_API_KEY") or os.getenv("LLM_API_KEY", ""),
        batch_size=batch_size or int(os.getenv("LLM_BATCH_SIZE", p["batch_size"])),
        max_calls=p["max_calls"] if max_calls is None else max_calls,
        rpm=int(os.getenv("LLM_RPM", p["rpm"])),
        timeout=int(os.getenv("LLM_TIMEOUT", p["timeout"])),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", 4096)),
        ollama_format=os.getenv("OLLAMA_FORMAT", "schema").strip().lower(),
    )
