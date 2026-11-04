import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_llm import start_server  # noqa: E402

from agent.config import Settings  # noqa: E402


@pytest.fixture
def fake_llm():
    server, behaviour, url = start_server()
    yield behaviour, url
    server.shutdown()


def make_settings(url: str, provider: str, tmp_path, max_calls: int = 500, batch_size: int = 25) -> Settings:
    base = url if provider == "ollama" else url + "/v1"
    return Settings(provider=provider, model="fake-model", base_url=base, batch_size=batch_size, max_calls=max_calls, timeout=10, cache_dir=str(tmp_path / "cache"))


@pytest.fixture
def settings_for(tmp_path):
    return lambda url, provider="ollama", **kw: make_settings(url, provider, tmp_path, **kw)
