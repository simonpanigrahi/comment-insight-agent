"""The two commands people run first when a model misbehaves."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_speed_test():
    spec = importlib.util.spec_from_file_location("speed_test", ROOT / "scripts" / "speed_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_speed_test_reports_placement_and_the_cost_of_the_schema(fake_llm, capsys):
    b, url = fake_llm
    assert load_speed_test().main(["--url", url, "--model", "fake-model"]) == 0
    out = capsys.readouterr().out
    assert "100% of it on the GPU" in out and "fully on the GPU" in out
    assert "JSON schema" in out and "plain JSON" in out and "no limit" in out
    assert "OLLAMA_FORMAT=" in out  # the pretend server is 4 times slower with a schema
    assert "labelling all 2,533" in out


def test_speed_test_says_when_the_model_is_on_the_cpu(fake_llm, capsys, monkeypatch):
    b, url = fake_llm
    b.vram_bytes = 0
    module = load_speed_test()
    monkeypatch.setattr(module, "shell", lambda *c: "NVIDIA RTX, 550, 10 MiB, 4096 MiB" if c[0] == "nvidia-smi" else "")
    module.main(["--url", url, "--model", "fake-model"])
    out = capsys.readouterr().out
    assert "entirely on the CPU" in out and "rmmod nvidia_uvm" in out


def test_speed_test_says_when_the_model_only_partly_fits(fake_llm, capsys, monkeypatch):
    b, url = fake_llm
    b.vram_bytes = 1500000000
    module = load_speed_test()
    monkeypatch.setattr(module, "shell", lambda *c: "NVIDIA RTX, 550, 10 MiB, 4096 MiB" if c[0] == "nvidia-smi" else "")
    module.main(["--url", url, "--model", "fake-model"])
    out = capsys.readouterr().out
    assert "Only 50% of the model fits" in out


def test_speed_test_with_no_server_does_not_crash(capsys):
    assert load_speed_test().main(["--url", "http://127.0.0.1:9", "--model", "fake-model"]) == 1
    assert "Could not reach Ollama" in capsys.readouterr().out


def run_cli(tmp_path, url, *args):
    env = {**os.environ, "LLM_PROVIDER": "ollama", "LLM_MODEL": "fake-model", "LLM_BASE_URL": url}
    return subprocess.run([sys.executable, str(ROOT / "run.py"), *args], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


def test_doctor_end_to_end(fake_llm, tmp_path):
    b, url = fake_llm
    b.vram_bytes = 1500000000
    done = run_cli(tmp_path, url, "doctor", "--provider", "ollama")
    assert done.returncode == 0, done.stderr
    assert "OK: valid JSON" in done.stdout and "Right on" in done.stdout
    assert "50% of it on the GPU" in done.stdout


def test_cli_errors_are_one_clear_line(fake_llm, tmp_path):
    b, url = fake_llm
    done = run_cli(tmp_path, url, "analyse", "does_not_exist.csv")
    assert done.returncode == 2 and "File not found" in done.stderr and "Traceback" not in done.stderr
