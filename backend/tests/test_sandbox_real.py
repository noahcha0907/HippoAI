"""Attacks the real bubblewrap sandbox. Runs only where it's available (WSL or Linux).

Run in WSL from backend/:  SANDBOX_VENV=~/hippo-venv ~/hippo-venv/bin/python -m pytest tests/test_sandbox_real.py
"""
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bwrap") is None or not os.environ.get("SANDBOX_VENV"),
    reason="Needs Linux, bubblewrap, and SANDBOX_VENV",
)

LAYER = "model.language_model.layers.0.mlp.down_proj.weight"


@pytest.fixture
def runner(tmp_path, monkeypatch):
    from repair.sandbox import BubblewrapRunner

    monkeypatch.setenv("FIXER_API_KEY", "SECRET-VALUE")
    weights = tmp_path / "model.safetensors"
    save_file({LAYER: torch.ones(2, 3).to(torch.bfloat16)}, weights)
    (tmp_path / "evaluator-only.txt").write_text("SEALED", encoding="utf-8")
    return BubblewrapRunner(tmp_path / "run" / "work", weights, Path(os.environ["SANDBOX_VENV"]))


def test_can_read_weights_and_write_to_work(runner):
    code = f"print(NAMES, shape('{LAYER}'), load('{LAYER}').sum())\nopen('/work/note.txt','w').write('ok')\nprint('wrote')"
    output = runner.run(code)
    assert LAYER in output and "6.0" in output and "wrote" in output
    assert (runner.work_dir / "note.txt").read_text() == "ok"


def test_save_writes_a_valid_edit(runner):
    output = runner.run(f"save('{LAYER}', np.zeros((2, 3)))\nprint('saved')")
    assert "saved" in output
    assert (runner.work_dir / "edits" / f"{LAYER}.npy").is_file()


def test_no_network(runner):
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3); print('CONNECTED')\n"
        "except OSError as e:\n"
        "    print('blocked', e)\n"
    )
    assert "blocked" in runner.run(code)


def test_environment_has_no_secrets(runner):
    assert "SECRET-VALUE" not in runner.run("print(dict(os.environ))")


def test_cannot_see_host_files(runner, tmp_path):
    for path in (str(tmp_path / "evaluator-only.txt"), "/mnt/c", str(Path.home() / ".cache"), "/etc/passwd"):
        assert "VISIBLE" not in runner.run(f"print('VISIBLE' if os.path.exists({path!r}) else 'hidden')"), path


def test_cannot_write_outside_work(runner):
    code = (
        "for p in ['/usr/x', '/weights/x', '/weights/model.safetensors']:\n"
        "    try:\n"
        "        open(p, 'w'); print('WROTE', p)\n"
        "    except OSError:\n"
        "        print('refused', p)\n"
    )
    assert "WROTE" not in runner.run(code)


def test_no_gpu(runner):
    assert "False" in runner.run("print(torch.cuda.is_available())")


def test_failures_and_stderr_come_back(runner):
    assert "ZeroDivisionError" in runner.run("1/0")


def test_memory_cap_stops_runaway_code(runner):
    output = runner.run("a = np.ones(int(3 * 1024**3 / 8)); print('ALLOCATED')")
    assert "ALLOCATED" not in output
    assert "memory limit" in output


def test_normal_code_still_runs_under_cap(runner):
    assert "ok" in runner.run("a = np.ones(10_000_000); print('ok')")
