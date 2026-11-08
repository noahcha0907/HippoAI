"""Runs the fixer's Python in a bubblewrap sandbox. Linux only.

Adapted from harness/sandbox.py in github.com/RikiyaT/lm_understand-logs.

The sandbox sees only the broken weights (read-only), its own work folder, and system
libraries. It has no network, GPU, API keys, sealed data, or clean model. There's
deliberately no fallback on other platforms: this code is written by an LLM.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT_SECONDS = 300
MAX_OUTPUT_CHARS = 20_000
MEMORY_BYTES = 32 << 30

PRELUDE = r'''
import json, os, numpy as np, torch
from safetensors import safe_open
np.set_printoptions(precision=4, suppress=True, linewidth=160)
_W = safe_open("/weights/model.safetensors", framework="pt")
NAMES = sorted(_W.keys())
def shape(name):
    return tuple(_W.get_slice(name).get_shape())
def load(name):
    """Current value of a tensor: the accepted edit if there is one, else the broken model."""
    p = f"/work/accepted/{name}.npy"
    return np.load(p, allow_pickle=False) if os.path.exists(p) else _W.get_tensor(name).float().numpy()
def save(name, array):
    """Write a candidate edit. Call the save_edit tool afterward to queue it."""
    a = np.asarray(array, np.float32)
    assert name in NAMES, f"unknown tensor {name}"
    assert a.shape == shape(name), (name, a.shape, shape(name))
    assert np.isfinite(a).all(), name
    os.makedirs("/work/edits", exist_ok=True)
    np.save(f"/work/edits/{name}.npy", a, allow_pickle=False)
'''


class SandboxUnavailable(RuntimeError):
    """Bubblewrap can't be used here."""


def build_command(code: str, work_dir: Path, weights_path: Path, venv: Path) -> list[str]:
    return [
        "bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", str(venv), str(venv),
        "--ro-bind", str(weights_path), "/weights/model.safetensors",
        "--bind", str(work_dir), "/work", "--chdir", "/work",
        "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin",
        "--setenv", "OMP_NUM_THREADS", "8", "--setenv", "CUDA_VISIBLE_DEVICES", "",
        "--remount-ro", "/",  # Only /work and /tmp stay writable.
        f"{venv}/bin/python", "-I", "-c", PRELUDE + "\n" + code,
    ]  # fmt: skip


def _limit_memory() -> None:
    import resource  # Not available on Windows.

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))


class BubblewrapRunner:
    def __init__(self, work_dir: Path, weights_path: Path, venv: Path | None = None):
        venv = venv or Path(os.environ.get("SANDBOX_VENV", ""))
        if not sys.platform.startswith("linux") or shutil.which("bwrap") is None:
            raise SandboxUnavailable("The sandbox needs Linux with bubblewrap installed")
        if not (venv / "bin" / "python").exists():
            raise SandboxUnavailable(f"No sandbox Python found in {venv!r}; set SANDBOX_VENV")
        self.work_dir, self.weights_path, self.venv = Path(work_dir), Path(weights_path), venv
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # A real memory cap (RLIMIT_AS can't be tight: importing PyTorch reserves ~8 GB of address space).
        self.memory_cap = os.environ.get("SANDBOX_MEMORY", "1500M")
        if self.memory_cap and shutil.which("systemd-run") is None:
            raise SandboxUnavailable("systemd-run is needed for the memory cap; set SANDBOX_MEMORY= to skip it")

    def _capped(self, command: list[str]) -> list[str]:
        if not self.memory_cap:
            return command
        return [
            "systemd-run", "--user", "--scope", "--quiet",
            "-p", f"MemoryMax={self.memory_cap}", "-p", "MemorySwapMax=0", *command,
        ]  # fmt: skip

    def run(self, code: str) -> str:
        command = self._capped(build_command(code, self.work_dir, self.weights_path, self.venv))
        started = time.time()
        try:
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, preexec_fn=_limit_memory
            )
            output = done.stdout + (f"\n[stderr]\n{done.stderr[-6000:]}" if done.stderr.strip() else "")
            if done.returncode in (137, 143, -9, -15):
                output += f"\n[stopped: the {self.memory_cap} memory limit was reached. Work on slices of large tensors]"
        except subprocess.TimeoutExpired:
            output = f"[timed out after {TIMEOUT_SECONDS} s]"
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n[output cut at {MAX_OUTPUT_CHARS:,} characters]"
        with (self.work_dir.parent / "python_log.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"t": started, "s": round(time.time() - started, 2), "code": code, "out": output[:6000]}) + "\n")
        return output or "(no output)"
