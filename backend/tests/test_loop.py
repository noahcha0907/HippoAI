"""Exercises the loop end to end against a scripted fake fixer over a fake HTTP server."""
import json
from pathlib import Path, PurePosixPath

import httpx
import numpy as np
import pytest
import torch

from repair.edits import EditSession
from repair.fixer_client import FixerClient, FixerConfig, FixerError
from repair.gate import AttemptScores
from repair.loop import run_repair
from repair.sandbox import PRELUDE, SandboxUnavailable, BubblewrapRunner, build_command
from repair.tools import ToolBox
from repair.validate import DevValidator, random_control_state, scaled_edit_state

LAYER = "model.language_model.layers.15.mlp.down_proj.weight"
EMBED = "model.language_model.embed_tokens.weight"
GOOD = AttemptScores(0.4, 0.5, 0.4, 0.0, 0.45)
BAD = AttemptScores(0.4, 0.4, 0.4, 0.0, 0.4)


def call(name, call_id="1", **arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def turn(*calls, content="", **extra):
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": list(calls), **extra}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}  # fmt: skip


class Server:
    """Replies with scripted turns, and records every request it receives."""

    def __init__(self, turns, status=200):
        self.turns = list(turns)
        self.requests = []
        self.status = status

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        if self.status != 200:
            return httpx.Response(self.status, text="nope")
        return httpx.Response(200, json=self.turns.pop(0))


def make_client(server, context_tokens=256_000):
    config = FixerConfig("http://fake/v1", "fake-model", "SECRET-KEY", context_tokens=context_tokens, max_retries=1)
    return FixerClient(config, transport=httpx.MockTransport(server), sleep=lambda _: None)


class FakeRunner:
    """Stands in for the sandbox: `code` names what to write to /work/edits."""

    def __init__(self, work_dir, session):
        self.work_dir, self.session = Path(work_dir), session
        self.codes = []

    def run(self, code):
        self.codes.append(code)
        (self.work_dir / "edits").mkdir(parents=True, exist_ok=True)
        edited = self.session.base[LAYER].numpy().copy()
        edited[0, 0] += 1
        np.save(self.work_dir / "edits" / f"{LAYER}.npy", edited)
        return "wrote edit"


@pytest.fixture
def box(tmp_path):
    def build(scores=(GOOD,), max_passes=25):
        torch.manual_seed(0)
        session = EditSession({LAYER: torch.randn(8, 4), EMBED: torch.randn(10, 2)})
        queue = list(scores)
        toolbox = ToolBox(
            session,
            FakeRunner(tmp_path / "work", session),
            tmp_path / "work",
            forward_pass_fn=lambda s, suite, category: {"accuracy": 0.4},
            attention_fn=lambda s, texts: {"attention": []},
            validator=lambda s: queue.pop(0),
            max_forward_passes=max_passes,
        )
        return toolbox

    return build


def repair_turns():
    return [
        turn(call("forward_pass", suite="clinical")),
        turn(call("run_python", code="save(...)")),
        turn(call("save_edit", tensor_name=LAYER)),
        turn(call("submit_attempt", summary="Fix it")),
        turn(call("finish", conclusion="repaired", cause="Damage in layer 15")),
    ]


def test_full_repair(box, tmp_path):
    toolbox = box()
    server = Server(repair_turns())
    result = run_repair(make_client(server), toolbox, "Wrong triage", tmp_path / "log.jsonl")
    assert result.outcome == "repaired" and result.stop_reason == "finished"
    assert result.diff["weights_changed"] == 1
    assert result.forward_passes == 1 and result.python_runs == 1
    assert toolbox.session.state()[LAYER][0, 0] == toolbox.session.base[LAYER][0, 0] + 1
    assert (tmp_path / "work" / "accepted" / f"{LAYER}.npy").exists()
    assert not (tmp_path / "work" / "edits").exists()
    log_text = (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    assert '"type": "result"' in log_text
    assert "SECRET-KEY" not in log_text


def test_tools_are_sent_and_reasoning_is_kept_for_later_turns(box, tmp_path):
    turns = [turn(call("forward_pass", suite="clinical"), reasoning_content="thinking"), *repair_turns()[-1:]]
    server = Server(turns)
    run_repair(make_client(server), box(), "Wrong triage", tmp_path / "log.jsonl")
    assert server.requests[0]["tools"]
    assert server.requests[1]["messages"][2]["reasoning_content"] == "thinking"


def test_fixer_cant_claim_repair_without_passing_gate(box, tmp_path):
    turns = [turn(call("finish", conclusion="repaired", cause="Trust me"))]
    result = run_repair(make_client(Server(turns)), box(), "x" * 5, tmp_path / "log.jsonl")
    assert result.outcome == "could_not_repair_safely"
    assert result.fixer_said == "repaired"


def test_failed_attempt_rolls_back_and_informs_fixer(box, tmp_path):
    toolbox = box(scores=(BAD,))
    turns = repair_turns()[1:]
    turns[-1] = turn(call("finish", conclusion="could_not_repair_safely", cause="No luck"))
    server = Server(turns)
    result = run_repair(make_client(server), toolbox, "Wrong triage", tmp_path / "log.jsonl")
    assert result.outcome == "could_not_repair_safely"
    assert result.diff["weights_changed"] == 0
    assert result.attempts[0].outcome == "rolled_back" and result.attempts[0].reasons
    tool_reply = server.requests[-1]["messages"][-1]["content"]
    assert "rolled_back" in tool_reply


def test_stops_after_three_rejections(box, tmp_path):
    turns = []
    for _ in range(3):
        turns += repair_turns()[1:4]
    result = run_repair(make_client(Server(turns)), box(scores=(BAD, BAD, BAD)), "Wrong triage", tmp_path / "l.jsonl")
    assert result.stop_reason == "no_improvement"
    assert len(result.attempts) == 3


def test_nothing_to_fix_control(box, tmp_path):
    turns = [turn(call("finish", conclusion="nothing_to_fix", cause="No damage found"))]
    result = run_repair(make_client(Server(turns)), box(), "Check it", tmp_path / "l.jsonl")
    assert result.outcome == "nothing_to_fix"
    assert result.diff["weights_changed"] == 0


def test_unsubmitted_edit_is_discarded(box, tmp_path):
    turns = repair_turns()[1:3] + [turn(call("finish", conclusion="repaired", cause="Done"))]
    toolbox = box()
    result = run_repair(make_client(Server(turns)), toolbox, "Wrong triage", tmp_path / "l.jsonl")
    assert result.outcome == "could_not_repair_safely"
    assert result.diff["weights_changed"] == 0


def test_time_limit(box, tmp_path):
    times = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0])
    turns = [turn(call("forward_pass", suite="clinical"))]
    result = run_repair(
        make_client(Server(turns)), box(), "Wrong triage", tmp_path / "l.jsonl", max_minutes=60, clock=lambda: next(times)
    )
    assert result.stop_reason == "time_limit"


def test_server_failure_ends_run_cleanly(box, tmp_path):
    result = run_repair(make_client(Server([], status=500)), box(), "Wrong triage", tmp_path / "l.jsonl")
    assert result.stop_reason == "fixer_error"
    assert result.outcome == "could_not_repair_safely"


def test_auth_failure_isnt_retried():
    server = Server([], status=401)
    with pytest.raises(FixerError, match="401"):
        make_client(server).chat([{"role": "user", "content": "hi"}])
    assert len(server.requests) == 1


def test_stops_nudging_silent_fixer(box, tmp_path):
    silent = {"choices": [{"message": {"role": "assistant", "content": "Hmm"}}], "usage": {}}
    result = run_repair(make_client(Server([silent] * 3)), box(), "Wrong triage", tmp_path / "l.jsonl")
    assert result.stop_reason == "fixer_stopped_without_finishing"


def test_handoff_starts_fresh_conversation_with_notes(box, tmp_path):
    turns = [
        turn(call("forward_pass", suite="clinical")),
        {"choices": [{"message": {"role": "assistant", "content": "MY NOTES"}}], "usage": {"prompt_tokens": 5}},
        turn(call("finish", conclusion="could_not_repair_safely", cause="Out of ideas")),
    ]
    server = Server(turns)
    result = run_repair(make_client(server, context_tokens=100), box(), "Wrong triage", tmp_path / "l.jsonl")
    assert result.handoffs == 1
    assert len(server.requests[2]["messages"]) == 2
    assert "MY NOTES" in server.requests[2]["messages"][1]["content"]
    assert "tools" not in server.requests[1]


# Tool-level failures come back as errors instead of crashing.
def test_forward_pass_budget_enforced_and_failures_are_free(box):
    toolbox = box(max_passes=1)
    toolbox.forward_pass_fn = lambda *a: 1 / 0
    assert "ZeroDivisionError" in toolbox.call("forward_pass", {"suite": "clinical"})
    assert toolbox.forward_passes_used == 0
    toolbox.forward_pass_fn = lambda *a: {"accuracy": 0.5}
    assert "forward_passes_left" in toolbox.call("forward_pass", {"suite": "clinical"})
    assert "used up" in toolbox.call("forward_pass", {"suite": "clinical"})


@pytest.mark.parametrize(
    "name,arguments,expected",
    [
        ("nope", "{}", "unknown tool"),
        ("run_python", "{not json", "Error"),
        ("run_python", "[1]", "must be an object"),
        ("run_python", {"wrong_arg": 1}, "Error"),
        ("run_python", {"code": "loss.backward()"}, "not allowed"),
        ("run_python", {"code": "AutoModel.from_pretrained('x')"}, "not allowed"),
        ("save_edit", {"tensor_name": "../../secret"}, "Unknown tensor"),
        ("save_edit", {"tensor_name": EMBED}, "No saved edit"),
        ("submit_attempt", {"summary": "nothing"}, "no edits"),
        ("attention", {"texts": []}, "between 1 and 16"),
        ("finish", {"conclusion": "bogus", "cause": "x"}, "Invalid conclusion"),
    ],
)
def test_bad_tool_calls_return_errors(box, name, arguments, expected):
    assert expected in box().call(name, arguments)


def test_save_edit_refuses_protected_wrong_shape_and_pickle(box, tmp_path):
    toolbox = box()
    edits = tmp_path / "work" / "edits"
    edits.mkdir(parents=True)
    np.save(edits / f"{EMBED}.npy", np.ones((10, 2), np.float32))
    assert "Protected" in toolbox.call("save_edit", {"tensor_name": EMBED})
    np.save(edits / f"{LAYER}.npy", np.ones((2, 2), np.float32))
    assert "doesn't match" in toolbox.call("save_edit", {"tensor_name": LAYER})
    np.save(edits / f"{LAYER}.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    assert "Error" in toolbox.call("save_edit", {"tensor_name": LAYER})
    assert toolbox.session.pending_names() == []


def test_save_edit_refuses_symlink(box, tmp_path):
    toolbox = box()
    edits = tmp_path / "work" / "edits"
    edits.mkdir(parents=True)
    target = tmp_path / "outside.npy"
    np.save(target, np.ones((8, 4), np.float32))
    try:
        (edits / f"{LAYER}.npy").symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks needs extra privileges here")
    assert "No saved edit" in toolbox.call("save_edit", {"tensor_name": LAYER})


# Sandbox isolation: checked from the command, since bubblewrap can't run on Windows.
def test_sandbox_command_isolates_the_code():
    work, weights, venv = PurePosixPath("/runs/a/work"), PurePosixPath("/models/broken.safetensors"), PurePosixPath("/venv")
    command = build_command("print(1)", work, weights, venv)
    for flag in ("--unshare-all", "--clearenv", "--die-with-parent", "--new-session"):
        assert flag in command
    binds = [command[i + 1] for i, part in enumerate(command) if part in ("--bind", "--ro-bind")]
    assert sorted(binds) == ["/models/broken.safetensors", "/runs/a/work", "/usr", "/venv"]
    assert command[command.index("--bind") + 1] == "/runs/a/work"  # The only writable path.
    assert command.count("--bind") == 1
    assert "print(1)" in command[-1] and PRELUDE in command[-1]


def test_sandbox_refuses_to_run_without_bubblewrap(tmp_path, monkeypatch):
    monkeypatch.setattr("repair.sandbox.shutil.which", lambda _: None)
    with pytest.raises(SandboxUnavailable):
        BubblewrapRunner(tmp_path, tmp_path / "w.safetensors", venv=tmp_path)


# Validator controls
@pytest.fixture
def edited_session():
    torch.manual_seed(0)
    base = {LAYER: torch.randn(50, 50), EMBED: torch.randn(10, 2)}
    session = EditSession(base)
    edited = base[LAYER].clone()
    edited[:10, :10] += 2.0
    session.save_edit(LAYER, edited)
    return session


def test_half_strength_is_halfway(edited_session):
    half = scaled_edit_state(edited_session, 0.5)
    expected = edited_session.base[LAYER][0, 0] + 1.0
    assert half[LAYER][0, 0].item() == pytest.approx(expected.item())
    assert half[EMBED] is edited_session.base[EMBED]


def test_random_control_matches_norm_and_weights_touched(edited_session):
    base = edited_session.base[LAYER]
    real = edited_session.state()[LAYER] - base
    control = random_control_state(edited_session)[LAYER] - base
    assert control.norm().item() == pytest.approx(real.norm().item(), rel=1e-4)
    assert torch.equal(control != 0, real != 0)
    assert not torch.allclose(control, real)


def test_dev_validator_computes_all_scores(edited_session):
    calls = []

    def score(state, suite):
        calls.append(suite)
        is_broken = state[LAYER] is edited_session.base[LAYER]
        return np.array([True, False, False, False]) if is_broken else np.array([True, True, False, False])

    scores = DevValidator(score)(edited_session)
    assert scores.broken_clinical == 0.25 and scores.edited_clinical == 0.5
    assert scores.collateral_loss == 0.0
    assert calls.count("medmcqa") == 2
