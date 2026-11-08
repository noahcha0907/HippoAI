"""Scores a pending edit on dev data, including the controls the gate needs.

`score_fn(state, suite)` returns one boolean per dev question, and `suite` is
"clinical" or "medmcqa". It's injected so this logic runs without a real model.
"""
from typing import Callable

import numpy as np
import torch

from eval.metrics import collateral_loss
from eval.scoring import known
from repair.edits import EditSession
from repair.gate import AttemptScores

ScoreFn = Callable[[dict[str, torch.Tensor], str], np.ndarray]


def scaled_edit_state(session: EditSession, strength: float) -> dict[str, torch.Tensor]:
    """The full state with every edit applied at a fraction of its size."""
    state = session.state()
    for name, base in session.base.items():
        if state[name] is not base:
            delta = state[name].to(torch.float32) - base.to(torch.float32)
            state[name] = (base.to(torch.float32) + strength * delta).to(base.dtype)
    return state


def random_control_state(session: EditSession, seed: int = 0) -> dict[str, torch.Tensor]:
    """The same tensors changed by the same amount, in a random direction over the same weights."""
    generator = torch.Generator().manual_seed(seed)
    state = session.state()
    for name, base in session.base.items():
        if state[name] is base:
            continue
        delta = state[name].to(torch.float32) - base.to(torch.float32)
        mask = delta != 0
        noise = torch.randn(delta.shape, generator=generator) * mask
        norm = noise.norm()
        if norm > 0:
            noise = noise * (delta.norm() / norm)
        state[name] = (base.to(torch.float32) + noise).to(base.dtype)
    return state


class DevValidator:
    """`general_suite` measures side effects. With `rotations` > 1 it holds every question in every
    answer order (see eval.scoring.rotated_set), and only questions the model gets right in most
    orders count as known, so lucky letter guesses can't register as lost knowledge."""

    def __init__(
        self,
        score_fn: ScoreFn,
        garbled_fn: Callable[[dict], float] | None = None,
        general_suite: str = "medmcqa",
        rotations: int = 1,
    ):
        self._score = score_fn
        self._garbled = garbled_fn
        self._general_suite = general_suite
        self._rotations = rotations
        self._broken_clinical: np.ndarray | None = None
        self._broken_medmcqa: np.ndarray | None = None
        self._broken_garbled = 0.0

    def _general(self, state) -> np.ndarray:
        correct = np.asarray(self._score(state, self._general_suite), dtype=bool)
        if self._rotations == 1:
            return correct
        return known(correct, self._rotations)

    def __call__(self, session: EditSession) -> AttemptScores:
        if self._broken_clinical is None:
            self._broken_clinical = np.asarray(self._score(session.base, "clinical"), dtype=bool)
            self._broken_medmcqa = self._general(session.base)
            self._broken_garbled = self._garbled(session.base) if self._garbled else 0.0
        edited = session.state()
        edited_medmcqa = self._general(edited)
        control = random_control_state(session)
        control_medmcqa = self._general(control)
        return AttemptScores(
            broken_clinical=float(self._broken_clinical.mean()),
            edited_clinical=float(np.mean(self._score(edited, "clinical"))),
            random_control_clinical=float(np.mean(self._score(control, "clinical"))),
            collateral_loss=collateral_loss(self._broken_medmcqa, edited_medmcqa),
            general_net_change=float(edited_medmcqa.mean() - self._broken_medmcqa.mean()),
            random_control_collateral=collateral_loss(self._broken_medmcqa, control_medmcqa),
            half_strength_clinical=float(np.mean(self._score(scaled_edit_state(session, 0.5), "clinical"))),
            garbled_rate_increase=max(0.0, self._garbled(edited) - self._broken_garbled) if self._garbled else 0.0,
        )
