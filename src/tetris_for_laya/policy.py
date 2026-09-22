"""One Laya keyboard decision per step, from board observations, with input legality checks."""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .game import (
    ACTIONS,
    GRAVITY_STEPS,
    Piece,
    StepTransition,
    TetrisGame,
    piece_cells,
)

DEFAULT_MODEL = Path("models/laya-multilingual-mlx")


class DecisionError(RuntimeError):
    """Raised when a model response cannot be executed safely."""


class AgentLike(Protocol):
    def predict(self, state: object, questions: dict[str, object]) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Candidate:
    label: str
    transition: StepTransition
    legal: bool


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    candidates: tuple[Candidate, ...]
    probabilities: dict[str, float]
    proposed: str
    executed: str
    intervened: bool
    inference_ms: float
    decision_ms: float
    input_tokens: int
    model_called: bool


def resolve_checkpoint(value: str | Path = DEFAULT_MODEL) -> Path:
    """Resolve a local checkpoint while guaranteeing that play never downloads."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    path = Path(value).expanduser()
    if path.is_dir():
        return path
    if str(value).startswith((".", "/", "~")) or "/" not in str(value):
        raise FileNotFoundError(
            f"Local Laya checkpoint does not exist: {value}\n"
            "Download it first with:\n"
            "  hf download aac6fef/laya-multilingual-mlx "
            "--local-dir models/laya-multilingual-mlx"
        )
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(str(value), local_files_only=True))
    except Exception as error:
        raise FileNotFoundError(
            f"{value} is not cached. Download the checkpoint before starting the game."
        ) from error


class LayaPolicy:
    """Observe the current piece and ask Laya for exactly one keyboard action."""

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        guarded: bool = True,
        optimize: bool = False,
        agent: AgentLike | None = None,
    ) -> None:
        self.guarded = guarded
        self._board_key: tuple | None = None
        self._history: deque[tuple[str, Piece, StepTransition]] = deque(maxlen=8)
        self._pending: tuple[int, str, Piece, StepTransition] | None = None
        if agent is not None:
            self.agent = agent
            self.model_path: Path | None = None
            return

        from laya_mlx import load

        self.model_path = resolve_checkpoint(model)
        self.agent = load(
            self.model_path,
            dtype="float16",
            device="gpu",
            batch_size=1,
            compile=optimize,
            pad_to_multiple=16 if optimize else None,
            cache_prompts=optimize,
        )

    def decide(self, game: TetrisGame) -> PolicyDecision:
        if game.game_over:
            raise DecisionError("The game is over")
        started = time.perf_counter()
        snapshot = game.snapshot()
        node = (snapshot.current, game.gravity_phase)
        key = (snapshot.pieces, snapshot.board, snapshot.current.kind, snapshot.next_kind)
        if key != self._board_key:
            self._board_key = key
            self._history.clear()
            self._pending = None
        if self._pending is not None:
            step, action, before, transition = self._pending
            if game.steps == step + 1 and node == (transition.piece, transition.gravity_phase):
                self._history.append((action, before, transition))
            elif game.steps != step:
                self._history.clear()
            self._pending = None
        # Only immediate input legality is inspected; no landing search or scoring.
        candidates = tuple(
            Candidate(action, transition, action in ("DOWN", "WAIT") or transition.moved)
            for action in ACTIONS
            for transition in (game.preview_step(action),)
        )
        piece = snapshot.current
        history = (
            "; ".join(
                f"{action}: ({before.x},{before.y},r{before.rotation}) -> "
                f"({result.piece.x},{result.piece.y},r{result.piece.rotation})"
                f"{' blocked' if action != 'WAIT' and not result.moved else ''}"
                for action, before, result in self._history
            )
            or "none (new piece)"
        )
        board = "\n".join(
            "".join("#" if cell is not None else "." for cell in row) for row in snapshot.board
        )
        state = (
            "Tetris 10x20. Coordinates x=0..9 left to right, y=0..19 top to bottom; "
            "negative y is above board. Board shows fixed cells only: # filled, . empty.\n"
            f"Board rows top to bottom:\n{board}\n"
            f"Current piece {piece.kind}: x={piece.x}, y={piece.y}, rotation={piece.rotation}. "
            f"Current occupied cells: {piece_cells(piece)}. "
            f"Next: {snapshot.next_kind}; spawn shape offsets: "
            f"{piece_cells(Piece(snapshot.next_kind, 0, 0, 0))}. "
            "Full rows clear; locking above the top loses. Each key consumes one tick; "
            f"gravity falls one row every {GRAVITY_STEPS} ticks, next in "
            f"{GRAVITY_STEPS - game.gravity_phase}. DOWN falls once, not twice on a gravity tick. "
            "Resting pieces lock on a gravity tick or DOWN. Rotation uses wall kicks. "
            f"Recent executed inputs (oldest first): {history}."
        )
        descriptions = {
            "LEFT": "Move one column left.",
            "RIGHT": "Move one column right.",
            "ROTATE": "Rotate clockwise once, with wall kicks.",
            "DOWN": "Move one row down; lock if resting.",
            "WAIT": "No movement input; advance gravity clock.",
        }
        criteria = {
            candidate.label: (
                f"{'Legal' if candidate.legal else 'Input blocked'}. "
                f"{descriptions[candidate.label]}"
            )
            for candidate in candidates
        }
        questions = {
            "action": {
                "type": "choice",
                "instructions": (
                    "Pick the best keyboard action: clear lines, avoid holes, keep the stack low and flat."
                ),
                "criteria": criteria,
            }
        }
        inference_started = time.perf_counter()
        output = self.agent.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        try:
            answer = output["answers"]["action"]
            raw_probabilities = answer["probabilities"]
            probabilities = {action: float(raw_probabilities[action]) for action in ACTIONS}
        except (KeyError, TypeError, ValueError) as error:
            raise DecisionError("Laya returned an incomplete action distribution") from error
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
            raise DecisionError("Laya returned a non-finite or out-of-range probability")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.005):
            raise DecisionError("Laya action probabilities do not sum to one")

        proposed = str(answer.get("choice", ""))
        by_label = {c.label: c for c in candidates}
        if proposed not in by_label and not self.guarded:
            raise DecisionError(f"Laya proposed an unknown action: {proposed!r}")
        executed = proposed
        if self.guarded:
            chosen = by_label.get(proposed)
            if chosen is None or not chosen.legal:
                # Keep the model's own preference among legal inputs, including WAIT.
                executed = max(
                    (c.label for c in candidates if c.legal), key=probabilities.__getitem__
                )
        self._pending = (game.steps, executed, piece, by_label[executed].transition)
        usage = output.get("usage", {})
        return PolicyDecision(
            candidates=candidates,
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            intervened=proposed != executed,
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=int(usage.get("input_tokens", 0)),
            model_called=True,
        )
