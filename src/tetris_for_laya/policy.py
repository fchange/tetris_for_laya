"""One Laya keyboard decision per step, with explicit lookahead and safety checks."""

from __future__ import annotations

import heapq
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Protocol

from .game import (
    ACTIONS,
    GRAVITY_STEPS,
    Landing,
    Piece,
    PlacementOption,
    StepTransition,
    TetrisGame,
)

DEFAULT_MODEL = Path("models/laya-multilingual-mlx")
Node = tuple[Piece, int]


class DecisionError(RuntimeError):
    """Raised when a model response cannot be executed safely."""


class AgentLike(Protocol):
    def predict(self, state: object, questions: dict[str, object]) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Candidate:
    label: str
    transition: StepTransition
    outcome: PlacementOption
    remaining_steps: int


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


def _quality(option: PlacementOption) -> float:
    return (
        0.760666 * option.lines
        - 0.510066 * option.aggregate_height
        - 0.35663 * option.holes
        - 0.184483 * option.bumpiness
    )


def _rank(option: PlacementOption, steps: int) -> tuple:
    return (
        int(option.top_out),
        -_quality(option),
        steps,
        option.landing.rotation,
        option.landing.x,
        option.landing.y,
    )


class _Lookahead:
    """Evaluate reachable futures with the same single-step physics as play.

    The board stays fixed while a piece falls. Cache this graph until it locks;
    each decision queries it from the actual current pose and gravity phase.
    No path or final placement is sent to the execution loop.
    """

    def __init__(self, game: TetrisGame) -> None:
        start = (game.current, game.gravity_phase)
        self.transitions: dict[Node, dict[str, StepTransition]] = {}
        self.values: dict[Node, tuple[tuple, PlacementOption]] = {}
        self.terminals: dict[Landing, PlacementOption] = {}
        reverse: dict[Node, list[Node]] = defaultdict(list)
        queue = deque([start])
        seen = {start}
        heap = []
        serial = count()
        while queue:
            node = queue.popleft()
            edges = self.transitions[node] = {}
            for action in ACTIONS:
                transition = game.preview_step(action, piece=node[0], gravity_phase=node[1])
                edges[action] = transition
                if transition.locked:
                    piece = transition.piece
                    landing = Landing(piece.rotation, piece.x, piece.y)
                    if landing not in self.terminals:
                        self.terminals[landing] = game.evaluate_landing(landing)
                    outcome = self.terminals[landing]
                    heapq.heappush(heap, (_rank(outcome, 1), next(serial), node, outcome))
                else:
                    target = (transition.piece, transition.gravity_phase)
                    reverse[target].append(node)
                    if target not in seen:
                        seen.add(target)
                        queue.append(target)

        # Reverse shortest paths: quality first, then fewer inputs.
        # Distance breaks reversible LEFT/RIGHT ties without executing a path.
        while heap:
            rank, _, node, outcome = heapq.heappop(heap)
            if node in self.values:
                continue
            self.values[node] = (rank, outcome)
            for parent in reverse[node]:
                if parent not in self.values:
                    heapq.heappush(
                        heap, (_rank(outcome, rank[2] + 1), next(serial), parent, outcome)
                    )

    def candidates(self, node: Node) -> tuple[Candidate, ...]:
        candidates = []
        for action, transition in self.transitions[node].items():
            if transition.locked:
                piece = transition.piece
                outcome = self.terminals[Landing(piece.rotation, piece.x, piece.y)]
                steps = 1
            else:
                rank, outcome = self.values[(transition.piece, transition.gravity_phase)]
                steps = rank[2] + 1
            candidates.append(Candidate(action, transition, outcome, steps))
        return tuple(candidates)


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
        self._seen: set[Node] = set()
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
        if key != self._board_key or node not in self._lookahead.transitions:
            self._lookahead = _Lookahead(game)
            self._board_key = key
            self._seen.clear()
        self._seen.add(node)
        candidates = self._lookahead.candidates(node)
        usable = [
            c for c in candidates if c.label == "WAIT" or c.transition.moved or c.transition.locked
        ]
        fresh = [
            c
            for c in usable
            if c.transition.locked
            or (c.transition.piece, c.transition.gravity_phase) not in self._seen
        ]
        best = min(fresh or usable, key=lambda c: _rank(c.outcome, c.remaining_steps))
        piece = snapshot.current
        state = (
            f"Tetris keyboard control. Current piece {piece.kind}: "
            f"x={piece.x}, y={piece.y}, rotation={piece.rotation}. Next: {snapshot.next_kind}. "
            f"Gravity in {GRAVITY_STEPS - game.gravity_phase} inputs. "
            f"Lookahead recommends {best.label}. Execute only ONE action, then observe again. "
            "Do not move merely to stay active; WAIT is a deliberate action."
        )
        descriptions = {
            "LEFT": "Move left ONE column",
            "RIGHT": "Move right ONE column",
            "ROTATE": "Rotate clockwise ONCE, using wall kicks if needed",
            "DOWN": "Move down ONE row; lock only if already resting",
            "WAIT": "Do not move or rotate; let the gravity clock advance",
        }
        criteria = {}
        for candidate in candidates:
            transition, outcome = candidate.transition, candidate.outcome
            criteria[candidate.label] = (
                f"{descriptions[candidate.label]}. "
                f"{'Input blocked. ' if candidate not in usable else ''}"
                f"{'Locks now. ' if transition.locked else ''}"
                f"Best reachable future after this key: {outcome.holes} holes, "
                f"{outcome.lines} lines cleared, height {outcome.max_height}, "
                f"{candidate.remaining_steps} inputs to lock. "
                f"{'TOP OUT. ' if outcome.top_out else ''}"
                f"{'Recommended next key.' if candidate.label == best.label else ''}"
            )
        questions = {
            "action": {
                "type": "choice",
                "instructions": (
                    "Choose the next single keyboard action. Avoid blocked keys and top-out, "
                    "minimize future holes and clear lines. Use LEFT, RIGHT or ROTATE only when "
                    "it improves the reachable board. Choose WAIT when the current column and "
                    "rotation are already suitable and natural gravity can safely continue. "
                    "Use the lookahead recommendation to avoid reversible side-to-side motion."
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
            if chosen is None or chosen not in (fresh or usable):
                executed = best.label
            elif chosen.outcome.top_out and not best.outcome.top_out:
                executed = best.label
            elif best.outcome.max_height >= 12 and (
                chosen.outcome.holes > best.outcome.holes
                or _quality(best.outcome) - _quality(chosen.outcome) > 0.5
            ):
                executed = best.label
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
