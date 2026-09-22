"""Laya placement policy with an explicit deterministic candidate planner."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .game import BOARD_HEIGHT, BOARD_WIDTH, Board, PlacementOption, TetrisGame

DEFAULT_MODEL = Path("models/laya-multilingual-mlx")


class DecisionError(RuntimeError):
    """Raised when a model response cannot be executed safely."""


class AgentLike(Protocol):
    def predict(self, state: object, questions: dict[str, object]) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Candidate:
    label: str
    placement: PlacementOption


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

    @property
    def placement(self) -> PlacementOption:
        return next(c.placement for c in self.candidates if c.label == self.executed)


def _quality(option: PlacementOption) -> float:
    """Classic linear board evaluation used only to build Laya's safe frontier."""

    return (
        0.760666 * option.lines
        - 0.510066 * option.aggregate_height
        - 0.35663 * option.holes
        - 0.184483 * option.bumpiness
    )


def _rank(option: PlacementOption) -> tuple[int, float, int, int]:
    """Keep the model's choice set small without secretly choosing its final move."""

    return (
        int(option.top_out),
        -_quality(option),
        option.landing.rotation,
        option.landing.x,
    )


def shortlist_options(
    options: tuple[PlacementOption, ...], limit: int = 8
) -> tuple[Candidate, ...]:
    """Return a bounded, stable choice set suitable for Laya's decision head."""

    if limit < 2:
        raise ValueError("candidate limit must be at least 2")
    if not options:
        return ()
    safe = [option for option in options if not option.top_out]
    pool = safe or list(options)
    selected = sorted(pool, key=_rank)[:limit]
    # Do not leak the planner rank through option order or label numbering.
    selected.sort(key=lambda option: (option.landing.rotation, option.landing.x))
    return tuple(Candidate(f"P{index + 1}", option) for index, option in enumerate(selected))


def board_metrics(board: Board) -> tuple[int, int, int, int]:
    heights: list[int] = []
    holes = 0
    for x in range(BOARD_WIDTH):
        first = next(
            (y for y in range(BOARD_HEIGHT) if board[y][x] is not None),
            BOARD_HEIGHT,
        )
        heights.append(BOARD_HEIGHT - first)
        holes += sum(board[y][x] is None for y in range(first + 1, BOARD_HEIGHT))
    return (
        holes,
        max(heights, default=0),
        sum(heights),
        sum(abs(a - b) for a, b in zip(heights, heights[1:])),
    )


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
    """Ask Laya to choose one final landing for each tetromino."""

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        *,
        candidate_limit: int = 2,
        guarded: bool = True,
        optimize: bool = False,
        agent: AgentLike | None = None,
    ) -> None:
        self.candidate_limit = candidate_limit
        self.guarded = guarded
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
        started = time.perf_counter()
        candidates = shortlist_options(game.placement_options(), self.candidate_limit)
        if not candidates:
            raise DecisionError("No legal placement is available")
        if len(candidates) == 1:
            only = candidates[0]
            return PolicyDecision(
                candidates=candidates,
                probabilities={only.label: 1.0},
                proposed="PLANNER ONLY",
                executed=only.label,
                intervened=True,
                inference_ms=0.0,
                decision_ms=(time.perf_counter() - started) * 1000,
                input_tokens=0,
                model_called=False,
            )

        snapshot = game.snapshot()
        holes, max_height, aggregate_height, bumpiness = board_metrics(snapshot.board)
        state = (
            f"Tetris. Current piece: {snapshot.current.kind}. Next piece: {snapshot.next_kind}. "
            f"Board holes: {holes}. Maximum height: {max_height}. "
            f"Aggregate height: {aggregate_height}. Bumpiness: {bumpiness}."
        )
        minimum_height = min(c.placement.max_height for c in candidates)
        minimum_bumpiness = min(c.placement.bumpiness for c in candidates)
        maximum_lines = max(c.placement.lines for c in candidates)
        criteria = {
            candidate.label: (
                f"Rotate to state {candidate.placement.landing.rotation}; "
                f"place at column {candidate.placement.landing.x}. "
                f"Result: {candidate.placement.holes} holes; "
                f"clears {candidate.placement.lines} lines"
                f"{' (best available)' if maximum_lines and candidate.placement.lines == maximum_lines else ''}; "
                f"maximum height {candidate.placement.max_height}"
                f"{' (lowest available)' if candidate.placement.max_height == minimum_height else ''}; "
                f"aggregate height {candidate.placement.aggregate_height}; "
                f"bumpiness {candidate.placement.bumpiness}"
                f"{' (smoothest available)' if candidate.placement.bumpiness == minimum_bumpiness else ''}."
            )
            for candidate in candidates
        }
        questions = {
            "placement": {
                "type": "choice",
                "instructions": (
                    "Choose the legal placement that best avoids holes and top-out, "
                    "clears lines, and keeps the board low and smooth."
                ),
                "criteria": criteria,
            }
        }
        inference_started = time.perf_counter()
        output = self.agent.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_started) * 1000

        try:
            answer = output["answers"]["placement"]
            raw_probabilities = answer["probabilities"]
            probabilities = {
                candidate.label: float(raw_probabilities[candidate.label])
                for candidate in candidates
            }
        except (KeyError, TypeError, ValueError) as error:
            raise DecisionError("Laya returned an incomplete placement distribution") from error
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
            raise DecisionError("Laya returned a non-finite or out-of-range probability")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.005):
            raise DecisionError("Laya placement probabilities do not sum to one")

        proposed = str(answer.get("choice", ""))
        valid = {candidate.label for candidate in candidates if not candidate.placement.top_out}
        if proposed not in valid:
            if not self.guarded:
                raise DecisionError(f"Laya proposed an unsafe placement: {proposed!r}")
            executed = (
                max(valid, key=probabilities.__getitem__)
                if valid
                else max(probabilities, key=probabilities.__getitem__)
            )
        else:
            executed = proposed
        if self.guarded and proposed in valid and max_height >= 12:
            by_label = {candidate.label: candidate.placement for candidate in candidates}
            planner_best = max(candidates, key=lambda candidate: _quality(candidate.placement))
            chosen = by_label[proposed]
            best = planner_best.placement
            if (
                chosen.holes > best.holes
                or chosen.max_height > best.max_height
                or _quality(best) - _quality(chosen) > 0.5
            ):
                executed = planner_best.label
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
