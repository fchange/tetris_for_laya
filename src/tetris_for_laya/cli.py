"""Command-line entry point for human and local Laya play."""

from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.live import Live

from .game import TETROMINOES, Piece, TetrisGame
from .policy import DEFAULT_MODEL, DecisionError, LayaPolicy, PolicyDecision
from .ui import CandidateView, DecisionView, render_game

MIN_COLUMNS = 78
MIN_ROWS = 22


class Keyboard:
    """Non-blocking raw-key reader that always restores terminal settings."""

    def __enter__(self) -> Keyboard:
        self.saved = None
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def read(self) -> str:
        if self.saved and select.select([sys.stdin], [], [], 0)[0]:
            import os

            return os.read(self.fd, 128).decode(errors="ignore")
        return ""

    def __exit__(self, *_: object) -> None:
        if self.saved:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def _decision_view(decision: PolicyDecision) -> DecisionView:
    labels = {
        candidate.label: (
            f"{candidate.label} r{candidate.placement.landing.rotation} "
            f"c{candidate.placement.landing.x}"
        )
        for candidate in decision.candidates
    }
    return DecisionView(
        candidates=tuple(
            CandidateView(labels[candidate.label], decision.probabilities[candidate.label])
            for candidate in decision.candidates
        ),
        proposed=labels.get(decision.proposed, decision.proposed),
        executed=labels.get(decision.executed, decision.executed),
        shield_applied=decision.intervened,
        inference_ms=decision.inference_ms if decision.model_called else None,
    )


def _animation_frames(game: TetrisGame, decision: PolicyDecision):
    snapshot = game.snapshot()
    landing = decision.placement.landing
    shape = TETROMINOES[snapshot.current.kind][landing.rotation]
    start_y = -max(y for _, y in shape) - 1
    for y in range(start_y, landing.y + 1):
        yield replace(
            snapshot,
            current=Piece(snapshot.current.kind, landing.rotation, landing.x, y),
            ghost_y=landing.y,
        )


def _summary(
    game: TetrisGame,
    started: float,
    inference: list[float],
    interventions: int = 0,
) -> dict[str, object]:
    snapshot = game.snapshot()
    return {
        "score": snapshot.score,
        "lines": snapshot.lines,
        "level": snapshot.level,
        "pieces": snapshot.pieces,
        "game_over": snapshot.game_over,
        "seconds": round(time.perf_counter() - started, 3),
        "decisions": len(inference),
        "shield_interventions": interventions,
        "mean_inference_ms": round(sum(inference) / len(inference), 3) if inference else None,
    }


def _run_laya(args: argparse.Namespace, console: Console) -> int:
    print("Loading local Laya MLX FP16 weights; gameplay stays offline...", file=sys.stderr)
    try:
        policy = LayaPolicy(args.model, optimize=args.optimize)
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    game = TetrisGame(seed=args.seed, start_level=args.level)
    started = time.perf_counter()
    inference: list[float] = []
    interventions = 0
    live = (
        Live(
            console=console,
            screen=not args.no_alt_screen,
            auto_refresh=False,
            vertical_overflow="crop",
        )
        if not args.headless
        else None
    )
    quit_requested = False
    try:
        with Keyboard() if live else nullcontext() as keys, live if live else nullcontext():
            while not game.game_over and not quit_requested:
                if args.pieces is not None and game.pieces >= args.pieces:
                    break
                if live and (console.width < MIN_COLUMNS or console.height < MIN_ROWS):
                    live.update(
                        f"Resize terminal to at least {MIN_COLUMNS} columns × {MIN_ROWS} rows. "
                        "Q quits.",
                        refresh=True,
                    )
                    if "q" in keys.read().lower():
                        break
                    time.sleep(0.1)
                    continue
                try:
                    decision = policy.decide(game)
                except DecisionError as error:
                    print(f"Laya decision rejected: {error}", file=sys.stderr)
                    return 3
                if decision.model_called:
                    inference.append(decision.inference_ms)
                interventions += decision.intervened
                view = _decision_view(decision)
                if live:
                    for frame in _animation_frames(game, decision):
                        pressed = keys.read().lower()
                        if "q" in pressed or "\x03" in pressed:
                            quit_requested = True
                            break
                        live.update(render_game(frame, view), refresh=True)
                        time.sleep(1 / args.fps)
                if quit_requested:
                    break
                game.apply_placement(decision.placement)
                if live:
                    live.update(render_game(game.snapshot(), view), refresh=True)
    except KeyboardInterrupt:
        pass
    print(json.dumps(_summary(game, started, inference, interventions), indent=2))
    return 0


def _run_human(args: argparse.Namespace, console: Console) -> int:
    if args.headless:
        print("Human mode needs an interactive terminal; remove --headless.", file=sys.stderr)
        return 2
    if not console.is_terminal:
        print("Human mode needs a TTY.", file=sys.stderr)
        return 2

    game = TetrisGame(seed=args.seed, start_level=args.level)
    started = time.perf_counter()
    next_tick = started + 0.4
    live = Live(
        render_game(game.snapshot()),
        console=console,
        screen=not args.no_alt_screen,
        auto_refresh=False,
        vertical_overflow="crop",
    )
    try:
        with Keyboard() as keys, live:
            while not game.game_over:
                pressed = keys.read()
                lower = pressed.lower()
                if "q" in lower or "\x03" in lower or pressed == "\x1b":
                    break
                if "\x1b[d" in lower or "h" in lower:
                    game.move_left()
                if "\x1b[c" in lower or "l" in lower:
                    game.move_right()
                if "\x1b[a" in lower or "k" in lower:
                    game.rotate()
                if "\x1b[b" in lower or "j" in lower:
                    game.soft_drop()
                if " " in pressed:
                    game.hard_drop()
                if "r" in lower:
                    game = TetrisGame(seed=args.seed, start_level=args.level)

                now = time.perf_counter()
                if now >= next_tick and not game.game_over:
                    game.tick()
                    delay = max(0.05, 0.4 * 0.85 ** (2 * game.level))
                    next_tick = now + delay
                live.update(render_game(game.snapshot()), refresh=True)
                time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    print(json.dumps(_summary(game, started, []), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", choices=("laya", "human"), default="laya")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--level", type=int, choices=range(10), default=0)
    parser.add_argument("--pieces", type=int, help="Stop after this many locked pieces")
    parser.add_argument("--fps", type=float, default=30, help="Laya drop animation FPS")
    parser.add_argument(
        "--optimize", action="store_true", help="Enable MLX compile and prompt cache"
    )
    parser.add_argument("--headless", action="store_true", help="Run Laya without terminal drawing")
    parser.add_argument("--no-alt-screen", action="store_true", help="Keep the final frame visible")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pieces is not None and args.pieces < 1:
        parser.error("--pieces must be positive")
    if not 1 <= args.fps <= 240:
        parser.error("--fps must be between 1 and 240")
    console = Console(highlight=False)
    if args.player == "human":
        return _run_human(args, console)
    if not args.headless and not console.is_terminal:
        parser.error("Interactive Laya display needs a TTY; use --headless otherwise")
    return _run_laya(args, console)
