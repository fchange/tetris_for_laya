"""Command-line entry point for human, local Laya and online Jev play."""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import termios
import time
import tty
from contextlib import nullcontext
from pathlib import Path

from rich.console import Console
from rich.live import Live

from .game import TetrisGame
from .jev import DEFAULT_JEV_MODEL, JevAgent
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


def _decision_view(
    decision: PolicyDecision, step: int = 0, policy_name: str = "LAYA"
) -> DecisionView:
    return DecisionView(
        candidates=tuple(
            CandidateView(candidate.label, decision.probabilities[candidate.label])
            for candidate in decision.candidates
        ),
        proposed=decision.proposed,
        executed=decision.executed,
        shield_applied=decision.intervened,
        inference_ms=decision.inference_ms if decision.model_called else None,
        step=step,
        policy_name=policy_name,
    )


def _summary(
    game: TetrisGame,
    started: float,
    inference: list[float],
    interventions: int = 0,
) -> dict[str, object]:
    snapshot = game.snapshot()
    elapsed = time.perf_counter() - started
    return {
        "score": snapshot.score,
        "lines": snapshot.lines,
        "level": snapshot.level,
        "pieces": snapshot.pieces,
        "steps": game.steps,
        "game_over": snapshot.game_over,
        "seconds": round(elapsed, 3),
        "steps_per_second": round(game.steps / elapsed, 3) if elapsed else 0.0,
        "decisions": len(inference),
        "shield_interventions": interventions,
        "mean_inference_ms": round(sum(inference) / len(inference), 3) if inference else None,
    }


def _run_laya(args: argparse.Namespace, console: Console) -> int:
    provider = args.player.upper()
    try:
        if args.player == "jev":
            agent = JevAgent(args.jev_model, timeout=args.jev_timeout)
            policy = LayaPolicy(agent=agent)
            print(f"Online Jev ({args.jev_model}); one API request per input...", file=sys.stderr)
        else:
            print("Loading local Laya MLX FP16 weights; gameplay stays offline...", file=sys.stderr)
            policy = LayaPolicy(args.model, optimize=args.optimize)
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    game = TetrisGame(seed=args.seed, start_level=args.level)
    started = time.perf_counter()
    inference: list[float] = []
    interventions = 0
    input_tokens = 0
    live = (
        Live(
            render_game(game.snapshot(), DecisionView(policy_name=provider)),
            console=console,
            screen=not args.no_alt_screen,
            auto_refresh=False,
            vertical_overflow="crop",
        )
        if not args.headless
        else None
    )
    try:
        with Keyboard() if live else nullcontext() as keys, live if live else nullcontext():
            while not game.game_over:
                if args.pieces is not None and game.pieces >= args.pieces:
                    break
                if args.steps is not None and game.steps >= args.steps:
                    break
                if live:
                    pressed = keys.read().lower()
                    if "q" in pressed or "\x03" in pressed:
                        break
                if live and (console.width < MIN_COLUMNS or console.height < MIN_ROWS):
                    live.update(
                        f"Resize terminal to at least {MIN_COLUMNS} columns × {MIN_ROWS} rows. "
                        "Q quits.",
                        refresh=True,
                    )
                    time.sleep(0.1)
                    continue
                step_started = time.perf_counter()
                try:
                    decision = policy.decide(game)
                except DecisionError as error:
                    print(f"{provider} decision rejected: {error}", file=sys.stderr)
                    return 3
                if decision.model_called:
                    inference.append(decision.inference_ms)
                input_tokens += decision.input_tokens
                interventions += decision.intervened
                game.step(decision.executed)
                if live:
                    view = _decision_view(decision, game.steps, provider)
                    live.update(render_game(game.snapshot(), view), refresh=True)
                if args.fps is not None:
                    time.sleep(max(0, 1 / args.fps - (time.perf_counter() - step_started)))
    except KeyboardInterrupt:
        pass
    summary = _summary(game, started, inference, interventions)
    summary.update(player=args.player, seed=args.seed, input_tokens=input_tokens)
    summary["model"] = (
        agent.resolved_model or args.jev_model if args.player == "jev" else str(args.model)
    )
    print(json.dumps(summary, indent=2))
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
    parser.add_argument("--player", choices=("laya", "jev", "human"), default="laya")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--jev-model", default=DEFAULT_JEV_MODEL)
    parser.add_argument("--jev-timeout", type=float, default=30, help="API timeout in seconds")
    parser.add_argument("--steps", type=int, help="Stop after this many inputs/model calls")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--level", type=int, choices=range(10), default=0)
    parser.add_argument("--pieces", type=int, help="Stop after this many locked pieces")
    parser.add_argument(
        "--fps", type=float, help="Optional AI action rate cap; default runs at model speed"
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Enable MLX compile and prompt cache"
    )
    parser.add_argument("--headless", action="store_true", help="Run AI without terminal drawing")
    parser.add_argument("--no-alt-screen", action="store_true", help="Keep the final frame visible")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pieces is not None and args.pieces < 1:
        parser.error("--pieces must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if not math.isfinite(args.jev_timeout) or args.jev_timeout <= 0:
        parser.error("--jev-timeout must be a positive finite number")
    if args.player == "jev" and args.optimize:
        parser.error("--optimize applies only to local Laya")
    if args.fps is not None and not 1 <= args.fps <= 240:
        parser.error("--fps must be between 1 and 240")
    console = Console(highlight=False)
    if args.player == "human":
        return _run_human(args, console)
    if not args.headless and not console.is_terminal:
        parser.error("Interactive AI display needs a TTY; use --headless otherwise")
    return _run_laya(args, console)
