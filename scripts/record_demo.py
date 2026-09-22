"""Record consecutive real Laya steps with Rich, macOS Quick Look and FFmpeg."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from rich.console import Console

from tetris_for_laya.cli import _decision_view
from tetris_for_laya.game import TetrisGame
from tetris_for_laya.policy import LayaPolicy
from tetris_for_laya.ui import render_game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup-pieces", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/demo.gif"))
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_pieces < 0:
        parser.error("steps must be positive and warmup-pieces nonnegative")
    for tool in ("qlmanage", "ffmpeg"):
        if shutil.which(tool) is None:
            parser.error(f"{tool} must be installed before recording")

    game = TetrisGame(seed=args.seed)
    policy = LayaPolicy()
    while game.pieces < args.warmup_pieces and not game.game_over:
        game.step(policy.decide(game).executed)
    first_step = game.steps + 1
    inference = []
    interventions = 0
    recorded = []
    started = time.perf_counter()
    for _ in range(args.steps):
        if game.game_over:
            break
        decision = policy.decide(game)
        game.step(decision.executed)
        inference.append(decision.inference_ms)
        interventions += decision.intervened
        recorded.append((game.snapshot(), _decision_view(decision, game.steps)))
    seconds = time.perf_counter() - started
    if not recorded:
        raise RuntimeError("Game ended during warmup; reduce --warmup-pieces")
    playback_fps = len(recorded) / seconds

    # Rendering happens after capture so SVG export cannot slow the measured ticks.
    with TemporaryDirectory(prefix="tetris-steps-") as directory:
        frames = Path(directory)
        for index, (snapshot, view) in enumerate(recorded):
            console = Console(
                file=io.StringIO(),
                record=True,
                width=78,
                height=22,
                color_system="truecolor",
                force_terminal=True,
            )
            console.print(render_game(snapshot, view))
            console.save_svg(
                str(frames / f"frame_{index:04d}.svg"),
                title="tetris_for_laya — one Laya call per step",
            )
        svg_files = sorted(frames.glob("*.svg"))
        subprocess.run(
            ["qlmanage", "-t", "-s", "900", "-o", str(frames), *map(str, svg_files)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        # Quick Look centers the SVG in a square thumbnail. Crop only that padding.
        viewbox = ElementTree.parse(svg_files[0]).getroot().attrib["viewBox"].split()
        height = round(900 * float(viewbox[3]) / float(viewbox[2]))
        top = (900 - height) // 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(playback_fps),
                "-i",
                str(frames / "frame_%04d.svg.png"),
                "-filter_complex",
                f"crop=900:{height}:0:{top},split[a][b];"
                "[a]palettegen=max_colors=128:stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                "-loop",
                "0",
                str(args.output),
            ],
            check=True,
        )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "first_recorded_step": first_step,
                "last_recorded_step": game.steps,
                "frames": len(inference),
                "model_calls": len(inference),
                "playback_fps": round(playback_fps, 3),
                "recording_seconds": round(seconds, 3),
                "shield_interventions": interventions,
                "mean_inference_ms": round(sum(inference) / len(inference), 3),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
