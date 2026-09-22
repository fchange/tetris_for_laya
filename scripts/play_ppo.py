"""Watch a trained PPO policy play through the shared Rich terminal UI."""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from rich.console import Console
from rich.live import Live
from stable_baselines3 import PPO

from tetris_for_laya.cli import MIN_COLUMNS, MIN_ROWS, Keyboard
from tetris_for_laya.game import ACTIONS
from tetris_for_laya.rl_env import TetrisEnv
from tetris_for_laya.ui import CandidateView, DecisionView, render_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/ppo-100k/model.zip"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-alt-screen", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.fps <= 240:
        parser.error("--fps must be between 1 and 240")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")
    console = Console(highlight=False)
    if not args.headless and not console.is_terminal:
        parser.error("Visualization needs a TTY; use --headless otherwise")
    torch.set_num_threads(1)
    model = PPO.load(args.model, device="cpu")
    env = TetrisEnv()
    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    terminated = truncated = False
    view = DecisionView(policy_name="PPO", shield_enabled=False)
    live = (
        None
        if args.headless
        else Live(
            render_game(env.game.snapshot(), view),
            console=console,
            screen=not args.no_alt_screen,
            auto_refresh=False,
            vertical_overflow="crop",
        )
    )
    try:
        with Keyboard() if live else nullcontext() as keys, live if live else nullcontext():
            while not (terminated or truncated):
                if live:
                    if "q" in keys.read().lower():
                        break
                    if console.width < MIN_COLUMNS or console.height < MIN_ROWS:
                        live.update(
                            f"Resize terminal to {MIN_COLUMNS} × {MIN_ROWS}. Q quits.", refresh=True
                        )
                        time.sleep(0.1)
                        continue
                started = time.perf_counter()
                with torch.no_grad():
                    tensor, _ = model.policy.obs_to_tensor(obs)
                    distribution = model.policy.get_distribution(tensor)
                    probabilities = distribution.distribution.probs[0].cpu().numpy()
                    action = int(distribution.get_actions(deterministic=True).item())
                inference_ms = (time.perf_counter() - started) * 1000
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if live:
                    view = DecisionView(
                        candidates=tuple(
                            CandidateView(key, float(p)) for key, p in zip(ACTIONS, probabilities)
                        ),
                        proposed=ACTIONS[action],
                        executed=ACTIONS[action],
                        inference_ms=inference_ms,
                        step=env.game.steps,
                        policy_name="PPO",
                        shield_enabled=False,
                    )
                    live.update(render_game(env.game.snapshot(), view), refresh=True)
                    time.sleep(max(0, 1 / args.fps - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    print(
        json.dumps(
            dict(info, reward=total_reward, terminated=terminated, truncated=truncated), indent=2
        )
    )


if __name__ == "__main__":
    main()
