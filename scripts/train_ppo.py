"""Train SB3 PPO and compare it with random play on held-out episode seeds."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from tetris_for_laya.rl_env import TetrisEnv


def evaluate(model, episodes, seed):
    env = TetrisEnv()
    results = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        env.action_space.seed(seed + episode)
        total = 0.0
        while True:
            action = (
                env.action_space.sample()
                if model is None
                else int(model.predict(obs, deterministic=True)[0])
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                results.append(dict(info, reward=total, terminated=terminated, truncated=truncated))
                break
    env.close()
    return {
        "mean": {
            key: float(np.mean([r[key] for r in results]))
            for key in ("lines", "pieces", "score", "steps", "reward")
        },
        "episodes": results,
    }


class EpisodeBudget(BaseCallback):
    """Count actual top-outs, checkpoint, and finish the final PPO rollout."""

    def __init__(self, target, output):
        super().__init__()
        self.target = target
        self.output = output
        self.episodes = 0
        self.next_checkpoint = 1000
        self.stop_after_update = False
        self.started = time.monotonic()

    def _on_step(self):
        self.episodes += sum(
            bool(done) and not info.get("TimeLimit.truncated", False)
            for done, info in zip(self.locals["dones"], self.locals["infos"])
        )
        return not self.stop_after_update

    def _on_rollout_start(self):
        # The preceding rollout has been trained before this hook runs.
        self.stop_after_update = self.episodes >= self.target
        self.write_status()
        if self.episodes >= self.next_checkpoint:
            self.model.save(self.output / "latest")
            self.next_checkpoint = (self.episodes // 1000 + 1) * 1000

    def write_status(self):
        status = {
            "completed_games": self.episodes,
            "target_games": self.target,
            "model_timesteps": self.model.num_timesteps,
            "elapsed_seconds": time.monotonic() - self.started,
        }
        temporary = self.output / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(self.output / "status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, help="Train at least this many complete games")
    parser.add_argument("--resume", type=Path, help="Continue an existing PPO checkpoint")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ppo-100k"))
    args = parser.parse_args()
    if args.timesteps <= 0 or args.eval_episodes <= 0:
        parser.error("timesteps and eval-episodes must be positive")
    if args.episodes is not None and args.episodes <= 0:
        parser.error("episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    check_env(TetrisEnv())
    # 4 * 250 = 1000 steps/rollout, so the default run is exactly 100,000.
    env = make_vec_env(
        TetrisEnv,
        n_envs=4,
        env_kwargs={"max_steps": None} if args.episodes else None,
        seed=args.seed,
        monitor_dir=str(args.output / "monitor"),
        monitor_kwargs={"info_keywords": ("lines", "pieces", "score")},
    )
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=250,
        batch_size=100,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=args.seed,
        device="cpu",
        verbose=1,
    )
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu")
        model.set_random_seed(args.seed)
    model.set_logger(configure(str(args.output), ["stdout", "csv"]))
    started = time.monotonic()
    initial_timesteps = model.num_timesteps
    callback = EpisodeBudget(args.episodes, args.output) if args.episodes else None
    (args.output / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "output": str(args.output),
                "resume": str(args.resume) if args.resume else None,
            },
            indent=2,
        )
        + "\n"
    )
    model.learn(
        total_timesteps=10**12 if callback else args.timesteps,
        callback=callback,
        reset_num_timesteps=not bool(args.resume),
    )
    if callback:
        callback.write_status()
    model.save(args.output / "model")
    elapsed = time.monotonic() - started
    model = PPO.load(args.output / "model", device="cpu")
    report = {
        "requested_timesteps": None if callback else args.timesteps,
        "requested_games": args.episodes,
        "completed_games": callback.episodes if callback else None,
        "additional_timesteps": model.num_timesteps - initial_timesteps,
        "actual_timesteps": model.num_timesteps,
        "training_seconds": elapsed,
        "seed": args.seed,
        "eval_seed_start": 10_000,
        "ppo": evaluate(model, args.eval_episodes, 10_000),
        "random": evaluate(None, args.eval_episodes, 10_000),
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    print(json.dumps({k: v for k, v in report.items() if k not in ("ppo", "random")}))
    print("PPO:", report["ppo"]["mean"])
    print("Random:", report["random"]["mean"])


if __name__ == "__main__":
    main()
