"""Gymnasium adapter sharing the human/Laya single-key game rules."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetris_for_laya.game import ACTIONS, GRAVITY_STEPS, TetrisGame, piece_cells


class TetrisEnv(gym.Env):
    """One action is one engine tick; no planner, shield or hard drop.

    Observation: locked board (200), visible active piece (200), current/next
    kind one-hots (14), rotation one-hot (4), normalized x/y/gravity (3).
    Seven-bag contents are hidden, as in normal play.
    Reward on lock: 10*clears² + 1 - .5*delta holes - .05*delta aggregate
    height - .02*delta bumpiness; top-out costs 5. Other ticks reward zero.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int | None = 10_000):
        super().__init__()
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-1.0, 1.0, (421,), dtype=np.float32)
        self.game = TetrisGame()
        self._metrics = (0, 0, 0, 0)
        self._done = False

    def _observation(self):
        game = self.game
        board = np.array([[c is not None for c in row] for row in game._board], np.float32)
        active = np.zeros((20, 10), np.float32)
        for x, y in piece_cells(game.current):
            if 0 <= x < 10 and 0 <= y < 20:
                active[y, x] = 1
        kinds = "IJLOSTZ"
        current = np.eye(7, dtype=np.float32)[kinds.index(game.current.kind)]
        next_kind = np.eye(7, dtype=np.float32)[kinds.index(game.next_kind)]
        rotation = np.eye(4, dtype=np.float32)[game.current.rotation]
        pose = np.array(
            [game.current.x / 10, game.current.y / 20, game.gravity_phase / (GRAVITY_STEPS - 1)],
            np.float32,
        )
        return np.concatenate((board.ravel(), active.ravel(), current, next_kind, rotation, pose))

    def _info(self):
        return {
            "lines": self.game.lines,
            "pieces": self.game.pieces,
            "score": self.game.score,
            "steps": self.game.steps,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game = TetrisGame(seed=int(self.np_random.integers(0, 2**31)))
        self._metrics = (0, 0, 0, 0)
        self._done = False
        return self._observation(), self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("reset() required after episode end")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        lines = self.game.lines
        transition = self.game.step(ACTIONS[int(action)])
        reward = 0.0
        if transition.locked:
            before = self._metrics
            after = self.game._board_metrics(self.game._board)
            self._metrics = after
            reward = (
                10.0 * (self.game.lines - lines) ** 2
                + 1.0
                - 0.5 * (after[0] - before[0])
                - 0.05 * (after[2] - before[2])
                - 0.02 * (after[3] - before[3])
            )
        terminated = self.game.game_over
        truncated = (
            self.max_steps is not None and self.game.steps >= self.max_steps and not terminated
        )
        if terminated:
            reward -= 5.0
        self._done = terminated or truncated
        return self._observation(), float(reward), terminated, truncated, self._info()
