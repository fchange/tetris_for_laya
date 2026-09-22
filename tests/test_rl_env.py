import copy

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from tetris_for_laya.game import ACTIONS, Piece  # noqa: E402
from tetris_for_laya.rl_env import TetrisEnv  # noqa: E402


def test_seed_and_engine_parity():
    env = TetrisEnv()
    first, _ = env.reset(seed=17)
    reference = copy.deepcopy(env.game)
    for action in [0, 1, 2, 3, 4] * 10:
        obs, _, _, _, _ = env.step(action)
        reference.step(ACTIONS[action])
        assert env.game.snapshot() == reference.snapshot()
        assert env.observation_space.contains(obs)
    second, _ = env.reset(seed=17)
    np.testing.assert_array_equal(first, second)


def test_truncation_and_reset_required():
    env = TetrisEnv(max_steps=1)
    env.reset(seed=1)
    _, _, terminated, truncated, _ = env.step(4)
    assert truncated and not terminated
    with pytest.raises(RuntimeError):
        env.step(4)


def test_clear_reward_and_top_out():
    env = TetrisEnv()
    env.reset(seed=1)
    env.game._board[-1] = ["I"] * 10
    env.game._board[-1][4:6] = [None, None]
    env.game.current = Piece("O", 0, 3, 18)
    _, reward, terminated, _, info = env.step(3)
    assert info["lines"] == 1 and reward > 10 and not terminated
    env.reset(seed=1)
    env.game._board[1][4] = "I"
    env.game.current = Piece("O", 0, 3, -1)
    _, reward, terminated, truncated, _ = env.step(3)
    assert terminated and not truncated and reward < 0
