import math
import re

import pytest

from tetris_for_laya.game import ACTIONS, TetrisGame
from tetris_for_laya.policy import DecisionError, LayaPolicy, resolve_checkpoint


class StubAgent:
    def __init__(self, choice=None, invalid=None):
        self.choice = choice
        self.invalid = invalid
        self.observations = []

    def predict(self, state, questions):
        self.observations.append((state, questions))
        choice = self.choice or re.search(r"recommends (\w+)", state).group(1)
        probabilities = {label: 1 / len(ACTIONS) for label in ACTIONS}
        if self.invalid is not None:
            probabilities["LEFT"] = self.invalid
        return {
            "answers": {"action": {"choice": choice, "probabilities": probabilities}},
            "usage": {"input_tokens": 123},
        }


def test_model_chooses_one_keyboard_action_from_the_current_position() -> None:
    game = TetrisGame(seed=7)
    agent = StubAgent(choice="LEFT")
    policy = LayaPolicy(agent=agent, guarded=False)
    before = game.snapshot()

    decision = policy.decide(game)

    state, questions = agent.observations[-1]
    assert "x=3" in state and "y=-1" in state
    assert set(questions) == {"action"}
    assert questions["action"]["type"] == "choice"
    assert set(questions["action"]["criteria"]) == set(ACTIONS)
    assert decision.executed == "LEFT"
    assert decision.model_called and decision.input_tokens == 123
    assert not decision.intervened
    assert game.snapshot() == before

    game.step(decision.executed)
    assert game.current.x == before.current.x - 1
    assert game.pieces == 0
    assert game.snapshot().board == before.board

    # A second decision observes the actual intermediate position.
    agent.choice = "RIGHT"
    second = policy.decide(game)
    assert "x=2" in agent.observations[-1][0]
    game.step(second.executed)
    assert game.current.x == before.current.x
    assert len(agent.observations) == game.steps == 2


def test_lookahead_futures_are_reachable_through_real_keyboard_steps() -> None:
    game = TetrisGame(seed=7)
    agent = StubAgent()
    policy = LayaPolicy(agent=agent)
    decision = policy.decide(game)
    target = next(c for c in decision.candidates if c.label == decision.executed)
    for _ in range(target.remaining_steps):
        assert not decision.intervened
        game.step(decision.executed)
        if game.pieces:
            break
        decision = policy.decide(game)

    assert game.pieces == 1
    assert game.steps == target.remaining_steps > 1
    assert game.lines == target.outcome.lines
    assert game.game_over == target.outcome.top_out
    assert len(agent.observations) == game.steps


def test_model_is_still_called_when_down_will_lock_the_piece() -> None:
    game = TetrisGame(seed=4)
    while game.current.y < game.snapshot().ghost_y:
        game.tick()
    agent = StubAgent(choice="DOWN")
    decision = LayaPolicy(agent=agent, guarded=False).decide(game)

    assert decision.model_called and len(agent.observations) == 1
    assert decision.executed == "DOWN"
    assert next(c for c in decision.candidates if c.label == "DOWN").transition.locked
    game.step(decision.executed)
    assert game.pieces == 1


def test_guard_replaces_unknown_action_without_changing_probabilities() -> None:
    decision = LayaPolicy(agent=StubAgent(choice="TELEPORT")).decide(TetrisGame(seed=5))

    assert decision.proposed == "TELEPORT"
    assert decision.executed in ACTIONS
    assert decision.intervened
    assert decision.probabilities == dict.fromkeys(ACTIONS, 1 / len(ACTIONS))


def test_unguarded_unknown_action_is_rejected() -> None:
    with pytest.raises(DecisionError, match="unknown action"):
        LayaPolicy(agent=StubAgent(choice="TELEPORT"), guarded=False).decide(TetrisGame())


def test_guard_marks_a_collision_override_explicitly() -> None:
    game = TetrisGame(seed=7)
    while game.move_left():
        pass
    agent = StubAgent(choice="LEFT")
    decision = LayaPolicy(agent=agent).decide(game)

    assert decision.proposed == "LEFT"
    assert decision.executed != "LEFT"
    assert decision.intervened
    assert decision.probabilities["LEFT"] == 1 / len(ACTIONS)
    assert len(agent.observations) == 1


def test_wait_is_an_intentional_action_not_a_collision() -> None:
    game = TetrisGame(seed=7)
    agent = StubAgent(choice="WAIT")
    policy = LayaPolicy(agent=agent)
    before = game.current
    for _ in range(4):
        decision = policy.decide(game)
        assert decision.proposed == decision.executed == "WAIT"
        assert not decision.intervened
        assert "Input blocked" not in agent.observations[-1][1]["action"]["criteria"]["WAIT"]
        game.step(decision.executed)

    assert game.current.x == before.x and game.current.rotation == before.rotation
    assert game.current.y == before.y + 1
    assert game.score == 0
    assert len(agent.observations) == game.steps == 4


def test_prompt_tells_laya_to_wait_instead_of_wasting_horizontal_moves() -> None:
    game = TetrisGame(seed=7)
    agent = StubAgent(choice="WAIT")
    LayaPolicy(agent=agent).decide(game)

    state, questions = agent.observations[-1]
    instructions = questions["action"]["instructions"]
    assert "WAIT is a deliberate action" in state
    assert "Choose WAIT when the current column and rotation are already suitable" in instructions
    assert "reversible side-to-side motion" in instructions


def test_new_tetromino_gets_fresh_geometry_and_lookahead() -> None:
    game = TetrisGame(seed=42)
    agent = StubAgent()
    policy = LayaPolicy(agent=agent)
    for _ in range(150):
        decision = policy.decide(game)
        game.step(decision.executed)
        if game.pieces == 3:
            break
    assert game.pieces == 3
    assert game.steps == len(agent.observations) > game.pieces
    assert not game.game_over


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.1, 0.0])
def test_invalid_probabilities_never_execute(bad: float) -> None:
    game = TetrisGame(seed=3)
    before = game.snapshot()
    with pytest.raises(DecisionError):
        LayaPolicy(agent=StubAgent(invalid=bad)).decide(game)
    assert game.snapshot() == before


def test_incomplete_probability_distribution_is_rejected() -> None:
    class IncompleteAgent:
        def predict(self, _state, _questions):
            return {"answers": {"action": {"choice": "DOWN", "probabilities": {"DOWN": 1}}}}

    with pytest.raises(DecisionError, match="incomplete"):
        LayaPolicy(agent=IncompleteAgent()).decide(TetrisGame())


def test_explicit_missing_checkpoint_fails_before_play() -> None:
    with pytest.raises(FileNotFoundError, match="Download it first"):
        resolve_checkpoint("./definitely-missing-laya-checkpoint")
