import math

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
        choice = self.choice or "DOWN"
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


def test_policy_never_searches_or_scores_landings(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Policy must not plan placements")

    for name in ("evaluate_landing", "placement_options", "legal_landings"):
        monkeypatch.setattr(TetrisGame, name, forbidden)
    game = TetrisGame(seed=7)
    policy = LayaPolicy(agent=StubAgent())
    while not game.pieces:
        decision = policy.decide(game)
        assert not decision.intervened
        game.step(decision.executed)
    assert game.pieces == 1


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


def test_short_upstream_instruction_keeps_wait_available() -> None:
    game = TetrisGame(seed=7)
    agent = StubAgent(choice="WAIT")
    LayaPolicy(agent=agent).decide(game)

    state, questions = agent.observations[-1]
    instructions = questions["action"]["instructions"]
    assert "clear lines, avoid holes, keep the stack low and flat" in instructions
    assert "WAIT" in questions["action"]["criteria"]


def test_new_tetromino_gets_fresh_observation() -> None:
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


def test_legal_reversal_is_not_overridden() -> None:
    game = TetrisGame(seed=7)
    agent = StubAgent(choice="LEFT")
    policy = LayaPolicy(agent=agent)
    first = policy.decide(game)
    game.step(first.executed)
    agent.choice = "RIGHT"
    second = policy.decide(game)
    assert second.proposed == second.executed == "RIGHT"
    assert not second.intervened
    assert second.probabilities == dict.fromkeys(ACTIONS, 1 / len(ACTIONS))
    assert "LEFT: (3,-1,r0) -> (2,-1,r0)" in agent.observations[-1][0]
    assert "Legal" in agent.observations[-1][1]["action"]["criteria"]["RIGHT"]


def test_useful_reversal_near_stack_is_allowed() -> None:
    from tetris_for_laya.game import Piece

    game = TetrisGame(seed=5)
    game.current = Piece(game.current.kind, 0, 3, 17)
    agent = StubAgent(choice="LEFT")
    policy = LayaPolicy(agent=agent)
    first = policy.decide(game)
    assert first.executed == "LEFT"
    game.step(first.executed)
    agent.choice = "RIGHT"
    second = policy.decide(game)
    assert second.executed == "RIGHT" and not second.intervened
    assert "Wasteful reversal" not in agent.observations[-1][1]["action"]["criteria"]["RIGHT"]


def test_history_is_bounded_confirmed_and_cleared_on_spawn() -> None:
    game = TetrisGame(seed=7)
    agent = StubAgent(choice="WAIT")
    policy = LayaPolicy(agent=agent, guarded=False)
    policy.decide(game)
    policy.decide(game)
    assert "none (new piece)" in agent.observations[-1][0]
    for _ in range(12):
        decision = policy.decide(game)
        game.step(decision.executed)
    policy.decide(game)
    assert agent.observations[-1][0].count("WAIT:") == 8
    game.hard_drop()
    policy.decide(game)
    assert "none (new piece)" in agent.observations[-1][0]


@pytest.mark.integration
@pytest.mark.parametrize("dense", [False, True])
def test_actual_tokenizer_preserves_instructions_options_and_history(dense) -> None:
    from pathlib import Path

    checkpoint = Path("models/laya-multilingual-mlx")
    if not (checkpoint / "tokenizer/tokenizer.json").exists():
        pytest.skip("Local checkpoint tokenizer is required")
    import json

    from laya_mlx.common import build_prefix, build_sequence, render_options
    from laya_mlx.tokenizer import Tokenizer

    tokenizer = Tokenizer(checkpoint / "tokenizer")
    config = json.loads((checkpoint / "rl_agent_config.json").read_text())
    game = TetrisGame(seed=7)
    if dense:
        from tetris_for_laya.game import Piece

        game.current = Piece(game.current.kind, 0, 3, -8)
        for y in range(2, 20):
            game._board[y] = ["T" if (x + y) % 2 else None for x in range(10)]
    agent = StubAgent(choice="WAIT")
    policy = LayaPolicy(agent=agent, guarded=False)
    for _ in range(12):
        decision = policy.decide(game)
        state, questions = agent.observations[-1]
        question = questions["action"]
        q = dict(t="choice", ins=question["instructions"], crit=question["criteria"])
        prefix, markers = build_prefix(tokenizer, q, config["head_max_len"])
        header = tokenizer("choice question: " + q["ins"])["input_ids"]
        assert prefix[1 : 1 + len(header)] == header
        for index, option in enumerate(render_options(q)):
            tokens = tokenizer(" " + option)["input_ids"]
            assert len(tokens) <= 48
            start = markers[index] + 1
            assert prefix[start : start + len(tokens)] == tokens
        full, _ = build_sequence(tokenizer, state, q, config["max_len"], config["head_max_len"])
        assert full == prefix + tokenizer(state)["input_ids"] + [tokenizer.sep_token_id]
        game.step(decision.executed)


def test_observation_contains_board_and_shapes_without_advice() -> None:
    from tetris_for_laya.game import Piece, piece_cells

    game = TetrisGame(seed=7)
    game._board[-1][2] = "T"
    agent = StubAgent()
    LayaPolicy(agent=agent).decide(game)
    state, questions = agent.observations[-1]
    rows = state.split("Board rows top to bottom:\n")[1].splitlines()[:20]
    assert rows == [".........."] * 19 + ["..#......."]
    assert str(piece_cells(game.current)) in state
    assert f"Next: {game.next_kind}" in state
    assert str(piece_cells(Piece(game.next_kind, 0, 0, 0))) in state
    text = (state + str(questions["action"]["criteria"])).lower()
    for advice in ("recommend", "lookahead", "best progress", "future", "landing", "quality"):
        assert advice not in text


def test_guard_uses_model_probabilities_for_illegal_input() -> None:
    class Agent(StubAgent):
        def predict(self, state, questions):
            result = super().predict(state, questions)
            result["answers"]["action"]["probabilities"] = dict(
                LEFT=0.6, RIGHT=0.05, ROTATE=0.05, DOWN=0.1, WAIT=0.2
            )
            return result

    game = TetrisGame(seed=7)
    while game.move_left():
        pass
    decision = LayaPolicy(agent=Agent(choice="LEFT")).decide(game)
    assert decision.proposed == "LEFT" and decision.executed == "WAIT"
    assert decision.intervened


def test_legal_losing_move_is_not_overridden() -> None:
    game = TetrisGame(seed=7)
    for row in game._board[1:]:
        row[:] = [None] + ["T"] * 9
    agent = StubAgent(choice="DOWN")
    policy = LayaPolicy(agent=agent)
    while not game.game_over:
        decision = policy.decide(game)
        assert decision.executed == "DOWN" and not decision.intervened
        game.step(decision.executed)
