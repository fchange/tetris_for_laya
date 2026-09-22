import math

import pytest

from tetris_for_laya.game import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    GameSnapshot,
    Landing,
    Piece,
    PlacementOption,
    TetrisGame,
)
from tetris_for_laya.policy import (
    DecisionError,
    LayaPolicy,
    resolve_checkpoint,
    shortlist_options,
)


class StubAgent:
    def __init__(self, *, choice: str | None = None, invalid: float | None = None) -> None:
        self.choice = choice
        self.invalid = invalid
        self.last_questions = None
        self.calls = 0

    def predict(self, _state, questions):
        self.calls += 1
        self.last_questions = questions
        labels = list(questions["placement"]["criteria"])
        probabilities = {label: 1 / len(labels) for label in labels}
        probabilities[labels[0]] += 1 - sum(probabilities.values())
        if self.invalid is not None:
            probabilities[labels[0]] = self.invalid
        return {
            "answers": {
                "placement": {
                    "choice": self.choice or labels[0],
                    "probabilities": probabilities,
                }
            },
            "usage": {"input_tokens": 123},
        }


def test_shortlist_is_bounded_safe_and_stable() -> None:
    game = TetrisGame(seed=9)
    first = shortlist_options(game.placement_options(), 8)
    second = shortlist_options(game.placement_options(), 8)

    assert first == second
    assert 2 <= len(first) <= 8
    assert [item.label for item in first] == [f"P{i}" for i in range(1, len(first) + 1)]
    assert all(not item.placement.top_out for item in first)


def test_laya_policy_builds_one_choice_question_and_returns_legal_placement() -> None:
    game = TetrisGame(seed=4)
    agent = StubAgent()
    decision = LayaPolicy(agent=agent).decide(game)

    assert set(agent.last_questions) == {"placement"}
    assert agent.last_questions["placement"]["type"] == "choice"
    assert decision.executed == "P1"
    assert decision.placement in game.placement_options()
    assert decision.input_tokens == 123
    assert not decision.intervened


def test_guard_replaces_an_unknown_choice_without_changing_probabilities() -> None:
    game = TetrisGame(seed=5)
    decision = LayaPolicy(agent=StubAgent(choice="NOT_A_PLACEMENT")).decide(game)

    assert decision.proposed == "NOT_A_PLACEMENT"
    assert decision.executed in decision.probabilities
    assert decision.intervened
    assert decision.placement in game.placement_options()


def test_guard_uses_planner_best_only_when_the_stack_is_dangerously_high() -> None:
    board = tuple(
        tuple("I" if y >= BOARD_HEIGHT - 12 and x == 0 else None for x in range(BOARD_WIDTH))
        for y in range(BOARD_HEIGHT)
    )
    best = PlacementOption("best", Landing(0, 0, 5), 0, 0, 12, 30, 3, False)
    worse = PlacementOption("worse", Landing(0, 1, 3), 0, 2, 14, 40, 8, False)

    class DangerousGame:
        def placement_options(self):
            return (best, worse)

        def snapshot(self):
            return GameSnapshot(board, Piece("T", 0, 3, -1), 17, "I", 0, 0, 0, False, 0)

    decision = LayaPolicy(agent=StubAgent(choice="P2")).decide(DangerousGame())  # type: ignore[arg-type]

    assert decision.proposed == "P2"
    assert decision.executed == "P1"
    assert decision.intervened
    assert decision.placement == best


def test_single_safe_candidate_is_marked_planner_only_without_model_call() -> None:
    board = tuple(tuple(None for _ in range(BOARD_WIDTH)) for _ in range(BOARD_HEIGHT))
    only = PlacementOption("only", Landing(0, 0, 18), 0, 0, 1, 4, 1, False)

    class ForcedGame:
        def placement_options(self):
            return (only,)

        def snapshot(self):
            return GameSnapshot(board, Piece("I", 0, 3, -1), 18, "O", 0, 0, 0, False, 0)

    agent = StubAgent()
    decision = LayaPolicy(agent=agent).decide(ForcedGame())  # type: ignore[arg-type]

    assert not decision.model_called
    assert decision.proposed == "PLANNER ONLY"
    assert decision.executed == "P1"
    assert decision.intervened
    assert decision.inference_ms == 0
    assert agent.calls == 0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.1])
def test_invalid_probabilities_never_execute(bad: float) -> None:
    with pytest.raises(DecisionError):
        LayaPolicy(agent=StubAgent(invalid=bad)).decide(TetrisGame(seed=3))


def test_explicit_missing_checkpoint_fails_before_play() -> None:
    with pytest.raises(FileNotFoundError, match="Download it first"):
        resolve_checkpoint("./definitely-missing-laya-checkpoint")
