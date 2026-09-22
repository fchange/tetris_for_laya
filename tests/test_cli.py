from tetris_for_laya.cli import _animation_frames, _decision_view, build_parser
from tetris_for_laya.game import TetrisGame
from tetris_for_laya.policy import LayaPolicy


class StubAgent:
    def predict(self, _state, questions):
        labels = list(questions["placement"]["criteria"])
        return {
            "answers": {
                "placement": {
                    "choice": labels[0],
                    "probabilities": {
                        label: (1.0 if index == 0 else 0.0) for index, label in enumerate(labels)
                    },
                }
            },
            "usage": {"input_tokens": 1},
        }


def test_parser_defaults_to_laya() -> None:
    args = build_parser().parse_args([])
    assert args.player == "laya"
    assert not args.headless


def test_policy_decision_converts_to_ui_and_animation_reaches_landing() -> None:
    game = TetrisGame(seed=7)
    decision = LayaPolicy(agent=StubAgent()).decide(game)
    view = _decision_view(decision)
    frames = list(_animation_frames(game, decision))

    assert view.executed.startswith(decision.executed)
    assert view.candidates[0].probability == 1.0
    assert frames[-1].current.y == decision.placement.landing.y
    assert frames[-1].ghost_y == decision.placement.landing.y
