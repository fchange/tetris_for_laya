import json
from io import StringIO

import pytest
from rich.console import Console

from tetris_for_laya import cli
from tetris_for_laya.cli import _decision_view, build_parser
from tetris_for_laya.game import TetrisGame
from tetris_for_laya.policy import LayaPolicy


class StubAgent:
    def __init__(self):
        self.calls = 0

    def predict(self, _state, questions):
        labels = list(questions["action"]["criteria"])
        choice = ("LEFT", "RIGHT")[self.calls] if self.calls < 2 else "DOWN"
        self.calls += 1
        return {
            "answers": {
                "action": {
                    "choice": choice,
                    "probabilities": {label: float(label == choice) for label in labels},
                }
            },
            "usage": {"input_tokens": 1},
        }


def test_parser_defaults_to_laya() -> None:
    args = build_parser().parse_args([])
    assert args.player == "laya"
    assert not args.headless
    assert args.fps is None


def test_policy_decision_displays_one_action_and_step() -> None:
    game = TetrisGame(seed=7)
    decision = LayaPolicy(agent=StubAgent()).decide(game)
    view = _decision_view(decision, step=12)

    assert view.proposed == "LEFT"
    assert view.executed == decision.executed
    assert (
        next(candidate for candidate in view.candidates if candidate.label == "LEFT").probability
        == 1.0
    )
    assert view.step == 12


def test_headless_piece_requires_multiple_model_actions(monkeypatch, capsys) -> None:
    agent = StubAgent()
    policy = LayaPolicy(agent=agent, guarded=False)
    monkeypatch.setattr(cli, "LayaPolicy", lambda *args, **kwargs: policy)

    def reject_placement(*_args, **_kwargs):
        raise AssertionError("The agent must never execute a complete placement")

    monkeypatch.setattr(TetrisGame, "apply_placement", reject_placement)
    monkeypatch.setattr(TetrisGame, "hard_drop", reject_placement)

    assert cli.main(["--headless", "--pieces", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["pieces"] == 1
    assert result["steps"] == result["decisions"] == agent.calls
    assert agent.calls > 3


@pytest.mark.parametrize("fps", [None, 8])
def test_display_renders_each_real_model_step(monkeypatch, capsys, fps) -> None:
    agent = StubAgent()
    policy = LayaPolicy(agent=agent, guarded=False)
    monkeypatch.setattr(cli, "LayaPolicy", lambda *args, **kwargs: policy)
    monkeypatch.setattr(cli.Keyboard, "read", lambda _self: "")
    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    frames = []

    def capture_frame(snapshot, decision):
        frames.append((snapshot, decision))
        return "frame"

    monkeypatch.setattr(cli, "render_game", capture_frame)
    args = build_parser().parse_args(["--pieces", "1", "--no-alt-screen"])
    args.fps = fps
    console = Console(file=StringIO(), force_terminal=True, width=100, height=30)

    assert cli._run_laya(args, console) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(frames) == agent.calls + 1 == result["steps"] + 1
    assert frames[1][0].current.x == frames[0][0].current.x - 1
    assert frames[2][0].current.x == frames[1][0].current.x + 1
    assert [view.step for _snapshot, view in frames] == list(range(len(frames)))
    assert all(snapshot.pieces == 0 for snapshot, _view in frames[:-1])
    assert frames[-1][0].pieces == 1
    if fps is None:
        assert sleeps == []
    else:
        assert len(sleeps) == result["steps"]
        assert all(0 <= duration <= 1 / fps for duration in sleeps)


def test_summary_reports_observed_action_rate(monkeypatch) -> None:
    game = TetrisGame(seed=7)
    game.step("DOWN")
    game.step("LEFT")
    monkeypatch.setattr(cli.time, "perf_counter", lambda: 12.0)

    summary = cli._summary(game, 10.0, [2.0, 3.0])

    assert summary["seconds"] == 2.0
    assert summary["steps_per_second"] == 1.0


def test_quit_is_read_before_another_model_action(monkeypatch, capsys) -> None:
    agent = StubAgent()
    policy = LayaPolicy(agent=agent)
    monkeypatch.setattr(cli, "LayaPolicy", lambda *args, **kwargs: policy)
    monkeypatch.setattr(cli.Keyboard, "read", lambda _self: "q")
    args = build_parser().parse_args(["--no-alt-screen"])
    console = Console(file=StringIO(), force_terminal=True, width=100, height=30)

    assert cli._run_laya(args, console) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["steps"] == result["decisions"] == agent.calls == 0
