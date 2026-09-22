import json
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from tetris_for_laya import cli
from tetris_for_laya.game import ACTIONS, TetrisGame
from tetris_for_laya.jev import ENDPOINT, JevAgent
from tetris_for_laya.policy import DecisionError, LayaPolicy


def response():
    return {
        "model": "jev-test-version",
        "answers": {
            "action": {"choice": "DOWN", "probabilities": {a: float(a == "DOWN") for a in ACTIONS}}
        },
        "usage": {"input_tokens": 123},
    }


class Transport:
    def __init__(self):
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        return BytesIO(json.dumps(response()).encode())


def test_identical_state_questions_history_and_actions(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-key")
    transport = Transport()
    remote = JevAgent(timeout=9)
    remote._opener = transport

    class LocalAgent:
        def predict(self, state, questions):
            self.latest = dict(state=state, questions=questions)
            return response()

    local = LocalAgent()
    games = [TetrisGame(seed=7), TetrisGame(seed=7)]
    policies = [LayaPolicy(agent=local), LayaPolicy(agent=remote)]
    for _ in range(12):
        decisions = [p.decide(g) for p, g in zip(policies, games)]
        request, timeout = transport.requests[-1]
        payload = json.loads(request.data)
        assert payload == dict(model="jev-latest", **local.latest)
        assert request.full_url == ENDPOINT
        assert request.get_header("Authorization") == "Bearer test-only-key"
        assert request.method == "POST" and timeout == 9
        assert decisions[0].executed == decisions[1].executed
        for g, d in zip(games, decisions):
            g.step(d.executed)
    assert remote.resolved_model == "jev-test-version"
    assert len(transport.requests) == 12


def test_missing_key_fails_before_request(monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert cli.main(["--player", "jev", "--headless", "--steps", "1"]) == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error",
    [
        HTTPError(ENDPOINT, 401, "secret", {}, None),
        HTTPError(ENDPOINT, 429, "secret", {}, None),
        URLError("secret"),
        TimeoutError("secret"),
    ],
)
def test_api_errors_are_safe_and_do_not_retry(monkeypatch, error):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    agent = JevAgent()
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(agent._opener, "open", fail)
    with pytest.raises(DecisionError) as exc:
        agent.predict("state", {})
    assert "secret" not in str(exc.value)
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"answers":null}'])
def test_invalid_response_rejected(monkeypatch, body):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-key")
    agent = JevAgent()
    monkeypatch.setattr(agent._opener, "open", lambda *a, **kw: BytesIO(body))
    with pytest.raises(DecisionError):
        agent.predict("state", {})


def test_online_cli_respects_call_limit_and_reports_model(monkeypatch, capsys):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only-key")
    agent = JevAgent()
    agent._opener = Transport()
    monkeypatch.setattr(cli, "JevAgent", lambda *a, **kw: agent)
    assert cli.main(["--player", "jev", "--headless", "--steps", "3"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["steps"] == summary["decisions"] == len(agent._opener.requests) == 3
    assert summary["player"] == "jev" and summary["model"] == "jev-test-version"
    assert summary["input_tokens"] == 369


@pytest.mark.parametrize(
    "args", [["--steps", "0"], ["--jev-timeout", "nan"], ["--player", "jev", "--optimize"]]
)
def test_invalid_cli_options(args):
    with pytest.raises(SystemExit):
        cli.main(args)
