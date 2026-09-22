"""Online Jev adapter for the same typed state/questions used by local Laya."""

from __future__ import annotations

import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .policy import DecisionError

DEFAULT_JEV_MODEL = "jev-latest"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class JevAgent:
    """One API call per input; never retries or silently falls back to another model."""

    def __init__(self, model: str = DEFAULT_JEV_MODEL, *, timeout: float = 30) -> None:
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            raise ValueError("Set TYPESAFE_API_KEY in your terminal before using --player jev.")
        if not model.strip():
            raise ValueError("Jev model must not be empty")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Jev timeout must be a positive finite number")
        self._key = key
        self.model = model
        self.timeout = timeout
        self.resolved_model: str | None = None
        self._opener = build_opener(_NoRedirect())

    def predict(self, state: object, questions: dict[str, object]) -> dict:
        request = Request(
            ENDPOINT,
            data=json.dumps(dict(model=self.model, state=state, questions=questions)).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                output = json.load(response)
        except HTTPError as error:
            hints = {
                401: "Check TYPESAFE_API_KEY.",
                403: "Check API key permissions and model access.",
                429: "Rate limited; try again later or use --fps 1.",
            }
            raise DecisionError(
                f"Jev HTTP {error.code}. {hints.get(error.code, 'Request failed.')}"
            ) from None
        except (URLError, TimeoutError, OSError):
            raise DecisionError(
                "Jev connection failed or timed out; check network/proxy settings."
            ) from None
        except (ValueError, UnicodeError):
            raise DecisionError("Jev returned invalid JSON") from None
        if not isinstance(output, dict) or not isinstance(output.get("answers"), dict):
            raise DecisionError("Jev returned an invalid response object")
        if isinstance(output.get("model"), str):
            self.resolved_model = output["model"]
        return output
