"""A tiny JSON-over-HTTPS client on the standard library, shared by the Supabase and OAuth code."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request


class HTTPCallError(RuntimeError):
    """The server answered with an error status. `status` is 0 when it couldn't be reached."""

    def __init__(self, status: int, message: str, data=None):
        super().__init__(message)
        self.status, self.data = status, data


def ssl_context() -> ssl.SSLContext:
    # Homebrew and python.org Pythons often lack system certificates; certifi (installed
    # with the Anthropic SDK) has them.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def error_message(data, fallback: str) -> str:
    """The human part of an error body from Supabase, Google or Twitch."""
    if isinstance(data, dict):
        for key in ("msg", "error_description", "message", "error"):
            val = data.get(key)
            if isinstance(val, str) and val:
                return val
            if isinstance(val, dict) and isinstance(val.get("message"), str):  # Google APIs
                return val["message"]
    if isinstance(data, str) and data.strip():
        return data.strip()[:300]
    return fallback


def call(method: str, url: str, *, headers: dict | None = None, json_body=None,
         form: dict | None = None, timeout: float = 20):
    """Send a request and return the decoded JSON body (None when empty).
    Raises HTTPCallError for error statuses and unreachable servers."""
    headers = dict(headers or {})
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    elif form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    headers.setdefault("Accept", "application/json")
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_context()) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = raw
        raise HTTPCallError(e.code, error_message(parsed, f"HTTP {e.code}"), parsed) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        host = urllib.parse.urlparse(url).netloc
        raise HTTPCallError(0, f"Couldn't reach {host}: {e}") from None
    if not body:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return body.decode(errors="replace")
