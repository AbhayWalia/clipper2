"""Recording runs and clips in Supabase, so each user's files in R2 have an owner.

The tables are in supabase/migrations/. This talks to Supabase's REST API with the standard
library, so there's nothing extra to install. Settings come from the environment (or .env):

    SUPABASE_URL                 https://PROJECT.supabase.co
    SUPABASE_SERVICE_ROLE_KEY    the service-role (secret) key: server only, never in a browser

Writes are upserts, so uploading the same run again updates its rows instead of failing.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass


class DatabaseError(RuntimeError):
    pass


def _ssl_context() -> ssl.SSLContext:
    # Homebrew and python.org Pythons often lack system certificates; certifi (installed
    # with the Anthropic SDK) has them.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


@dataclass
class Database:
    url: str
    key: str

    def __post_init__(self):
        # Accept the project URL with or without the "/rest/v1" the dashboard sometimes shows,
        # and a trailing "/", which would otherwise double up in request paths.
        url = self.url.strip().rstrip("/")
        self.url = url.removesuffix("/rest/v1").rstrip("/")

    @classmethod
    def from_env(cls) -> "Database | None":
        """The configured database, or None when Supabase isn't set up."""
        url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            return None
        return cls(url, key)

    def upsert(self, table: str, rows: list[dict], on_conflict: str) -> None:
        if not rows:
            return
        req = urllib.request.Request(
            f"{self.url}/rest/v1/{table}?on_conflict={on_conflict}",
            data=json.dumps(rows).encode(), method="POST",
            headers={"apikey": self.key, "Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json",
                     "Prefer": "resolution=merge-duplicates,return=minimal"})
        try:
            with urllib.request.urlopen(req, timeout=30, context=_ssl_context()):
                pass
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise DatabaseError(f"Supabase refused the {table} rows ({e.code}): {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise DatabaseError(f"Couldn't reach Supabase at {self.url}: {e}") from None

    def record_run(self, run: dict, clips: list[dict]) -> None:
        """Save a run, then its clips (they reference the run, so it goes first)."""
        self.upsert("runs", [run], "id")
        self.upsert("clips", clips, "run_id,idx")
