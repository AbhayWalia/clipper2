"""Recording runs and clips in Supabase, so each user's files in R2 have an owner.

The tables are in supabase/migrations/. This talks to Supabase's REST API with the standard
library, so there's nothing extra to install. Settings come from the environment (or .env):

    SUPABASE_URL                 https://PROJECT.supabase.co
    SUPABASE_SERVICE_ROLE_KEY    the service-role (secret) key: server only, never in a browser

Writes are upserts, so uploading the same run again updates its rows instead of failing.
The service-role key bypasses row level security, so every query here that acts for a
signed-in user must filter on that user's id itself (accounts.py does).
"""

from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass

from .httpjson import HTTPCallError, call


class DatabaseError(RuntimeError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def _filters(where: dict) -> str:
    """{"user_id": "abc", "id": 3} -> "user_id=eq.abc&id=eq.3" (PostgREST equality filters)."""
    return "&".join(f"{urllib.parse.quote(k)}=eq.{urllib.parse.quote(str(v), safe='')}"
                    for k, v in where.items())


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

    def _request(self, method: str, path: str, what: str, body=None, prefer: str | None = None):
        headers = {"apikey": self.key, "Authorization": f"Bearer {self.key}"}
        if prefer:
            headers["Prefer"] = prefer
        try:
            return call(method, f"{self.url}/rest/v1/{path}", headers=headers, json_body=body,
                        timeout=30)
        except HTTPCallError as e:
            if e.status == 0:
                raise DatabaseError(f"Couldn't reach Supabase at {self.url}: {e}") from None
            detail = e.data if isinstance(e.data, str) else str(e.data)
            raise DatabaseError(f"Supabase refused {what} ({e.status}): {detail[:300]}",
                                e.status) from None

    # -- generic table access ----------------------------------------------

    def upsert(self, table: str, rows: list[dict], on_conflict: str) -> None:
        if not rows:
            return
        self._request("POST", f"{table}?on_conflict={on_conflict}", f"the {table} rows", rows,
                      "resolution=merge-duplicates,return=minimal")

    def select(self, table: str, where: dict, columns: str = "*") -> list[dict]:
        query = f"select={urllib.parse.quote(columns, safe=',*()')}"
        if where:
            query += "&" + _filters(where)
        return self._request("GET", f"{table}?{query}", f"the {table} query") or []

    def insert(self, table: str, row: dict) -> dict:
        """Insert one row and return it. A duplicate raises DatabaseError with status 409."""
        rows = self._request("POST", table, f"the new {table} row", [row], "return=representation")
        return rows[0] if rows else {}

    def update(self, table: str, where: dict, values: dict) -> list[dict]:
        """Change matching rows and return them (an empty list when nothing matched)."""
        if not where:
            raise ValueError("update() needs a filter")  # never rewrite a whole table by accident
        return self._request("PATCH", f"{table}?{_filters(where)}", f"the {table} change",
                             values, "return=representation") or []

    def delete(self, table: str, where: dict) -> list[dict]:
        if not where:
            raise ValueError("delete() needs a filter")
        return self._request("DELETE", f"{table}?{_filters(where)}", f"the {table} delete",
                             prefer="return=representation") or []

    # -- runs ----------------------------------------------------------------

    def record_run(self, run: dict, clips: list[dict]) -> None:
        """Save a run, then its clips (they reference the run, so it goes first)."""
        self.upsert("runs", [run], "id")
        self.upsert("clips", clips, "run_id,idx")
