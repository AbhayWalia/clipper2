"""Sign-up, sign-in and sessions with Supabase Auth (its REST API, standard library only).

Settings come from the environment (or .env):

    SUPABASE_URL                 https://PROJECT.supabase.co
    SUPABASE_ANON_KEY            the publishable (anon) key, used for sign-up and sign-in
    SUPABASE_SERVICE_ROLE_KEY    the secret key, used to delete users

The browser never sees a key or a token: the web server signs in on the user's behalf and
keeps the session in HttpOnly cookies (see web.py).
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass

from .httpjson import HTTPCallError, call

MIN_PASSWORD = 8
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USER_CACHE_SECONDS = 60  # how long a checked access token is trusted without asking Supabase


class AuthError(RuntimeError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class User:
    id: str
    email: str
    created_at: str | None = None

    @classmethod
    def from_api(cls, data: dict) -> "User":
        return cls(id=data["id"], email=data.get("email") or "", created_at=data.get("created_at"))


@dataclass
class Session:
    access_token: str
    refresh_token: str
    expires_in: int
    user: User

    @classmethod
    def from_api(cls, data: dict) -> "Session":
        return cls(data["access_token"], data["refresh_token"], int(data.get("expires_in") or 3600),
                   User.from_api(data["user"]))


def check_credentials(email: str, password: str) -> tuple[str, str]:
    email = (email or "").strip().lower()
    if not EMAIL.match(email):
        raise AuthError("Enter a valid email address.")
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        raise AuthError(f"Use a password of at least {MIN_PASSWORD} characters.")
    return email, password


class Auth:
    def __init__(self, url: str, anon_key: str, service_key: str):
        self.url = url.strip().rstrip("/").removesuffix("/rest/v1").rstrip("/")
        self.anon_key, self.service_key = anon_key, service_key
        self._cache: dict[str, tuple[User, float]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "Auth | None":
        url = os.environ.get("SUPABASE_URL")
        service = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not service:
            return None
        # The anon key is what Supabase expects for user-facing calls; the service key works
        # too, so a missing anon key only matters once rate limits per key do.
        return cls(url, os.environ.get("SUPABASE_ANON_KEY") or service, service)

    def _call(self, method: str, path: str, body=None, token: str | None = None,
              admin: bool = False):
        key = self.service_key if admin else self.anon_key
        headers = {"apikey": key, "Authorization": f"Bearer {self.service_key if admin else token or key}"}
        try:
            return call(method, f"{self.url}/auth/v1/{path}", headers=headers, json_body=body)
        except HTTPCallError as e:
            if e.status == 0:
                raise AuthError("Couldn't reach the sign-in service. Try again in a minute.", 503) from None
            if e.status == 429:
                raise AuthError("Too many attempts. Wait a minute and try again.", 429) from None
            raise AuthError(str(e), e.status if 400 <= e.status < 500 else 502) from None

    # -- the user's own session ------------------------------------------------

    def sign_up(self, email: str, password: str, redirect_to: str | None = None) -> Session | None:
        """Create a user. Returns their session, or None when Supabase sends a confirmation
        email first (the default for new projects)."""
        email, password = check_credentials(email, password)
        path = "signup"
        if redirect_to:
            path += "?redirect_to=" + urllib.parse.quote(redirect_to, safe="")
        data = self._call("POST", path, {"email": email, "password": password}) or {}
        return Session.from_api(data) if data.get("access_token") else None

    def sign_in(self, email: str, password: str) -> Session:
        email = (email or "").strip().lower()
        if not email or not password:
            raise AuthError("Enter your email and password.")
        try:
            data = self._call("POST", "token?grant_type=password",
                              {"email": email, "password": password})
        except AuthError as e:
            if e.status == 400:  # wrong password, unknown email or unconfirmed: say it plainly
                raise AuthError(str(e) if "confirm" in str(e).lower()
                                else "Wrong email or password.", 400) from None
            raise
        return Session.from_api(data)

    def refresh(self, refresh_token: str) -> Session:
        data = self._call("POST", "token?grant_type=refresh_token", {"refresh_token": refresh_token})
        return Session.from_api(data)

    def sign_out(self, access_token: str) -> None:
        self._forget(access_token)
        try:
            self._call("POST", "logout", token=access_token)
        except AuthError:
            pass  # the token was already invalid: the user is signed out either way

    def user(self, access_token: str) -> User:
        """The user an access token belongs to. Supabase checks the token; the answer is
        cached for a minute so every page load doesn't cost a round trip."""
        digest = hashlib.sha256(access_token.encode()).hexdigest()
        now = time.time()
        with self._lock:
            hit = self._cache.get(digest)
            if hit and hit[1] > now:
                return hit[0]
        user = User.from_api(self._call("GET", "user", token=access_token))
        with self._lock:
            if len(self._cache) > 1000:
                self._cache = {k: v for k, v in self._cache.items() if v[1] > now}
            self._cache[digest] = (user, now + USER_CACHE_SECONDS)
        return user

    def _forget(self, access_token: str) -> None:
        with self._lock:
            self._cache.pop(hashlib.sha256(access_token.encode()).hexdigest(), None)

    def forget_user(self, user_id: str) -> None:
        with self._lock:
            self._cache = {k: v for k, v in self._cache.items() if v[0].id != user_id}

    # -- admin ---------------------------------------------------------------------

    def delete_user(self, user_id: str) -> None:
        """Delete the user from Supabase Auth. Their profile, runs, clips, credits and linked
        accounts go with it (on delete cascade)."""
        self._call("DELETE", f"admin/users/{urllib.parse.quote(user_id, safe='')}", admin=True)
        self.forget_user(user_id)
