"""Accounts on top of Supabase: the user's profile, deleting an account, and linking YouTube
and Twitch channels with OAuth so later features (pulling clip stats, fetching VODs,
posting Shorts) can act for the user.

OAuth settings come from the environment (or .env). A provider without them is shown as
"not set up" and can't be linked:

    GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET    YouTube (Google Cloud console, OAuth client
                                               of type "Web application")
    TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET    Twitch (dev.twitch.tv/console/apps)

Both need the redirect URL <CLIPPER_PUBLIC_URL>/api/connect/<youtube|twitch>/callback
registered. Tokens are stored in linked_accounts, readable by the server only.

The database calls here use the service-role key, which bypasses row level security, so
every query filters on the signed-in user's id itself.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .auth import Auth, User
from .db import Database, DatabaseError
from .httpjson import HTTPCallError, call
from .storage import Store, StorageError, user_id as safe_user_id

STATE_MAX_AGE = 600  # seconds the user has to finish the provider's consent screen
DISPLAY_NAME_MAX = 50
# What the browser may see of a linked account. Never the tokens.
PUBLIC_LINK_COLUMNS = "id,provider,provider_user_id,handle,display_name,avatar_url,scopes,created_at"


class AccountError(RuntimeError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #

def profile(db: Database, user: User) -> dict:
    """Everything the account page shows about the signed-in user."""
    rows = db.select("profiles", {"id": user.id}, "display_name,plan,created_at")
    prof = rows[0] if rows else {}
    balance = db.select("credit_balance", {"user_id": user.id}, "balance")
    links = db.select("linked_accounts", {"user_id": user.id}, PUBLIC_LINK_COLUMNS)
    return {"id": user.id, "email": user.email,
            "display_name": prof.get("display_name"), "plan": prof.get("plan", "free"),
            "created_at": prof.get("created_at") or user.created_at,
            "credits": balance[0]["balance"] if balance else 0,
            "linked": sorted(links, key=lambda r: (r["provider"], r.get("created_at") or ""))}


def update_profile(db: Database, user: User, form: dict) -> dict:
    """Change the fields a user may edit themselves. Only display_name for now."""
    changes: dict = {}
    if "display_name" in form:
        name = form["display_name"]
        if name is not None and not isinstance(name, str):
            raise AccountError("Display name must be text.")
        name = " ".join((name or "").split())  # collapse whitespace, drop control characters
        if len(name) > DISPLAY_NAME_MAX:
            raise AccountError(f"Keep the display name to {DISPLAY_NAME_MAX} characters.")
        changes["display_name"] = name or None
    if not changes:
        raise AccountError("Nothing to change.")
    changes["updated_at"] = _now().isoformat()
    if not db.update("profiles", {"id": user.id}, changes):
        raise AccountError("Your profile wasn't found. Sign out and in again.", 404)
    return profile(db, user)


# --------------------------------------------------------------------------- #
# Deleting an account
# --------------------------------------------------------------------------- #

def _store_from_env() -> Store | None:
    """The bucket, or None when cloud storage isn't set up (then there are no files to delete)."""
    if not os.environ.get("CLIPPER_BUCKET"):
        return None
    return Store.from_env()  # set up but broken (e.g. boto3 missing): raises, and we stop


def delete_account(db: Database, auth: Auth, user: User, store_factory=_store_from_env) -> dict:
    """Delete everything a user has: revoke linked channels, delete their files in the bucket,
    then the Supabase user, which cascades to their profile, runs, clips, credits and links.
    Files go first: if that fails the account stays, so nothing is left without an owner."""
    links = db.select("linked_accounts", {"user_id": user.id}, "*")
    for row in links:
        revoke(row)
    files = 0
    try:
        store = store_factory()
        if store:
            uid = safe_user_id(user.id)
            for area in ("clips", "recipes", "vods"):
                files += store.delete_prefix(f"{area}/{uid}/")
    except StorageError as e:
        raise AccountError(f"Couldn't delete your files, so your account was kept: {e}", 502) from None
    except Exception as e:  # boto3 errors: same outcome, keep the account
        raise AccountError(f"Couldn't delete your files, so your account was kept ({type(e).__name__}). "
                           f"Try again in a few minutes.", 502) from None
    auth.delete_user(user.id)
    return {"files_deleted": files, "unlinked": len(links)}


# --------------------------------------------------------------------------- #
# Linking YouTube and Twitch
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    env: str                    # prefix of the client id/secret variables
    authorize_url: str
    token_url: str
    revoke_url: str
    scopes: tuple[str, ...]
    extra_params: dict = field(default_factory=dict)

    def client(self) -> tuple[str, str] | None:
        cid, secret = os.environ.get(f"{self.env}_CLIENT_ID"), os.environ.get(f"{self.env}_CLIENT_SECRET")
        return (cid, secret) if cid and secret else None

    @property
    def configured(self) -> bool:
        return self.client() is not None

    def _client_or_fail(self) -> tuple[str, str]:
        client = self.client()
        if not client:
            raise AccountError(f"{self.label} linking isn't set up on this server: add "
                               f"{self.env}_CLIENT_ID and {self.env}_CLIENT_SECRET to .env.", 503)
        return client


PROVIDERS = {
    "youtube": Provider(
        key="youtube", label="YouTube", env="GOOGLE",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        revoke_url="https://oauth2.googleapis.com/revoke",
        # Read the channel and its analytics (for clip stats). Uploading Shorts will need
        # youtube.upload, asked for when that feature exists (include_granted_scopes adds it).
        scopes=("https://www.googleapis.com/auth/youtube.readonly",
                "https://www.googleapis.com/auth/yt-analytics.readonly"),
        # offline + consent: Google only returns a refresh token this way.
        extra_params={"access_type": "offline", "prompt": "consent",
                      "include_granted_scopes": "true"}),
    "twitch": Provider(
        key="twitch", label="Twitch", env="TWITCH",
        authorize_url="https://id.twitch.tv/oauth2/authorize",
        token_url="https://id.twitch.tv/oauth2/token",
        revoke_url="https://id.twitch.tv/oauth2/revoke",
        # Identity only: public VODs and chat need no scope. Add scopes as features need them.
        scopes=(),
        extra_params={"force_verify": "true"}),
}


def get_provider(key: str) -> Provider:
    try:
        return PROVIDERS[key]
    except KeyError:
        raise AccountError(f"Unknown service {key!r}.", 404) from None


# -- the state parameter: proves the callback answers a request this user started ---------

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_state(secret: bytes, user_id: str, provider: str, now: float | None = None) -> str:
    payload = _b64(json.dumps({"u": user_id, "p": provider, "t": int(now or time.time()),
                               "n": secrets.token_hex(8)}, separators=(",", ":")).encode())
    sig = _b64(hmac.new(secret, payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def check_state(secret: bytes, state: str, user_id: str, provider: str,
                now: float | None = None) -> None:
    bad = AccountError("That link request expired or came from another session. Try linking again.")
    payload, _, sig = (state or "").partition(".")
    good_sig = _b64(hmac.new(secret, payload.encode(), hashlib.sha256).digest())
    if not payload or not hmac.compare_digest(sig, good_sig):
        raise bad
    try:
        data = json.loads(_unb64(payload))
    except ValueError:
        raise bad from None
    age = (now or time.time()) - data.get("t", 0)
    if data.get("u") != user_id or data.get("p") != provider or not 0 <= age <= STATE_MAX_AGE:
        raise bad


def authorize_url(p: Provider, state: str, redirect_uri: str) -> str:
    cid, _ = p._client_or_fail()
    params = {"client_id": cid, "redirect_uri": redirect_uri, "response_type": "code",
              "scope": " ".join(p.scopes), "state": state, **p.extra_params}
    return f"{p.authorize_url}?{urllib.parse.urlencode(params)}"


# -- talking to the providers ---------------------------------------------------------------

def _provider_call(p: Provider, method: str, url: str, **kw):
    try:
        return call(method, url, **kw)
    except HTTPCallError as e:
        status = 503 if e.status == 0 else 502
        raise AccountError(f"{p.label} said no: {e}", status) from None


def _token_fields(tokens: dict) -> dict:
    scope = tokens.get("scope") or []
    row = {"access_token": tokens["access_token"],
           "token_expires_at": (_now() + timedelta(seconds=int(tokens.get("expires_in") or 3600))).isoformat(),
           "scopes": scope.split() if isinstance(scope, str) else list(scope)}
    if tokens.get("refresh_token"):  # Google leaves it out on refresh: keep the old one
        row["refresh_token"] = tokens["refresh_token"]
    return row


def exchange_code(p: Provider, code: str, redirect_uri: str) -> dict:
    cid, secret = p._client_or_fail()
    return _provider_call(p, "POST", p.token_url, form={
        "client_id": cid, "client_secret": secret, "code": code,
        "grant_type": "authorization_code", "redirect_uri": redirect_uri})


def fetch_identity(p: Provider, access_token: str) -> dict:
    """The channel or user the token belongs to, as linked_accounts columns."""
    if p.key == "youtube":
        data = _provider_call(p, "GET", "https://www.googleapis.com/youtube/v3/channels"
                              "?part=snippet&mine=true",
                              headers={"Authorization": f"Bearer {access_token}"}) or {}
        items = data.get("items") or []
        if not items:
            raise AccountError("That Google account has no YouTube channel. Pick the account "
                               "(or brand account) that owns the channel.")
        ch, snip = items[0], items[0].get("snippet") or {}
        return {"provider_user_id": ch["id"], "handle": snip.get("customUrl"),
                "display_name": snip.get("title"),
                "avatar_url": ((snip.get("thumbnails") or {}).get("default") or {}).get("url")}
    if p.key == "twitch":
        cid, _ = p._client_or_fail()
        data = _provider_call(p, "GET", "https://api.twitch.tv/helix/users",
                              headers={"Authorization": f"Bearer {access_token}",
                                       "Client-Id": cid}) or {}
        users = data.get("data") or []
        u = users[0] if users else None
        if not u:
            raise AccountError("Twitch didn't say which account that is. Try linking again.", 502)
        return {"provider_user_id": u["id"], "handle": u.get("login"),
                "display_name": u.get("display_name"), "avatar_url": u.get("profile_image_url")}
    raise AccountError(f"Unknown service {p.key!r}.", 404)


def save_link(db: Database, p: Provider, user_id: str, identity: dict, tokens: dict) -> dict:
    """Store the link. Re-linking the same channel refreshes it; a channel already linked
    to someone else is refused rather than moved."""
    row = {**identity, **_token_fields(tokens), "updated_at": _now().isoformat()}
    taken = AccountError(f"That {p.label} account is already linked to another Clipper account. "
                         f"Unlink it there first.", 409)
    existing = db.select("linked_accounts", {"provider": p.key,
                                             "provider_user_id": identity["provider_user_id"]},
                         "id,user_id")
    if existing:
        if existing[0]["user_id"] != user_id:
            raise taken
        db.update("linked_accounts", {"id": existing[0]["id"], "user_id": user_id}, row)
        return {**row, "id": existing[0]["id"]}
    try:
        return db.insert("linked_accounts", {**row, "user_id": user_id, "provider": p.key})
    except DatabaseError as e:
        if e.status == 409:  # linked by someone else between the check and the insert
            raise taken from None
        raise


def finish_link(db: Database, p: Provider, user: User, code: str, state: str,
                redirect_uri: str, secret: bytes) -> dict:
    """The OAuth callback: check the state, trade the code for tokens, store the link."""
    check_state(secret, state, user.id, p.key)
    if not code:
        raise AccountError(f"{p.label} didn't send an authorization code. Try linking again.")
    tokens = exchange_code(p, code, redirect_uri)
    if not tokens or not tokens.get("access_token"):
        raise AccountError(f"{p.label} didn't send a token. Try linking again.", 502)
    identity = fetch_identity(p, tokens["access_token"])
    try:
        return save_link(db, p, user.id, identity, tokens)
    except AccountError:
        revoke({"provider": p.key, **_token_fields(tokens)})  # don't keep a grant we won't use
        raise


def revoke(row: dict) -> bool:
    """Tell the provider to forget our token. Best effort: unlinking goes ahead either way
    (the user can also remove access in their Google or Twitch settings)."""
    p = PROVIDERS.get(row.get("provider"))
    client = p.client() if p else None
    if not client:
        return False
    try:
        if p.key == "youtube":
            # Revoking the refresh token revokes the whole grant.
            token = row.get("refresh_token") or row.get("access_token")
            if token:
                call("POST", p.revoke_url, form={"token": token})
        elif row.get("access_token"):
            call("POST", p.revoke_url, form={"client_id": client[0], "token": row["access_token"]})
        return True
    except HTTPCallError:
        return False


def unlink(db: Database, user: User, account_id) -> dict:
    if not isinstance(account_id, int):
        raise AccountError("Pick a linked account to remove.")
    rows = db.select("linked_accounts", {"id": account_id, "user_id": user.id}, "*")
    if not rows:
        raise AccountError("That linked account wasn't found.", 404)
    revoke(rows[0])
    db.delete("linked_accounts", {"id": account_id, "user_id": user.id})
    return {"unlinked": rows[0]["provider"]}


def access_token(db: Database, row: dict) -> str:
    """A working access token for a linked account, refreshed if it's about to expire.
    For later features (YouTube Analytics, Twitch API calls)."""
    p = get_provider(row["provider"])
    expires = row.get("token_expires_at")
    if expires and datetime.fromisoformat(expires) > _now() + timedelta(seconds=60):
        return row["access_token"]
    if not row.get("refresh_token"):
        raise AccountError(f"Your {p.label} link has expired. Link it again on the account page.", 401)
    cid, secret = p._client_or_fail()
    try:
        tokens = call("POST", p.token_url, form={
            "client_id": cid, "client_secret": secret, "grant_type": "refresh_token",
            "refresh_token": row["refresh_token"]})
    except HTTPCallError as e:
        if 400 <= e.status < 500:  # the user removed our access
            raise AccountError(f"Your {p.label} link has expired. Link it again on the "
                               f"account page.", 401) from None
        raise AccountError(f"{p.label} said no: {e}", 502) from None
    fields = {**_token_fields(tokens), "updated_at": _now().isoformat()}
    db.update("linked_accounts", {"id": row["id"], "user_id": row["user_id"]}, fields)
    row.update(fields)
    return row["access_token"]
