"""Sign-up/sign-in, the profile, deleting an account and linking YouTube/Twitch.
No network: Supabase, Google and Twitch are faked at urllib.request.urlopen."""

import http.client
import io
import json
import re
import threading
import urllib.error
import urllib.parse
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer

import pytest

from clipfinder import accounts, web
from clipfinder.accounts import AccountError
from clipfinder.auth import Auth, AuthError, User
from clipfinder.db import Database
from clipfinder.storage import StorageError

SB = "https://proj.supabase.co"
USER = User("3f1c9a2e-7b4d-4e8a-9c1f-0a2b3c4d5e6f", "a@example.com", "2026-11-01T00:00:00Z")
OTHER = "99999999-7b4d-4e8a-9c1f-0a2b3c4d5e6f"
SECRET = b"test-secret"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeNet:
    """Routes requests by method and URL regex to handlers returning (status, json body)."""

    def __init__(self, monkeypatch):
        self.routes, self.sent = [], []
        monkeypatch.setattr("urllib.request.urlopen", self.urlopen)

    def on(self, method, pattern, handler):
        self.routes.insert(0, (method, re.compile(pattern), handler))  # later routes win

    def urlopen(self, req, timeout, context):
        self.sent.append(req)
        for method, pattern, handler in self.routes:
            if req.get_method() == method and pattern.search(req.full_url):
                status, body = handler(req)
                raw = b"" if body is None else json.dumps(body).encode()
                if status >= 400:
                    raise urllib.error.HTTPError(req.full_url, status, "err", {}, io.BytesIO(raw))
                return FakeResponse(raw)
        raise AssertionError(f"unexpected request {req.get_method()} {req.full_url}")

    def calls(self, method=None, pattern=""):
        return [r for r in self.sent
                if (method is None or r.get_method() == method) and re.search(pattern, r.full_url)]


def body(req):
    ctype = req.get_header("Content-type") or ""
    if "json" in ctype:
        return json.loads(req.data)
    return dict(urllib.parse.parse_qsl(req.data.decode()))


def session_json(user=USER, token="access-1", refresh="refresh-1"):
    return {"access_token": token, "refresh_token": refresh, "expires_in": 3600,
            "user": {"id": user.id, "email": user.email, "created_at": user.created_at}}


@pytest.fixture
def net(monkeypatch):
    return FakeNet(monkeypatch)


@pytest.fixture
def auth():
    return Auth(SB + "/", "anon-key", "service-key")


@pytest.fixture
def db():
    return Database(SB, "service-key")


@pytest.fixture
def oauth_env(monkeypatch):
    for name in ("GOOGLE", "TWITCH"):
        monkeypatch.setenv(f"{name}_CLIENT_ID", f"{name.lower()}-id")
        monkeypatch.setenv(f"{name}_CLIENT_SECRET", f"{name.lower()}-secret")


# -- Supabase Auth ---------------------------------------------------------------------------

def test_sign_in_returns_the_session_and_hides_why_it_failed(net, auth):
    net.on("POST", r"/auth/v1/token\?grant_type=password",
           lambda req: (200, session_json()) if body(req)["password"] == "right-password"
           else (400, {"error_code": "invalid_credentials", "msg": "Invalid login credentials"}))
    s = auth.sign_in(" A@Example.com ", "right-password")
    assert s.user == USER and s.access_token == "access-1"
    req = net.sent[0]
    assert req.get_header("Apikey") == "anon-key" and body(req)["email"] == "a@example.com"
    with pytest.raises(AuthError, match="Wrong email or password") as e:
        auth.sign_in("a@example.com", "wrong-password")
    assert e.value.status == 400


def test_sign_up_checks_input_and_reports_email_confirmation(net, auth):
    net.on("POST", r"/auth/v1/signup", lambda req: (200, {"id": USER.id, "email": USER.email}))
    for email, pw in (("not-an-email", "longenough"), ("a@example.com", "short")):
        with pytest.raises(AuthError):
            auth.sign_up(email, pw)
    assert not net.sent  # bad input never reaches Supabase
    assert auth.sign_up("a@example.com", "longenough", "http://localhost:8000/account") is None
    assert "redirect_to=http%3A%2F%2Flocalhost%3A8000%2Faccount" in net.sent[0].full_url


def test_checked_tokens_are_cached_and_forgotten_on_sign_out(net, auth):
    net.on("GET", r"/auth/v1/user$", lambda req: (200, {"id": USER.id, "email": USER.email}))
    net.on("POST", r"/auth/v1/logout", lambda req: (204, None))
    assert auth.user("tok").id == USER.id
    assert auth.user("tok").id == USER.id
    assert len(net.calls("GET", "/user")) == 1
    assert net.sent[0].get_header("Authorization") == "Bearer tok"
    auth.sign_out("tok")
    auth.user("tok")
    assert len(net.calls("GET", "/user")) == 2


def test_supabase_down_is_a_503_not_a_wrong_password(net, auth):
    def down(req):
        raise urllib.error.URLError("connection refused")
    net.on("POST", r"/auth/v1/", down)
    with pytest.raises(AuthError) as e:
        auth.sign_in("a@example.com", "whatever-pw")
    assert e.value.status == 503


# -- profile ---------------------------------------------------------------------------------

def test_profile_combines_profile_credits_and_links_without_tokens(net, db):
    net.on("GET", r"/rest/v1/profiles\?", lambda req: (200, [{"display_name": "Al", "plan": "free",
                                                               "created_at": "2026-11-01"}]))
    net.on("GET", r"/rest/v1/credit_balance\?", lambda req: (200, [{"balance": 4}]))
    net.on("GET", r"/rest/v1/linked_accounts\?", lambda req: (200, [
        {"id": 1, "provider": "youtube", "provider_user_id": "UC1", "handle": "@al"}]))
    me = accounts.profile(db, USER)
    assert me["credits"] == 4 and me["display_name"] == "Al" and me["linked"][0]["handle"] == "@al"
    links_query = urllib.parse.unquote(net.calls("GET", "linked_accounts")[0].full_url)
    assert f"user_id=eq.{USER.id}" in links_query
    assert "token" not in links_query.split("select=")[1].split("&")[0].replace("token_expires_at", "")


def test_display_name_is_trimmed_validated_and_scoped_to_the_user(net, db):
    sent = []
    net.on("PATCH", r"/rest/v1/profiles\?", lambda req: sent.append(req) or (200, [{"id": USER.id}]))
    net.on("GET", r"/rest/v1/", lambda req: (200, []))
    accounts.update_profile(db, USER, {"display_name": "  Big   Al \n"})
    assert body(sent[0])["display_name"] == "Big Al"
    assert f"id=eq.{USER.id}" in sent[0].full_url
    accounts.update_profile(db, USER, {"display_name": ""})
    assert body(sent[1])["display_name"] is None
    for bad in ({"display_name": "x" * 51}, {"display_name": 5}, {}, {"plan": "paid"}):
        with pytest.raises(AccountError):
            accounts.update_profile(db, USER, bad)


# -- OAuth state -----------------------------------------------------------------------------

def test_state_only_works_for_the_same_user_provider_and_ten_minutes():
    state = accounts.sign_state(SECRET, USER.id, "youtube", now=1000)
    accounts.check_state(SECRET, state, USER.id, "youtube", now=1000 + 599)
    bad = [(SECRET, state, OTHER, "youtube", 1000),          # someone else's callback
           (SECRET, state, USER.id, "twitch", 1000),         # other provider
           (SECRET, state, USER.id, "youtube", 1000 + 601),  # too old
           (b"other-secret", state, USER.id, "youtube", 1000),
           (SECRET, state[:-2] + "xx", USER.id, "youtube", 1000),
           (SECRET, "", USER.id, "youtube", 1000)]
    for secret, st, uid, prov, now in bad:
        with pytest.raises(AccountError):
            accounts.check_state(secret, st, uid, prov, now=now)


def test_authorize_url_asks_google_for_a_refresh_token(oauth_env):
    url = accounts.authorize_url(accounts.PROVIDERS["youtube"], "st", "http://localhost:8000/cb")
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
    assert q["access_type"] == "offline" and q["prompt"] == "consent" and q["state"] == "st"
    assert q["client_id"] == "google-id" and "youtube.readonly" in q["scope"]


def test_unconfigured_provider_says_what_to_set(monkeypatch):
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    with pytest.raises(AccountError, match="TWITCH_CLIENT_ID") as e:
        accounts.authorize_url(accounts.PROVIDERS["twitch"], "st", "http://x/cb")
    assert e.value.status == 503


# -- linking -----------------------------------------------------------------------------------

def youtube_fakes(net, existing=()):
    net.on("POST", r"oauth2\.googleapis\.com/token", lambda req: (200, {
        "access_token": "ya29", "refresh_token": "1//r", "expires_in": 3599,
        "scope": "https://www.googleapis.com/auth/youtube.readonly"}))
    net.on("GET", r"youtube/v3/channels", lambda req: (200, {"items": [{"id": "UC123", "snippet": {
        "title": "Al Plays", "customUrl": "@alplays",
        "thumbnails": {"default": {"url": "https://yt3/al.jpg"}}}}]}))
    net.on("GET", r"/rest/v1/linked_accounts\?", lambda req: (200, list(existing)))
    net.on("POST", r"/rest/v1/linked_accounts$", lambda req: (201, [{"id": 7, **body(req)[0]}]))
    net.on("PATCH", r"/rest/v1/linked_accounts\?", lambda req: (200, [{"id": 7}]))
    net.on("POST", r"oauth2\.googleapis\.com/revoke", lambda req: (200, {}))


def finish(db, provider="youtube", user=USER):
    state = accounts.sign_state(SECRET, user.id, provider)
    return accounts.finish_link(db, accounts.PROVIDERS[provider], user, "the-code", state,
                                "http://localhost:8000/api/connect/youtube/callback", SECRET)


def test_linking_youtube_stores_the_channel_and_tokens(net, db, oauth_env):
    youtube_fakes(net)
    finish(db)
    token_req = net.calls("POST", "oauth2.googleapis.com/token")[0]
    assert body(token_req) == {"client_id": "google-id", "client_secret": "google-secret",
                               "code": "the-code", "grant_type": "authorization_code",
                               "redirect_uri": "http://localhost:8000/api/connect/youtube/callback"}
    assert net.calls("GET", "channels")[0].get_header("Authorization") == "Bearer ya29"
    [row] = body(net.calls("POST", "/rest/v1/linked_accounts")[0])
    assert row["user_id"] == USER.id and row["provider"] == "youtube"
    assert row["provider_user_id"] == "UC123" and row["handle"] == "@alplays"
    assert row["access_token"] == "ya29" and row["refresh_token"] == "1//r"
    assert row["scopes"] == ["https://www.googleapis.com/auth/youtube.readonly"]


def test_relinking_updates_and_someone_elses_channel_is_refused(net, db, oauth_env):
    youtube_fakes(net, existing=[{"id": 7, "user_id": USER.id}])
    finish(db)
    assert net.calls("PATCH", "linked_accounts") and not net.calls("POST", "/rest/v1/linked_accounts")

    net.sent.clear()
    youtube_fakes(net, existing=[{"id": 7, "user_id": OTHER}])
    with pytest.raises(AccountError, match="already linked to another") as e:
        finish(db)
    assert e.value.status == 409
    assert not net.calls("PATCH", "linked_accounts")
    assert net.calls("POST", "revoke")  # the unused grant is given back


def test_linking_a_tampered_state_never_spends_the_code(net, db, oauth_env):
    youtube_fakes(net)
    state = accounts.sign_state(SECRET, OTHER, "youtube")
    with pytest.raises(AccountError, match="expired or came from another session"):
        accounts.finish_link(db, accounts.PROVIDERS["youtube"], USER, "code", state, "cb", SECRET)
    assert not net.sent


def test_google_account_without_a_channel(net, db, oauth_env):
    youtube_fakes(net)
    net.on("GET", r"youtube/v3/channels", lambda req: (200, {"items": []}))
    with pytest.raises(AccountError, match="no YouTube channel"):
        finish(db)


def test_linking_twitch_sends_the_client_id(net, db, oauth_env):
    net.on("POST", r"id\.twitch\.tv/oauth2/token", lambda req: (200, {
        "access_token": "tw-a", "refresh_token": "tw-r", "expires_in": 14000, "scope": []}))
    net.on("GET", r"api\.twitch\.tv/helix/users", lambda req: (200, {"data": [{
        "id": "1234", "login": "alplays", "display_name": "AlPlays", "profile_image_url": "u"}]}))
    net.on("GET", r"/rest/v1/linked_accounts\?", lambda req: (200, []))
    net.on("POST", r"/rest/v1/linked_accounts$", lambda req: (201, [body(req)[0]]))
    finish(db, "twitch")
    assert net.calls("GET", "helix")[0].get_header("Client-id") == "twitch-id"
    [row] = body(net.calls("POST", "/rest/v1/linked_accounts")[0])
    assert row["provider"] == "twitch" and row["handle"] == "alplays" and row["scopes"] == []


def test_unlink_revokes_and_only_touches_the_users_own_row(net, db, oauth_env):
    net.on("GET", r"/rest/v1/linked_accounts\?", lambda req: (200, [
        {"id": 7, "user_id": USER.id, "provider": "youtube", "refresh_token": "1//r"}]))
    net.on("DELETE", r"/rest/v1/linked_accounts\?", lambda req: (200, [{"id": 7}]))
    net.on("POST", r"oauth2\.googleapis\.com/revoke", lambda req: (200, {}))
    assert accounts.unlink(db, USER, 7) == {"unlinked": "youtube"}
    assert body(net.calls("POST", "revoke")[0]) == {"token": "1//r"}
    for req in net.calls(pattern="linked_accounts"):
        assert f"user_id=eq.{USER.id}" in req.full_url and "id=eq.7" in req.full_url
    with pytest.raises(AccountError):
        accounts.unlink(db, USER, "7")


def test_expired_tokens_are_refreshed_and_kept(net, db, oauth_env):
    net.on("POST", r"oauth2\.googleapis\.com/token", lambda req: (200, {"access_token": "new",
                                                                         "expires_in": 3600}))
    net.on("PATCH", r"/rest/v1/linked_accounts\?", lambda req: (200, [{}]))
    soon = (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat()
    later = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    row = {"id": 7, "user_id": USER.id, "provider": "youtube", "access_token": "old",
           "refresh_token": "1//r", "token_expires_at": later}
    assert accounts.access_token(db, row) == "old" and not net.sent
    row["token_expires_at"] = soon
    assert accounts.access_token(db, row) == "new"
    assert body(net.calls("POST", "token")[0])["grant_type"] == "refresh_token"
    patch = body(net.calls("PATCH", "linked_accounts")[0])
    assert patch["access_token"] == "new" and "refresh_token" not in patch  # Google sent none

    net.on("POST", r"oauth2\.googleapis\.com/token", lambda req: (400, {"error": "invalid_grant"}))
    row["token_expires_at"] = soon
    with pytest.raises(AccountError, match="Link it again") as e:
        accounts.access_token(db, row)
    assert e.value.status == 401


# -- deleting an account ----------------------------------------------------------------------

class FakeStore:
    def __init__(self, fail=False):
        self.deleted, self.fail = [], fail

    def delete_prefix(self, prefix):
        if self.fail:
            raise StorageError("bucket says no")
        self.deleted.append(prefix)
        return 2


def test_delete_account_revokes_deletes_files_then_the_user(net, db, auth, oauth_env):
    net.on("GET", r"/rest/v1/linked_accounts\?", lambda req: (200, [
        {"id": 1, "provider": "twitch", "access_token": "tw-a"}]))
    net.on("POST", r"id\.twitch\.tv/oauth2/revoke", lambda req: (200, None))
    net.on("DELETE", r"/auth/v1/admin/users/", lambda req: (200, {}))
    store = FakeStore()
    out = accounts.delete_account(db, auth, USER, store_factory=lambda: store)
    assert out == {"files_deleted": 6, "unlinked": 1}
    assert store.deleted == [f"clips/{USER.id}/", f"recipes/{USER.id}/", f"vods/{USER.id}/"]
    admin = net.calls("DELETE", "admin/users")[0]
    assert admin.full_url.endswith(f"/auth/v1/admin/users/{USER.id}")
    assert admin.get_header("Authorization") == "Bearer service-key"
    assert [r.get_method() for r in net.sent][-1] == "DELETE"  # the user goes last


def test_account_is_kept_when_its_files_cant_be_deleted(net, db, auth):
    net.on("GET", r"/rest/v1/linked_accounts\?", lambda req: (200, []))
    with pytest.raises(AccountError, match="account was kept"):
        accounts.delete_account(db, auth, USER, store_factory=lambda: FakeStore(fail=True))
    assert not net.calls("DELETE")


def test_delete_prefix_refuses_the_whole_bucket():
    from clipfinder.storage import Store
    store = Store(client=None, bucket="b")
    for prefix in ("", "clips/", "clips"):
        with pytest.raises(StorageError):
            store.delete_prefix(prefix)


# -- the web server ------------------------------------------------------------------------------

@pytest.fixture
def server(net, monkeypatch):
    monkeypatch.setattr(web, "AUTH", Auth(SB, "anon-key", "service-key"))
    monkeypatch.setattr(web, "DB", Database(SB, "service-key"))
    monkeypatch.setattr(web, "SECRET", SECRET)
    monkeypatch.setattr(web, "PUBLIC_URL", "http://localhost:8000")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def request(srv, method, path, data=None, cookies="", ctype="application/json"):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    headers = {"Cookie": cookies} if cookies else {}
    if data is not None:
        headers["Content-Type"] = ctype
    conn.request(method, path, body=None if data is None else json.dumps(data), headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    try:
        payload = json.loads(raw) if raw else None
    except ValueError:
        payload = raw.decode()
    return resp.status, payload, resp.getheader("Location"), resp.msg.get_all("Set-Cookie") or []


def test_sign_in_sets_httponly_cookies_and_me_reads_them(server, net):
    net.on("POST", r"/auth/v1/token\?grant_type=password", lambda req: (200, session_json()))
    net.on("GET", r"/auth/v1/user$", lambda req: (200, {"id": USER.id, "email": USER.email}))
    net.on("GET", r"/rest/v1/", lambda req: (200, []))

    status, me, _, _ = request(server, "GET", "/api/me")
    assert status == 200 and me["configured"] and me["user"] is None

    status, _, _, cookies = request(server, "POST", "/api/auth/login",
                                    {"email": "a@example.com", "password": "pw-123456"})
    assert status == 200
    assert any(c.startswith("clipper_at=access-1;") and "HttpOnly" in c and "SameSite=Lax" in c
               for c in cookies)
    assert any(c.startswith("clipper_rt=refresh-1;") for c in cookies)

    status, me, _, _ = request(server, "GET", "/api/me", cookies="clipper_at=access-1")
    assert me["user"]["email"] == USER.email and me["user"]["credits"] == 0
    assert set(me["providers"]) == {"youtube", "twitch"}


def test_expired_access_token_is_refreshed_from_the_cookie(server, net):
    net.on("GET", r"/auth/v1/user$", lambda req: (401, {"msg": "JWT expired"}))
    net.on("POST", r"grant_type=refresh_token", lambda req: (200, session_json(token="access-2",
                                                                               refresh="refresh-2")))
    net.on("GET", r"/rest/v1/", lambda req: (200, []))
    status, me, _, cookies = request(server, "GET", "/api/me",
                                     cookies="clipper_at=old; clipper_rt=refresh-1")
    assert me["user"]["id"] == USER.id
    assert any(c.startswith("clipper_at=access-2;") for c in cookies)


def test_account_posts_need_json_and_a_session(server, net):
    status, out, _, _ = request(server, "POST", "/api/auth/login", {"email": "x"}, ctype="text/plain")
    assert status == 415
    status, out, _, _ = request(server, "POST", "/api/me", {"display_name": "x"})
    assert status == 401 and out["error"] == "Sign in first."
    assert not net.sent


def test_delete_needs_the_password(server, net):
    net.on("GET", r"/auth/v1/user$", lambda req: (200, {"id": USER.id, "email": USER.email}))
    net.on("POST", r"grant_type=password", lambda req: (400, {"msg": "Invalid login credentials"}))
    status, out, _, _ = request(server, "POST", "/api/me/delete", {"password": "nope-nope"},
                                cookies="clipper_at=access-1")
    assert status == 403 and out["error"] == "Wrong password."
    assert not net.calls("DELETE")


def test_link_start_redirects_to_the_provider_and_callback_back(server, net, oauth_env):
    net.on("GET", r"/auth/v1/user$", lambda req: (200, {"id": USER.id, "email": USER.email}))
    status, _, location, _ = request(server, "GET", "/api/connect/youtube/start")
    assert status == 302 and location.startswith("/account?link_error=Sign%20in")

    status, _, location, _ = request(server, "GET", "/api/connect/youtube/start",
                                     cookies="clipper_at=access-1")
    assert status == 302 and location.startswith("https://accounts.google.com/")
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(location).query))
    assert q["redirect_uri"] == "http://localhost:8000/api/connect/youtube/callback"

    youtube_fakes(net)
    status, _, location, _ = request(
        server, "GET", f"/api/connect/youtube/callback?code=c&state={urllib.parse.quote(q['state'])}",
        cookies="clipper_at=access-1")
    assert location == "/account?linked=youtube"

    status, _, location, _ = request(server, "GET", "/api/connect/youtube/callback?error=access_denied",
                                     cookies="clipper_at=access-1")
    assert "link_error=YouTube%20wasn" in location

    status, _, location, _ = request(server, "GET", "/api/connect/myspace/start",
                                     cookies="clipper_at=access-1")
    assert "link_error=Unknown%20service" in location
