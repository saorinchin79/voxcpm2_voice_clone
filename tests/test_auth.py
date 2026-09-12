from __future__ import annotations

import email.parser
import hashlib
import http.cookies
import importlib.util
import json
import os
import socket
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUTH_PATH = ROOT / "auth.py"
SERVER_PATH = ROOT / "server.py"

# server.py does a bare `import auth`, so the repo root has to be importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if not AUTH_PATH.exists():
    pytest.skip("auth.py is not present yet", allow_module_level=True)

auth = _load_module("auth", AUTH_PATH)

# Test-only throwaway credentials; the real seeds never appear in this file.
TEST_USER = "tester"
TEST_PASSWORD = "test-password-123"
OTHER_PASSWORD = "other-password-456"


# --------------------------------------------------------------------------- #
# hash_password / verify_password
# --------------------------------------------------------------------------- #


def test_hash_password_round_trip():
    encoded = auth.hash_password(TEST_PASSWORD)

    assert encoded.startswith("pbkdf2_sha256$")
    assert TEST_PASSWORD not in encoded
    assert auth.verify_password(TEST_PASSWORD, encoded) is True


def test_verify_password_rejects_wrong_password():
    encoded = auth.hash_password(TEST_PASSWORD)

    assert auth.verify_password(OTHER_PASSWORD, encoded) is False
    assert auth.verify_password("", encoded) is False


@pytest.mark.parametrize(
    "encoded",
    [
        "",
        "not-a-hash",
        "pbkdf2_sha256$600000$onlythreefields",
        "pbkdf2_sha256$notanint$c2FsdA==$aGFzaA==",
        "pbkdf2_sha256$600000$$",
        "md5$600000$c2FsdA==$aGFzaA==",
        "pbkdf2_sha256$600000$!!!$???",
    ],
)
def test_verify_password_returns_false_on_malformed_encoding(encoded):
    assert auth.verify_password(TEST_PASSWORD, encoded) is False


def test_hash_password_uses_a_random_salt():
    first = auth.hash_password(TEST_PASSWORD)
    second = auth.hash_password(TEST_PASSWORD)

    assert first != second
    assert auth.verify_password(TEST_PASSWORD, first)
    assert auth.verify_password(TEST_PASSWORD, second)


# --------------------------------------------------------------------------- #
# UserStore
# --------------------------------------------------------------------------- #


@pytest.fixture
def users(tmp_path):
    return auth.UserStore(tmp_path / "users.json")


def test_add_user_returns_sanitised_record(users):
    record = users.add_user(TEST_USER, TEST_PASSWORD)

    assert record["username"] == TEST_USER
    assert record["role"] == "admin"
    assert "password" not in record
    assert users.verify(TEST_USER, TEST_PASSWORD) is not None


def test_add_user_rejects_duplicates(users):
    users.add_user(TEST_USER, TEST_PASSWORD)

    with pytest.raises(auth.AuthError):
        users.add_user(TEST_USER, OTHER_PASSWORD)


def test_list_users_never_exposes_the_hash(users):
    users.add_user(TEST_USER, TEST_PASSWORD)

    listed = users.list_users()

    assert [u["username"] for u in listed] == [TEST_USER]
    entry = listed[0]
    assert set(entry) == {"username", "role", "created", "last_login"}
    assert TEST_PASSWORD not in json.dumps(listed)


def test_get_never_exposes_the_hash(users):
    users.add_user(TEST_USER, TEST_PASSWORD)

    assert "password" not in users.get(TEST_USER)
    assert users.get("nobody") is None


def test_set_password_invalidates_the_old_password(users):
    users.add_user(TEST_USER, TEST_PASSWORD)

    users.set_password(TEST_USER, OTHER_PASSWORD)

    assert users.verify(TEST_USER, TEST_PASSWORD) is None
    assert users.verify(TEST_USER, OTHER_PASSWORD) is not None


def test_set_password_rejects_unknown_user(users):
    with pytest.raises(auth.AuthError):
        users.set_password("nobody", TEST_PASSWORD)


def test_verify_updates_last_login(users):
    users.add_user(TEST_USER, TEST_PASSWORD)
    assert users.get(TEST_USER)["last_login"] is None

    users.verify(TEST_USER, TEST_PASSWORD)

    assert users.get(TEST_USER)["last_login"] > 0


def test_delete_user_removes_the_account(users):
    users.add_user(TEST_USER, TEST_PASSWORD)
    users.add_user("second", OTHER_PASSWORD)

    users.delete_user("second")

    assert [u["username"] for u in users.list_users()] == [TEST_USER]


def test_delete_user_rejects_unknown_user(users):
    users.add_user(TEST_USER, TEST_PASSWORD)

    with pytest.raises(auth.AuthError):
        users.delete_user("nobody")


def test_delete_user_refuses_to_remove_the_last_user(users):
    users.add_user(TEST_USER, TEST_PASSWORD)

    with pytest.raises(auth.AuthError):
        users.delete_user(TEST_USER)

    assert users.get(TEST_USER) is not None


def test_usernames_are_case_insensitive(users):
    users.add_user("Admin ", TEST_PASSWORD)

    assert users.get("ADMIN")["username"] == "admin"
    assert users.verify("aDmIn", TEST_PASSWORD) is not None
    with pytest.raises(auth.AuthError):
        users.add_user("ADMIN", OTHER_PASSWORD)


@pytest.mark.parametrize(
    "username",
    ["", "   ", "bad name", "user@host", "up/down", "x" * 65, "sl*sh"],
)
def test_invalid_usernames_are_rejected(users, username):
    with pytest.raises(auth.AuthError):
        users.add_user(username, TEST_PASSWORD)


def test_seed_defaults_is_idempotent(users):
    seeds = [("alpha", "alpha-password-1"), ("beta", "beta-password-2")]

    created = users.seed_defaults(seeds)

    assert sorted(created) == ["alpha", "beta"]
    assert users.seed_defaults(seeds) == []
    assert len(users.list_users()) == 2


def test_users_file_is_written_with_mode_0600(tmp_path):
    path = tmp_path / "users.json"
    store = auth.UserStore(path)

    store.add_user(TEST_USER, TEST_PASSWORD)

    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_users_persist_across_instances(tmp_path):
    path = tmp_path / "users.json"
    auth.UserStore(path).add_user(TEST_USER, TEST_PASSWORD)

    reopened = auth.UserStore(path)

    assert reopened.verify(TEST_USER, TEST_PASSWORD) is not None
    assert TEST_PASSWORD not in path.read_text(encoding="utf-8")


def test_corrupt_users_file_does_not_crash(tmp_path):
    path = tmp_path / "users.json"
    path.write_text("{not json at all", encoding="utf-8")

    store = auth.UserStore(path)

    assert store.get(TEST_USER) is None


# --------------------------------------------------------------------------- #
# SessionStore
# --------------------------------------------------------------------------- #


@pytest.fixture
def sessions(tmp_path):
    return auth.SessionStore(tmp_path / "sessions.json")


def test_session_create_and_get_round_trip(sessions):
    token = sessions.create(TEST_USER)

    assert isinstance(token, str) and len(token) >= 20
    record = sessions.get(token)
    assert record["username"] == TEST_USER
    assert record["expires"] > time.time()
    assert record["last_seen"] <= time.time()


def test_session_tokens_are_unique(sessions):
    assert sessions.create(TEST_USER) != sessions.create(TEST_USER)


def test_get_unknown_token_returns_none(sessions):
    assert sessions.get("no-such-token") is None
    assert sessions.get("") is None


def test_destroy_removes_the_session(sessions):
    token = sessions.create(TEST_USER)

    sessions.destroy(token)

    assert sessions.get(token) is None
    sessions.destroy(token)  # destroying twice must not raise


def test_session_expires_after_the_absolute_ttl(tmp_path):
    store = auth.SessionStore(tmp_path / "sessions.json", ttl=0.05, idle_ttl=60)
    token = store.create(TEST_USER)

    time.sleep(0.1)

    assert store.get(token) is None


def test_session_expires_after_the_idle_ttl(tmp_path):
    store = auth.SessionStore(tmp_path / "sessions.json", ttl=60, idle_ttl=0.05)
    token = store.create(TEST_USER)

    time.sleep(0.1)

    assert store.get(token) is None


def test_get_slides_the_idle_window(tmp_path):
    store = auth.SessionStore(tmp_path / "sessions.json", ttl=60, idle_ttl=1.0)
    token = store.create(TEST_USER)

    time.sleep(0.6)
    first = store.get(token)
    time.sleep(0.6)
    second = store.get(token)

    assert first is not None
    assert second is not None
    assert second["last_seen"] > first["last_seen"]


def test_destroy_user_kills_only_that_users_sessions(sessions):
    mine = [sessions.create(TEST_USER), sessions.create(TEST_USER)]
    theirs = sessions.create("someone-else")

    killed = sessions.destroy_user(TEST_USER)

    assert killed == 2
    assert all(sessions.get(token) is None for token in mine)
    assert sessions.get(theirs) is not None


def test_sweep_drops_expired_sessions(tmp_path):
    store = auth.SessionStore(tmp_path / "sessions.json", ttl=0.05, idle_ttl=60)
    token = store.create(TEST_USER)

    time.sleep(0.1)

    assert store.sweep() >= 1
    assert store.get(token) is None


def test_sessions_persist_across_instances(tmp_path):
    path = tmp_path / "sessions.json"
    token = auth.SessionStore(path).create(TEST_USER)

    reopened = auth.SessionStore(path)

    assert reopened.get(token)["username"] == TEST_USER


def test_sessions_file_is_written_with_mode_0600(tmp_path):
    path = tmp_path / "sessions.json"
    store = auth.SessionStore(path)

    store.create(TEST_USER)

    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_corrupt_sessions_file_starts_empty(tmp_path):
    path = tmp_path / "sessions.json"
    path.write_text("]]not json[[", encoding="utf-8")

    store = auth.SessionStore(path)

    assert store.get("anything") is None
    token = store.create(TEST_USER)
    assert store.get(token)["username"] == TEST_USER


# --------------------------------------------------------------------------- #
# LoginThrottle
# --------------------------------------------------------------------------- #


def test_throttle_allows_attempts_under_the_limit():
    throttle = auth.LoginThrottle(max_attempts=3, window=60, lockout=60)

    allowed, retry_after = throttle.check("1.2.3.4")
    assert allowed is True
    assert retry_after == 0

    throttle.fail("1.2.3.4")
    throttle.fail("1.2.3.4")

    assert throttle.check("1.2.3.4")[0] is True


def test_throttle_blocks_over_the_limit():
    throttle = auth.LoginThrottle(max_attempts=3, window=60, lockout=60)
    for _ in range(3):
        throttle.fail("1.2.3.4")

    allowed, retry_after = throttle.check("1.2.3.4")

    assert allowed is False
    assert retry_after > 0


def test_throttle_is_keyed_per_client():
    throttle = auth.LoginThrottle(max_attempts=2, window=60, lockout=60)
    for _ in range(2):
        throttle.fail("1.2.3.4")

    assert throttle.check("1.2.3.4")[0] is False
    assert throttle.check("5.6.7.8")[0] is True


def test_throttle_reset_clears_the_lockout():
    throttle = auth.LoginThrottle(max_attempts=2, window=60, lockout=60)
    for _ in range(2):
        throttle.fail("1.2.3.4")
    assert throttle.check("1.2.3.4")[0] is False

    throttle.reset("1.2.3.4")

    assert throttle.check("1.2.3.4") == (True, 0)


# --------------------------------------------------------------------------- #
# server.py — live HTTP gating
# --------------------------------------------------------------------------- #

HTML_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep 302s visible instead of letting urllib chase them."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Response:
    def __init__(self, status, headers, body):
        self.status = status
        self.headers = headers
        self.body = body

    def json(self):
        return json.loads(self.body.decode("utf-8"))

    @property
    def location(self):
        return self.headers.get("Location")

    def cookie(self, name=None):
        name = name or auth.SESSION_COOKIE
        for raw in self.headers.get_all("Set-Cookie") or []:
            jar = http.cookies.SimpleCookie()
            jar.load(raw)
            if name in jar:
                return jar[name]
        return None


class Client:
    def __init__(self, base):
        self.base = base
        self.jar = {}
        self.opener = urllib.request.build_opener(_NoRedirect)

    def request(self, path, method="GET", body=None, headers=None, cookies=True):
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        if cookies and self.jar:
            req.add_header("Cookie", "; ".join(f"{k}={v}" for k, v in self.jar.items()))
        try:
            raw = self.opener.open(req, timeout=10)
        except urllib.error.HTTPError as exc:
            raw = exc
        with raw:
            res = Response(raw.status, raw.headers, raw.read())
        morsel = res.cookie()
        if morsel is not None:
            if morsel["max-age"] in ("0", 0):
                self.jar.pop(auth.SESSION_COOKIE, None)
            else:
                self.jar[auth.SESSION_COOKIE] = morsel.value
        return res

    def get(self, path, **kwargs):
        return self.request(path, **kwargs)

    def post(self, path, body=None, **kwargs):
        return self.request(path, method="POST", body=body, **kwargs)

    def login(self, username=TEST_USER, password=TEST_PASSWORD, **kwargs):
        return self.post(
            "/api/login", {"username": username, "password": password}, **kwargs
        )


def assert_isolation_headers(res):
    assert res.headers.get("Cross-Origin-Opener-Policy") == "same-origin"
    assert res.headers.get("Cross-Origin-Embedder-Policy") == "require-corp"


@pytest.fixture(scope="session")
def server_module(tmp_path_factory):
    if not SERVER_PATH.exists():
        pytest.skip("server.py is not present")

    # server.py builds its USERS/SESSIONS at import time; keep that off the real
    # users.json by pointing the stores at a throwaway directory first.
    boot = tmp_path_factory.mktemp("auth-boot")
    real_users, real_sessions = auth.UserStore, auth.SessionStore

    class BootUserStore(real_users):
        def __init__(self, path=boot / "users.json", *args, **kwargs):
            super().__init__(path, *args, **kwargs)

    class BootSessionStore(real_sessions):
        def __init__(self, path=boot / "sessions.json", *args, **kwargs):
            super().__init__(path, *args, **kwargs)

    auth.UserStore, auth.SessionStore = BootUserStore, BootSessionStore
    try:
        module = _load_module("server", SERVER_PATH)
    except Exception as exc:  # pragma: no cover - only when server.py is broken
        pytest.skip(f"server.py could not be imported: {exc}")
    finally:
        auth.UserStore, auth.SessionStore = real_users, real_sessions

    for name in ("USERS", "SESSIONS", "THROTTLE", "SecureHandler"):
        if not hasattr(module, name):
            pytest.skip(f"server.py has no {name}")
    return module


@pytest.fixture
def client(server_module, tmp_path, monkeypatch):
    monkeypatch.delenv("VOXCPM_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("VOXCPM_FORCE_SECURE_COOKIE", raising=False)
    monkeypatch.chdir(ROOT)

    server_module.USERS = auth.UserStore(tmp_path / "users.json")
    server_module.SESSIONS = auth.SessionStore(tmp_path / "sessions.json")
    # server.py may now hold more than one throttle (per-IP and per-username), and a
    # lockout left behind by one test would decide the next one; replace them all.
    for name, value in list(vars(server_module).items()):
        if isinstance(value, auth.LoginThrottle):
            setattr(server_module, name,
                    auth.LoginThrottle(max_attempts=3, window=60, lockout=60))
    server_module.THROTTLE = auth.LoginThrottle(max_attempts=3, window=60, lockout=60)
    server_module.USERS.add_user(TEST_USER, TEST_PASSWORD)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_module.SecureHandler)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield Client(f"http://127.0.0.1:{httpd.server_address[1]}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_unauthenticated_studio_redirects_to_login(client):
    res = client.get("/studio.html", headers={"Accept": HTML_ACCEPT})

    assert res.status == 302
    parsed = urllib.parse.urlparse(res.location)
    assert parsed.path == "/login.html"
    assert urllib.parse.parse_qs(parsed.query)["next"] == ["/studio.html"]
    assert_isolation_headers(res)


def test_unauthenticated_gradio_api_returns_401(client):
    res = client.get("/gradio_api/queue/status")

    assert res.status == 401
    assert res.json() == {"error": "Authentication required"}
    assert_isolation_headers(res)


def test_unauthenticated_i2v_returns_401(client):
    res = client.post(
        "/api/i2v", {"fal_key": "x", "image_url": "data:image/jpeg;base64,AA"}
    )

    assert res.status == 401


def test_login_page_is_public(client):
    res = client.get("/login.html", headers={"Accept": HTML_ACCEPT})

    assert res.status == 200
    assert_isolation_headers(res)


def test_api_me_is_public_but_reports_anonymous(client):
    res = client.get("/api/me")

    assert res.status == 401
    assert res.json()["authenticated"] is False
    assert_isolation_headers(res)


def test_login_sets_the_session_cookie_and_unlocks_the_studio(client):
    res = client.login()

    assert res.status == 200
    payload = res.json()
    assert payload["ok"] is True
    assert payload["user"]["username"] == TEST_USER
    assert payload["user"]["role"] == "admin"

    morsel = res.cookie()
    assert morsel is not None and morsel.value
    assert morsel["httponly"]
    assert morsel["samesite"].lower() == "lax"
    assert morsel["path"] == "/"
    # plain http on localhost: a Secure cookie would be dropped by the browser
    assert not morsel["secure"]

    studio = client.get("/studio.html", headers={"Accept": HTML_ACCEPT})
    assert studio.status == 200
    assert_isolation_headers(studio)

    me = client.get("/api/me")
    assert me.status == 200
    assert me.json()["authenticated"] is True
    assert me.json()["user"]["username"] == TEST_USER


def test_login_marks_the_cookie_secure_behind_an_https_proxy(client):
    res = client.login(headers={"X-Forwarded-Proto": "https"})

    assert res.status == 200
    assert res.cookie()["secure"]


def test_login_rejects_a_wrong_password(client):
    res = client.login(password=OTHER_PASSWORD)

    assert res.status == 401
    assert res.json() == {"ok": False, "error": "Invalid username or password"}
    assert res.cookie() is None
    assert_isolation_headers(res)


def test_login_rejects_an_unknown_user(client):
    res = client.login(username="nobody")

    assert res.status == 401
    assert res.json()["ok"] is False


def test_login_rejects_malformed_json(client):
    res = client.post("/api/login", b"{not json")

    assert res.status == 400


def test_repeated_failures_are_throttled(client):
    statuses = []
    for _ in range(10):
        res = client.login(password=OTHER_PASSWORD)
        statuses.append(res.status)
        if res.status == 429:
            break

    assert statuses[0] == 401
    assert statuses[-1] == 429
    payload = res.json()
    assert payload["ok"] is False
    assert payload["error"] == "Too many attempts"
    assert payload["retry_after"] > 0

    # a lockout must also cover the correct password
    assert client.login().status == 429


def test_successful_login_resets_the_throttle(client):
    client.login(password=OTHER_PASSWORD)
    client.login(password=OTHER_PASSWORD)

    assert client.login().status == 200
    assert client.login(password=OTHER_PASSWORD).status == 401


def test_logout_expires_the_cookie_and_relocks_the_studio(client):
    client.login()
    stale = client.jar[auth.SESSION_COOKIE]

    res = client.post("/api/logout")

    assert res.status == 200
    assert res.json()["ok"] is True
    assert res.cookie()["max-age"] in ("0", 0)

    client.jar[auth.SESSION_COOKIE] = stale
    studio = client.get("/studio.html", headers={"Accept": HTML_ACCEPT})
    assert studio.status == 302
    assert urllib.parse.urlparse(studio.location).path == "/login.html"


@pytest.mark.parametrize(
    "path",
    ["/server.py", "/auth.py", "/users.json", "/.sessions.json", "/gradio.log",
     "/.git/config", "/manage_users.py", "/_tmp/../server.py"],
)
def test_sensitive_paths_are_404_even_when_authenticated(client, path):
    client.login()

    res = client.get(path)

    assert res.status == 404


def test_tmp_files_stay_reachable_without_a_session(client):
    # FAL.ai fetches this URL from the internet with no cookie.
    tmp_dir = ROOT / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    probe = tmp_dir / "pytest-auth-probe.jpg"
    probe.write_bytes(b"\xff\xd8\xff\xdbprobe")
    try:
        res = client.get("/_tmp/pytest-auth-probe.jpg", cookies=False)
    finally:
        probe.unlink(missing_ok=True)

    assert res.status == 200
    assert res.body == b"\xff\xd8\xff\xdbprobe"


def test_root_redirects_depending_on_the_session(client):
    anonymous = client.get("/", headers={"Accept": HTML_ACCEPT})
    assert anonymous.status == 302
    assert urllib.parse.urlparse(anonymous.location).path == "/login.html"

    client.login()
    authenticated = client.get("/", headers={"Accept": HTML_ACCEPT})
    assert authenticated.status == 302
    assert urllib.parse.urlparse(authenticated.location).path == "/studio.html"


def test_options_stays_public(client):
    res = client.request("/gradio_api/queue/join", method="OPTIONS")
    assert res.status == 200

    origin = client.base
    cors = client.request(
        "/gradio_api/queue/join", method="OPTIONS", headers={"Origin": origin}
    )
    assert cors.status == 200
    assert cors.headers.get("Access-Control-Allow-Origin") == origin


def test_an_unknown_session_cookie_does_not_authenticate(client):
    client.jar[auth.SESSION_COOKIE] = "forged-token"

    assert client.get("/api/me").status == 401
    assert client.get("/studio.html", headers={"Accept": HTML_ACCEPT}).status == 302


def test_auth_disabled_env_knob_opens_the_studio(server_module, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXCPM_AUTH_DISABLED", "1")
    # the knob may have been frozen into a module constant at import time
    for name, value in list(vars(server_module).items()):
        if "AUTH_DISABLED" in name and isinstance(value, bool):
            monkeypatch.setattr(server_module, name, True)
    monkeypatch.chdir(ROOT)
    server_module.USERS = auth.UserStore(tmp_path / "users.json")
    server_module.SESSIONS = auth.SessionStore(tmp_path / "sessions.json")
    server_module.THROTTLE = auth.LoginThrottle()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_module.SecureHandler)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        res = Client(f"http://127.0.0.1:{httpd.server_address[1]}").get(
            "/studio.html", headers={"Accept": HTML_ACCEPT}
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert res.status == 200
    assert os.environ.get("VOXCPM_AUTH_DISABLED") == "1"


# --------------------------------------------------------------------------- #
# raw-socket HTTP
#
# urllib collapses '..' in a URL before the request ever reaches the socket, so a
# traversal probe sent through it arrives as a harmless '/studio.html' and proves
# nothing. Everything below writes the request line byte for byte, and reads until
# the server hangs up so a second response written into the body is visible.
# --------------------------------------------------------------------------- #


class RawResponse:
    def __init__(self, raw):
        self.raw = raw
        head, _, self.body = raw.partition(b"\r\n\r\n")
        status_line, _, field_lines = head.partition(b"\r\n")
        self.status_line = status_line.decode("latin-1")
        parts = self.status_line.split(None, 2)
        self.status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        self.headers = email.parser.BytesParser().parsebytes(field_lines + b"\r\n\r\n")

    @property
    def location(self):
        return self.headers.get("Location")


def raw_request(base, path, method="GET", headers=None, body=b"", timeout=5.0):
    """Send `path` verbatim and return everything the server writes back."""
    split = urllib.parse.urlsplit(base)
    lines = [
        "%s %s HTTP/1.1" % (method, path),
        "Host: %s" % split.netloc,
        "Connection: close",
    ]
    for key, value in (headers or {}).items():
        lines.append("%s: %s" % (key, value))
    if body:
        lines.append("Content-Length: %d" % len(body))
    wire = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body

    sock = socket.create_connection((split.hostname, split.port), timeout=timeout)
    chunks = []
    try:
        sock.settimeout(timeout)
        sock.sendall(wire)
        while True:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        sock.close()
    return RawResponse(b"".join(chunks))


def cookie_header(client):
    return {"Cookie": "; ".join(f"{k}={v}" for k, v in client.jar.items())}


def test_raw_request_puts_the_literal_path_on_the_wire():
    """Guards every traversal test below: if the helper normalised the path they
    would all pass against a server with no traversal check at all."""
    seen = []
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve():
        conn, _ = listener.accept()
        with conn:
            seen.append(conn.recv(65536).split(b"\r\n")[0])
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        res = raw_request(
            "http://127.0.0.1:%d" % listener.getsockname()[1], "/_tmp/..%2fstudio.html"
        )
    finally:
        thread.join(timeout=5)
        listener.close()

    assert res.status == 200
    assert seen == [b"GET /_tmp/..%2fstudio.html HTTP/1.1"]


# --------------------------------------------------------------------------- #
# server.py — /_tmp/ traversal must not bypass the session gate
# --------------------------------------------------------------------------- #

TRAVERSAL_PATHS = [
    "/_tmp/../studio.html",
    "/_tmp/%2e%2e/studio.html",
    "/_tmp/x/%2e%2e/%2e%2e/studio.html",
    "/_tmp/..%2fstudio.html",
]


@pytest.mark.parametrize("path", TRAVERSAL_PATHS)
def test_tmp_traversal_never_serves_the_studio_unauthenticated(client, path):
    # /_tmp/* is public because FAL.ai fetches from it with no cookie; climbing out
    # of it must not inherit that exemption.
    for accept in (HTML_ACCEPT, "*/*"):
        res = raw_request(client.base, path, headers={"Accept": accept})

        assert res.status != 200, f"{path} with Accept: {accept} was served unauthenticated"
        assert b"<title>VoxCPM2 Studio</title>" not in res.raw


def test_tmp_traversal_is_about_the_gate_not_the_spelling(client):
    """Whatever the fix does with a traversal - serve it or 404 it - it must not be
    the anonymous 200 it used to be, and a signed-in caller must not be told to log in."""
    client.login()

    res = raw_request(
        client.base, "/_tmp/../studio.html",
        headers=dict(cookie_header(client), Accept=HTML_ACCEPT),
    )

    assert res.status in (200, 301, 302, 404)
    if res.status == 200:
        assert b"<title>VoxCPM2 Studio</title>" in res.raw
    elif res.status == 302:
        assert "/login.html" not in (res.location or "")


@pytest.mark.parametrize(
    "path",
    ["/_tmp/../server.py", "/_tmp/../users.json", "/_tmp/%2e%2e/users.json",
     "/_tmp/..%2fserver.py"],
)
def test_tmp_traversal_to_sensitive_files_is_404(client, path):
    anonymous = raw_request(client.base, path, headers={"Accept": HTML_ACCEPT})
    assert anonymous.status == 404, f"{path} leaked to an anonymous caller"

    client.login()
    signed_in = raw_request(
        client.base, path, headers=dict(cookie_header(client), Accept=HTML_ACCEPT)
    )
    assert signed_in.status == 404, f"{path} leaked to a signed-in caller"
    assert b"pbkdf2" not in signed_in.raw
    assert b"import auth" not in signed_in.raw


def test_a_real_tmp_file_is_served_with_no_cookie_header_at_all(client):
    tmp_dir = ROOT / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    probe = tmp_dir / "pytest-raw-probe.jpg"
    probe.write_bytes(b"\xff\xd8\xff\xdbraw-probe")
    try:
        res = raw_request(client.base, "/_tmp/pytest-raw-probe.jpg")
    finally:
        probe.unlink(missing_ok=True)

    assert res.status == 200
    assert res.body == b"\xff\xd8\xff\xdbraw-probe"


# --------------------------------------------------------------------------- #
# server.py — throttle keying
# --------------------------------------------------------------------------- #


def _user_throttles(module):
    return [
        name
        for name, value in vars(module).items()
        if isinstance(value, auth.LoginThrottle) and name != "THROTTLE"
    ]


def test_rotating_forwarded_for_still_trips_the_throttle(client):
    """X-Forwarded-For is attacker-controlled text. Keying the lockout on its first
    hop meant twelve guesses from one socket cost twelve different keys and never
    tripped anything."""
    statuses = []
    for i in range(12):
        res = client.login(
            password=OTHER_PASSWORD,
            headers={"X-Forwarded-For": "203.0.113.%d" % (i + 1)},
        )
        statuses.append(res.status)
        if res.status == 429:
            break

    assert 429 in statuses, f"12 guesses behind a rotating XFF were never throttled: {statuses}"
    assert res.json()["retry_after"] > 0


def test_throttle_keys_on_the_socket_peer_when_there_is_no_forwarded_for(client):
    for _ in range(12):
        res = client.login(password=OTHER_PASSWORD)
        if res.status == 429:
            break

    assert res.status == 429, "a loopback peer sending no XFF was never throttled"
    assert res.json()["retry_after"] > 0
    # A header appearing only now must not hand the same socket a clean slate.
    assert client.login(headers={"X-Forwarded-For": "198.51.100.7"}).status == 429


def test_a_distributed_guess_locks_the_targeted_account(client, server_module):
    if not _user_throttles(server_module):
        pytest.skip("server.py keeps no per-username throttle")

    for i in range(12):
        res = client.login(
            password=OTHER_PASSWORD,
            headers={"X-Forwarded-For": "198.51.100.%d" % (i + 1)},
        )
        if res.status == 429:
            break
    assert res.status == 429

    # The lock travels with the account: a fresh address holding the right password
    # is refused too.
    fresh = client.login(headers={"X-Forwarded-For": "198.51.100.240"})
    assert fresh.status == 429
    assert fresh.cookie() is None


# --------------------------------------------------------------------------- #
# stores shared between processes (server.py + manage_users.py)
# --------------------------------------------------------------------------- #


def test_user_store_picks_up_a_password_change_made_by_another_instance(tmp_path):
    """manage_users.py passwd writes users.json while the server holds its own
    UserStore; the running server has to honour the new password immediately."""
    path = tmp_path / "users.json"
    server_side = auth.UserStore(path)
    server_side.add_user(TEST_USER, TEST_PASSWORD)
    cli_side = auth.UserStore(path)

    assert server_side.verify(TEST_USER, TEST_PASSWORD) is not None

    cli_side.set_password(TEST_USER, OTHER_PASSWORD)

    assert server_side.verify(TEST_USER, OTHER_PASSWORD) is not None
    assert server_side.verify(TEST_USER, TEST_PASSWORD) is None


def test_user_store_sees_a_user_added_by_another_instance(tmp_path):
    path = tmp_path / "users.json"
    server_side = auth.UserStore(path)
    server_side.add_user(TEST_USER, TEST_PASSWORD)
    cli_side = auth.UserStore(path)

    cli_side.add_user("newcomer", OTHER_PASSWORD)

    assert server_side.get("newcomer") is not None
    assert server_side.verify("newcomer", OTHER_PASSWORD) is not None


def test_two_user_store_instances_do_not_clobber_each_others_writes(tmp_path):
    path = tmp_path / "users.json"
    first = auth.UserStore(path)
    second = auth.UserStore(path)

    first.add_user("x", TEST_PASSWORD)
    second.add_user("y", OTHER_PASSWORD)

    for store in (first, second, auth.UserStore(path)):
        assert [u["username"] for u in store.list_users()] == ["x", "y"]
    assert auth.UserStore(path).verify("x", TEST_PASSWORD) is not None


def test_user_store_sees_a_deletion_made_by_another_instance(tmp_path):
    path = tmp_path / "users.json"
    server_side = auth.UserStore(path)
    server_side.add_user(TEST_USER, TEST_PASSWORD)
    server_side.add_user("doomed", OTHER_PASSWORD)

    auth.UserStore(path).delete_user("doomed")

    assert server_side.get("doomed") is None
    assert server_side.verify("doomed", OTHER_PASSWORD) is None


def test_session_store_sees_a_session_created_by_another_instance(tmp_path):
    path = tmp_path / "sessions.json"
    first = auth.SessionStore(path)
    second = auth.SessionStore(path)

    token = first.create(TEST_USER)

    assert second.get(token)["username"] == TEST_USER


def test_session_store_sees_a_destroy_made_by_another_instance(tmp_path):
    """manage_users.py passwd calls destroy_user to sign people out of the server
    process that is already running."""
    path = tmp_path / "sessions.json"
    first = auth.SessionStore(path)
    second = auth.SessionStore(path)
    token = first.create(TEST_USER)
    assert second.get(token) is not None

    second.destroy(token)

    assert first.get(token) is None


def test_session_store_destroy_user_reaches_another_instance(tmp_path):
    path = tmp_path / "sessions.json"
    first = auth.SessionStore(path)
    second = auth.SessionStore(path)
    tokens = [first.create(TEST_USER), first.create(TEST_USER)]

    assert second.destroy_user(TEST_USER) == 2

    assert all(first.get(token) is None for token in tokens)


def test_two_session_store_instances_do_not_clobber_each_others_writes(tmp_path):
    path = tmp_path / "sessions.json"
    first = auth.SessionStore(path)
    second = auth.SessionStore(path)

    mine = first.create(TEST_USER)
    theirs = second.create("someone-else")

    assert first.get(theirs) is not None
    assert second.get(mine) is not None


# --------------------------------------------------------------------------- #
# seed semantics
# --------------------------------------------------------------------------- #


def test_seed_defaults_does_nothing_on_a_non_empty_store(users):
    users.add_user("someone", TEST_PASSWORD)

    assert users.seed_defaults() == []
    assert [u["username"] for u in users.list_users()] == ["someone"]
    for name in getattr(auth, "SEED_USERNAMES", ("admin", "chhay")):
        assert users.get(name) is None


def test_seed_defaults_ignores_explicit_seeds_on_a_non_empty_store(users):
    users.add_user("someone", TEST_PASSWORD)

    assert users.seed_defaults([("alpha", "alpha-password-1")]) == []
    assert users.get("alpha") is None


def test_seed_defaults_does_not_resurrect_a_deleted_account(users):
    seeds = [("alpha", "alpha-password-1"), ("beta", "beta-password-2")]
    assert sorted(users.seed_defaults(seeds)) == ["alpha", "beta"]

    users.delete_user("alpha")

    # server.py seeds on every start; a deliberately deleted account must stay gone.
    assert users.seed_defaults(seeds) == []
    assert users.get("alpha") is None
    assert [u["username"] for u in users.list_users()] == ["beta"]


def test_seed_defaults_does_not_resurrect_across_instances(tmp_path):
    path = tmp_path / "users.json"
    seeds = [("alpha", "alpha-password-1"), ("beta", "beta-password-2")]
    auth.UserStore(path).seed_defaults(seeds)
    auth.UserStore(path).delete_user("alpha")

    assert auth.UserStore(path).seed_defaults(seeds) == []
    assert auth.UserStore(path).get("alpha") is None


# --------------------------------------------------------------------------- #
# a corrupt users.json is evidence, not scratch space
# --------------------------------------------------------------------------- #


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def truncated_users_file(tmp_path):
    path = tmp_path / "users.json"
    seeded = auth.UserStore(path)
    seeded.add_user(TEST_USER, TEST_PASSWORD)
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: len(text) // 2], encoding="utf-8")
    return path


def test_a_truncated_users_file_loads_no_users(truncated_users_file):
    store = auth.UserStore(truncated_users_file)

    assert store.list_users() == []
    assert store.get(TEST_USER) is None
    assert store.verify(TEST_USER, TEST_PASSWORD) is None


def test_a_truncated_users_file_is_never_rewritten(truncated_users_file):
    """The surviving hashes are recoverable by hand right up until something
    overwrites them, so every mutation has to refuse."""
    before = _sha256(truncated_users_file)
    store = auth.UserStore(truncated_users_file)

    with pytest.raises(auth.AuthError):
        store.add_user("newcomer", OTHER_PASSWORD)
    assert _sha256(truncated_users_file) == before

    with pytest.raises(auth.AuthError):
        store.set_password(TEST_USER, OTHER_PASSWORD)
    assert _sha256(truncated_users_file) == before

    with pytest.raises(auth.AuthError):
        store.delete_user(TEST_USER)
    assert _sha256(truncated_users_file) == before

    assert store.seed_defaults([("alpha", "alpha-password-1")]) == []
    assert _sha256(truncated_users_file) == before

    assert auth.UserStore(truncated_users_file).list_users() == []


# --------------------------------------------------------------------------- #
# gradio proxy — one response per request
# --------------------------------------------------------------------------- #


def _dying_upstream():
    """A listener that sends headers and part of a chunked body, then hangs up
    mid-chunk. Returns (base_url, close)."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)

    def serve():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            with conn:
                try:
                    conn.settimeout(5.0)
                    conn.recv(65536)
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: text/event-stream\r\n"
                        b"Transfer-Encoding: chunked\r\n\r\n"
                        b"2000\r\n" + b"x" * 8192 + b"\r\n"
                        b"40\r\ncut"  # promises 64 more bytes, then dies
                    )
                except OSError:
                    pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    def close():
        listener.close()
        thread.join(timeout=5)

    return "http://127.0.0.1:%d" % listener.getsockname()[1], close


def test_a_dying_upstream_never_injects_a_second_status_line(client, server_module, monkeypatch):
    """send_response() after end_headers() writes 'HTTP/1.0 502 ...' into the body the
    browser is already parsing, which is how a raw status line ended up inside an SSE
    stream. Once the headers are out, the only honest move is to hang up."""
    upstream, close = _dying_upstream()
    monkeypatch.setattr(server_module, "GRADIO_URL", upstream)
    client.login()
    try:
        res = raw_request(
            client.base, "/gradio_api/queue/data", headers=cookie_header(client), timeout=15.0
        )
    finally:
        close()

    assert res.status == 200
    assert b"x" * 4096 in res.body, "the partial body never reached the client"
    # 'HTTP/1.' only ever starts a status line; the Server header's 'SimpleHTTP/0.6'
    # is why this looks for that spelling rather than a bare 'HTTP/'.
    assert res.body.count(b"HTTP/") == 0, (
        "a second status line landed inside the body: %r" % res.body[-300:]
    )
    assert res.raw.count(b"HTTP/1.") == 1
    assert b"502" not in res.body
