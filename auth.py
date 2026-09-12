#!/usr/bin/env python3
"""Password hashing, user records, sessions and login throttling for VoxCPM2 Studio.

Standard library only, so the studio server keeps its zero-dependency install.
Every store is safe to share between the threads of a ThreadingHTTPServer: an
RLock guards the in-memory dict and the file write together, and files are
replaced atomically at mode 0600 so a crash can never leave a half-written or
world-readable secret behind.
"""
import base64
import contextlib
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sys
import tempfile
import threading
import time
from pathlib import Path

try:
    import fcntl
except ImportError:
    # No flock on this platform (Windows). The stores still serialise their own
    # threads through the RLock; they simply cannot coordinate with a second
    # process the way they do on macOS and Linux.
    fcntl = None

__all__ = [
    'AuthError', 'DEFAULT_USERS_PATH', 'DEFAULT_SESSIONS_PATH', 'SESSION_COOKIE',
    'SEED_USERS', 'SEED_USERNAMES', 'hash_password', 'verify_password',
    'normalize_username', 'UserStore', 'SessionStore', 'LoginThrottle',
]

DEFAULT_USERS_PATH    = Path(__file__).resolve().parent / 'users.json'
DEFAULT_SESSIONS_PATH = Path(__file__).resolve().parent / '.sessions.json'
SEED_USERS_PATH       = Path(__file__).resolve().parent / '.seed_users.json'
SESSION_COOKIE = 'voxcpm_session'

PBKDF2_ALGORITHM  = 'pbkdf2_sha256'
PBKDF2_ITERATIONS = 600000
PBKDF2_SALT_BYTES = 16

SESSION_TOKEN_BYTES = 32
# last_seen slides on every authenticated request; only rewrite the file once a
# session has aged past this, otherwise a busy studio page rewrites it per fetch.
SESSION_TOUCH_INTERVAL = 60

USERNAME_MAX_LENGTH = 64
_USERNAME_RE = re.compile(r'^[a-z0-9._-]{1,64}$')

SEED_USERNAMES = ('admin', 'chhay')


class AuthError(Exception):
    pass


def _b64(raw):
    return base64.b64encode(raw).decode('ascii')


def hash_password(password, iterations=PBKDF2_ITERATIONS):
    """Return 'pbkdf2_sha256$<iterations>$<b64salt>$<b64hash>'."""
    if not isinstance(password, str) or not password:
        raise AuthError('Password must be a non-empty string')
    salt = secrets.token_bytes(PBKDF2_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    return '%s$%d$%s$%s' % (PBKDF2_ALGORITHM, iterations, _b64(salt), _b64(digest))


def verify_password(password, encoded):
    """Constant-time check of a password against an encoded hash. False if malformed."""
    if not isinstance(password, str) or not isinstance(encoded, str):
        return False
    try:
        algorithm, raw_iterations, raw_salt, raw_digest = encoded.split('$')
        if algorithm != PBKDF2_ALGORITHM:
            return False
        iterations = int(raw_iterations)
        salt = base64.b64decode(raw_salt, validate=True)
        expected = base64.b64decode(raw_digest, validate=True)
    except (ValueError, TypeError):
        return False
    if iterations < 1 or not salt or not expected:
        return False
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt,
                                 iterations, dklen=len(expected))
    return hmac.compare_digest(digest, expected)


_dummy_lock = threading.Lock()
_dummy_cache = []


def _dummy_hash():
    """A throwaway hash so an unknown username costs the same time as a known one."""
    with _dummy_lock:
        if not _dummy_cache:
            _dummy_cache.append(hash_password(secrets.token_urlsafe(24)))
        return _dummy_cache[0]


def normalize_username(username):
    """Strip and lowercase, then validate. Raises AuthError on anything unusable."""
    if not isinstance(username, str):
        raise AuthError('Username must be a string')
    name = username.strip().lower()
    if not name:
        raise AuthError('Username must not be empty')
    if len(name) > USERNAME_MAX_LENGTH:
        raise AuthError('Username must be at most %d characters' % USERNAME_MAX_LENGTH)
    if not _USERNAME_RE.match(name):
        raise AuthError('Username may only contain a-z, 0-9, dot, underscore and hyphen')
    return name


def _as_float(value, default):
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_JSON_OK      = 'ok'
_JSON_MISSING = 'missing'
_JSON_CORRUPT = 'corrupt'


def _read_json_state(path):
    """Return (state, payload) where state is 'ok', 'missing' or 'corrupt'.

    A store has to tell those last two apart: 'missing' means bootstrap me, while
    'corrupt' means the file holds data we could not understand and must not
    clobber.
    """
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return _JSON_OK, json.load(fh)
    except FileNotFoundError:
        return _JSON_MISSING, None
    except (OSError, ValueError):
        return _JSON_CORRUPT, None


def _read_json(path):
    """Return the parsed file, or None when it is missing, unreadable or corrupt."""
    return _read_json_state(path)[1]


def _file_signature(path):
    """(mtime_ns, size, inode), or None when the file is absent.

    Cheap enough to run before every read: one stat, and the inode alone already
    catches a rewrite because _atomic_write_json always publishes a new one.
    """
    try:
        st = os.stat(str(path))
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


@contextlib.contextmanager
def _file_lock(path):
    """Hold an exclusive inter-process lock for path, via a sibling '<name>.lock'.

    The lock lives on its own file because _atomic_write_json replaces the data
    file's inode, which would drop a lock held on the data file itself. Where
    fcntl is unavailable this degrades to a no-op and callers fall back to the
    in-process RLock alone.
    """
    if fcntl is None:
        yield None
        return
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path.parent / (path.name + '.lock')),
                     os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        yield None
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _atomic_write_json(path, payload):
    """Replace path with payload. The temp file is chmod 0600 before it is published,
    so the hashes never exist world-readable even for an instant."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_seed_users():
    """Bootstrap credentials, resolved at import time.

    They are deliberately NOT literals in this module: the repo is public, so a
    password written here would be published on the first push. Order of lookup:
    the VOXCPM_SEED_USERS env var ('user:password,user:password'), then a
    gitignored .seed_users.json next to this file ({"user": "password"}), and
    finally a freshly generated password per required account which
    seed_defaults() prints once, at the moment it creates the account.
    """
    configured = {}

    raw = os.environ.get('VOXCPM_SEED_USERS', '').strip()
    for chunk in raw.split(','):
        chunk = chunk.strip()
        if not chunk or ':' not in chunk:
            continue
        name, password = chunk.split(':', 1)
        if name.strip() and password:
            configured[name.strip().lower()] = password

    from_file = _read_json(SEED_USERS_PATH)
    if isinstance(from_file, dict):
        for name, password in from_file.items():
            if isinstance(name, str) and isinstance(password, str) and name and password:
                configured.setdefault(name.strip().lower(), password)

    generated = {}
    seeds = []
    for name in SEED_USERNAMES:
        password = configured.pop(name, None)
        if password is None:
            password = secrets.token_urlsafe(12)
            generated[name] = password
        seeds.append((name, password))
    for name, password in sorted(configured.items()):
        seeds.append((name, password))
    return seeds, generated


SEED_USERS, _GENERATED_SEED_PASSWORDS = _load_seed_users()


def _stdout_is_tty():
    try:
        return bool(sys.stdout is not None and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _write_seed_credentials(directory, entries):
    """Drop generated bootstrap passwords into a fresh 0600 file, return its path.

    Printing them is only safe when a human is watching: under launchd or systemd
    stdout is a log file (~/VoxCPM/server.log is 0644), so a printed password is
    world-readable forever. The file lands next to users.json at the repo root,
    which the server never serves - its static allowlist is studio.html,
    login.html, favicon.ico and the assets/, examples/ and _tmp/ trees only.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    attempt = 0
    while True:
        name = 'seed-credentials-%d.txt' % stamp if not attempt else \
               'seed-credentials-%d-%d.txt' % (stamp, attempt)
        target = directory / name
        try:
            fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            attempt += 1
            continue
        break
    lines = ['VoxCPM2 Studio bootstrap credentials',
             'Sign in with these, change them, then delete this file.', '']
    for username, password in entries:
        lines.append('%s\t%s' % (username, password))
    lines.append('')
    lines.append('Change one with: python manage_users.py passwd <user>')
    lines.append('')
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    os.chmod(str(target), 0o600)
    return target


def _announce_seed_credentials(entries, directory):
    """Tell the operator about freshly generated passwords without leaking them."""
    if _stdout_is_tty():
        for username, password in entries:
            print('[auth] created user %r with a generated password: %s' % (username, password),
                  flush=True)
            print('[auth] change it now: python manage_users.py passwd %s' % username, flush=True)
        return
    try:
        target = _write_seed_credentials(directory, entries)
    except OSError as exc:
        print('[auth] created %d bootstrap account(s), but the generated passwords could '
              'not be written: %s' % (len(entries), exc), flush=True)
        print('[auth] set a password by hand: python manage_users.py passwd <user>', flush=True)
        return
    print('[auth] created %d bootstrap account(s) with generated passwords.' % len(entries),
          flush=True)
    print('[auth] stdout is not a terminal, so the passwords are NOT printed here.', flush=True)
    print('[auth] read them from %s then delete that file.' % target, flush=True)



class UserStore:
    """users.json: {"version":1,"users":{"<name>":{"password","role","created","last_login"}}}

    The file, not this object, is the source of truth. A long-lived server process
    and a short-lived manage_users.py run share it, so reads re-read the file when
    it has moved on and every mutation is a read-modify-write under _file_lock.
    """

    def __init__(self, path=DEFAULT_USERS_PATH):
        self.path = Path(path)
        self.corrupt = False
        self._lock = threading.RLock()
        self._users = {}
        self._signature = None
        self._load()

    def _warn_corrupt(self):
        banner = '[auth] ' + '=' * 64
        print(banner, flush=True)
        print('[auth] THE USER FILE EXISTS BUT WILL NOT PARSE: %s' % self.path, flush=True)
        print('[auth] Refusing to read or write it. The password hashes inside are', flush=True)
        print('[auth] still recoverable by hand; overwriting it would destroy them.', flush=True)
        print('[auth] Until it is repaired this store holds no users, so every login', flush=True)
        print('[auth] fails and nothing on disk is touched.', flush=True)
        print('[auth] Restore it from a backup or fix the JSON, then restart.', flush=True)
        print(banner, flush=True)

    def _load(self):
        # Stat before reading: a signature taken first can only ever be older than
        # the bytes parsed, which costs a redundant reload, never a stale read.
        signature = _file_signature(self.path)
        state, data = _read_json_state(self.path)
        if state == _JSON_CORRUPT:
            with self._lock:
                self._users = {}
                self._signature = signature
                self.corrupt = True
            self._warn_corrupt()
            return
        users = {}
        if isinstance(data, dict) and isinstance(data.get('users'), dict):
            for name, record in data['users'].items():
                if not isinstance(name, str) or not isinstance(record, dict):
                    continue
                if not isinstance(record.get('password'), str):
                    continue
                try:
                    key = normalize_username(name)
                except AuthError:
                    continue
                users[key] = {
                    'password': record['password'],
                    'role': record.get('role') or 'admin',
                    'created': _as_float(record.get('created'), time.time()),
                    'last_login': _as_float(record.get('last_login'), None),
                }
        with self._lock:
            self._users = users
            self._signature = signature
            self.corrupt = False

    def _refresh_if_changed(self):
        """Pick up another process's writes. One stat per call; the JSON is only
        re-parsed when the file actually changed underneath us."""
        with self._lock:
            unchanged = _file_signature(self.path) == self._signature
        if not unchanged:
            self._load()

    def _save(self):
        if self.corrupt:
            raise AuthError('Refusing to overwrite the corrupt user file %s' % self.path)
        _atomic_write_json(self.path, {'version': 1, 'users': self._users})
        self._signature = _file_signature(self.path)

    def _public(self, name, record):
        return {
            'username': name,
            'role': record.get('role') or 'admin',
            'created': record.get('created'),
            'last_login': record.get('last_login'),
        }

    def _put_user(self, name, encoded, role='admin'):
        """Insert one already-hashed record. The caller holds both locks, has just
        re-read the file, and is responsible for saving."""
        if name in self._users:
            raise AuthError('User %r already exists' % name)
        self._users[name] = {
            'password': encoded,
            'role': role or 'admin',
            'created': time.time(),
            'last_login': None,
        }
        return self._users[name]

    def add_user(self, username, password, role='admin'):
        name = normalize_username(username)
        encoded = hash_password(password)  # 600k rounds: never under the file lock
        with self._lock, _file_lock(self.path):
            self._load()
            record = self._put_user(name, encoded, role)
            self._save()
            return self._public(name, record)

    def set_password(self, username, password):
        name = normalize_username(username)
        encoded = hash_password(password)
        with self._lock, _file_lock(self.path):
            self._load()
            record = self._users.get(name)
            if record is None:
                raise AuthError('No such user: %r' % name)
            record['password'] = encoded
            self._save()
            return self._public(name, record)

    def delete_user(self, username):
        name = normalize_username(username)
        with self._lock, _file_lock(self.path):
            self._load()
            if name not in self._users:
                raise AuthError('No such user: %r' % name)
            if len(self._users) <= 1:
                raise AuthError('Refusing to delete the last remaining user')
            del self._users[name]
            self._save()

    def list_users(self):
        self._refresh_if_changed()
        with self._lock:
            return [self._public(name, self._users[name]) for name in sorted(self._users)]

    def get(self, username):
        try:
            name = normalize_username(username)
        except AuthError:
            return None
        self._refresh_if_changed()
        with self._lock:
            record = self._users.get(name)
            return self._public(name, record) if record is not None else None

    def verify(self, username, password):
        self._refresh_if_changed()
        try:
            name = normalize_username(username)
        except AuthError:
            name = None
        with self._lock:
            record = self._users.get(name) if name else None
            encoded = record['password'] if record is not None else None
        # Hash outside the lock (600k rounds is slow) and hash even for an unknown
        # user, so response timing does not disclose which usernames exist.
        matched = verify_password(password, encoded if encoded is not None else _dummy_hash())
        if record is None or not matched:
            return None
        with self._lock, _file_lock(self.path):
            self._load()
            live = self._users.get(name)
            if live is None:
                return None
            live['last_login'] = time.time()
            self._save()
            return self._public(name, live)

    def seed_defaults(self, seeds=None):
        """Bootstrap an EMPTY store, and only an empty one. Returns the names created.

        server.py runs this on every start, so it must never resurrect an account
        somebody deliberately deleted: if the store already holds any user at all
        this returns [] and writes nothing.
        """
        if seeds is None:
            seeds = SEED_USERS
        self._refresh_if_changed()
        with self._lock:
            if self.corrupt:
                self._warn_corrupt()
                return []
            if self._users:
                return []
        prepared = []
        for username, password in seeds:
            try:
                name = normalize_username(username)
            except AuthError as exc:
                print('[auth] skipping invalid seed user %r: %s' % (username, exc), flush=True)
                continue
            prepared.append((name, password, hash_password(password)))
        created = []
        with self._lock, _file_lock(self.path):
            self._load()
            if self.corrupt or self._users:
                return []
            for name, password, encoded in prepared:
                try:
                    self._put_user(name, encoded)
                except AuthError as exc:
                    print('[auth] could not seed user %r: %s' % (name, exc), flush=True)
                    continue
                created.append((name, password))
            if created:
                self._save()
        generated = [(name, password) for name, password in created
                     if _GENERATED_SEED_PASSWORDS.get(name) == password]
        if generated:
            _announce_seed_credentials(generated, self.path.parent)
        return [name for name, _ in created]


class SessionStore:
    """.sessions.json: {"version":1,"sessions":{"<token>":{"username","created","expires","last_seen"}}}

    Persisted so a LaunchAgent/systemd restart does not sign everybody out, and
    re-read before every lookup so that manage_users.py can sign a user out of the
    already-running server.
    """

    def __init__(self, path=DEFAULT_SESSIONS_PATH, ttl=60 * 60 * 12, idle_ttl=60 * 60 * 4):
        self.path = Path(path)
        self.ttl = ttl
        self.idle_ttl = idle_ttl
        self._lock = threading.RLock()
        self._sessions = {}
        self._signature = None
        self._load()
        self.sweep()

    def _load(self):
        signature = _file_signature(self.path)
        state, data = _read_json_state(self.path)
        if state == _JSON_CORRUPT:
            # Unlike users.json this file is disposable - losing it only signs
            # people out - so keep the damaged copy for a post-mortem and carry on.
            quarantine = self.path.parent / (self.path.name + '.corrupt')
            try:
                os.replace(str(self.path), str(quarantine))
                print('[auth] %s was corrupt; moved it to %s and started with no sessions'
                      % (self.path, quarantine), flush=True)
            except OSError as exc:
                print('[auth] %s is corrupt and could not be moved aside: %s'
                      % (self.path, exc), flush=True)
            with self._lock:
                self._sessions = {}
                self._signature = _file_signature(self.path)
            return
        sessions = {}
        if isinstance(data, dict) and isinstance(data.get('sessions'), dict):
            for token, record in data['sessions'].items():
                if not isinstance(token, str) or not isinstance(record, dict):
                    continue
                username = record.get('username')
                if not isinstance(username, str) or not username:
                    continue
                created = _as_float(record.get('created'), None)
                expires = _as_float(record.get('expires'), None)
                last_seen = _as_float(record.get('last_seen'), created)
                if created is None or expires is None or last_seen is None:
                    continue
                sessions[token] = {
                    'username': username,
                    'created': created,
                    'expires': expires,
                    'last_seen': last_seen,
                }
        with self._lock:
            self._sessions = sessions
            self._signature = signature

    def _refresh_if_changed(self):
        """Same one-stat check as UserStore: re-parse only on a real change."""
        with self._lock:
            unchanged = _file_signature(self.path) == self._signature
        if not unchanged:
            self._load()

    def _save(self):
        _atomic_write_json(self.path, {'version': 1, 'sessions': self._sessions})
        self._signature = _file_signature(self.path)

    def _expired(self, record, now):
        return now >= record['expires'] or (now - record['last_seen']) >= self.idle_ttl

    def create(self, username):
        name = normalize_username(username)
        now = time.time()
        token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
        with self._lock, _file_lock(self.path):
            self._load()
            for stale in [t for t, r in self._sessions.items() if self._expired(r, now)]:
                del self._sessions[stale]
            self._sessions[token] = {
                'username': name,
                'created': now,
                'expires': now + self.ttl,
                'last_seen': now,
            }
            self._save()
        return token

    def get(self, token):
        if not isinstance(token, str) or not token:
            return None
        self._refresh_if_changed()
        now = time.time()
        with self._lock:
            record = self._sessions.get(token)
            if record is None:
                return None
            expired = self._expired(record, now)
            drifted = (now - record['last_seen']) >= SESSION_TOUCH_INTERVAL
            result = None
            if not expired:
                record['last_seen'] = now
                result = dict(record)
        if expired or drifted:
            with self._lock, _file_lock(self.path):
                self._load()
                live = self._sessions.get(token)
                if live is not None:
                    if expired:
                        del self._sessions[token]
                    else:
                        live['last_seen'] = now
                    self._save()
        return result

    def destroy(self, token):
        if not isinstance(token, str) or not token:
            return
        with self._lock, _file_lock(self.path):
            self._load()
            if self._sessions.pop(token, None) is not None:
                self._save()

    def destroy_user(self, username):
        try:
            name = normalize_username(username)
        except AuthError:
            return 0
        with self._lock, _file_lock(self.path):
            self._load()
            doomed = [t for t, r in self._sessions.items() if r['username'] == name]
            for token in doomed:
                del self._sessions[token]
            if doomed:
                self._save()
            return len(doomed)

    def sweep(self):
        now = time.time()
        with self._lock, _file_lock(self.path):
            self._load()
            doomed = [t for t, r in self._sessions.items() if self._expired(r, now)]
            for token in doomed:
                del self._sessions[token]
            if doomed:
                self._save()
            return len(doomed)


class LoginThrottle:
    """In-memory failed-login counter, keyed by client IP."""

    def __init__(self, max_attempts=8, window=300, lockout=300):
        self.max_attempts = max_attempts
        self.window = window
        self.lockout = lockout
        self._lock = threading.RLock()
        self._failures = {}
        self._blocked = {}

    def check(self, key):
        now = time.time()
        with self._lock:
            until = self._blocked.get(key)
            if until is None:
                return True, 0
            if now < until:
                return False, max(1, int(math.ceil(until - now)))
            del self._blocked[key]
            self._failures.pop(key, None)
            return True, 0

    def fail(self, key):
        now = time.time()
        with self._lock:
            hits = [t for t in self._failures.get(key, []) if now - t < self.window]
            hits.append(now)
            if len(hits) >= self.max_attempts:
                self._blocked[key] = now + self.lockout
                self._failures.pop(key, None)
            else:
                self._failures[key] = hits
            self._prune(now)

    def reset(self, key):
        with self._lock:
            self._failures.pop(key, None)
            self._blocked.pop(key, None)

    def _prune(self, now):
        """Keep a flood of one-shot IPs from growing the dicts without bound."""
        if len(self._failures) + len(self._blocked) < 1024:
            return
        for key in [k for k, v in self._failures.items() if not v or now - v[-1] >= self.window]:
            del self._failures[key]
        for key in [k for k, v in self._blocked.items() if now >= v]:
            del self._blocked[key]
