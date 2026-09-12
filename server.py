#!/usr/bin/env python3
import os
import json
import base64
import ipaddress
import uuid
import time
import posixpath
import threading
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from http.cookies import CookieError, SimpleCookie
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import auth

FAL_MODEL   = 'fal-ai/ltx-video/image-to-video'
FAL_QUEUE   = f'https://queue.fal.run/{FAL_MODEL}'
# Explicit override only. Left unset, the URL handed to FAL.ai is derived per request from
# the Host nginx forwarded, because a hardcoded deploy host rots in silence: the old default
# below now answers 526 while the live studio is studio.saorin.me.
PUBLIC_HOST          = os.environ.get('PUBLIC_HOST')
PUBLIC_HOST_FALLBACK = 'https://aivoice.saorin.me'
GRADIO_PORT = int(os.environ.get('GRADIO_PORT', 8808))
# Full override for when the Gradio engine is not on this host — e.g. DP Server
# reaches the Mac's Gradio through the cloudflared tunnel at https://tts.saorin.me
GRADIO_URL  = os.environ.get('GRADIO_URL', f'http://localhost:{GRADIO_PORT}').rstrip('/')

ROOT    = Path(__file__).resolve().parent
TMP_DIR = ROOT / '_tmp'
TMP_DIR.mkdir(exist_ok=True)

USERS    = auth.UserStore()
SESSIONS = auth.SessionStore()
THROTTLE = auth.LoginThrottle()
# A per-IP counter alone never stops a spray from many IPs against one account. The budget
# is deliberately generous so guessing at a name cannot lock its owner out of it.
USER_THROTTLE = auth.LoginThrottle(max_attempts=50, window=900, lockout=300)

# Local debugging escape hatch only; every gate below honours it.
AUTH_DISABLED = os.environ.get('VOXCPM_AUTH_DISABLED') == '1'

# The handler used to serve the whole repo (source, logs, .git). Only these are reachable.
STATIC_FILES = ('studio.html', 'login.html', 'favicon.ico')
STATIC_DIRS  = ('assets', 'examples', '_tmp')
# /_tmp/* stays public: FAL.ai fetches the jpg from the internet with no cookie.
PUBLIC_PATHS = ('/login.html', '/login', '/favicon.ico', '/api/login', '/api/me')
MAX_LOGIN_BODY = 8192


def _normalize_ip(value):
    """Canonical text of an IP address, or None when the value is not one at all."""
    try:
        parsed = ipaddress.ip_address((value or '').strip())
    except ValueError:
        return None
    mapped = getattr(parsed, 'ipv4_mapped', None)
    return str(mapped if mapped is not None else parsed)


def _parse_trusted_proxies(raw):
    values = [v for v in (raw or '').replace(',', ' ').split() if v]
    if not values:
        # Exactly the DP Server setup: nginx on the same box, talking to us over loopback.
        values = ['127.0.0.1', '::1']
    return frozenset(ip for ip in (_normalize_ip(v) for v in values) if ip)


TRUSTED_PROXIES = _parse_trusted_proxies(os.environ.get('VOXCPM_TRUSTED_PROXIES'))


def _is_loopback_host(host):
    """FAL.ai fetches the jpg from the internet, so a loopback host can never serve it."""
    name = (host or '').split('://', 1)[-1].split('/', 1)[0].strip().lower()
    if name.startswith('['):
        name = name[1:].split(']', 1)[0]
    elif name.count(':') == 1:
        name = name.rsplit(':', 1)[0]
    if not name or name == 'localhost' or name.endswith('.localhost'):
        return True
    parsed = _normalize_ip(name)
    if parsed is None:
        return False
    address = ipaddress.ip_address(parsed)
    return address.is_loopback or address.is_unspecified


def _cleanup_old_tmp():
    while True:
        time.sleep(300)
        cutoff = time.time() - 600
        for f in TMP_DIR.glob('*.jpg'):
            f.unlink(missing_ok=True) if f.stat().st_mtime < cutoff else None

threading.Thread(target=_cleanup_old_tmp, daemon=True).start()


class SecureHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

    def log_message(self, format, *args):
        pass

    # ---------- request helpers ----------

    def _url_path(self):
        return self.path.split('?', 1)[0].split('#', 1)[0]

    def _client_ip(self):
        """Throttle key: the socket peer, unless that peer is a proxy we trust.

        nginx's proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for APPENDS to
        whatever the client sent, so hop 0 is attacker text and rotating it walked straight
        past the lockout. The last hop is the one the adjacent proxy wrote itself. Anything
        that is not an IP is dropped so junk cannot mint unbounded throttle keys.
        """
        try:
            peer = _normalize_ip(self.client_address[0]) or str(self.client_address[0])[:64]
        except Exception:
            peer = 'unknown'
        if peer in TRUSTED_PROXIES:
            forwarded = self.headers.get('X-Forwarded-For')
            if forwarded:
                candidate = _normalize_ip(forwarded.split(',')[-1])
                if candidate:
                    return candidate[:64]
        return peer[:64]

    def _is_https(self):
        """HTTPS terminates at nginx, so trust X-Forwarded-Proto for the Secure flag.
        Getting this wrong drops the cookie on plain-http localhost and login does nothing."""
        if os.environ.get('VOXCPM_FORCE_SECURE_COOKIE') == '1':
            return True
        proto = (self.headers.get('X-Forwarded-Proto') or '').split(',')[0].strip().lower()
        return proto == 'https'

    def _cookie(self, name):
        raw = self.headers.get('Cookie')
        if not raw:
            return None
        jar = SimpleCookie()
        try:
            jar.load(raw)
        except CookieError:
            return None
        found = jar.get(name)
        return found.value if found else None

    def _session_cookie(self, token, max_age=None):
        ttl = int(getattr(SESSIONS, 'ttl', 60 * 60 * 12)) if max_age is None else max_age
        parts = [f'{auth.SESSION_COOKIE}={token}', 'Path=/', 'HttpOnly',
                 'SameSite=Lax', f'Max-Age={ttl}']
        if self._is_https():
            parts.append('Secure')
        return '; '.join(parts)

    def _send_cors(self):
        """'*' plus cookies leaks authenticated responses to any origin that asks, so echo
        only an Origin that is this very host and send nothing at all to anyone else."""
        origin = self.headers.get('Origin')
        if not origin:
            return
        try:
            netloc = urllib.parse.urlsplit(origin).netloc.lower()
        except ValueError:
            return
        if netloc and netloc == (self.headers.get('Host') or '').lower():
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Access-Control-Allow-Credentials', 'true')
            self.send_header('Vary', 'Origin')

    def _json(self, code, body_dict, set_cookie=None):
        body = json.dumps(body_dict).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        if set_cookie:
            self.send_header('Set-Cookie', set_cookie)
        if self.close_connection:
            self.send_header('Connection', 'close')
        self._send_cors()
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _redirect(self, location, code=302):
        self.send_response(code)
        self.send_header('Location', location)
        self.send_header('Content-Length', '0')
        if self.close_connection:
            self.send_header('Connection', 'close')
        self._send_cors()
        self.end_headers()

    # ---------- auth ----------

    def _current_user(self):
        if AUTH_DISABLED:
            return {'username': 'local', 'role': 'admin'}
        token = self._cookie(auth.SESSION_COOKIE)
        if not token:
            return None
        session = SESSIONS.get(token)
        if not session:
            return None
        record = USERS.get(session['username'])
        if not record:
            SESSIONS.destroy(token)
            return None
        return {'username': session['username'], 'role': record.get('role') or 'admin'}

    def _is_public(self):
        """Decide on the DECODED, NORMALIZED path, the one the allowlist actually resolves.

        Testing the raw path meant /_tmp/../studio.html and /_tmp/%2e%2e/studio.html read as
        public and then resolved to studio.html: the whole studio was readable with no cookie.
        A request is public only when it asks for a public file by that exact spelling, so no
        traversal can wear the /_tmp/ prefix as a disguise.
        """
        parts = self._path_parts()
        if not parts:
            return False
        normalized = '/' + '/'.join(parts)
        if normalized != self._url_path():
            return False
        return parts[0] == '_tmp' or normalized in PUBLIC_PATHS

    def _gate(self):
        """True when the request may proceed; otherwise the 302/401 has already been sent."""
        if AUTH_DISABLED or self._is_public() or self._current_user():
            return True
        if self.command in ('GET', 'HEAD') and 'text/html' in (self.headers.get('Accept') or ''):
            nxt = urllib.parse.quote(self.path[:512], safe='')
            self._redirect(f'/login.html?next={nxt}')
        else:
            # The body of a rejected POST is never drained, so this connection is done.
            if self.command not in ('GET', 'HEAD'):
                self.close_connection = True
            self._json(401, {'error': 'Authentication required'})
        return False

    # ---------- static allowlist ----------

    def _path_parts(self, path=None):
        """Decoded, normalized path segments, or None when the path is unusable.

        SimpleHTTPRequestHandler URL-decodes before touching the filesystem, so the
        allowlist has to be applied to the decoded path or /assets/..%2fusers.json
        walks straight out of it. _is_public() and _resolve_static() both read this one
        spelling; the result is cached because a single request asks for it several times.
        """
        raw = self.path if path is None else path
        cached = getattr(self, '_path_parts_cache', None)
        if cached is not None and cached[0] == raw:
            return cached[1]
        decoded = raw.split('?', 1)[0].split('#', 1)[0]
        try:
            decoded = urllib.parse.unquote(decoded, errors='surrogatepass')
        except UnicodeDecodeError:
            decoded = urllib.parse.unquote(decoded)
        decoded = decoded.replace('\\', '/')
        parts = [p for p in posixpath.normpath(decoded).split('/') if p and p != '.']
        if not parts or any(p == '..' or p.startswith('.') for p in parts):
            parts = None
        self._path_parts_cache = (raw, parts)
        return parts

    def _resolve_static(self, path):
        """Map a URL path to a real file inside the allowlist, or None.

        resolve() collapses symlinks, which is the only way to notice a link inside
        assets/ pointing at the rest of the disk.
        """
        parts = self._path_parts(path)
        if not parts:
            return None
        if len(parts) == 1:
            if parts[0] not in STATIC_FILES:
                return None
        elif parts[0] not in STATIC_DIRS:
            return None
        elif parts[0] == '_tmp' and len(parts) != 2:
            return None
        try:
            resolved = ROOT.joinpath(*parts).resolve()
            resolved.relative_to(ROOT)
            if not resolved.is_file():
                return None
        except (OSError, ValueError):
            return None
        return resolved

    def translate_path(self, path):
        target = self._resolve_static(path)
        if target is None:
            # Anything off the allowlist maps to a name that cannot exist, so even a
            # code path that skips the dispatch below still 404s.
            return str(ROOT / '.__forbidden__')
        return str(target)

    # ---------- dispatch ----------

    def do_OPTIONS(self):
        # Preflight stays public (it carries no data); _send_cors keeps the answer
        # scoped to this host so a foreign origin still cannot read a reply.
        self.send_response(200)
        self.send_header('Content-Length', '0')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Accept')
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        self._get_or_head()

    def do_HEAD(self):
        self._get_or_head()

    def _get_or_head(self):
        path = self._url_path()
        if path == '/api/me':
            self._handle_me()
            return
        if path == '/login':
            query = self.path.split('?', 1)[1] if '?' in self.path else ''
            self._redirect('/login.html' + (f'?{query}' if query else ''))
            return
        if path in ('/', '/index.html'):
            self._redirect('/studio.html' if self._current_user() else '/login.html')
            return
        if path.startswith('/gradio_api/'):
            if self._gate():
                self._proxy_gradio()
            return
        if self._resolve_static(self.path) is None:
            # Off the allowlist is a flat 404 for everyone, signed in or not: a session
            # is no reason to hand out server.py, users.json or the logs, and a uniform
            # 404 keeps an unauthenticated GET/HEAD from confirming what exists.
            self.send_error(404)
            return
        if not self._gate():
            return
        if self.command == 'HEAD':
            super().do_HEAD()
        else:
            super().do_GET()

    def do_POST(self):
        path = self._url_path()
        if path == '/login':
            # login.html posts here when JS is off. The body carries a password, so it is
            # never read, parsed or logged; the form reloads and submits /api/login instead.
            # Leaving that body unread means this connection cannot be reused.
            self.close_connection = True
            self._redirect('/login.html', code=303)
            return
        if path == '/api/login':
            self._handle_login()
            return
        if not self._gate():
            return
        if path.startswith('/gradio_api/'):
            self._proxy_gradio()
        elif path == '/api/logout':
            self._handle_logout()
        elif path == '/api/i2v':
            self._handle_i2v()
        else:
            self.send_error(404)

    # ---------- auth endpoints ----------

    def _handle_login(self):
        key = self._client_ip()
        allowed, retry_after = THROTTLE.check(key)
        if not allowed:
            self.close_connection = True
            self._json(429, {'ok': False, 'error': 'Too many attempts',
                             'retry_after': retry_after})
            return
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_LOGIN_BODY:
            # Refuse an oversized body without reading a byte of it.
            self.close_connection = True
            self._json(400, {'ok': False, 'error': 'Invalid request'})
            return
        try:
            data = json.loads(self.rfile.read(length).decode('utf-8'))
            username = data['username']
            password = data['password']
        except Exception:
            self._json(400, {'ok': False, 'error': 'Invalid request'})
            return
        if not isinstance(username, str) or not isinstance(password, str):
            self._json(400, {'ok': False, 'error': 'Invalid request'})
            return

        user_key = self._user_throttle_key(username)
        allowed, retry_after = USER_THROTTLE.check(user_key)
        if not allowed:
            # Unlike the per-IP check above, the body is already drained here, so the
            # connection stays reusable.
            self._json(429, {'ok': False, 'error': 'Too many attempts',
                             'retry_after': retry_after})
            return

        user = USERS.verify(username, password)
        if not user:
            THROTTLE.fail(key)
            USER_THROTTLE.fail(user_key)
            self._json(401, {'ok': False, 'error': 'Invalid username or password'})
            return

        THROTTLE.reset(key)
        USER_THROTTLE.reset(user_key)
        name  = user.get('username') or username.strip().lower()
        token = SESSIONS.create(name)
        self._json(200, {'ok': True, 'user': {'username': name,
                                              'role': user.get('role') or 'admin'}},
                   set_cookie=self._session_cookie(token))

    def _user_throttle_key(self, username):
        """Throttle the account too, not just the IP, so a spray from many addresses at one
        name still trips. Unusable names never verify but still spend budget, under a key
        clamped in length so they cannot grow the table."""
        try:
            return auth.normalize_username(username)
        except auth.AuthError:
            return username.strip().lower()[:64]

    def _handle_logout(self):
        token = self._cookie(auth.SESSION_COOKIE)
        if token:
            SESSIONS.destroy(token)
        self._json(200, {'ok': True}, set_cookie=self._session_cookie('', max_age=0))

    def _handle_me(self):
        user = self._current_user()
        if user:
            self._json(200, {'authenticated': True, 'user': user})
        else:
            self._json(401, {'authenticated': False})

    # ---------- proxy / fal ----------

    def _forward_headers(self):
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ('host', 'connection', 'content-length', 'cookie')}
        # Keep whatever cookies the engine itself set, but never hand it our session token.
        kept = [c for c in (self.headers.get('Cookie') or '').split(';')
                if c.strip() and not c.strip().startswith(auth.SESSION_COOKIE + '=')]
        if kept:
            headers['Cookie'] = ';'.join(kept)
        return headers

    def _proxy_gradio(self):
        """Forward /gradio_api/* to the Gradio server, streaming the response.

        Once the status line is out there is no way to take it back: a second
        send_response() lands inside the body the client is already reading, which is how
        a raw 'HTTP/1.0 502 ...' ended up embedded in an SSE stream. After end_headers()
        the only honest move is to hang up and let the client see a truncated response.
        """
        target = GRADIO_URL + self.path
        started = False
        try:
            length = int(self.headers.get('Content-Length', 0))
            body   = self.rfile.read(length) if length else None

            req = urllib.request.Request(target, data=body, headers=self._forward_headers())
            req.method = self.command

            with urllib.request.urlopen(req, timeout=120) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ('transfer-encoding', 'connection'):
                        self.send_header(k, v)
                self._send_cors()
                self.end_headers()
                started = True
                # Stream the body in chunks (important for SSE responses)
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            # The browser cancelled a generation. Nobody is listening and nothing is wrong,
            # so do not spray a traceback into server.log over it.
            self.close_connection = True
        except urllib.error.HTTPError as e:
            if started:
                self.close_connection = True
                print(f'[gradio proxy] upstream {e.code} after response started', flush=True)
                return
            body = e.read()
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            if started:
                self.close_connection = True
                print(f'[gradio proxy] {e} after response started', flush=True)
                return
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def _public_host(self):
        """The origin FAL.ai will fetch the jpg from: PUBLIC_HOST when the operator set it,
        otherwise this request's own Host and forwarded scheme."""
        if PUBLIC_HOST:
            return PUBLIC_HOST.rstrip('/')
        host = (self.headers.get('Host') or '').strip()
        if not host:
            return PUBLIC_HOST_FALLBACK.rstrip('/')
        proto = (self.headers.get('X-Forwarded-Proto') or '').split(',')[0].strip().lower()
        return '%s://%s' % (proto if proto in ('http', 'https') else 'http', host)

    def _handle_i2v(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            data   = json.loads(self.rfile.read(length))

            fal_key   = data.get('fal_key', '').strip()
            image_b64 = data.get('image_url', '')
            prompt    = data.get('prompt', 'camera tilt horizontal')

            if not fal_key:   raise ValueError('Missing fal_key')
            if not image_b64: raise ValueError('Missing image_url')
            if ',' not in image_b64:
                raise ValueError('image_url must be a base64 data URL')

            public_host = self._public_host()
            if _is_loopback_host(public_host):
                self._json(500, {'error': 'FAL.ai cannot fetch an image from a loopback '
                                          'address. Set PUBLIC_HOST to the public URL of '
                                          'this studio.'})
                return

            _, b64data = image_b64.split(',', 1)
            img_bytes  = base64.b64decode(b64data)
            filename   = f'{uuid.uuid4().hex}.jpg'
            (TMP_DIR / filename).write_bytes(img_bytes)
            image_url  = f'{public_host}/_tmp/{filename}'

            payload = json.dumps({
                'image_url': image_url,
                'prompt': prompt,
                'num_inference_steps': 30,
                'guidance_scale': 3,
                'negative_prompt': 'blur, distort, low quality, worst quality'
            }).encode()
            req = urllib.request.Request(
                FAL_QUEUE, data=payload,
                headers={'Authorization': f'Key {fal_key}', 'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                result = json.loads(r.read())

            self._json(200, result)

        except Exception:
            import traceback
            # The trace names absolute paths and quotes source lines, so it goes to the log
            # and nowhere near the browser.
            print('[i2v error]', traceback.format_exc(), flush=True)
            self._json(500, {'error': 'Image-to-video request failed'})


if __name__ == '__main__':
    os.chdir(ROOT)
    port = int(os.environ.get('PORT', 9090))
    host = os.environ.get('HOST', '127.0.0.1')

    created = USERS.seed_defaults()
    if created:
        print(f'Seeded users: {", ".join(created)}', flush=True)
    if AUTH_DISABLED:
        print('*** WARNING: VOXCPM_AUTH_DISABLED=1 - every endpoint is OPEN, no login required ***',
              flush=True)
    else:
        names = ', '.join(u['username'] for u in USERS.list_users()) or 'none'
        print(f'Auth enabled - users: {names} - sign in at /login.html', flush=True)

    server = ThreadingHTTPServer((host, port), SecureHandler)
    print(f'Studio → http://localhost:{port}/studio.html  (bound {host}:{port})', flush=True)
    print(f'Gradio proxy → /gradio_api/* → {GRADIO_URL}', flush=True)
    server.serve_forever()
