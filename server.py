#!/usr/bin/env python3
import os
import json
import base64
import uuid
import time
import threading
import urllib.request
import urllib.error
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

FAL_MODEL   = 'fal-ai/ltx-video/image-to-video'
FAL_QUEUE   = f'https://queue.fal.run/{FAL_MODEL}'
PUBLIC_HOST = os.environ.get('PUBLIC_HOST', 'https://aivoice.saorin.me')
GRADIO_PORT = int(os.environ.get('GRADIO_PORT', 8808))
GRADIO_URL  = f'http://localhost:{GRADIO_PORT}'

TMP_DIR = Path(__file__).parent / '_tmp'
TMP_DIR.mkdir(exist_ok=True)

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

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Accept')
        self.end_headers()

    def do_GET(self):
        if self.path.startswith('/gradio_api/'):
            self._proxy_gradio()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith('/gradio_api/'):
            self._proxy_gradio()
        elif self.path == '/api/i2v':
            self._handle_i2v()
        else:
            self.send_error(404)

    def _proxy_gradio(self):
        """Forward /gradio_api/* to the Gradio server, streaming the response."""
        target = GRADIO_URL + self.path
        try:
            length = int(self.headers.get('Content-Length', 0))
            body   = self.rfile.read(length) if length else None

            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ('host', 'connection', 'content-length')}

            req = urllib.request.Request(target, data=body, headers=headers)
            req.method = self.command

            with urllib.request.urlopen(req, timeout=120) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ('transfer-encoding', 'connection'):
                        self.send_header(k, v)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                # Stream the body in chunks (important for SSE responses)
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

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

            _, b64data = image_b64.split(',', 1)
            img_bytes  = base64.b64decode(b64data)
            filename   = f'{uuid.uuid4().hex}.jpg'
            (TMP_DIR / filename).write_bytes(img_bytes)
            image_url  = f'{PUBLIC_HOST}/_tmp/{filename}'

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

        except Exception as e:
            import traceback
            msg = traceback.format_exc()
            print('[i2v error]', msg, flush=True)
            self._json(500, {'error': str(e), 'detail': msg[-500:]})

    def _json(self, code, body_dict):
        body = json.dumps(body_dict).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('', port), SecureHandler)
    print(f'Studio → http://localhost:{port}/studio.html')
    print(f'Gradio proxy → /gradio_api/* → {GRADIO_URL}')
    server.serve_forever()
