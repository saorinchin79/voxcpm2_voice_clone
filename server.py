#!/usr/bin/env python3
import os
import json
import base64
import uuid
import time
import threading
import urllib.request
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

FAL_MODEL  = 'fal-ai/ltx-video/image-to-video'
FAL_QUEUE  = f'https://queue.fal.run/{FAL_MODEL}'
PUBLIC_HOST = os.environ.get('PUBLIC_HOST', 'https://aivoice.saorin.me')

# Temp images are saved here and served publicly
TMP_DIR = Path(__file__).parent / '_tmp'
TMP_DIR.mkdir(exist_ok=True)

def _cleanup_old_tmp():
    """Delete temp images older than 10 minutes."""
    while True:
        time.sleep(300)
        cutoff = time.time() - 600
        for f in TMP_DIR.glob('*.jpg'):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)

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
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/i2v':
            self._handle_i2v()
        else:
            self.send_error(404)

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

            # Decode and save as temp file — give FAL.ai a real HTTPS URL
            _, b64data = image_b64.split(',', 1)
            img_bytes  = base64.b64decode(b64data)
            filename   = f'{uuid.uuid4().hex}.jpg'
            (TMP_DIR / filename).write_bytes(img_bytes)
            image_url  = f'{PUBLIC_HOST}/_tmp/{filename}'

            # Submit to FAL.ai queue
            payload = json.dumps({
                'image_url': image_url,
                'prompt': prompt,
                'num_inference_steps': 30,
                'guidance_scale': 3,
                'negative_prompt': 'blur, distort, low quality, worst quality'
            }).encode()
            req = urllib.request.Request(
                FAL_QUEUE, data=payload,
                headers={
                    'Authorization': f'Key {fal_key}',
                    'Content-Type': 'application/json'
                }
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
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('', port), SecureHandler)
    print(f'Studio running at http://localhost:{port}/studio.html')
    server.serve_forever()
