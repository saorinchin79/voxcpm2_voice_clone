#!/usr/bin/env python3
import os
import json
import base64
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler

FAL_MODEL = 'fal-ai/ltx-video/image-to-video'
FAL_QUEUE = f'https://queue.fal.run/{FAL_MODEL}'
FAL_STORAGE = 'https://storage.fal.run/'

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
            data = json.loads(self.rfile.read(length))

            fal_key   = data.get('fal_key', '').strip()
            image_b64 = data.get('image_url', '')   # data:image/jpeg;base64,...
            prompt    = data.get('prompt', 'camera tilt horizontal')

            if not fal_key:   raise ValueError('Missing fal_key')
            if not image_b64: raise ValueError('Missing image_url')

            # Decode base64 data URL
            if ',' not in image_b64:
                raise ValueError('image_url must be a base64 data URL')
            header, b64 = image_b64.split(',', 1)
            img_bytes = base64.b64decode(b64)
            mime = header.split(':')[1].split(';')[0]   # e.g. image/jpeg

            # Upload to FAL.ai storage (server-to-server — no CORS)
            boundary = b'X-BOUNDARY'
            form = (
                b'--' + boundary + b'\r\n'
                b'Content-Disposition: form-data; name="file"; filename="image.jpg"\r\n'
                b'Content-Type: ' + mime.encode() + b'\r\n\r\n' +
                img_bytes +
                b'\r\n--' + boundary + b'--\r\n'
            )
            up_req = urllib.request.Request(
                FAL_STORAGE, data=form,
                headers={
                    'Authorization': f'Key {fal_key}',
                    'Content-Type': f'multipart/form-data; boundary={boundary.decode()}'
                }
            )
            with urllib.request.urlopen(up_req, timeout=30) as r:
                up = json.loads(r.read())
            image_url = up.get('url')
            if not image_url:
                raise ValueError('FAL.ai storage returned no URL: ' + str(up)[:200])

            # Submit inference job with the hosted URL
            payload = json.dumps({
                'image_url': image_url,
                'prompt': prompt,
                'num_inference_steps': 30,
                'guidance_scale': 3,
                'negative_prompt': 'blur, distort, low quality, worst quality'
            }).encode()
            infer_req = urllib.request.Request(
                FAL_QUEUE, data=payload,
                headers={
                    'Authorization': f'Key {fal_key}',
                    'Content-Type': 'application/json'
                }
            )
            with urllib.request.urlopen(infer_req, timeout=30) as r:
                result = json.loads(r.read())

            self._json(200, result)

        except Exception as e:
            import traceback
            msg = traceback.format_exc()
            print('[i2v error]', msg, flush=True)
            self._json(500, {'error': str(e), 'detail': msg[-400:]})

    def _json(self, code, data):
        body = json.dumps(data).encode()
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
