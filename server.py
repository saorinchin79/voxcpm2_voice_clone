#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler

class SecureHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Required for FFmpeg.wasm SharedArrayBuffer
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

    def log_message(self, format, *args):
        pass  # silent logging

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('', port), SecureHandler)
    print(f'Studio running at http://localhost:{port}/studio.html')
    server.serve_forever()
