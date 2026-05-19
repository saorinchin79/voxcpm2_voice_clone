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
    server = HTTPServer(('', 8080), SecureHandler)
    print('Studio running at http://localhost:8080/studio.html')
    server.serve_forever()
