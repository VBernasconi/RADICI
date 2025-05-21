# https_server.py
import http.server
import ssl

port = 8080  # puoi usare 443 se sei root

httpd = http.server.HTTPServer(('0.0.0.0', port), http.server.SimpleHTTPRequestHandler)

httpd.socket = ssl.wrap_socket(httpd.socket,
                               server_side=True,
                               certfile="cert.pem",
                               keyfile="key.pem",
                               ssl_version=ssl.PROTOCOL_TLS)

print(f"Serving HTTPS on port {port}...")
httpd.serve_forever()