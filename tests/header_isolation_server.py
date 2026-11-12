"""Mock HTTP server for navigation header isolation tests.

Records every request it receives and serves configurable redirect chains,
including cross-origin redirects expressed through absolute ``Location``
headers. Records can be read back from ``/requests`` (optionally reset with
``?reset=1``) so that tests can assert on the headers, method and body actually
received on every hop of a chain.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import urlopen


class _HeaderIsolationHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        pass

    def _handle(self) -> None:
        parsed = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        path = parsed.path

        # the inspection endpoint itself must not pollute the recorded traffic
        if path == "/requests":
            records = list(self.server.records)  # type: ignore[attr-defined]
            if query.get("reset"):
                self.server.records.clear()  # type: ignore[attr-defined]
            self._send_json({"records": records})
            return

        content_length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(content_length) if content_length else b""
        self.server.records.append(  # type: ignore[attr-defined]
            {
                "method": self.command,
                "path": self.path,
                "headers": dict(self.headers),
                "body": body.decode("utf8", "replace"),
            }
        )

        if path in ("/headers", "/final"):
            self._send_json(dict(self.headers))
            return

        # configurable redirect: /redirect?status=307&to=<absolute or relative url>
        if path == "/redirect":
            status = int(query.get("status") or 302)
            location = query.get("to") or "/final"
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.send_header("Location", location)
            self.end_headers()
            return

        # form target: echo method and body
        if path == "/form":
            self._send_json({"method": self.command, "body": body.decode("utf8", "replace")})
            return

        self._send_json({"error": "unknown path"}, status=404)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload, indent=2).encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_DELETE = _handle


class HeaderIsolationMockServer:
    """Threaded HTTP server recording requests for header isolation tests.

    To be used as a context manager:

        with HeaderIsolationMockServer() as server:
            url = server.urljoin("/redirect?status=307")
    """

    def __enter__(self) -> "HeaderIsolationMockServer":
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _HeaderIsolationHandler)
        self.httpd.records = []  # type: ignore[attr-defined]
        self.httpd.daemon_threads = True
        server_address = self.httpd.server_address
        self.address: str = str(server_address[0])
        self.port: int = server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.httpd.shutdown()
        self.thread.join()

    @property
    def origin(self) -> str:
        return "http://" + self.address + ":" + str(self.port)

    def urljoin(self, url: Optional[str] = None) -> str:
        return urljoin(self.origin, url)

    def fetch_records(self, reset: bool = False) -> List[dict]:
        """Return the requests received by this server, optionally resetting."""
        url = self.urljoin("/requests?reset=1" if reset else "/requests")
        with urlopen(url) as response:  # nosec - local loopback
            payload = json.loads(response.read().decode("utf8"))
        return payload["records"]
