"""The contour lab's window: a page in the browser, served from this computer.

    python -m actis.contourlab [images ...]

starts a small server on 127.0.0.1, opens the page, and keeps running until
Ctrl+C. The server holds the images and runs the workflows (session.py); the
page (static/) shows them. Only this computer can reach it, and only with the
key in the address the page is opened with.
"""

import argparse
import json
import secrets
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .session import MAX_UPLOAD_BYTES, LabError, Session, browse, meta, parse_steps

STATIC = Path(__file__).with_name("static")
PORT = 8765
# The page's files and what they are. Not asked of the operating system: on some
# Windows PCs the registry calls .js plain text, and a browser will not run that.
KINDS = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
}


class Lab(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, session=None, key=None):
        super().__init__(address, Handler)
        self.session = session or Session()
        self.key = key or secrets.token_urlsafe(18)

    @property
    def url(self):
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/?key={self.key}"


class Handler(BaseHTTPRequestHandler):
    server_version = "ContourLab"
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # quiet: the page shows what matters
        pass

    # --- replies ---

    def send(self, body, kind="application/json", status=HTTPStatus.OK, headers=()):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def json(self, data, status=HTTPStatus.OK):
        self.send(json.dumps(data, allow_nan=False).encode("utf-8"), status=status)

    def fail(self, status, message):
        self.json({"error": message}, status)

    # --- who may ask ---

    def allowed(self, query):
        """Only this computer's own page: the address must be the server's, and
        the key must be given (in the page's address, or as a header)."""
        host, port = self.server.server_address[:2]
        if self.headers.get("Host", "") not in (f"{host}:{port}", f"localhost:{port}"):
            return False
        given = self.headers.get("X-Lab-Key") or (query.get("key") or [""])[0]
        return secrets.compare_digest(given, self.server.key)

    # --- requests ---

    def do_GET(self):
        self.answer()

    def do_POST(self):
        self.answer()

    def answer(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        parts = [unquote(part) for part in url.path.split("/") if part]
        try:
            if parts == ["favicon.ico"]:  # the browser asks by itself, without the key
                return self.send(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
            if not self.allowed(query):
                self.close_connection = True  # what it sent is not read
                return self.fail(
                    HTTPStatus.FORBIDDEN, "open the address the lab printed"
                )
            # Whatever was sent is read before anything is answered: the connection is
            # kept for the page's next request, which would otherwise begin with it.
            upload = parts == ["api", "upload"]
            self.data = self.received(MAX_UPLOAD_BYTES if upload else 2**24)
            if not parts and self.command == "GET":
                return self.page()
            if parts[0] == "static" and len(parts) == 2 and self.command == "GET":
                return self.static(parts[1])
            if parts[0] == "api":
                return self.api(parts[1:], query)
            self.fail(HTTPStatus.NOT_FOUND, "no such page")
        except LabError as error:
            self.fail(HTTPStatus.BAD_REQUEST, str(error))
        except (BrokenPipeError, ConnectionError):
            pass  # the page went away
        except Exception as error:  # noqa: BLE001
            # A bug: say so on the page rather than leave it waiting for an answer.
            self.fail(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(error).__name__}: {error}"
            )

    def page(self):
        text = (STATIC / "index.html").read_text(encoding="utf-8")
        self.send(
            text.replace("{{key}}", self.server.key).encode("utf-8"),
            "text/html; charset=utf-8",
        )

    def static(self, name):
        path = STATIC / name
        if path.parent != STATIC or not path.is_file():
            return self.fail(HTTPStatus.NOT_FOUND, "no such file")
        kind = KINDS.get(path.suffix.lower())
        if kind is None:
            return self.fail(HTTPStatus.NOT_FOUND, "no such file")
        self.send(path.read_bytes(), kind)

    def received(self, limit):
        """The bytes a request sent (b"" if none)."""
        length = int(self.headers.get("Content-Length") or 0)
        if length > limit:
            self.close_connection = True
            raise LabError("too much was sent")
        return self.rfile.read(length) if length > 0 else b""

    def sent(self):
        """The JSON object a POST sent ({} if none)."""
        data = self.data
        try:
            value = json.loads(data) if data else {}
        except ValueError:
            raise LabError("what was sent is not JSON") from None
        if not isinstance(value, dict):
            raise LabError("what was sent is not a JSON object")
        return value

    def api(self, parts, query):
        session = self.server.session
        get, post = self.command == "GET", self.command == "POST"

        def one(name, default=None):
            return (query.get(name) or [default])[0]

        if parts == ["meta"] and get:
            return self.json(meta())
        if parts == ["session"] and get:
            return self.json(session.describe())
        if parts == ["browse"] and get:
            return self.json(browse(one("folder")))
        if parts == ["open"] and post:
            opened, failed = [], []
            for path in self.sent().get("paths", []):
                try:
                    opened.append(session.open(path).describe())
                except LabError as error:
                    failed.append(str(error))
            return self.json({"tabs": opened, "errors": failed})
        if parts == ["upload"] and post:
            tab = session.upload(one("name", "image.png"), self.data)
            return self.json({"tabs": [tab.describe()], "errors": []})
        if len(parts) >= 2 and parts[0] == "tabs":
            tab, what = session.get(parts[1]), parts[2:]
            if what == ["close"] and post:
                session.close(tab.id)
                return self.json({})
            if what == ["duplicate"] and post:
                return self.json(session.duplicate(tab.id).describe())
            if what == ["view"] and post:
                try:
                    tab.set_view(**self.sent())
                except (TypeError, ValueError) as error:
                    raise LabError(str(error)) from error
                return self.json(tab.describe())
            if what == ["run"] and post:
                sent = self.sent()
                steps = parse_steps(sent.get("steps", []))
                tab.request(int(sent.get("request", 0)), steps, sent.get("region"))
                return self.json({"serial": tab.serial})
            if what == ["result"] and get:
                timeout = min(float(one("timeout", 25)), 55.0)
                return self.json(tab.wait(int(one("after", -1)), timeout))
            if len(what) == 5 and what[0] == "tile" and get:
                layer = tab.layer(what[1])
                data, kind, width, height = layer.tile(*(int(v) for v in what[2:]))
                headers = (("X-Tile", f"{kind} {width} {height}"),)
                return self.send(data, "application/octet-stream", headers=headers)
            if len(what) == 2 and what[0] == "contours" and get:
                return self.send(
                    tab.contour_set(what[1]).payload(), "application/octet-stream"
                )
            if len(what) == 3 and what[0] == "contours" and get:
                return self.json(tab.contour_set(what[1]).detail(int(what[2])))
            if len(what) == 3 and what[0] == "export" and get:
                pixel_nm = float(one("pixel_nm", 0)) or None
                data, name, kind = tab.export(what[1], what[2], pixel_nm)
                headers = (("Content-Disposition", f'attachment; filename="{name}"'),)
                return self.send(data, kind, headers=headers)
        self.fail(HTTPStatus.NOT_FOUND, "the lab has no such request")


def start(port=PORT, session=None, tries=20):
    """A lab server listening on 127.0.0.1, on `port` or the next free one
    (port 0: any free one); call serve_forever() on it."""
    last = None
    for number in [0] if port == 0 else range(port, port + tries):
        try:
            return Lab(("127.0.0.1", number), session)
        except OSError as error:
            last = error
    raise OSError(f"no free port from {port} to {port + tries - 1}: {last}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="actis-contours",
        description="Design a workflow that finds the contours in ACTIS images, in "
        "a window in the browser.",
    )
    parser.add_argument(
        "images", nargs="*", type=Path, help="PNG or TIFF images to open"
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"port (default {PORT}; 0: any free one)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open the page: only print its address",
    )
    args = parser.parse_args(argv)
    try:
        lab = start(args.port)
    except OSError as error:
        print(f"Contour lab: {error}", file=sys.stderr)
        return 1
    for path in args.images:
        try:
            lab.session.open(path)
        except LabError as error:
            print(f"Contour lab: {error}", file=sys.stderr)
    print(f"Contour lab: {lab.url}", flush=True)
    print("Leave this window open while you work; Ctrl+C ends the lab.", flush=True)
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, (lab.url,)).start()
    try:
        lab.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        lab.session.shutdown()
        lab.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
