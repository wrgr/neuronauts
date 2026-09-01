"""Serve a mesh bundle to Neuroglancer over HTTP with CORS.

Neuroglancer runs in the browser and fetches precomputed sources itself, so a
bundle on disk has to be reachable over HTTP and the server has to send
``Access-Control-Allow-Origin`` or the viewer silently shows nothing. Python's
``http.server`` does the file serving; this module adds the headers, an
``OPTIONS`` handler, and a JSON content type for the extension-less ``info``
files. Browsers treat ``http://localhost`` as a secure context, so the hosted
viewer at ``https://neuroglancer-demo.appspot.com`` can load from it.

Only binds to loopback unless told otherwise: the bundle is your data.
"""

from __future__ import annotations

import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class CorsHandler(SimpleHTTPRequestHandler):
    quiet = False

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "x-requested-with, range, content-type")
        self.send_header("Access-Control-Expose-Headers", "content-range, content-length")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 (http.server naming)
        self.send_response(204)
        self.end_headers()

    def guess_type(self, path):
        name = Path(str(path)).name
        if name == "info" or name.endswith(".json"):
            return "application/json"
        if name.endswith(".mesh") or ":" in name or name.isdigit():
            return "application/octet-stream"
        return super().guess_type(path)

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)


def make_server(directory: str | Path, *, host: str = DEFAULT_HOST,
                port: int = DEFAULT_PORT, quiet: bool = False) -> ThreadingHTTPServer:
    """Bind a CORS file server on ``directory``. ``port=0`` picks a free port."""
    directory = str(Path(directory).resolve())
    handler = partial(CorsHandler, directory=directory)
    handler_cls = type("BoundCorsHandler", (CorsHandler,), {"quiet": quiet})
    handler = partial(handler_cls, directory=directory)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def server_url(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


def serve_in_thread(directory: str | Path, *, host: str = DEFAULT_HOST, port: int = 0,
                    quiet: bool = True) -> tuple[ThreadingHTTPServer, str]:
    """Start serving in a daemon thread; returns ``(server, base_url)``."""
    server = make_server(directory, host=host, port=port, quiet=quiet)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, server_url(server)


def serve_forever(directory: str | Path, *, host: str = DEFAULT_HOST,
                  port: int = DEFAULT_PORT, quiet: bool = False) -> None:
    server = make_server(directory, host=host, port=port, quiet=quiet)
    url = server_url(server)
    print(f"serving {Path(directory).resolve()} at {url}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def port_is_free(port: int, host: str = DEFAULT_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def launch_local_viewer(state: dict):
    """Open the state in a viewer served by the ``neuroglancer`` Python package
    (no hosted viewer, no mixed-content concerns). Returns the viewer, whose
    ``str()`` is its URL. Raises ImportError if the package is missing."""
    import neuroglancer  # noqa: WPS433 (optional dependency)

    viewer = neuroglancer.Viewer()
    viewer.set_state(neuroglancer.ViewerState(state))
    return viewer


__all__ = ["CorsHandler", "DEFAULT_HOST", "DEFAULT_PORT", "launch_local_viewer",
           "make_server", "port_is_free", "serve_forever", "serve_in_thread", "server_url"]
