from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mcp_contrast.config import Settings
from mcp_contrast.embed import LexicalEmbedder
from mcp_contrast.model import Server, Tool

HERE = Path(__file__).parent
STDIO_SERVER = [sys.executable, str(HERE / "servers" / "stdio_server.py")]


def server(name: str, *tools: tuple[str, str]) -> Server:
    return Server(name, name, tuple(Tool(n, d, name) for n, d in tools))


@pytest.fixture
def lexical() -> LexicalEmbedder:
    return LexicalEmbedder()


@pytest.fixture
def settings() -> Settings:
    return Settings(embedder="lexical")


@pytest.fixture
def write_json(tmp_path: Path):
    def write(name: str, data: Any) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(data, indent=2))
        return path

    return write


TOOLS = [
    {"name": "get_user", "description": "Get a user by id."},
    {"name": "fetch_user", "description": "Fetch a user by id."},
]


class _Handler(BaseHTTPRequestHandler):
    mode = "json"
    require_auth: str | None = None
    seen_headers: ClassVar[list[dict[str, str]]] = []

    def log_message(self, *args: Any) -> None:
        pass

    def do_DELETE(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_POST(self) -> None:
        type(self).seen_headers.append(dict(self.headers.items()))
        if self.require_auth and self.headers.get("Authorization") != self.require_auth:
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"unauthorized")
            return
        message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "id" not in message:
            self.send_response(202)
            self.end_headers()
            return
        if message["method"] == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "http-test", "version": "0"},
            }
        else:
            assert self.headers.get("Mcp-Session-Id") == "session-123"
            assert self.headers.get("MCP-Protocol-Version") == "2025-06-18"
            cursor = message.get("params", {}).get("cursor")
            result = {"tools": TOOLS[1:]} if cursor else {"tools": TOOLS[:1], "nextCursor": "p2"}
        body = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        self.send_response(200)
        self.send_header("Mcp-Session-Id", "session-123")
        if self.mode == "sse":
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": keepalive\n\n")
            self.wfile.write(b'data: {"jsonrpc":"2.0","method":"notifications/message"}\n\n')
            self.wfile.write(f"event: message\ndata: {json.dumps(body)}\n\n".encode())
        else:
            payload = json.dumps(body).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


@pytest.fixture(params=["json", "sse"])
def http_server(request: pytest.FixtureRequest) -> Iterator[str]:
    handler = type("Handler", (_Handler,), {"mode": request.param, "seen_headers": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/mcp"
    httpd.shutdown()


@pytest.fixture
def auth_http_server() -> Iterator[tuple[str, type[_Handler]]]:
    handler = type("Handler", (_Handler,), {"require_auth": "Bearer s3cret", "seen_headers": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/mcp", handler
    httpd.shutdown()
