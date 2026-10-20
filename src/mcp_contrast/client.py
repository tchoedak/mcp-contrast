"""Minimal MCP client: initialize, then page through ``tools/list``.

Supports stdio and Streamable HTTP. Uses only the standard library and never
calls ``tools/call``.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import threading
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from mcp_contrast import __version__

PROTOCOL_VERSION = "2025-06-18"
MAX_PAGES = 1000


class SourceError(RuntimeError):
    pass


def _initialize_params() -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "mcp-contrast", "version": __version__},
    }


def _list_tools(request: Any) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        result = request("tools/list", {"cursor": cursor} if cursor else {})
        page = result.get("tools")
        if not isinstance(page, list):
            raise SourceError("tools/list result has no 'tools' array")
        tools.extend(page)
        cursor = result.get("nextCursor")
        if not cursor:
            return tools
    raise SourceError(f"tools/list did not finish after {MAX_PAGES} pages")


class _StdioSession:
    def __init__(
        self,
        command: Sequence[str],
        env: Mapping[str, str] | None,
        cwd: str | None,
        timeout: float,
    ):
        self.timeout = timeout
        self._next_id = 0
        self._stderr: deque[str] = deque(maxlen=20)
        self._lines: queue.Queue[str | None] = queue.Queue()
        try:
            self.proc = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, **(env or {})},
                cwd=cwd,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise SourceError(f"could not start {command[0]!r}: {exc}") from exc
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        self._stderr_reader = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_reader.start()

    def _pump_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr.append(line.rstrip())

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._died("server closed its input") from exc

    def _died(self, reason: str, exiting: bool = True) -> SourceError:
        code = self.proc.poll()
        if exiting:
            # A server that closed its pipes is usually exiting; give it a moment so the
            # exit code and its last stderr lines make it into the message.
            try:
                code = self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                code = None
            self._stderr_reader.join(timeout=1)
        tail = "\n".join(f"    {line}" for line in self._stderr)
        status = f" (exit code {code})" if code is not None else ""
        return SourceError(f"{reason}{status}" + (f"\n  server stderr:\n{tail}" if tail else ""))

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            try:
                line = self._lines.get(timeout=self.timeout)
            except queue.Empty:
                raise self._died(
                    f"timed out after {self.timeout:g}s waiting for {method}", exiting=False
                ) from None
            if line is None:
                raise self._died(f"server exited before answering {method}")
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                # Servers that log to stdout break the protocol; skip the noise.
                continue
            if not isinstance(message, dict):
                continue
            if "method" in message and "id" in message:
                self._answer_server_request(message)
                continue
            if message.get("id") != request_id:
                continue
            return _unwrap(message, method)

    def _answer_server_request(self, message: dict[str, Any]) -> None:
        if message["method"] == "ping":
            self._send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
        else:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32601, "message": "mcp-contrast does not support this"},
                }
            )

    def close(self) -> None:
        with contextlib.suppress(OSError):
            if self.proc.stdin:
                self.proc.stdin.close()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def _unwrap(message: dict[str, Any], method: str) -> dict[str, Any]:
    if "error" in message:
        error = message["error"] or {}
        raise SourceError(f"{method} failed: {error.get('message', error)}")
    result = message.get("result")
    if not isinstance(result, dict):
        raise SourceError(f"{method} returned no result object")
    return result


def list_tools_stdio(
    command: Sequence[str],
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    if not command:
        raise SourceError("empty stdio command")
    session = _StdioSession(command, env, cwd, timeout)
    try:
        session.request("initialize", _initialize_params())
        session.notify("notifications/initialized")
        return _list_tools(session.request)
    finally:
        session.close()


class _HttpSession:
    def __init__(self, url: str, headers: Mapping[str, str] | None, timeout: float):
        self.url = url
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.session_id: str | None = None
        self.protocol: str | None = None
        self._next_id = 0

    def _post(self, message: dict[str, Any]) -> Iterator[dict[str, Any]]:
        headers = {
            **self.headers,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol:
            headers["MCP-Protocol-Version"] = self.protocol
        request = urllib.request.Request(
            self.url, data=json.dumps(message).encode(), headers=headers, method="POST"
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode(errors="replace")[:500]
            except OSError:
                # The server may reject and close before reading our request body.
                body = ""
            hint = " (check --header for auth)" if exc.code in (401, 403) else ""
            detail = f": {body.strip()}" if body.strip() else ""
            raise SourceError(f"{self.url}: HTTP {exc.code}{hint}{detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SourceError(f"{self.url}: {exc}") from exc
        with response:
            if session_id := response.headers.get("Mcp-Session-Id"):
                self.session_id = session_id
            if response.status == 202:
                return
            content_type = response.headers.get("Content-Type", "")
            if content_type.startswith("text/event-stream"):
                yield from _sse_messages(response)
            else:
                body = response.read().decode()
                if body.strip():
                    parsed = json.loads(body)
                    yield from parsed if isinstance(parsed, list) else [parsed]

    def notify(self, method: str) -> None:
        for _ in self._post({"jsonrpc": "2.0", "method": method}):
            pass

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        try:
            for reply in self._post(message):
                if isinstance(reply, dict) and reply.get("id") == request_id:
                    return _unwrap(reply, method)
        except json.JSONDecodeError as exc:
            raise SourceError(f"{self.url}: response to {method} is not JSON: {exc}") from exc
        raise SourceError(f"{self.url}: no response to {method}")

    def close(self) -> None:
        if not self.session_id:
            return
        headers = {**self.headers, "Mcp-Session-Id": self.session_id}
        request = urllib.request.Request(self.url, headers=headers, method="DELETE")
        with contextlib.suppress(Exception), urllib.request.urlopen(request, timeout=5):
            pass


def _sse_messages(response: Any) -> Iterator[dict[str, Any]]:
    data: list[str] = []
    for raw in response:
        line = raw.decode().rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            yield json.loads("\n".join(data))
            data = []
    if data:
        yield json.loads("\n".join(data))


def list_tools_http(
    url: str, headers: Mapping[str, str] | None = None, timeout: float = 30.0
) -> list[dict[str, Any]]:
    session = _HttpSession(url, headers, timeout)
    try:
        result = session.request("initialize", _initialize_params())
        session.protocol = str(result.get("protocolVersion") or PROTOCOL_VERSION)
        session.notify("notifications/initialized")
        return _list_tools(session.request)
    finally:
        session.close()
