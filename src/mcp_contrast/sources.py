"""Turn CLI inputs into servers with tool lists.

A JSON file can be any of:

* a ``tools/list`` result: ``{"tools": [...]}``
* a full JSON-RPC response: ``{"result": {"tools": [...]}}``
* a bare array of tool objects
* a client config with ``mcpServers`` (Claude, Cursor) or ``servers`` (VS Code);
  every server in it is started or connected to
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mcp_contrast.client import SourceError, list_tools_http, list_tools_stdio
from mcp_contrast.model import Server, Tool

_ENV_VAR = re.compile(r"\$\{(?:env:)?([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


@dataclass(frozen=True)
class Connect:
    timeout: float = 30.0
    headers: Mapping[str, str] | None = None


def from_file(path: Path, connect: Connect, only: str | None = None) -> list[Server]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SourceError(f"{path}: {exc.strerror or exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SourceError(f"{path}: not valid JSON ({exc})") from exc

    if isinstance(data, dict) and isinstance(data.get("mcpServers") or data.get("servers"), dict):
        return _from_client_config(path, text, data, connect, only)

    tools = _tool_array(data)
    if tools is None:
        raise SourceError(
            f"{path}: expected a tools/list result, a JSON array of tools, "
            "or a client config with mcpServers"
        )
    name = path.stem
    if name in {"tools", "mcp-tools"} and path.parent.name:
        name = path.resolve().parent.name
    if only is not None and only != name:
        return []
    return [_server(name, str(path), tools, file=path, text=text)]


def split_command(command: str) -> list[str]:
    """Split like a shell. On Windows, keep backslashes in paths such as ``C:\\tools``."""
    if os.name != "nt":
        return shlex.split(command)
    return [
        token[1:-1] if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'" else token
        for token in shlex.split(command, posix=False)
    ]


def from_stdio(command: str, connect: Connect) -> Server:
    argv = split_command(command)
    if not argv:
        raise SourceError("--stdio needs a command")
    tools = list_tools_stdio(argv, timeout=connect.timeout)
    return _server(_command_name(argv), command, tools)


def from_http(url: str, connect: Connect) -> Server:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SourceError(f"--http needs an http(s) URL, got {url!r}")
    tools = list_tools_http(url, headers=connect.headers, timeout=connect.timeout)
    return _server(parsed.hostname or url, url, tools)


def _tool_array(data: Any) -> list[Any] | None:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("tools"), list):
            return list(data["tools"])
        result = data.get("result")
        if isinstance(result, dict) and isinstance(result.get("tools"), list):
            return list(result["tools"])
    return None


def _from_client_config(
    path: Path, text: str, data: dict[str, Any], connect: Connect, only: str | None
) -> list[Server]:
    entries: dict[str, Any] = data.get("mcpServers") or data.get("servers") or {}
    if only is not None:
        if only not in entries:
            raise SourceError(f"{path}: no server named {only!r} (have: {', '.join(entries)})")
        entries = {only: entries[only]}
    servers = []
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise SourceError(f"{path}: server {name!r} is not an object")
        if entry.get("disabled") is True:
            continue
        try:
            tools = _connect_entry(entry, connect, cwd=path.parent)
        except SourceError as exc:
            raise SourceError(f"{path}: server {name!r}: {exc}") from exc
        line = _line_of(text, rf'"{re.escape(name)}"\s*:')
        servers.append(_server(name, f"{path}#{name}", tools, file=path, line=line))
    return servers


def _connect_entry(entry: dict[str, Any], connect: Connect, cwd: Path) -> list[Any]:
    kind = str(entry.get("type") or entry.get("transport") or "")
    if "url" in entry:
        if kind == "sse":
            raise SourceError("legacy SSE transport is not supported; use Streamable HTTP")
        headers = {k: expand(str(v)) for k, v in (entry.get("headers") or {}).items()}
        return list_tools_http(
            expand(str(entry["url"])),
            headers={**headers, **(connect.headers or {})},
            timeout=connect.timeout,
        )
    if "command" in entry:
        argv = [expand(str(entry["command"])), *(expand(str(a)) for a in entry.get("args", []))]
        env = {k: expand(str(v)) for k, v in (entry.get("env") or {}).items()}
        entry_cwd = entry.get("cwd")
        return list_tools_stdio(
            argv,
            env=env,
            cwd=str(cwd / expand(str(entry_cwd))) if entry_cwd else str(cwd),
            timeout=connect.timeout,
        )
    raise SourceError("needs either 'command' (stdio) or 'url' (HTTP)")


def expand(value: str, environ: Mapping[str, str] = os.environ) -> str:
    """Expand ``${VAR}``, ``${VAR:-default}``, and VS Code's ``${env:VAR}``."""

    def substitute(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in environ:
            return environ[name]
        if default is not None:
            return default
        raise SourceError(f"environment variable {name} is not set")

    return _ENV_VAR.sub(substitute, value)


def _server(
    name: str,
    origin: str,
    raw_tools: list[Any],
    file: Path | None = None,
    text: str | None = None,
    line: int | None = None,
) -> Server:
    tools = []
    for index, raw in enumerate(raw_tools):
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not raw["name"]:
            raise SourceError(f"{origin}: tool #{index + 1} has no string 'name'")
        description = raw.get("description")
        if not isinstance(description, str):
            description = ""
        tool_line = line
        if text is not None:
            tool_line = _line_of(text, rf'"name"\s*:\s*"{re.escape(raw["name"])}"')
        tools.append(Tool(raw["name"], description, name, file=file, line=tool_line))
    return Server(name, origin, tuple(tools), file=file, line=line, raw_tools=tuple(raw_tools))


def _line_of(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return text.count("\n", 0, match.start()) + 1 if match else None


def _command_name(argv: list[str]) -> str:
    """A readable server name: the package or script, not the launcher."""
    launchers = {"npx", "uvx", "pipx", "bunx", "node", "python", "python3", "uv", "deno", "docker"}
    for arg in argv:
        base = Path(arg).name
        if base in launchers or arg.startswith("-") or base in {"run", "tool", "-y"}:
            continue
        return re.sub(r"@[^/@]*$", "", base) or base
    return Path(argv[0]).name


def snapshot(server: Server) -> dict[str, Any]:
    """The server's tools exactly as listed, in tools/list shape, for committing to a repo."""
    return {"tools": list(server.raw_tools)}
