import json
import sys

import pytest

from mcp_contrast.client import SourceError, list_tools_http, list_tools_stdio
from mcp_contrast.sources import Connect, expand, from_file, from_http, from_stdio, snapshot
from tests.conftest import STDIO_SERVER

CONNECT = Connect(timeout=10)


@pytest.mark.parametrize(
    "payload",
    [
        {"tools": [{"name": "a", "description": "A."}]},
        {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "a", "description": "A."}]}},
        [{"name": "a", "description": "A."}],
    ],
)
def test_file_shapes(write_json, payload) -> None:
    [server] = from_file(write_json("crm.json", payload), CONNECT)
    assert server.name == "crm"
    assert [(t.name, t.description) for t in server.tools] == [("a", "A.")]


def test_tool_lines_point_into_the_file(write_json) -> None:
    path = write_json("s.json", {"tools": [{"name": "first"}, {"name": "second"}]})
    [server] = from_file(path, CONNECT)
    lines = path.read_text().splitlines()
    for tool in server.tools:
        assert tool.file == path
        assert f'"{tool.name}"' in lines[tool.line - 1]


def test_generic_file_name_uses_directory(tmp_path) -> None:
    directory = tmp_path / "billing-server"
    directory.mkdir()
    (directory / "tools.json").write_text('{"tools": []}')
    [server] = from_file(directory / "tools.json", CONNECT)
    assert server.name == "billing-server"


def test_missing_description_becomes_empty(write_json) -> None:
    [server] = from_file(write_json("s.json", [{"name": "a", "description": None}]), CONNECT)
    assert server.tools[0].description == ""


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("not json", "not valid JSON"),
        ('{"name": "x"}', "expected a tools/list result"),
        ('[{"description": "no name"}]', "tool #1 has no string 'name'"),
    ],
)
def test_bad_files(tmp_path, content, error) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(SourceError, match=error):
        from_file(path, CONNECT)


def test_missing_file(tmp_path) -> None:
    with pytest.raises(SourceError, match="No such file"):
        from_file(tmp_path / "nope.json", CONNECT)


def test_stdio_with_pagination_noise_and_ping() -> None:
    tools = list_tools_stdio([*STDIO_SERVER, "--page-size", "1", "--noisy", "--ping"], timeout=10)
    assert [t["name"] for t in tools] == ["get_user", "fetch_user", "delete_user"]


def test_stdio_crash_reports_stderr() -> None:
    with pytest.raises(
        SourceError, match=r"exited before answering initialize \(exit code 3\)"
    ) as e:
        list_tools_stdio([*STDIO_SERVER, "--crash"], timeout=10)
    assert "boom: missing API key" in str(e.value)


def test_stdio_timeout() -> None:
    with pytest.raises(SourceError, match=r"timed out after 0\.5s waiting for tools/list"):
        list_tools_stdio([*STDIO_SERVER, "--hang"], timeout=0.5)


def test_stdio_jsonrpc_error() -> None:
    with pytest.raises(SourceError, match="tools/list failed: listing disabled"):
        list_tools_stdio([*STDIO_SERVER, "--error"], timeout=10)


def test_stdio_missing_binary() -> None:
    with pytest.raises(SourceError, match="could not start"):
        list_tools_stdio(["definitely-not-a-real-binary-xyz"], timeout=1)


def test_from_stdio_names_server_after_script() -> None:
    command = " ".join([sys.executable, STDIO_SERVER[1]])
    server = from_stdio(command, CONNECT)
    assert server.name == "stdio_server.py"
    assert len(server.tools) == 3


def test_http_json_and_sse(http_server) -> None:
    tools = list_tools_http(http_server, timeout=10)
    assert [t["name"] for t in tools] == ["get_user", "fetch_user"]
    server = from_http(http_server, CONNECT)
    assert server.name == "127.0.0.1"


def test_http_auth_header(auth_http_server) -> None:
    url, _ = auth_http_server
    with pytest.raises(SourceError, match=r"HTTP 401 \(check --header for auth\)"):
        list_tools_http(url, timeout=10)
    tools = list_tools_http(url, headers={"Authorization": "Bearer s3cret"}, timeout=10)
    assert len(tools) == 2


def test_http_rejects_non_http_url() -> None:
    with pytest.raises(SourceError, match="needs an http"):
        from_http("ftp://example.com", CONNECT)


def test_http_unreachable() -> None:
    with pytest.raises(SourceError, match=r"127\.0\.0\.1:9"):
        list_tools_http("http://127.0.0.1:9/mcp", timeout=2)


def test_client_config_starts_every_server(tmp_path, http_server, monkeypatch) -> None:
    monkeypatch.setenv("TEST_TOKEN", "s3cret")
    config = {
        "mcpServers": {
            "users": {"command": STDIO_SERVER[0], "args": [STDIO_SERVER[1]]},
            "remote": {"type": "http", "url": http_server, "headers": {"X-Token": "${TEST_TOKEN}"}},
            "off": {"command": "nope", "disabled": True},
        }
    }
    path = tmp_path / ".mcp.json"
    path.write_text(json.dumps(config, indent=2))
    servers = from_file(path, CONNECT)
    assert [s.name for s in servers] == ["users", "remote"]
    assert [len(s.tools) for s in servers] == [3, 2]
    assert servers[0].tools[0].file == path
    assert '"users"' in path.read_text().splitlines()[servers[0].tools[0].line - 1]

    [only] = from_file(path, CONNECT, only="remote")
    assert only.name == "remote"
    with pytest.raises(SourceError, match="no server named 'x'"):
        from_file(path, CONNECT, only="x")


def test_vscode_config_and_legacy_sse(tmp_path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"old": {"type": "sse", "url": "http://x/sse"}}}))
    with pytest.raises(SourceError, match="server 'old': legacy SSE transport is not supported"):
        from_file(path, CONNECT)


def test_expand() -> None:
    env = {"A": "1"}
    assert expand("${A}-${env:A}-${B:-two}", env) == "1-1-two"
    with pytest.raises(SourceError, match="B is not set"):
        expand("${B}", env)


def test_snapshot_keeps_full_tool_objects() -> None:
    server = from_stdio(" ".join(STDIO_SERVER), CONNECT)
    data = snapshot(server)
    assert data["tools"][0] == {
        "name": "get_user",
        "description": "Get a user by id.",
        "inputSchema": {"type": "object"},
    }
