import json

import pytest

from mcp_contrast import __version__
from mcp_contrast.cli import main
from tests.conftest import STDIO_SERVER

SYNONYMS = {
    "tools": [
        {"name": "get_user", "description": "Get a user."},
        {"name": "fetch_user", "description": "Fetch a user."},
    ]
}
OVERLAP = {
    "tools": [
        {"name": "search_notes", "description": "Search notes by keyword across all records."},
        {"name": "find_text", "description": "Search notes by keyword across all records."},
    ]
}
CLEAN = {
    "tools": [
        {"name": "send_email", "description": "Send an email to recipients."},
        {"name": "list_events", "description": "List calendar events in a range."},
    ]
}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FORCE_COLOR", raising=False)


def run(*args: str) -> int:
    return main(["--embedder", "lexical", *args] if args and args[0] != "rules" else list(args))


def test_exit_codes(write_json, capsys) -> None:
    assert main(["check", "--embedder", "lexical", str(write_json("ok.json", CLEAN))]) == 0
    assert "No contrast problems" in capsys.readouterr().out
    assert run(str(write_json("bad.json", SYNONYMS))) == 1
    assert run(str(write_json("warn.json", OVERLAP))) == 0
    assert run(str(write_json("warn.json", OVERLAP)), "--fail-on", "warning") == 1
    assert run(str(write_json("bad.json", SYNONYMS)), "--fail-on", "never") == 0


def test_text_output(write_json, capsys) -> None:
    run(str(write_json("bad.json", SYNONYMS)))
    out = capsys.readouterr().out
    assert "synonym-contrast" in out
    assert "get_user <-> fetch_user" in out
    assert "bad.json:4" in out
    assert "\033[" not in out
    assert out.rstrip().splitlines()[-1] == (
        "\u2716 1 problem (1 error) - 2 tools in 1 server, 0 clean contrasts"
    )


def test_json_output(write_json, capsys) -> None:
    run(str(write_json("bad.json", SYNONYMS)), "--format", "json")
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["errors"] == 1
    finding = data["findings"][0]
    assert finding["rule"] == "synonym-contrast"
    assert finding["terms"] == [["get"], ["fetch"]]
    assert finding["tools"][0] == {
        "server": "bad",
        "name": "get_user",
        "file": "bad.json",
        "line": 4,
    }


def test_sarif_output(write_json, tmp_path) -> None:
    out = tmp_path / "out.sarif"
    run(str(write_json("bad.json", SYNONYMS)), "--format", "sarif", "-o", str(out))
    doc = json.loads(out.read_text())
    assert doc["version"] == "2.1.0"
    run_ = doc["runs"][0]
    assert run_["tool"]["driver"]["version"] == __version__
    assert {r["id"] for r in run_["tool"]["driver"]["rules"]} == {
        "name-collision",
        "synonym-contrast",
        "unmarked-contrast",
        "hidden-overlap",
    }
    [result] = run_["results"]
    assert result["ruleId"] == "synonym-contrast"
    assert result["level"] == "error"
    location = result["locations"][0]["physicalLocation"]
    assert location == {"artifactLocation": {"uri": "bad.json"}, "region": {"startLine": 4}}
    assert result["relatedLocations"][0]["physicalLocation"]["region"] == {"startLine": 8}


def test_sarif_alongside_text(write_json, tmp_path, capsys) -> None:
    out = tmp_path / "side.sarif"
    run(str(write_json("bad.json", SYNONYMS)), "--sarif", str(out))
    assert "synonym-contrast" in capsys.readouterr().out
    assert json.loads(out.read_text())["runs"][0]["results"][0]["ruleId"] == "synonym-contrast"


def test_github_output(write_json, capsys) -> None:
    run(str(write_json("bad.json", SYNONYMS)), "--format", "github")
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("::error title=mcp-contrast synonym-contrast,file=bad.json,line=4::")


def test_baseline_round_trip(write_json, tmp_path, capsys) -> None:
    source = str(write_json("bad.json", SYNONYMS))
    base = tmp_path / "baseline.json"
    assert run(source, "--baseline", str(base), "--write-baseline") == 0
    assert json.loads(base.read_text())["findings"][0]["rule"] == "synonym-contrast"
    capsys.readouterr()
    assert run(source, "--baseline", str(base)) == 0
    assert "1 suppressed" in capsys.readouterr().out

    grown = {"tools": [*SYNONYMS["tools"], {"name": "retrieve_user", "description": "Retrieve."}]}
    assert run(str(write_json("bad.json", grown)), "--baseline", str(base)) == 1


def test_config_file_is_discovered(write_json, tmp_path) -> None:
    (tmp_path / "mcp-contrast.toml").write_text(
        'embedder = "lexical"\n[severity]\nsynonym-contrast = "warning"\n'
    )
    assert main(["check", str(write_json("bad.json", SYNONYMS))]) == 0


def test_pyproject_config(write_json, tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mcp-contrast]\nembedder = "lexical"\nignore = [{ tools = ["get_*", "fetch_*"] }]\n'
    )
    assert main(["check", str(write_json("bad.json", SYNONYMS))]) == 0


@pytest.mark.parametrize(
    ("toml", "error"),
    [
        ("bogus = 1\n", "unknown setting(s): bogus"),
        ("threshold = 2\n", "threshold must be a number in (0, 1]"),
        ('[severity]\nnope = "error"\n', "unknown rule 'nope'"),
        ('ignore = [{ tools = ["a"] }]\n', "ignore.tools must be a pair"),
        ("threshold = \n", "mcp-contrast.toml"),
    ],
)
def test_bad_config(write_json, tmp_path, capsys, toml, error) -> None:
    (tmp_path / "mcp-contrast.toml").write_text(toml)
    assert main(["check", str(write_json("ok.json", CLEAN))]) == 2
    assert error in capsys.readouterr().err


def test_input_errors_exit_2(tmp_path, capsys) -> None:
    assert run(str(tmp_path / "missing.json")) == 2
    assert "mcp-contrast: error:" in capsys.readouterr().err
    assert main(["check"]) == 2
    assert "no sources given" in capsys.readouterr().err
    assert run("--write-baseline", str(tmp_path)) == 2
    assert run("x.json", "--threshold", "1.5") == 2


def test_live_stdio_check(capsys) -> None:
    assert run("--stdio", " ".join(STDIO_SERVER)) == 1
    assert "get_user <-> fetch_user" in capsys.readouterr().out


def test_snapshot_then_check(tmp_path, capsys) -> None:
    out = tmp_path / "tools.json"
    assert main(["snapshot", "--stdio", " ".join(STDIO_SERVER), "-o", str(out)]) == 0
    assert "saved 3 tool(s)" in capsys.readouterr().err
    assert len(json.loads(out.read_text())["tools"]) == 3
    assert run(str(out)) == 1


def test_snapshot_needs_one_server(write_json, capsys) -> None:
    a, b = write_json("a.json", CLEAN), write_json("b.json", CLEAN)
    assert main(["snapshot", str(a), str(b)]) == 2
    assert "exactly one server" in capsys.readouterr().err


def test_duplicate_server_names_are_numbered(tmp_path, capsys) -> None:
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "s.json").write_text(json.dumps(SYNONYMS))
    run(str(tmp_path / "one" / "s.json"), str(tmp_path / "two" / "s.json"), "--format", "json")
    servers = {f["tools"][0]["server"] for f in json.loads(capsys.readouterr().out)["findings"]}
    assert servers == {"s", "s-2"}


def test_rules_command(capsys) -> None:
    assert main(["rules"]) == 0
    out = capsys.readouterr().out
    for rule in ("name-collision", "synonym-contrast", "unmarked-contrast", "hidden-overlap"):
        assert rule in out


def test_version(capsys) -> None:
    with pytest.raises(SystemExit) as exit_:
        main(["--version"])
    assert exit_.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys) -> None:
    assert main([]) == 2
    assert "usage: mcp-contrast" in capsys.readouterr().out
