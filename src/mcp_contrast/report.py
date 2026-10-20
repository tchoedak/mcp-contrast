"""Reporters: human text, JSON, SARIF 2.1.0, and GitHub workflow annotations."""

from __future__ import annotations

import json
import os
import shutil
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp_contrast import __version__
from mcp_contrast.model import Finding, Report, Severity, Tool
from mcp_contrast.rules import RULES

REPO = "https://github.com/tchoedak/mcp-contrast"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _where(tool: Tool) -> str | None:
    if tool.file is None:
        return None
    return _relative(tool.file) + (f":{tool.line}" if tool.line else "")


# --- text ---------------------------------------------------------------------------------


class _Style:
    def __init__(self, color: bool):
        self.color = color

    def __call__(self, text: str, *codes: str) -> str:
        if not self.color or not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"


_RULE_WIDTH = max(len(rule) for rule in RULES) + 1
_SEVERITY_STYLE = {
    Severity.ERROR: ("1", "31"),
    Severity.WARNING: ("1", "33"),
    Severity.INFO: ("1", "36"),
}


def text(report: Report, *, color: bool = False, show_clean: bool = False) -> str:
    s = _Style(color)
    width = max(60, min(100, shutil.get_terminal_size((100, 24)).columns))
    lines: list[str] = []
    by_server: dict[str, list[Finding]] = {}
    for finding in report.findings:
        by_server.setdefault(finding.a.server, []).append(finding)

    indent = " " * 11
    for server in report.servers:
        findings = by_server.get(server.name, [])
        if not findings:
            continue
        count = _plural(len(server.tools), "tool")
        lines.append(f"{s(server.name, '1', '4')} {s(f'({count})', '2')}")
        for f in findings:
            pair = f"{_label(f.a, server.name)} {s('<->', '2')} {_label(f.b, server.name)}"
            where = _where(f.a)
            lines.append(
                f"  {s(f'{f.severity.value:<8}', *_SEVERITY_STYLE[f.severity])} "
                f"{s(f'{f.rule:<{_RULE_WIDTH}}', '2')} {s(pair, '1')}"
                + (f"  {s(where, '2')}" if where else "")
            )
            for chunk in textwrap.wrap(f.message, width - len(indent)):
                lines.append(indent + chunk)
            for i, chunk in enumerate(textwrap.wrap(f.fix, width - len(indent) - 5)):
                prefix = s("fix: ", "36") if i == 0 else "     "
                lines.append(indent + prefix + chunk)
            lines.append("")

    if show_clean and report.clean:
        lines.append(
            s("clean contrasts", "1", "4") + s(" (similar, but the difference is stated)", "2")
        )
        single = len(report.servers) == 1
        for c in report.clean:
            a, b = (c.a.name, c.b.name) if single else (c.a.qualified, c.b.qualified)
            detail = s(f"similarity {c.similarity:.2f}, {c.reason}", "2")
            lines.append(f"  {s('ok', '1', '32')}  {a} {s('<->', '2')} {b}  {detail}")
        lines.append("")

    errors, warnings, infos = (report.count(sev) for sev in Severity)
    parts = [
        f"{_plural(report.tool_count, 'tool')} in {_plural(len(report.servers), 'server')}",
        _plural(len(report.clean), "clean contrast"),
    ]
    if report.suppressed:
        parts.append(f"{report.suppressed} suppressed")
    if report.findings:
        counts = [
            _plural(n, label)
            for n, label in ((errors, "error"), (warnings, "warning"), (infos, "note"))
            if n
        ]
        head = s(
            f"\u2716 {_plural(len(report.findings), 'problem')} ({', '.join(counts)})",
            "1",
            "31" if errors else "33",
        )
    else:
        head = s("\u2714 No contrast problems", "1", "32")
    lines.append(head + s(" - " + ", ".join(parts), "2"))
    return "\n".join(lines) + "\n"


def _label(tool: Tool, server: str) -> str:
    return tool.name if tool.server == server else tool.qualified


# --- json ---------------------------------------------------------------------------------


def _tool_json(tool: Tool) -> dict[str, Any]:
    out: dict[str, Any] = {"server": tool.server, "name": tool.name}
    if tool.file is not None:
        out["file"] = _relative(tool.file)
    if tool.line is not None:
        out["line"] = tool.line
    return out


def as_json(report: Report) -> str:
    payload = {
        "version": __version__,
        "summary": {
            "servers": len(report.servers),
            "tools": report.tool_count,
            "errors": report.count(Severity.ERROR),
            "warnings": report.count(Severity.WARNING),
            "infos": report.count(Severity.INFO),
            "clean": len(report.clean),
            "suppressed": report.suppressed,
        },
        "findings": [
            {
                "rule": f.rule,
                "severity": f.severity.value,
                "fingerprint": f.fingerprint,
                "tools": [_tool_json(f.a), _tool_json(f.b)],
                "similarity": None if f.similarity is None else round(f.similarity, 4),
                "terms": None if f.slot is None else [list(f.slot.a_terms), list(f.slot.b_terms)],
                "message": f.message,
                "fix": f.fix,
            }
            for f in report.findings
        ],
        "clean": [
            {
                "tools": [_tool_json(c.a), _tool_json(c.b)],
                "similarity": round(c.similarity, 4),
                "reason": c.reason,
            }
            for c in report.clean
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


# --- sarif --------------------------------------------------------------------------------

_SARIF_LEVEL = {Severity.ERROR: "error", Severity.WARNING: "warning", Severity.INFO: "note"}


def _sarif_location(tool: Tool) -> dict[str, Any] | None:
    if tool.file is None:
        return None
    physical: dict[str, Any] = {"artifactLocation": {"uri": _relative(tool.file)}}
    if tool.line is not None:
        physical["region"] = {"startLine": tool.line}
    return {
        "physicalLocation": physical,
        "logicalLocations": [{"name": tool.name, "fullyQualifiedName": tool.qualified}],
    }


def sarif(report: Report) -> str:
    results = []
    for f in report.findings:
        result: dict[str, Any] = {
            "ruleId": f.rule,
            "level": _SARIF_LEVEL[f.severity],
            "message": {"text": f"{f.message} Fix: {f.fix}"},
            "partialFingerprints": {"mcpContrast/v1": f.fingerprint},
        }
        primary = _sarif_location(f.a)
        if primary is not None:
            result["locations"] = [primary]
        related = _sarif_location(f.b)
        if related is not None:
            result["relatedLocations"] = [{"id": 1, **related}]
        if f.similarity is not None:
            result["properties"] = {"similarity": round(f.similarity, 4)}
        results.append(result)

    document = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "mcp-contrast",
                        "version": __version__,
                        "informationUri": REPO,
                        "rules": [
                            {
                                "id": rule.id,
                                "name": "".join(p.title() for p in rule.id.split("-")),
                                "shortDescription": {"text": rule.summary},
                                "fullDescription": {"text": f"{rule.summary} {rule.rationale}"},
                                "helpUri": rule.help_uri,
                                "defaultConfiguration": {"level": _SARIF_LEVEL[rule.severity]},
                            }
                            for rule in RULES.values()
                        ],
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(document, indent=2) + "\n"


# --- github -------------------------------------------------------------------------------


def _escape_data(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(value: str) -> str:
    return _escape_data(value).replace(":", "%3A").replace(",", "%2C")


def github(report: Report) -> str:
    """GitHub Actions workflow commands, which render as inline PR annotations."""
    command = {Severity.ERROR: "error", Severity.WARNING: "warning", Severity.INFO: "notice"}
    lines = []
    for f in report.findings:
        props = {"title": f"mcp-contrast {f.rule}"}
        if f.a.file is not None:
            props["file"] = _relative(f.a.file)
            if f.a.line is not None:
                props["line"] = str(f.a.line)
        prop_text = ",".join(f"{k}={_escape_property(v)}" for k, v in props.items())
        lines.append(
            f"::{command[f.severity]} {prop_text}::{_escape_data(f.message + ' Fix: ' + f.fix)}"
        )
    lines.append(text(report, color=False).rstrip("\n").splitlines()[-1])
    return "\n".join(lines) + "\n"


FORMATS: dict[str, Callable[[Report], str]] = {
    "json": as_json,
    "sarif": sarif,
    "github": github,
}


def use_color(stream_isatty: bool) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return stream_isatty and os.environ.get("TERM") != "dumb"
