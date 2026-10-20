"""Baselines let an existing server adopt the linter and gate only new findings."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from mcp_contrast.model import Finding

VERSION = 1


class BaselineError(ValueError):
    pass


def load(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BaselineError(
            f"{path}: baseline not found (create it with --write-baseline)"
        ) from None
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"{path}: unreadable baseline ({exc})") from exc
    if not isinstance(data, dict) or data.get("version") != VERSION:
        raise BaselineError(f"{path}: not an mcp-contrast baseline (version {VERSION})")
    return {entry["fingerprint"] for entry in data.get("findings", [])}


def write(path: Path, findings: Iterable[Finding]) -> int:
    entries = sorted(
        (
            {
                "fingerprint": f.fingerprint,
                "rule": f.rule,
                "tools": sorted((f.a.qualified, f.b.qualified)),
            }
            for f in findings
        ),
        key=lambda e: (e["rule"], e["tools"]),
    )
    unique = list({e["fingerprint"]: e for e in entries}.values())
    path.write_text(
        json.dumps({"version": VERSION, "findings": unique}, indent=2) + "\n", encoding="utf-8"
    )
    return len(unique)
