"""Core data types shared by sources, analysis, and reporters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @property
    def rank(self) -> int:
        return {"error": 3, "warning": 2, "info": 1}[self.value]


@dataclass(frozen=True)
class Server:
    """One MCP server's tool list and where it came from."""

    name: str
    origin: str
    tools: tuple[Tool, ...]
    # File that a reporter can point at (a snapshot or the client config), if any.
    file: Path | None = None
    line: int | None = None
    # Tool objects exactly as the server listed them, for snapshots.
    raw_tools: tuple[Any, ...] = field(default=(), repr=False, compare=False)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    server: str
    file: Path | None = None
    line: int | None = None

    @property
    def qualified(self) -> str:
        return f"{self.server}/{self.name}"


@dataclass(frozen=True)
class Slot:
    """The single position where two aligned tool names differ.

    ``a_terms`` and ``b_terms`` are the differing name tokens. One side is empty
    when a name is the other plus an inserted word (``get_records`` vs
    ``get_all_records``).
    """

    a_terms: tuple[str, ...]
    b_terms: tuple[str, ...]

    @property
    def is_insertion(self) -> bool:
        return not self.a_terms or not self.b_terms


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: Severity
    a: Tool
    b: Tool
    message: str
    fix: str
    similarity: float | None = None
    slot: Slot | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Stable id for baselines: rule plus the unordered pair of qualified names."""
        first, second = sorted((self.a.qualified, self.b.qualified))
        digest = hashlib.sha256(f"{self.rule}|{first}|{second}".encode()).hexdigest()
        return digest[:16]


@dataclass(frozen=True)
class CleanContrast:
    """A similar-looking pair whose descriptions mark the difference. Not a problem."""

    a: Tool
    b: Tool
    similarity: float
    reason: str


@dataclass(frozen=True)
class Report:
    servers: tuple[Server, ...]
    findings: tuple[Finding, ...]
    clean: tuple[CleanContrast, ...]
    suppressed: int = 0

    @property
    def tool_count(self) -> int:
        return sum(len(s.tools) for s in self.servers)

    def count(self, severity: Severity) -> int:
        return sum(1 for f in self.findings if f.severity is severity)
