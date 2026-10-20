"""Settings from ``mcp-contrast.toml`` or ``[tool.mcp-contrast]`` in ``pyproject.toml``."""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mcp_contrast.embed import DEFAULT_MODEL, DEFAULT_REVISION
from mcp_contrast.model import Finding, Severity
from mcp_contrast.rules import RULES

CONFIG_FILES = ("mcp-contrast.toml", ".mcp-contrast.toml")

# Similarity scales differ by embedder, so each has its own defaults:
# (overlap, aligned, near-identical).
DEFAULT_THRESHOLDS = {"model2vec": (0.85, 0.70, 0.95), "lexical": (0.70, 0.50, 0.90)}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Ignore:
    tools: tuple[str, str]
    rule: str | None = None

    def matches(self, finding: Finding) -> bool:
        if self.rule is not None and self.rule != finding.rule:
            return False
        names = (finding.a.name, finding.b.name)
        quals = (finding.a.qualified, finding.b.qualified)
        first, second = self.tools

        def hit(pattern: str, i: int) -> bool:
            return fnmatch.fnmatchcase(names[i], pattern) or fnmatch.fnmatchcase(quals[i], pattern)

        return (hit(first, 0) and hit(second, 1)) or (hit(first, 1) and hit(second, 0))


@dataclass(frozen=True)
class Settings:
    embedder: str = "model2vec"
    model: str = DEFAULT_MODEL
    model_revision: str | None = DEFAULT_REVISION
    threshold: float | None = None
    aligned: float | None = None
    near_identical: float | None = None
    fail_on: str = "error"
    cross_server: bool = False
    synonyms: tuple[tuple[str, ...], ...] = ()
    ignore: tuple[Ignore, ...] = ()
    severity: dict[str, Severity | None] = field(default_factory=dict)

    @property
    def overlap_threshold(self) -> float:
        return self.threshold if self.threshold is not None else self._defaults[0]

    @property
    def aligned_threshold(self) -> float:
        """Lower bar for pairs whose names already align in one slot."""
        return self.aligned if self.aligned is not None else self._defaults[1]

    @property
    def near_identical_threshold(self) -> float:
        if self.near_identical is not None:
            return self.near_identical
        return self._defaults[2]

    @property
    def _defaults(self) -> tuple[float, float, float]:
        return DEFAULT_THRESHOLDS.get(self.embedder, DEFAULT_THRESHOLDS["model2vec"])

    def severity_for(self, rule: str) -> Severity | None:
        """Configured severity, or None when the rule is turned off."""
        if rule in self.severity:
            return self.severity[rule]
        return RULES[rule].severity

    def with_overrides(self, **overrides: Any) -> Settings:
        values = {k: v for k, v in overrides.items() if v is not None}
        if (
            "model" in values
            and "model_revision" not in values
            and values["model"] != DEFAULT_MODEL
        ):
            values["model_revision"] = None
        return replace(self, **values)


def discover(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        for name in CONFIG_FILES:
            if (directory / name).is_file():
                return directory / name
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file() and "mcp-contrast" in _read(pyproject).get("tool", {}):
            return pyproject
        if (directory / ".git").exists():
            break
    return None


def load(path: Path | None) -> Settings:
    if path is None:
        return Settings()
    data = _read(path)
    if path.name == "pyproject.toml":
        data = data.get("tool", {}).get("mcp-contrast", {})
    return parse(data, source=str(path))


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


_KEYS = {
    "embedder",
    "model",
    "model-revision",
    "threshold",
    "aligned-threshold",
    "near-identical-threshold",
    "fail-on",
    "cross-server",
    "synonyms",
    "ignore",
    "severity",
}


def parse(data: dict[str, Any], source: str = "config") -> Settings:
    unknown = set(data) - _KEYS
    if unknown:
        raise ConfigError(f"{source}: unknown setting(s): {', '.join(sorted(unknown))}")
    try:
        return _parse(data, source)
    except ConfigError as exc:
        if str(exc).startswith(source):
            raise
        raise ConfigError(f"{source}: {exc}") from exc


def _parse(data: dict[str, Any], source: str) -> Settings:
    overrides: dict[str, Any] = {}

    if "embedder" in data:
        overrides["embedder"] = _choice(data["embedder"], {"model2vec", "lexical"}, "embedder")
    if "model" in data:
        overrides["model"] = _str(data["model"], "model")
    if "model-revision" in data:
        overrides["model_revision"] = _str(data["model-revision"], "model-revision")
    if "threshold" in data:
        overrides["threshold"] = _unit(data["threshold"], "threshold")
    if "aligned-threshold" in data:
        overrides["aligned"] = _unit(data["aligned-threshold"], "aligned-threshold")
    if "near-identical-threshold" in data:
        overrides["near_identical"] = _unit(
            data["near-identical-threshold"], "near-identical-threshold"
        )
    if "fail-on" in data:
        overrides["fail_on"] = _choice(data["fail-on"], set(FAIL_ON), "fail-on")
    if "cross-server" in data:
        if not isinstance(data["cross-server"], bool):
            raise ConfigError(f"{source}: cross-server must be true or false")
        overrides["cross_server"] = data["cross-server"]
    if "synonyms" in data:
        groups = data["synonyms"]
        if not isinstance(groups, list) or not all(
            isinstance(g, list) and len(g) >= 2 and all(isinstance(w, str) for w in g)
            for g in groups
        ):
            raise ConfigError(f"{source}: synonyms must be a list of word lists")
        overrides["synonyms"] = tuple(tuple(g) for g in groups)
    if "ignore" in data:
        overrides["ignore"] = tuple(_ignore(entry, source) for entry in data["ignore"])
    if "severity" in data:
        overrides["severity"] = _severities(data["severity"], source)

    return Settings().with_overrides(**overrides)


FAIL_ON = ("error", "warning", "info", "never")


def _choice(value: Any, allowed: set[str], key: str) -> str:
    if value not in allowed:
        raise ConfigError(f"{key} must be one of: {', '.join(sorted(allowed))}")
    return str(value)


def _str(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _unit(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value <= 1:
        raise ConfigError(f"{key} must be a number in (0, 1]")
    return float(value)


def _ignore(entry: Any, source: str) -> Ignore:
    if not isinstance(entry, dict) or set(entry) - {"tools", "rule"}:
        raise ConfigError(f"{source}: each ignore entry is {{ tools = [a, b], rule = '...' }}")
    tools = entry.get("tools")
    if not (isinstance(tools, list) and len(tools) == 2 and all(isinstance(t, str) for t in tools)):
        raise ConfigError(f"{source}: ignore.tools must be a pair of tool names or globs")
    rule = entry.get("rule")
    if rule is not None and rule not in RULES:
        raise ConfigError(f"{source}: unknown rule {rule!r} in ignore")
    return Ignore((tools[0], tools[1]), rule)


def _severities(table: Any, source: str) -> dict[str, Severity | None]:
    if not isinstance(table, dict):
        raise ConfigError(f"{source}: severity must be a table of rule = level")
    result: dict[str, Severity | None] = {}
    for rule, level in table.items():
        if rule not in RULES:
            raise ConfigError(f"{source}: unknown rule {rule!r} in severity")
        if level == "off":
            result[rule] = None
        elif level in {s.value for s in Severity}:
            result[rule] = Severity(level)
        else:
            raise ConfigError(f"{source}: severity.{rule} must be error, warning, info, or off")
    return result
