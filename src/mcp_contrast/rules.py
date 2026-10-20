"""Rule catalog. Each rule names one way a pair of tools fails to contrast."""

from __future__ import annotations

from dataclasses import dataclass

from mcp_contrast.model import Severity

DOCS = "https://github.com/tchoedak/mcp-contrast/blob/main/docs/rules.md"


@dataclass(frozen=True)
class Rule:
    id: str
    severity: Severity
    summary: str
    rationale: str

    @property
    def help_uri(self) -> str:
        return f"{DOCS}#{self.id}"


RULES: dict[str, Rule] = {
    rule.id: rule
    for rule in (
        Rule(
            "name-collision",
            Severity.ERROR,
            "Two tools have the same name, or names that differ only in case or separators.",
            "Clients either reject the duplicate or silently route calls to one of them.",
        ),
        Rule(
            "synonym-contrast",
            Severity.ERROR,
            "Tool names differ only by near-synonyms, such as get/fetch or all/full.",
            "A model assumes different words mean different things and will guess "
            "at a difference the names do not state.",
        ),
        Rule(
            "unmarked-contrast",
            Severity.WARNING,
            "Tool names differ in one word, but the descriptions read the same and do not "
            "say what that word changes.",
            "The name signals a contrast the description never explains, so the model "
            "cannot tell when to pick each tool.",
        ),
        Rule(
            "hidden-overlap",
            Severity.WARNING,
            "Differently named tools have near-identical descriptions and neither "
            "points to the other.",
            "Two tools that both look correct for a request make the model pick at random.",
        ),
    )
}
