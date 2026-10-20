"""Near-synonym sets common in tool names.

Two names that differ only by words from the same set give a model nothing to
choose on. The list is curated rather than learned: static word embeddings score
``get``/``fetch`` lower than many unrelated pairs, and a reviewable list keeps
the rule deterministic. Extend it per project with the ``synonyms`` setting.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Set as AbstractSet

from mcp_contrast.names import stem

SYNONYM_SETS: tuple[frozenset[str], ...] = tuple(
    frozenset(group)
    for group in (
        # Reading one thing.
        {"get", "fetch", "retrieve", "read", "load", "obtain", "show", "view", "describe"},
        # Finding things.
        {"search", "find", "query", "lookup", "locate", "seek"},
        {"list", "enumerate", "browse"},
        # "post" is also REST's create (Notion's API-post-page).
        {"create", "add", "new", "make", "insert", "append", "post"},
        # Not "write": write_file (replace) and edit_file (patch lines) differ.
        {"update", "modify", "edit", "change", "patch", "alter"},
        {"delete", "remove", "destroy", "erase", "drop"},
        {"send", "post", "submit", "dispatch"},
        {"run", "execute", "invoke", "trigger", "perform"},
        {"analyze", "analyse", "inspect", "examine"},
        {"check", "verify", "validate"},
        {"stop", "halt", "abort", "terminate", "kill"},
        {"summary", "summarize", "summarise", "overview", "digest"},
        # Scope modifiers.
        {"all", "full", "complete", "entire", "whole"},
        {"info", "information", "details", "detail"},
        # Abbreviations.
        {"message", "msg"},
        {"document", "doc"},
        {"repository", "repo"},
        {"organization", "organisation", "org"},
        {"environment", "env"},
        {"database", "db"},
        {"configuration", "config", "cfg"},
        {"directory", "dir", "folder"},
        {"image", "img"},
        {"parameter", "param"},
        {"identifier", "id"},
        {"statistics", "stats", "stat"},
    )
)


class Lexicon:
    def __init__(self, extra: Iterable[Iterable[str]] = ()) -> None:
        self._groups: dict[str, set[int]] = {}
        self._members: list[frozenset[str]] = []
        for index, group in enumerate([*SYNONYM_SETS, *(frozenset(g) for g in extra)]):
            self._members.append(frozenset(stem(word) for word in group))
            for word in group:
                self._groups.setdefault(stem(word), set()).add(index)

    def synonymous(self, a: str, b: str) -> bool:
        sa, sb = stem(a), stem(b)
        if sa == sb:
            return True
        return bool(self._groups.get(sa, set()) & self._groups.get(sb, set()))

    def synonymous_terms(self, a: tuple[str, ...], b: tuple[str, ...]) -> bool:
        """Equal-length term runs that are pairwise synonyms (``get_info`` vs ``fetch_details``)."""
        return (
            len(a) == len(b) > 0
            and any(x != y for x, y in zip(a, b, strict=True))
            and all(self.synonymous(x, y) for x, y in zip(a, b, strict=True))
        )

    def mentions(
        self, description_stems: AbstractSet[str], terms: tuple[str, ...], other: AbstractSet[str]
    ) -> bool:
        """True if the description uses every term, or a synonym the other description lacks.

        ``get_page`` described as "Retrieve a page" marks "get". A synonym both
        descriptions share ("query" in two identical texts) marks nothing.
        """
        own = description_stems - other
        return bool(terms) and all(
            stem(term) in description_stems
            or any(not self._members[i].isdisjoint(own) for i in self._groups.get(stem(term), ()))
            for term in terms
        )
