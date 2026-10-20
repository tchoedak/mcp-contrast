"""Tokenizing, stemming, and aligning tool names."""

from __future__ import annotations

import re
from collections.abc import Set as AbstractSet
from difflib import SequenceMatcher

from mcp_contrast.model import Slot

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    """Split names and prose alike: snake, kebab, dot, camel, and whitespace."""
    tokens: list[str] = []
    for chunk in _NON_ALNUM.split(text):
        if chunk:
            tokens.extend(part.lower() for part in _CAMEL.split(chunk) if part)
    return tokens


def stem(word: str) -> str:
    """Crude suffix stripping, applied the same way to names and descriptions.

    Only needs to be consistent, not linguistically correct: ``logs``/``log`` and
    ``update``/``updated`` must land on the same stem.
    """
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("sses"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def stems(text: str) -> set[str]:
    return {stem(t) for t in tokenize(text)}


def align(a: list[str], b: list[str]) -> Slot | None:
    """Return the one slot where two token lists differ, or None.

    None means the names differ in zero places (same tokens, different spelling)
    or in more than one place; callers distinguish the two with ``a == b``.
    """
    ops = [
        op for op in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"
    ]
    if len(ops) != 1:
        return None
    _, i1, i2, j1, j2 = ops[0]
    return Slot(tuple(a[i1:i2]), tuple(b[j1:j2]))


def mentions(description_stems: AbstractSet[str], terms: tuple[str, ...]) -> bool:
    return bool(terms) and all(stem(t) in description_stems for t in terms)


def references(description: str, name: str) -> bool:
    """True if a description names another tool explicitly.

    Single-word names like ``search`` are ignored: the word appearing in prose
    is not a reference to the tool.
    """
    if len(tokenize(name)) < 2:
        return False
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
    return re.search(pattern, description, re.IGNORECASE) is not None
