"""Find pairs of tools a model cannot tell apart.

Candidate pairs come from two places: names that align in a single slot
(``update_person`` / ``update_company``) and descriptions whose embeddings are
close. Each candidate is then classified by *how* the two tools contrast:

* names differ by near-synonyms                    -> synonym-contrast
* names differ only in inflection (user / users)   -> as a one-slot pair, but
  each description must use its own word form
* names differ in one slot the descriptions mark   -> clean (no finding)
* names differ in one slot the descriptions ignore -> unmarked-contrast
* names unrelated, descriptions near-identical,
  and neither uses a name word the other lacks     -> hidden-overlap

A description that names the other tool (``use list_all_records instead
when...``) marks the contrast explicitly and clears every rule but
name-collision.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from mcp_contrast.config import Settings
from mcp_contrast.embed import Embedder
from mcp_contrast.lexicon import Lexicon
from mcp_contrast.model import CleanContrast, Finding, Report, Server, Severity, Slot, Tool
from mcp_contrast.names import align, references, stem, stems, tokenize

# Descriptions shorter than this carry too little signal to compare.
MIN_DESCRIPTION_TOKENS = 3

# A sentence shared by this many of a server's descriptions is boilerplate.
BOILERPLATE_MIN_TOOLS = 3


@dataclass(frozen=True)
class _Entry:
    tool: Tool
    tokens: tuple[str, ...]
    text: str  # the description without server-wide boilerplate
    stems: frozenset[str]
    words: frozenset[str]
    has_description: bool

    @property
    def compact(self) -> str:
        """Lowercase letters and digits only, so "Function App" contains "functionapp"."""
        return re.sub(r"[^a-z0-9]", "", self.text.lower())


def analyze(servers: Sequence[Server], settings: Settings, embedder: Embedder) -> Report:
    lexicon = Lexicon(settings.synonyms)
    groups: Iterable[list[Tool]]
    if settings.cross_server:
        groups = [[tool for server in servers for tool in server.tools]]
    else:
        groups = [list(server.tools) for server in servers]

    findings: list[Finding] = []
    clean: list[CleanContrast] = []
    for tools in groups:
        group_findings, group_clean = _analyze_group(tools, settings, embedder, lexicon)
        findings.extend(group_findings)
        clean.extend(group_clean)

    kept: list[Finding] = []
    suppressed = 0
    for finding in findings:
        severity = settings.severity_for(finding.rule)
        if severity is None or any(rule.matches(finding) for rule in settings.ignore):
            suppressed += 1
            continue
        if severity is not finding.severity:
            finding = _with_severity(finding, severity)
        kept.append(finding)

    kept.sort(key=lambda f: (-f.severity.rank, f.a.server, f.a.name, f.b.name, f.rule))
    clean.sort(key=lambda c: (c.a.server, c.a.name, c.b.name))
    return Report(tuple(servers), tuple(kept), tuple(clean), suppressed)


def _analyze_group(
    tools: list[Tool], settings: Settings, embedder: Embedder, lexicon: Lexicon
) -> tuple[list[Finding], list[CleanContrast]]:
    names = [tokenize(tool.name) for tool in tools]
    prefix = _shared_prefix(names)
    entries = [
        _Entry(
            tool,
            tuple(name[prefix:]),
            text,
            frozenset(stems(text)),
            frozenset(tokenize(text)),
            len(tokenize(text)) >= MIN_DESCRIPTION_TOKENS,
        )
        for tool, name, text in zip(
            tools, names, _without_boilerplate([t.description for t in tools]), strict=True
        )
    ]
    if len(entries) < 2:
        return [], []

    texts = [e.text if e.has_description else "" for e in entries]
    vectors = embedder.embed(texts)
    similarity = vectors @ vectors.T

    candidates = _name_candidates(entries)
    described = [i for i, e in enumerate(entries) if e.has_description]
    for x, y in combinations(described, 2):
        if similarity[x, y] >= settings.overlap_threshold:
            candidates.add((x, y))

    findings: list[Finding] = []
    clean: list[CleanContrast] = []
    for i, j in sorted(candidates):
        a, b = entries[i], entries[j]
        score = (
            float(np.clip(similarity[i, j], -1.0, 1.0))
            if a.has_description and b.has_description
            else None
        )
        result = _classify(a, b, score, settings, lexicon)
        if isinstance(result, Finding):
            findings.append(result)
        elif isinstance(result, CleanContrast):
            clean.append(result)
    return findings, clean


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _without_boilerplate(descriptions: list[str]) -> list[str]:
    """Drop sentences that recur in BOILERPLATE_MIN_TOOLS or more descriptions.

    Azure MCP ends every description with the same two sentences about its command
    router; shared text tells tools apart from nothing but inflates similarity.
    A description that would be left too short keeps its original text.
    """
    split = [[s.strip() for s in _SENTENCE_END.split(d) if s.strip()] for d in descriptions]
    counts = Counter(s for sentences in split for s in set(sentences))
    result = []
    for original, sentences in zip(descriptions, split, strict=True):
        kept = " ".join(s for s in sentences if counts[s] < BOILERPLATE_MIN_TOOLS)
        result.append(kept if len(tokenize(kept)) >= MIN_DESCRIPTION_TOKENS else original)
    return result


def _shared_prefix(names: list[list[str]]) -> int:
    """How many leading tokens every name shares, such as Notion's ``API-``.

    A namespace on every tool (``API-post-page``, ``slack_send_message``) tells
    the tools apart from nothing, so it must not make unrelated names align.
    """
    if len(names) < 3:
        return 0
    shortest = min(len(name) for name in names)
    count = 0
    while count < shortest - 1 and len({name[count] for name in names}) == 1:
        count += 1
    return count


def _name_candidates(entries: list[_Entry]) -> set[tuple[int, int]]:
    """Pairs whose names are equal or differ by one token, found by bucketing stems.

    Stems, so ``get_users`` / ``fetch_user`` differ in one token, not two.
    """
    exact: dict[tuple[str, ...], list[int]] = defaultdict(list)
    replaced: dict[tuple[str, ...], list[int]] = defaultdict(list)
    names = [tuple(stem(t) for t in entry.tokens) for entry in entries]
    for index, tokens in enumerate(names):
        exact[tokens].append(index)
        for k in range(len(tokens)):
            replaced[(*tokens[:k], "\0", *tokens[k + 1 :])].append(index)

    pairs: set[tuple[int, int]] = set()
    for bucket in (*exact.values(), *replaced.values()):
        pairs.update(combinations(sorted(set(bucket)), 2))
    # Insertions: dropping one token from a name yields another tool's name.
    for index, tokens in enumerate(names):
        for k in range(len(tokens)):
            for other in exact.get((*tokens[:k], *tokens[k + 1 :]), ()):
                if other != index:
                    pairs.add((min(index, other), max(index, other)))
    return pairs


def _classify(
    a: _Entry, b: _Entry, score: float | None, settings: Settings, lexicon: Lexicon
) -> Finding | CleanContrast | None:
    ta, tb = a.tool, b.tool

    if a.tokens == b.tokens:
        same = ta.name == tb.name
        return Finding(
            rule="name-collision",
            severity=Severity.ERROR,
            a=ta,
            b=tb,
            message=(
                f"Two tools are both named {ta.name!r}."
                if same and ta.server == tb.server
                else f"{ta.qualified!r} and {tb.qualified!r} share a name; clients that do not "
                "namespace tools by server will collide."
                if same
                else f"{ta.name!r} and {tb.name!r} differ only in case or separators."
            ),
            fix="Rename one tool, or merge them if they do the same thing.",
            similarity=score,
        )

    if references(ta.description, tb.name) or references(tb.description, ta.name):
        if score is not None:
            return CleanContrast(ta, tb, score, "a description names the other tool")
        return None

    slot = align(list(a.tokens), list(b.tokens))

    if (
        slot is not None
        and lexicon.synonymous_terms(slot.a_terms, slot.b_terms)
        and not _inflection(slot)
    ):
        return Finding(
            rule="synonym-contrast",
            severity=Severity.ERROR,
            a=ta,
            b=tb,
            message=(
                f"{ta.name!r} and {tb.name!r} differ only by near-synonyms "
                f"{_words(slot.a_terms)} / {_words(slot.b_terms)}, which give a model "
                "nothing to choose on."
            ),
            fix=(
                "Merge the tools, or rename so the differing word states the real "
                "difference (scope, target, side effect)."
            ),
            similarity=score,
            slot=slot,
        )

    if slot is not None:
        return _classify_slot(a, b, slot, score, settings, lexicon)

    if score is not None and score >= settings.overlap_threshold:
        a_word, b_word = _distinctive(a, b, lexicon), _distinctive(b, a, lexicon)
        if a_word and b_word:
            return CleanContrast(ta, tb, score, f'descriptions mark "{a_word}" / "{b_word}"')
        return Finding(
            rule="hidden-overlap",
            severity=Severity.WARNING,
            a=ta,
            b=tb,
            message=(
                f"{ta.name!r} and {tb.name!r} have near-identical descriptions "
                f"(similarity {score:.2f}) and neither says when to use it instead of the other."
            ),
            fix=(
                f"Merge the tools, or add a line to each description such as "
                f'"Use {tb.name} instead when ...".'
            ),
            similarity=score,
        )
    return None


def _distinctive(a: _Entry, b: _Entry, lexicon: Lexicon) -> str | None:
    """A word from a's name that a's description uses and b's does not, if any.

    Short descriptions of unrelated tools ("Create a new branch in a GitHub
    repository" / "Get details for a commit from a GitHub repository") embed close
    together; the name word each one uses and the other lacks is the contrast.
    """
    other_name = {stem(t) for t in b.tokens}
    for token in a.tokens:
        s = stem(token)
        if s in other_name:
            continue
        if lexicon.mentions(a.stems - b.stems, (token,), b.stems):
            return token
        # Compound and clipped names: "functionapp" in "Function App", "postgres" in
        # "PostgreSQL". Short tokens would match inside unrelated words.
        if len(token) >= 4 and token in a.compact and token not in b.compact:
            return token
    return None


def _classify_slot(
    a: _Entry, b: _Entry, slot: Slot, score: float | None, settings: Settings, lexicon: Lexicon
) -> Finding | CleanContrast | None:
    ta, tb = a.tool, b.tool
    missing: list[Tool] = []
    if not a.has_description:
        missing.append(ta)
    if not b.has_description:
        missing.append(tb)
    if missing:
        names = " and ".join(repr(t.name) for t in missing)
        return _unmarked(
            a,
            b,
            slot,
            score,
            f"{ta.name!r} and {tb.name!r} differ only by {_slot_words(slot)}, "
            f"and {names} {'has' if len(missing) == 1 else 'have'} no usable description "
            "to explain it.",
        )

    assert score is not None
    if score < settings.aligned_threshold:
        return None

    if _inflection(slot):
        # Stems cannot tell "user" from "users"; the word forms themselves must.
        a_marks = all(t in a.words for t in slot.a_terms)
        b_marks = all(t in b.words for t in slot.b_terms)
    else:
        a_marks = lexicon.mentions(a.stems, slot.a_terms, b.stems)
        b_marks = lexicon.mentions(b.stems, slot.b_terms, a.stems)

    if slot.is_insertion:
        marked_tool, marked = (ta, a_marks) if slot.a_terms else (tb, b_marks)
        plain_tool = tb if marked_tool is ta else ta
        word = _words(slot.a_terms or slot.b_terms)
        if not marked:
            return _unmarked(
                a,
                b,
                slot,
                score,
                f"{marked_tool.name!r} adds {word} to {plain_tool.name!r}, but its description "
                f"never says what {word} changes (similarity {score:.2f}).",
            )
        if score >= settings.near_identical_threshold:
            return _unmarked(
                a,
                b,
                slot,
                score,
                f"{marked_tool.name!r} and {plain_tool.name!r} differ by {word} but their "
                f"descriptions are near-identical (similarity {score:.2f}); "
                f"{plain_tool.name!r} should say what it does not cover.",
            )
        return CleanContrast(ta, tb, score, f"description of {marked_tool.name} explains {word}")

    if a_marks and b_marks:
        return CleanContrast(ta, tb, score, f"descriptions mark {_slot_words(slot)}")
    unmarked = [t for t, ok in ((ta, a_marks), (tb, b_marks)) if not ok]
    terms = [
        _words(terms) for terms, ok in ((slot.a_terms, a_marks), (slot.b_terms, b_marks)) if not ok
    ]
    detail = "; ".join(
        f"{tool.name!r} never mentions {term}" for tool, term in zip(unmarked, terms, strict=True)
    )
    return _unmarked(
        a,
        b,
        slot,
        score,
        f"{ta.name!r} and {tb.name!r} differ by {_slot_words(slot)}, but the descriptions "
        f"read the same (similarity {score:.2f}): {detail}.",
    )


def _inflection(slot: Slot) -> bool:
    """True if the slot differs only in word forms: ``user`` / ``users``."""
    return (
        len(slot.a_terms) == len(slot.b_terms) > 0
        and slot.a_terms != slot.b_terms
        and all(stem(x) == stem(y) for x, y in zip(slot.a_terms, slot.b_terms, strict=True))
    )


def _unmarked(a: _Entry, b: _Entry, slot: Slot, score: float | None, message: str) -> Finding:
    return Finding(
        rule="unmarked-contrast",
        severity=Severity.WARNING,
        a=a.tool,
        b=b.tool,
        message=message,
        fix=(
            "State the difference in each description using the word that differs in "
            "the name, or name the other tool explicitly."
        ),
        similarity=score,
        slot=slot,
    )


def _words(terms: tuple[str, ...]) -> str:
    return '"' + " ".join(terms) + '"'


def _slot_words(slot: Slot) -> str:
    if slot.is_insertion:
        return _words(slot.a_terms or slot.b_terms)
    return f"{_words(slot.a_terms)} / {_words(slot.b_terms)}"


def _with_severity(finding: Finding, severity: Severity) -> Finding:
    return replace(finding, severity=severity)
