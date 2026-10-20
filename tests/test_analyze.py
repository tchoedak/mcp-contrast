from dataclasses import replace

import numpy as np

from mcp_contrast.analyze import analyze
from mcp_contrast.config import Ignore, Settings
from mcp_contrast.embed import LexicalEmbedder, Matrix
from mcp_contrast.model import Severity
from tests.conftest import server


class Scripted:
    """Embedder that returns chosen similarities: texts sharing a key are identical."""

    name = "scripted"

    def __init__(self, scores: dict[frozenset[str], float]):
        self.scores = scores

    def embed(self, texts: list[str]) -> Matrix:
        n = len(texts)
        sim = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                sim[i, j] = sim[j, i] = self.scores.get(frozenset((texts[i], texts[j])), 0.0)
        # Factor the similarity matrix into vectors whose dot products reproduce it.
        values, vectors = np.linalg.eigh(sim)
        return (vectors * np.sqrt(np.clip(values, 0, None))).astype(np.float32)


def rules(report):
    return [(f.rule, f.a.name, f.b.name) for f in report.findings]


def test_synonym_contrast_fires_regardless_of_descriptions(settings, lexical) -> None:
    report = analyze(
        [server("s", ("get_user", "Return one user."), ("fetch_user", "Stream audit events."))],
        settings,
        lexical,
    )
    assert rules(report) == [("synonym-contrast", "get_user", "fetch_user")]
    assert report.findings[0].severity is Severity.ERROR


def test_clean_crud_family_is_not_flagged(settings, lexical) -> None:
    report = analyze(
        [
            server(
                "crm",
                ("update_person", "Update fields on a person record in the CRM."),
                ("update_company", "Update fields on a company record in the CRM."),
            )
        ],
        settings,
        lexical,
    )
    assert report.findings == ()
    assert [(c.a.name, c.b.name) for c in report.clean] == [("update_person", "update_company")]


def test_unmarked_substitution(settings, lexical) -> None:
    report = analyze(
        [
            server(
                "logs",
                ("search_logs", "Return log lines matching a query over a time range."),
                ("analyze_logs", "Return log lines matching a query over a time range."),
            )
        ],
        settings,
        lexical,
    )
    assert rules(report) == [("unmarked-contrast", "search_logs", "analyze_logs")]
    message = report.findings[0].message
    assert "'search_logs' never mentions \"search\"" in message
    assert "'analyze_logs' never mentions \"analyze\"" in message


def test_unmarked_insertion(settings, lexical) -> None:
    report = analyze(
        [
            server(
                "crm",
                ("list_deals", "List deals in the pipeline, newest first."),
                ("list_all_deals", "List deals in the pipeline, newest first, with filters."),
            )
        ],
        settings,
        lexical,
    )
    assert rules(report) == [("unmarked-contrast", "list_deals", "list_all_deals")]
    assert 'adds "all"' in report.findings[0].message


def test_marked_insertion_with_near_identical_descriptions_still_warns() -> None:
    plain, marked = "List deals in the pipeline.", "List all deals in the pipeline."
    report = analyze(
        [server("crm", ("list_deals", plain), ("list_all_deals", marked))],
        Settings(),
        Scripted({frozenset((plain, marked)): 0.97}),
    )
    assert rules(report) == [("unmarked-contrast", "list_deals", "list_all_deals")]
    assert "should say what it does not cover" in report.findings[0].message


def test_marked_insertion_is_clean() -> None:
    plain, marked = "List the first page of deals.", "List all deals across every page."
    report = analyze(
        [server("crm", ("list_deals", plain), ("list_all_deals", marked))],
        Settings(),
        Scripted({frozenset((plain, marked)): 0.9}),
    )
    assert report.findings == ()
    assert len(report.clean) == 1


def test_aligned_pairs_use_the_lower_threshold() -> None:
    a, b = "Return deals in the pipeline.", "Return deals in the pipeline with filters."
    tools = server("crm", ("list_deals", a), ("list_all_deals", b))
    below = analyze([tools], Settings(), Scripted({frozenset((a, b)): 0.69}))
    above = analyze([tools], Settings(), Scripted({frozenset((a, b)): 0.71}))
    assert below.findings == ()
    assert rules(above) == [("unmarked-contrast", "list_deals", "list_all_deals")]


def test_hidden_overlap(settings, lexical) -> None:
    text = "Search notes attached to people and companies by keyword."
    report = analyze(
        [server("crm", ("search_notes", text), ("find_notes_by_text", text))],
        settings,
        lexical,
    )
    assert rules(report) == [("hidden-overlap", "search_notes", "find_notes_by_text")]
    assert report.findings[0].similarity is not None
    assert report.findings[0].similarity > 0.99


def test_unrelated_tools_with_dissimilar_descriptions_pass(settings, lexical) -> None:
    report = analyze(
        [
            server(
                "mixed",
                ("send_email", "Send an email message to one or more recipients."),
                ("list_events", "List calendar events in a date range."),
            )
        ],
        settings,
        lexical,
    )
    assert report.findings == ()
    assert report.clean == ()


def test_cross_reference_clears_overlap() -> None:
    a = "Search notes by keyword. Use find_notes_by_text for regex."
    b = "Search notes by keyword."
    report = analyze(
        [server("crm", ("search_notes", a), ("find_notes_by_text", b))],
        Settings(),
        Scripted({frozenset((a, b)): 0.9}),
    )
    assert report.findings == ()
    assert report.clean[0].reason == "a description names the other tool"


def test_cross_reference_does_not_clear_collisions(settings, lexical) -> None:
    report = analyze(
        [server("s", ("getUser", "Use get_user instead."), ("get_user", "Use getUser instead."))],
        settings,
        lexical,
    )
    assert rules(report) == [("name-collision", "getUser", "get_user")]


def test_name_collision_exact_duplicate(settings, lexical) -> None:
    report = analyze([server("s", ("ping", "Ping."), ("ping", "Ping again."))], settings, lexical)
    assert rules(report) == [("name-collision", "ping", "ping")]
    assert "both named 'ping'" in report.findings[0].message


def test_missing_description_on_aligned_pair(settings, lexical) -> None:
    report = analyze(
        [server("s", ("list_deals", "List deals in the pipeline."), ("list_open_deals", ""))],
        settings,
        lexical,
    )
    assert rules(report) == [("unmarked-contrast", "list_deals", "list_open_deals")]
    assert "no usable description" in report.findings[0].message
    assert report.findings[0].similarity is None


def test_servers_are_isolated_by_default(settings, lexical) -> None:
    servers = [server("a", ("search", "Search everything.")), server("b", ("search", "Search."))]
    assert analyze(servers, settings, lexical).findings == ()
    crossed = analyze(servers, replace(settings, cross_server=True), lexical)
    assert rules(crossed) == [("name-collision", "search", "search")]
    assert "share a name" in crossed.findings[0].message


def test_ignore_and_severity_overrides(settings, lexical) -> None:
    tools = server("s", ("get_user", "Get a user."), ("fetch_user", "Fetch a user."))
    ignored = analyze([tools], replace(settings, ignore=(Ignore(("fetch_*", "get_*")),)), lexical)
    assert ignored.findings == ()
    assert ignored.suppressed == 1

    other_rule = analyze(
        [tools], replace(settings, ignore=(Ignore(("*", "*"), "hidden-overlap"),)), lexical
    )
    assert len(other_rule.findings) == 1

    off = analyze([tools], replace(settings, severity={"synonym-contrast": None}), lexical)
    assert off.findings == ()
    assert off.suppressed == 1

    downgraded = analyze(
        [tools], replace(settings, severity={"synonym-contrast": Severity.INFO}), lexical
    )
    assert downgraded.findings[0].severity is Severity.INFO


def test_custom_synonyms(settings, lexical) -> None:
    tools = server(
        "s",
        ("create_ticket", "Open a support case for a customer."),
        ("create_issue", "Track a bug in the engineering backlog."),
    )
    assert analyze([tools], settings, lexical).findings == ()
    flagged = analyze([tools], replace(settings, synonyms=(("ticket", "issue"),)), lexical)
    assert rules(flagged) == [("synonym-contrast", "create_ticket", "create_issue")]


def test_findings_sorted_by_severity_then_name(settings, lexical) -> None:
    text = "Search notes attached to people and companies by keyword."
    report = analyze(
        [
            server(
                "s",
                ("search_notes", text),
                ("find_notes_by_text", text),
                ("get_user", "Get a user."),
                ("fetch_user", "Fetch a user."),
            )
        ],
        settings,
        lexical,
    )
    assert [f.rule for f in report.findings] == ["synonym-contrast", "hidden-overlap"]


def test_fingerprint_is_order_independent(settings) -> None:
    a = analyze(
        [server("s", ("get_user", "Get a user."), ("fetch_user", "Fetch a user."))],
        settings,
        LexicalEmbedder(),
    )
    b = analyze(
        [server("s", ("fetch_user", "Fetch a user."), ("get_user", "Get a user."))],
        settings,
        LexicalEmbedder(),
    )
    assert a.findings[0].fingerprint == b.findings[0].fingerprint


def test_single_tool_and_empty_server(settings, lexical) -> None:
    assert analyze([server("s", ("only", "The only tool."))], settings, lexical).findings == ()
    assert analyze([server("s")], settings, lexical).findings == ()


def test_documented_synonym_examples(settings, lexical) -> None:
    pairs = [
        ("get_user", "fetch_user"),
        ("list_all_deals", "list_full_deals"),
        ("get_repo_info", "get_repository_info"),
        ("search_logs", "search_log"),
    ]
    for a, b in pairs:
        report = analyze(
            [server("s", (a, "Does one thing well."), (b, "Does another thing entirely."))],
            settings,
            lexical,
        )
        assert rules(report) == [("synonym-contrast", a, b)], (a, b)
