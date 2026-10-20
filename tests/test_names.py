import pytest

from mcp_contrast.lexicon import Lexicon
from mcp_contrast.model import Slot
from mcp_contrast.names import align, mentions, references, stem, stems, tokenize


@pytest.mark.parametrize(
    ("name", "tokens"),
    [
        ("get_user", ["get", "user"]),
        ("getUser", ["get", "user"]),
        ("get-user", ["get", "user"]),
        ("users.get", ["users", "get"]),
        ("parseHTMLDocument", ["parse", "html", "document"]),
        ("list_v2_items", ["list", "v2", "items"]),
        ("  ", []),
    ],
)
def test_tokenize(name: str, tokens: list[str]) -> None:
    assert tokenize(name) == tokens


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("logs", "log"),
        ("update", "updated"),
        ("issues", "issue"),
        ("queries", "query"),
        ("listing", "list"),
        ("messages", "message"),
    ],
)
def test_stem_conflates_inflections(a: str, b: str) -> None:
    assert stem(a) == stem(b)


def test_stem_keeps_short_words() -> None:
    assert stem("ss") == "ss"
    assert stem("is") == "is"
    assert stem("bus") == "bus"


def test_align_substitution() -> None:
    assert align(["update", "person"], ["update", "company"]) == Slot(("person",), ("company",))


def test_align_insertion() -> None:
    assert align(["list", "deals"], ["list", "all", "deals"]) == Slot((), ("all",))


def test_align_rejects_two_differences() -> None:
    assert align(["get", "user", "info"], ["fetch", "user", "details"]) is None


def test_align_identical_tokens_is_none() -> None:
    assert align(["get", "user"], ["get", "user"]) is None


def test_mentions_uses_stems() -> None:
    assert mentions(stems("Lists every page of results"), ("page",))
    assert not mentions(stems("Lists results"), ("all",))
    assert not mentions(stems("anything"), ())


def test_references_requires_compound_name() -> None:
    assert references("Use archive_person instead.", "archive_person")
    assert references("Prefer `listAllDeals` for exports", "listAllDeals")
    assert not references("Use archive_person_v2 instead.", "archive_person")
    assert not references("search for things", "search")


def test_lexicon_synonyms() -> None:
    lexicon = Lexicon()
    assert lexicon.synonymous("get", "fetch")
    assert lexicon.synonymous("logs", "log")
    assert lexicon.synonymous("all", "full")
    assert not lexicon.synonymous("create", "delete")
    assert not lexicon.synonymous("person", "company")


def test_lexicon_terms_must_all_be_synonyms() -> None:
    lexicon = Lexicon()
    assert lexicon.synonymous_terms(("get", "info"), ("fetch", "details"))
    assert not lexicon.synonymous_terms(("get", "info"), ("delete", "details"))
    assert not lexicon.synonymous_terms(("get",), ("get",))
    assert not lexicon.synonymous_terms(("get",), ("fetch", "all"))


def test_lexicon_extra_groups() -> None:
    assert Lexicon([["ticket", "issue"]]).synonymous("tickets", "issue")
    assert not Lexicon().synonymous("ticket", "issue")
