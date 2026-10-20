# Rules

Every rule looks at a **pair** of tools. Change a rule's severity or turn it off with the
`[severity]` table in your config; skip specific pairs with `ignore`.

A description that names the other tool explicitly (for example `Use list_all_deals to
export every page`) marks the contrast and clears every rule except `name-collision`.

| Rule | Default | Catches |
|---|---|---|
| [`name-collision`](#name-collision) | error | identical names, or names differing only in case or separators |
| [`synonym-contrast`](#synonym-contrast) | error | names differing only by near-synonyms |
| [`unmarked-contrast`](#unmarked-contrast) | warning | names differing in one word that the descriptions never explain |
| [`hidden-overlap`](#hidden-overlap) | warning | unrelated names with near-identical descriptions |

## name-collision

Two tools have the same name, or names that tokenize the same: `getUser`, `get_user`,
`get-user`.

**Why it matters.** Clients either reject the duplicate or route every call to one of the
two. With `--cross-server`, identical names on different servers are also reported, since
clients that do not namespace tools by server will collide.

**Fix.** Rename one tool, or merge them.

## synonym-contrast

The names align except for one run of words, and those words are near-synonyms:

```
get_user        / fetch_user           get / fetch
list_all_deals  / list_full_deals      all / full
get_repo_info   / get_repository_info  repo / repository
search_logs     / search_log           logs / log
```

**Why it matters.** A model assumes that different words mean different things. Given
these names it will guess at a difference that is not there, or that is there but is not
written down.

This rule fires regardless of the descriptions: even with good descriptions, the names
themselves invite the wrong guess.

**Fix.** Merge the tools, or rename so the differing word names the real difference:
`get_user` / `get_user_with_audit_log`, `list_deals_page` / `export_all_deals`.

**Synonym list.** Synonyms come from a curated list in
[`lexicon.py`](../src/mcp_contrast/lexicon.py) (common verbs, scope modifiers, and
abbreviations) plus a stem check that catches singular/plural pairs. Add domain words in
your config:

```toml
synonyms = [["ticket", "issue"], ["workspace", "project"]]
```

## unmarked-contrast

The names differ in exactly one place, the descriptions are similar (at or above
`aligned-threshold`, default 0.70), and the descriptions do not use the differing word to
explain the difference.

```
list_deals      "List deals in the pipeline, newest first."
list_all_deals  "List deals in the pipeline, newest first, with optional filters."
                 -> "all" is never explained
```

It also fires when:

- the descriptions are near-identical (at or above `near-identical-threshold`, default
  0.95) even though the longer name's description mentions its extra word, because the
  shorter tool still does not say what it leaves out;
- either tool in an aligned pair has no usable description (fewer than three words).

**What passes.** `update_person` / `update_company` with descriptions that mention "person"
and "company" respectively. The names differ in one slot, and each description says which
side of the contrast it is on.

**Fix.** Use the differing word in each description to say what it changes, or name the
other tool: `List the first page of deals. Use list_all_deals to export every page.`

## hidden-overlap

The names do not align (they differ in more than one place, or share no structure), but the
descriptions score at or above `threshold` (default 0.85).

```
search_notes        "Search notes attached to people and companies by keyword."
find_notes_by_text  "Search notes attached to people and companies by keyword."
```

**Why it matters.** Both tools look correct for the same request, so the model picks one at
random, and retries or fails when it picks the wrong one.

**Fix.** Merge the tools, or add a sentence to each that says when to use the other.

## Thresholds

Similarity is the cosine between description embeddings. Defaults depend on the embedder:

| Setting | `model2vec` | `lexical` |
|---|---|---|
| `threshold` | 0.85 | 0.70 |
| `aligned-threshold` | 0.70 | 0.50 |
| `near-identical-threshold` | 0.95 | 0.90 |

These are starting points, not tuned values. If a rule fires on a pair a model would
clearly tell apart, please
[open a false-positive report](https://github.com/tchoedak/mcp-contrast/issues/new?template=false-positive.yml)
with both tool definitions.
