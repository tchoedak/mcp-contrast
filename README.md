# mcp-contrast

[![CI](https://github.com/tchoedak/mcp-contrast/actions/workflows/ci.yml/badge.svg)](https://github.com/tchoedak/mcp-contrast/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

**Lint MCP servers for tools a model cannot tell apart.**

`mcp-contrast` finds pairs of tools that look interchangeable to an LLM: `get_user` next
to `fetch_user`, `list_deals` next to an unexplained `list_all_deals`, two differently
named tools with the same description. It runs in CI, fully in memory, in under a second
once its 30 MB model is cached.

```console
$ mcp-contrast check examples/crm.json
crm (10 tools)
  error    synonym-contrast   get_person <-> fetch_person  examples/crm.json:3
           'get_person' and 'fetch_person' differ only by near-synonyms "get" / "fetch", which
           give a model nothing to choose on.
           fix: Merge the tools, or rename so the differing word states the real difference
                (scope, target, side effect).

  warning  unmarked-contrast  list_deals <-> list_all_deals  examples/crm.json:7
           'list_all_deals' adds "all" to 'list_deals', but its description never says what
           "all" changes (similarity 0.83).
           fix: State the difference in each description using the word that differs in the
                name, or name the other tool explicitly.

  warning  hidden-overlap     search_notes <-> find_notes_by_text  examples/crm.json:9
           'search_notes' and 'find_notes_by_text' have near-identical descriptions (similarity
           1.00) and neither says when to use it instead of the other.
           fix: Merge the tools, or add a line to each description such as "Use
                find_notes_by_text instead when ...".

✖ 3 problems (1 error, 2 warnings) - 10 tools in 1 server, 3 clean contrasts
```

## Why

Linguists call it the *principle of contrast*: listeners assume that two different words
mean two different things. Eve Clark proposed it to explain how children learn words
([Clark 1987](#references)), and it is the idea this project is built on. Models do the same. Given `get_user` and
`fetch_user`, a model looks for a difference, finds none in the descriptions, and picks
one at random.

The naive fix, flagging tools whose descriptions are similar, does not work. Well-designed
families like `update_person` / `update_company` are *supposed* to look alike: they
differ in exactly the word the model needs. An earlier linter
([mcp-doctor](https://github.com/vishalhabib99/mcp-doctor)) tried a similarity score,
found it mostly flagged these families, and fell back to near-exact matching.

`mcp-contrast` keeps the semantic signal and asks a second question: **where do the two
names differ, and do the descriptions say so?**

| Pair | Names differ by | Descriptions | Result |
|---|---|---|---|
| `update_person` / `update_company` | person / company | each mentions its own noun | clean |
| `get_user` / `fetch_user` | get / fetch (synonyms) | n/a | `synonym-contrast` |
| `list_deals` / `list_all_deals` | inserted "all" | "all" never explained | `unmarked-contrast` |
| `search_notes` / `find_notes_by_text` | no shared structure | near-identical | `hidden-overlap` |
| `read_file` / `read_text_file` | inserted "text" | `read_file` says "use read_text_file instead" | clean |

## Install

```sh
uvx mcp-contrast check tools.json          # run without installing
pipx install mcp-contrast                  # or install the CLI
pip install mcp-contrast                   # or into an environment
```

Until the first PyPI release, install from GitHub:
`uvx --from git+https://github.com/tchoedak/mcp-contrast mcp-contrast check tools.json`.

Requires Python 3.11+. No PyTorch, no GPU, no API keys.

## Usage

```sh
# A saved tools/list response (or a bare array of tools)
mcp-contrast check tools.json

# A live stdio server
mcp-contrast check --stdio 'npx -y @modelcontextprotocol/server-filesystem /tmp'

# A live Streamable HTTP server
mcp-contrast check --http https://mcp.example.com/mcp --header "Authorization: Bearer $TOKEN"

# Every server in a client config (Claude, Cursor, or VS Code format)
mcp-contrast check .mcp.json
mcp-contrast check .mcp.json --cross-server   # also compare tools across servers

# Save a live server's tools for linting in CI
mcp-contrast snapshot --stdio 'node dist/index.js' -o tools.json
```

`mcp-contrast` only performs the MCP handshake and `tools/list`. It never calls a tool.

| Flag | Default | |
|---|---|---|
| `--format` | `text` | `text`, `json`, `sarif`, or `github` (workflow annotations) |
| `-o`, `--output` | stdout | write the report to a file |
| `--sarif PATH` | | also write SARIF, alongside the main format |
| `--fail-on` | `error` | lowest severity that exits 1: `error`, `warning`, `info`, `never` |
| `--show-clean` | off | list the similar pairs that passed, and why |
| `--baseline PATH` | | ignore findings recorded in a baseline |
| `--write-baseline` | | record current findings into `--baseline` and exit 0 |
| `--cross-server` | off | compare tools across servers, as a client loading all of them would |
| `--embedder` | `model2vec` | `lexical` for a zero-download fallback |
| `--threshold` | `0.85` | description similarity that makes an unrelated pair a candidate |
| `--aligned-threshold` | `0.70` | the same, for pairs whose names differ in one word |
| `--near-identical` | `0.95` | similarity above which descriptions count as the same |

Exit codes: `0` clean, `1` findings at or above `--fail-on`, `2` usage or input error.

Run `mcp-contrast rules` for the rule list, or see [docs/rules.md](docs/rules.md).

## CI

### GitHub Action

```yaml
name: MCP tools
on: [pull_request]

permissions:
  contents: read
  security-events: write   # only for the SARIF upload

jobs:
  contrast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: tchoedak/mcp-contrast@v0.1.0
        with:
          sources: tools.json
      - uses: github/codeql-action/upload-sarif@v4
        if: always()
        with:
          sarif_file: mcp-contrast.sarif
```

Findings appear as annotations on the pull request. The SARIF upload is optional and puts
them in the repository's code scanning tab.

To lint a server you build in the same job, start it through `args`:

```yaml
      - run: npm ci && npm run build
      - uses: tchoedak/mcp-contrast@v0.1.0
        with:
          args: --stdio 'node dist/index.js'
```

| Input | Default | |
|---|---|---|
| `sources` | | space-separated snapshots or client configs |
| `args` | | extra CLI arguments, shell-quoted |
| `fail-on` | `error` | |
| `sarif-file` | `mcp-contrast.sarif` | empty to skip |
| `working-directory` | `.` | |

The action caches the embedding model between runs.

### Anywhere else

The CLI is the integration: `uvx mcp-contrast check tools.json --format sarif -o
report.sarif` works in GitLab CI, CircleCI, Buildkite, or a pre-commit hook.

### Adopting on an existing server

```sh
mcp-contrast check tools.json --baseline .mcp-contrast-baseline.json --write-baseline
```

Commit the baseline. Later runs with `--baseline` fail only on new findings. Fingerprints
are the rule plus the pair of tool names, so unrelated edits do not invalidate them.

## Configuration

Settings live in `mcp-contrast.toml`, `.mcp-contrast.toml`, or `[tool.mcp-contrast]` in
`pyproject.toml`, discovered from the working directory up to the repository root. CLI
flags win.

```toml
# mcp-contrast.toml
fail-on = "warning"
threshold = 0.85
aligned-threshold = 0.70

# Domain words your users treat as interchangeable.
synonyms = [["ticket", "issue"], ["workspace", "project"]]

# Known pairs to skip. Names or globs, in either order.
ignore = [
  { tools = ["search_logs", "analyze_logs"], rule = "hidden-overlap" },
  { tools = ["legacy_*", "*"] },
]

[severity]
unmarked-contrast = "error"   # error, warning, info, or off
```

## How it works

1. **Load** each server's tools from a snapshot, a client config, or a live connection.
2. **Embed** every description in memory with
   [model2vec](https://github.com/MinishLab/model2vec)'s `potion-base-8M`: a 30 MB static
   embedding model (a lookup table plus mean pooling) downloaded once from the Hugging
   Face Hub at a pinned revision and cached. A few hundred tools take milliseconds.
3. **Find candidates**: pairs whose names are equal or differ in one word (found by
   bucketing name tokens), plus pairs whose descriptions score above `--threshold`.
4. **Align** the two names token by token and **classify** the pair:
   - names equal apart from case or separators: `name-collision`
   - the differing words are near-synonyms (a curated, extensible list):
     `synonym-contrast`
   - names differ in one slot and each description mentions its own word: clean
   - names differ in one slot and a description does not: `unmarked-contrast`
   - names unrelated, descriptions near-identical: `hidden-overlap`
   - a description names the other tool (`use read_text_file instead`): clean

Servers are analyzed independently unless you pass `--cross-server`.

Synonyms come from a reviewable list rather than the embedding model, because static word
embeddings score `get`/`fetch` (0.36) lower than many unrelated pairs.

**Offline CI**: set `HF_HUB_OFFLINE=1` with a warm cache, point `--model` at a local copy
of the model directory, or use `--embedder lexical` (TF-IDF, no download, misses
paraphrases).

## Limitations

- **Thresholds are starting points.** They come from a small set of hand-built examples
  and the MCP reference servers, not a labeled benchmark. Tune them, and please
  [report false positives](https://github.com/tchoedak/mcp-contrast/issues/new?template=false-positive.yml).
- **It checks that a contrast is stated, not that it is meaningful.** If
  `search_logs` says "Search logs..." and `analyze_logs` says "Analyze logs...", each
  description mentions its word and the pair passes.
- **English only.** Tokenizing, stemming, and the synonym list assume English names.
- **Legacy HTTP+SSE servers are not supported.** Use Streamable HTTP, stdio, or a snapshot.
- Code scanning needs file locations, so live `--stdio`/`--http` findings carry none.
  Lint a snapshot or a client config to get annotations on a line.

## Related projects

Other MCP linters cover ground this one does not; they work well alongside it.

- [mcplint](https://github.com/antonsoo/mcplint): token cost, naming rules, word-overlap
  near-duplicates, hidden instructions.
- [lintlang](https://github.com/Katerina637/lintlang): static checks for tool
  descriptions, schemas, and agent prompts.
- [mcp-lint](https://github.com/hahahahahahahahah6/mcp-lint): per-tool design scores.
- [mcp-doctor](https://github.com/vishalhabib99/mcp-doctor): server quality checks and
  fuzzing.
- [mcp-tef](https://github.com/StacklokLabs/mcp-tef): an evaluation service with
  embedding similarity matrices and LLM-based tool-selection tests.

## References

- Clark, E. V. (1987). The principle of contrast: A constraint on language acquisition.
  In B. MacWhinney (Ed.),
  [*Mechanisms of Language Acquisition*](https://www.routledge.com/Mechanisms-of-Language-Acquisition-The-20th-Annual-Carnegie-Mellon-Symposium-on-Cognition/MacWhinney/p/book/9780898599732)
  (pp. 1-33). Lawrence Erlbaum Associates.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). False-positive reports with the two tool
definitions are the most useful contribution right now.

## License

[MIT](LICENSE)
