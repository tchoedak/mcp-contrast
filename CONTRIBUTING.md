# Contributing

Thanks for helping. Bug reports, false-positive reports, and pull requests are all welcome.

## Reporting a false positive or a miss

These are the most useful reports right now, because thresholds and the synonym list are
still being tuned. Use the
[false-positive template](https://github.com/tchoedak/mcp-contrast/issues/new?template=false-positive.yml)
and include:

- both tool definitions (name and description), or the server they come from
- the output of `mcp-contrast check ... --format json`
- why a model *can* (or *cannot*) tell the two apart

## Development

You need [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/tchoedak/mcp-contrast
cd mcp-contrast
uv sync
uv run pytest                 # the first run downloads the 30 MB embedding model
uv run pytest -m "not model"  # skip tests that need the model
uv run ruff format . && uv run ruff check . && uv run mypy
```

CI runs the same commands on Linux (Python 3.11 to 3.14), macOS, and Windows, plus a
self-test of the GitHub Action.

## Pull requests

- Keep each pull request to one change, with a test that fails without it.
- New rules need an entry in `docs/rules.md` with an example pair that fires and one that
  passes.
- Additions to the synonym list need a real-world tool pair that motivates them.
- Explain how you tested the change in the pull request description.

## Releasing

Maintainers bump `__version__` in `src/mcp_contrast/__init__.py`, merge, then push a tag
`vX.Y.Z`. The release workflow checks the tag matches the version, publishes to PyPI with
trusted publishing, and creates a GitHub release with generated notes. Move the major tag
(`v0`, `v1`) so `uses: tchoedak/mcp-contrast@v1` picks up the release.
