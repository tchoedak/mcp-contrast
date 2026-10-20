# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub security advisories](https://github.com/tchoedak/mcp-contrast/security/advisories/new),
not in public issues. You should get a response within a week.

## What mcp-contrast executes

- `--stdio CMD` and client configs (`.mcp.json`) **start the commands they name**, with
  the environment variables they list. Only lint configs you would run yourself.
- Against any server, `mcp-contrast` sends `initialize` and `tools/list` and nothing else.
  It never calls `tools/call`.
- The default embedding model is downloaded from the Hugging Face Hub at a pinned revision.
  Analysis runs locally; tool definitions are never sent anywhere.
