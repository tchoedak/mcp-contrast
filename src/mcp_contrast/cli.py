"""Command line interface.

Exit codes: 0 clean, 1 findings at or above --fail-on, 2 usage or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from mcp_contrast import __version__, baseline, config, report
from mcp_contrast.analyze import analyze
from mcp_contrast.client import SourceError
from mcp_contrast.embed import EmbedderError, make_embedder
from mcp_contrast.model import Report, Server, Severity
from mcp_contrast.rules import RULES
from mcp_contrast.sources import Connect, from_file, from_http, from_stdio, snapshot

EXIT_OK, EXIT_FINDINGS, EXIT_ERROR = 0, 1, 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-contrast",
        description="Lint MCP servers for tools a model cannot tell apart.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    def add_sources(sub: argparse.ArgumentParser) -> None:
        group = sub.add_argument_group("sources")
        group.add_argument(
            "files",
            nargs="*",
            metavar="FILE",
            help="tools/list JSON snapshot, or a client config (.mcp.json) whose servers to start",
        )
        group.add_argument(
            "--stdio",
            action="append",
            default=[],
            metavar="CMD",
            help="start a stdio server, e.g. --stdio 'npx -y @modelcontextprotocol/server-memory'",
        )
        group.add_argument(
            "--http",
            action="append",
            default=[],
            metavar="URL",
            help="connect to a Streamable HTTP server",
        )
        group.add_argument(
            "--header",
            action="append",
            default=[],
            metavar="'NAME: VALUE'",
            help="HTTP header for --http and config servers (repeatable)",
        )
        group.add_argument(
            "--server", metavar="NAME", help="only use this server from a client config"
        )
        group.add_argument(
            "--timeout",
            type=float,
            default=30.0,
            metavar="SECONDS",
            help="per-request timeout for live servers (default: 30)",
        )

    check = commands.add_parser(
        "check",
        help="lint tool lists (default command)",
        description="Lint tool lists for pairs of tools a model cannot tell apart.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            examples:
              mcp-contrast check tools.json
              mcp-contrast check .mcp.json --format sarif -o mcp-contrast.sarif
              mcp-contrast check --stdio 'npx -y @modelcontextprotocol/server-everything'
              mcp-contrast check --http https://mcp.example.com --header "Authorization: Bearer $T"

            exit codes: 0 clean, 1 findings at or above --fail-on, 2 usage or input error
            """
        ),
    )
    add_sources(check)
    out = check.add_argument_group("output")
    out.add_argument(
        "--format",
        choices=["text", *report.FORMATS],
        default="text",
        help="output format (default: text)",
    )
    out.add_argument("-o", "--output", type=Path, metavar="PATH", help="write output to a file")
    out.add_argument(
        "--sarif",
        type=Path,
        metavar="PATH",
        help="also write SARIF to PATH, alongside the main --format output",
    )
    out.add_argument(
        "--show-clean",
        action="store_true",
        help="also list similar pairs whose difference is stated (text format)",
    )
    out.add_argument("--no-color", action="store_true", help="disable color in text output")
    analysis = check.add_argument_group("analysis")
    analysis.add_argument(
        "--fail-on",
        choices=config.FAIL_ON,
        help="lowest severity that fails the run (default: error)",
    )
    analysis.add_argument(
        "--threshold",
        type=float,
        metavar="0-1",
        help="description similarity that makes two tools a candidate pair",
    )
    analysis.add_argument(
        "--aligned-threshold",
        type=float,
        metavar="0-1",
        help="lower similarity bar for pairs whose names differ in one word",
    )
    analysis.add_argument(
        "--near-identical",
        type=float,
        metavar="0-1",
        help="similarity above which descriptions count as near-identical",
    )
    analysis.add_argument("--embedder", choices=["model2vec", "lexical"])
    analysis.add_argument("--model", help="model2vec model id or local directory")
    analysis.add_argument("--model-revision", help="Hugging Face revision to pin")
    analysis.add_argument(
        "--cross-server",
        action="store_true",
        default=None,
        help="compare tools across servers too, as a client loading all of them would see them",
    )
    analysis.add_argument("--config", type=Path, metavar="PATH", help="settings file")
    analysis.add_argument(
        "--baseline",
        type=Path,
        metavar="PATH",
        help="ignore findings recorded in this baseline file",
    )
    analysis.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the current findings into --baseline and exit 0",
    )

    snap = commands.add_parser(
        "snapshot",
        help="save a live server's tools/list to a file",
        description="Save one server's tools/list response, for committing and linting in CI.",
    )
    add_sources(snap)
    snap.add_argument("-o", "--output", type=Path, metavar="PATH", help="default: stdout")

    commands.add_parser("rules", help="list the rules", description="List the rules.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] not in {
        "check",
        "snapshot",
        "rules",
        "-h",
        "--help",
        "--version",
    }:
        args_list.insert(0, "check")
    parser = _parser()
    args = parser.parse_args(args_list)
    if args.command is None:
        parser.print_help()
        return EXIT_ERROR
    try:
        if args.command == "rules":
            return _rules()
        if args.command == "snapshot":
            return _snapshot(args)
        return _check(args)
    except (SourceError, config.ConfigError, baseline.BaselineError, EmbedderError) as exc:
        print(f"mcp-contrast: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        return 130


def _headers(raw: list[str]) -> dict[str, str]:
    headers = {}
    for item in raw:
        name, sep, value = item.partition(":")
        if not sep or not name.strip():
            raise SourceError(f"--header must look like 'Name: value', got {item!r}")
        headers[name.strip()] = value.strip()
    return headers


def _load_servers(args: argparse.Namespace) -> list[Server]:
    if not (args.files or args.stdio or args.http):
        raise SourceError("no sources given; pass a FILE, --stdio CMD, or --http URL")
    connect = Connect(timeout=args.timeout, headers=_headers(args.header))
    servers: list[Server] = []
    for name in args.files:
        servers.extend(from_file(Path(name), connect, only=args.server))
    servers.extend(from_stdio(command, connect) for command in args.stdio)
    servers.extend(from_http(url, connect) for url in args.http)
    if not servers:
        raise SourceError("no servers found in the given sources")
    _dedupe_names(servers)
    return servers


def _dedupe_names(servers: list[Server]) -> None:
    """Two sources with the same derived name would merge in reports; number them."""
    from dataclasses import replace

    seen: dict[str, int] = {}
    for index, server in enumerate(servers):
        count = seen.get(server.name, 0)
        seen[server.name] = count + 1
        if count:
            name = f"{server.name}-{count + 1}"
            tools = tuple(replace(t, server=name) for t in server.tools)
            servers[index] = replace(server, name=name, tools=tools)


def _check(args: argparse.Namespace) -> int:
    config_path = args.config or config.discover(Path.cwd())
    settings = config.load(config_path).with_overrides(
        embedder=args.embedder,
        model=args.model,
        model_revision=args.model_revision,
        threshold=_unit(args.threshold, "--threshold"),
        aligned=_unit(args.aligned_threshold, "--aligned-threshold"),
        near_identical=_unit(args.near_identical, "--near-identical"),
        fail_on=args.fail_on,
        cross_server=args.cross_server,
    )
    if args.write_baseline and args.baseline is None:
        raise config.ConfigError("--write-baseline needs --baseline PATH")

    servers = _load_servers(args)
    embedder = make_embedder(settings.embedder, settings.model, settings.model_revision)
    result = analyze(servers, settings, embedder)

    if args.write_baseline:
        written = baseline.write(args.baseline, result.findings)
        print(f"mcp-contrast: wrote {written} finding(s) to {args.baseline}", file=sys.stderr)
        return EXIT_OK
    if args.baseline is not None:
        known = baseline.load(args.baseline)
        kept = tuple(f for f in result.findings if f.fingerprint not in known)
        result = Report(
            result.servers, kept, result.clean, result.suppressed + len(result.findings) - len(kept)
        )

    if args.format == "text":
        stream_tty = args.output is None and sys.stdout.isatty()
        output = report.text(
            result,
            color=not args.no_color and report.use_color(stream_tty),
            show_clean=args.show_clean,
        )
    else:
        output = report.FORMATS[args.format](result)
    _emit(output, args.output)
    if args.sarif is not None:
        _emit(report.sarif(result), args.sarif)
    return _exit_code(result, settings.fail_on)


def _unit(value: float | None, flag: str) -> float | None:
    if value is not None and not 0 < value <= 1:
        raise config.ConfigError(f"{flag} must be in (0, 1]")
    return value


def _exit_code(result: Report, fail_on: str) -> int:
    if fail_on == "never":
        return EXIT_OK
    floor = Severity(fail_on).rank
    return EXIT_FINDINGS if any(f.severity.rank >= floor for f in result.findings) else EXIT_OK


def _emit(output: str, path: Path | None) -> None:
    if path is None:
        sys.stdout.write(output)
    else:
        path.write_text(output, encoding="utf-8")


def _snapshot(args: argparse.Namespace) -> int:
    servers = _load_servers(args)
    if len(servers) != 1:
        names = ", ".join(s.name for s in servers)
        raise SourceError(f"snapshot needs exactly one server, got {len(servers)}: {names}")
    _emit(json.dumps(snapshot(servers[0]), indent=2, ensure_ascii=False) + "\n", args.output)
    if args.output is not None:
        count = len(servers[0].tools)
        print(f"mcp-contrast: saved {count} tool(s) to {args.output}", file=sys.stderr)
    return EXIT_OK


def _rules() -> int:
    for rule in RULES.values():
        print(f"{rule.id:<18} {rule.severity.value:<8} {rule.summary}")
    print(f"\ndetails: {RULES['name-collision'].help_uri.split('#')[0]}")
    return EXIT_OK
