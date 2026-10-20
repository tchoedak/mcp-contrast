"""A tiny stdio MCP server for tests. Behavior is controlled by argv flags.

--page-size N   paginate tools/list
--noisy         print a non-JSON log line to stdout before answering
--ping          send a ping request to the client before answering tools/list
--crash         exit before answering initialize
--hang          never answer tools/list
--error         answer tools/list with a JSON-RPC error
"""

import json
import sys

TOOLS = [
    {"name": "get_user", "description": "Get a user by id.", "inputSchema": {"type": "object"}},
    {"name": "fetch_user", "description": "Fetch a user by id.", "inputSchema": {"type": "object"}},
    {"name": "delete_user", "description": "Delete a user permanently."},
]


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main():
    args = sys.argv[1:]
    page_size = int(args[args.index("--page-size") + 1]) if "--page-size" in args else len(TOOLS)
    pinged = False
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if "id" not in message or method is None:
            continue
        if method == "initialize":
            if "--crash" in args:
                print("boom: missing API key", file=sys.stderr)
                sys.exit(3)
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test", "version": "0"},
            }
            send({"jsonrpc": "2.0", "id": message["id"], "result": result})
        elif method == "tools/list":
            if "--hang" in args:
                continue
            if "--error" in args:
                error = {"code": -32603, "message": "listing disabled"}
                send({"jsonrpc": "2.0", "id": message["id"], "error": error})
                continue
            if "--noisy" in args:
                print("starting up...", flush=True)
            if "--ping" in args and not pinged:
                pinged = True
                send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
                reply = json.loads(sys.stdin.readline())
                assert reply == {"jsonrpc": "2.0", "id": "srv-1", "result": {}}, reply
            start = int(message.get("params", {}).get("cursor") or 0)
            page = TOOLS[start : start + page_size]
            result = {"tools": page}
            if start + page_size < len(TOOLS):
                result["nextCursor"] = str(start + page_size)
            send({"jsonrpc": "2.0", "id": message["id"], "result": result})


if __name__ == "__main__":
    main()
