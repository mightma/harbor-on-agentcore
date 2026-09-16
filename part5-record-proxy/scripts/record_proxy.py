#!/usr/bin/env python
"""Record token ids and logprobs for a harness that calls the model itself.

    scripts/record_proxy.py --upstream http://127.0.0.1:8000/v1 --port 8010 \
        --record "$HARBOR_JOBS/rollouts.jsonl"

Why this exists
---------------
Step-wise RL needs per-turn ``prompt_token_ids``, ``completion_token_ids`` and
``logprobs``. Harbor's own LLM layer collects them (``llms/lite_llm.py``), so
``terminus-2`` is trainable -- but every SWE-specialised harness
(``mini-swe-agent``, ``swe-agent``, ``claude-code``) is an installed CLI that calls
the model with its own client, and nothing records those calls.

Harbor already decides *where* those clients send their traffic: it injects the
endpoint per harness (``mini_swe_agent.py`` -> ``OPENAI_BASE_URL`` /
``OPENAI_API_BASE``). Point that at this proxy and the recording happens at the API
boundary, which is a more trustworthy place to capture than inside the agent: what
is recorded is what the engine emitted.

The two parameters, and where they actually live
-----------------------------------------------
Harbor's LLM layer adds exactly two things when collecting rollout details::

    completion_kwargs["logprobs"] = True
    extra_body["return_token_ids"] = True

``extra_body`` is a *client-side* LiteLLM concept -- on the wire both are plain
top-level JSON fields. So this proxy sets ``logprobs`` and ``return_token_ids`` on
the request body, and reads back what vLLM returns in the same shape harbor reads:

    prompt_token_ids        body["prompt_token_ids"]                 (root)
    completion_token_ids    body["choices"][0]["token_ids"]
    logprobs                body["choices"][0]["logprobs"]["content"][*]["logprob"]

The response is relayed to the client **byte for byte**, so the harness sees a
completely ordinary OpenAI response and no agent needs changing.

Stitching turns into trajectories, without the client's help
------------------------------------------------------------
A proxy sees interleaved requests from many concurrent trials, so it has to group
them. Two mechanisms, in order:

1. **A session header.** ``mini-swe-agent`` sends Harbor's ``session_id`` as
   ``X-Session-ID`` when one is set (``mini_swe_agent.py``:
   ``model.model_kwargs.extra_headers.<name>``), which is what SkyRL supplies per
   rollout. That is the authoritative key.
2. **Prefix chaining.** Otherwise: turn *N*'s messages start with turn *N*-1's
   messages. A request that extends a known conversation continues it; one that
   extends nothing starts a new rollout.

Prefix chaining is also the honest detector for the cases the design says a proxy
*cannot* resolve. A request that extends a conversation somewhere other than its
tip is a **fork** -- a subagent, or a client that rewrote its own history -- and it
is recorded with ``forked: true`` in a separate rollout rather than silently
concatenated into one trajectory. ``RolloutDetail``'s own docstring concedes this
class ("agents with subagents, summarization, or other non-linear chat
histories"); what this proxy will not do is pretend.

What it refuses to do
---------------------
``stream: true`` is rejected by default. A streamed turn cannot be recorded by this
code path, and a silently unrecorded turn inside a trajectory is worse than a loud
failure -- it trains on a hole. ``--allow-unrecorded-stream`` relays it anyway and
counts it, for when you are evaluating rather than training.

Output
------
Append-only JSONL, one line per recorded turn. ``scripts/attach_rollouts.py``
aggregates those lines into ``RolloutDetail`` objects and attaches them to the
trials they belong to.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Headers that must not be forwarded: hop-by-hop, or ones the upstream recomputes.
_SKIP_REQUEST_HEADERS = {"host", "content-length", "connection", "accept-encoding"}
_SKIP_RESPONSE_HEADERS = {"content-length", "connection", "transfer-encoding"}


def _message_keys(messages: list[dict]) -> list[tuple[str, str]]:
    """A comparable fingerprint per message: (role, sha1 of the content)."""
    keys = []
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, str):
            # Multimodal or tool-call content: hash the canonical JSON instead.
            content = json.dumps(content, sort_keys=True, default=str)
        keys.append(
            (
                str(message.get("role", "")),
                hashlib.sha1(content.encode("utf-8", "replace")).hexdigest(),
            )
        )
    return keys


def _first_user(messages: list[dict]) -> str:
    for message in messages or []:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            return json.dumps(content, default=str)
    return ""


class Recorder:
    """Groups turns into rollouts and appends them to a JSONL file."""

    def __init__(self, record_path: Path, session_headers: list[str]):
        self.path = record_path
        self.session_headers = [h.lower() for h in session_headers]
        self._lock = threading.Lock()
        # rollout_id -> {"keys": [...], "turns": int}
        self._rollouts: dict[str, dict] = {}
        self.counters = {
            "recorded": 0,
            "forked": 0,
            "unrecorded_stream": 0,
            "missing_token_ids": 0,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def session_id_from(self, headers) -> str | None:
        for name in self.session_headers:
            value = headers.get(name)
            if value:
                return value.strip()
        return None

    def _place(self, session_id: str | None, keys: list[tuple[str, str]]) -> tuple[str, bool]:
        """Return (rollout_id, forked) for a request with these message keys."""
        if session_id:
            state = self._rollouts.get(session_id)
            if state is None:
                return session_id, False
            if keys[: len(state["keys"])] == state["keys"]:
                return session_id, False
            # Same session, but the history is not an extension of what we saw:
            # the client rewrote it (compaction) or this is a subagent.
            forks = sum(1 for k in self._rollouts if k.startswith(f"{session_id}#"))
            return f"{session_id}#{forks + 2}", True

        # Two ways an incoming request can relate to a known rollout:
        #
        #   extension  its whole tip is a prefix of these messages -> the next turn
        #   fork       they agree for a while and then diverge *before* the tip ->
        #              a subagent, or a client that rewrote its own history
        #
        # A fork has to share at least one user message to count as one: unrelated
        # trials share an identical system prompt, and that must not chain them.
        extend_id, extend_len = None, -1
        fork_id, fork_len = None, -1
        for rollout_id, state in self._rollouts.items():
            tip = state["keys"]
            common = 0
            for mine, theirs in zip(keys, tip):
                if mine != theirs:
                    break
                common += 1
            if common == len(tip):
                if common > extend_len:
                    extend_id, extend_len = rollout_id, common
            elif (
                common > fork_len
                and any(role == "user" for role, _ in keys[:common])
            ):
                fork_id, fork_len = rollout_id, common

        if extend_id is not None:
            return extend_id, False
        if fork_id is not None:
            root = fork_id.split("#", 1)[0]
            forks = sum(1 for k in self._rollouts if k.startswith(f"{root}#"))
            return f"{root}#{forks + 2}", True
        return f"r{len(self._rollouts) + 1}-{keys[0][1][:8] if keys else 'empty'}", False

    def record(self, request_body: dict, response_body: dict, headers) -> dict:
        messages = request_body.get("messages") or []
        keys = _message_keys(messages)
        session_id = self.session_id_from(headers)

        choice = (response_body.get("choices") or [{}])[0]
        completion_token_ids = choice.get("token_ids")
        prompt_token_ids = response_body.get("prompt_token_ids")
        logprob_content = ((choice.get("logprobs") or {}).get("content")) or []
        logprobs = [c["logprob"] for c in logprob_content if "logprob" in c]

        with self._lock:
            rollout_id, forked = self._place(session_id, keys)
            state = self._rollouts.setdefault(rollout_id, {"keys": [], "turns": 0})
            state["keys"] = keys
            state["turns"] += 1
            turn = state["turns"]
            if forked:
                self.counters["forked"] += 1
            if not completion_token_ids or prompt_token_ids is None:
                self.counters["missing_token_ids"] += 1
            self.counters["recorded"] += 1

            entry = {
                "rollout_id": rollout_id,
                "turn": turn,
                "forked": forked,
                "ts": time.time(),
                "model": request_body.get("model"),
                "session_id": session_id,
                "n_messages": len(messages),
                # Enough to match a rollout to a trial without storing the prompt.
                "first_user_sha256": hashlib.sha256(
                    _first_user(messages).encode("utf-8", "replace")
                ).hexdigest(),
                "first_user_head": _first_user(messages)[:200],
                "finish_reason": choice.get("finish_reason"),
                "prompt_token_ids": prompt_token_ids,
                "completion_token_ids": completion_token_ids,
                "logprobs": logprobs,
            }
            with self.path.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
        return entry


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "harbor-record-proxy"

    # Set on the server instance in serve().
    upstream: str
    recorder: Recorder
    allow_unrecorded_stream: bool

    def log_message(self, fmt, *args):  # noqa: A002 - stdlib signature
        # The per-turn line in _handle_chat is the useful log; this would double it.
        pass

    def _upstream_url(self, path: str) -> str:
        """Join the upstream base with the requested path without doubling /v1.

        Clients are configured with ``.../v1`` (that is what OPENAI_BASE_URL means)
        and then request ``/v1/chat/completions``, so a naive concatenation asks
        the server for ``/v1/v1/chat/completions``.
        """
        base = self.upstream.rstrip("/")
        if base.endswith("/v1") and path.startswith("/v1/"):
            path = path[len("/v1") :]
        return base + path

    def _forward(self, path: str, body: bytes | None) -> tuple[int, dict, bytes]:
        url = self._upstream_url(path)
        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _SKIP_REQUEST_HEADERS
        }
        request = urllib.request.Request(
            url, data=body, headers=headers, method=self.command
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as err:
            return err.code, dict(err.headers), err.read()
        except Exception as err:  # noqa: BLE001 - relay the failure as a 502
            payload = json.dumps(
                {"error": {"message": f"record-proxy upstream failure: {err}", "type": "proxy"}}
            ).encode()
            return 502, {"content-type": "application/json"}, payload

    def _respond(self, status: int, headers: dict, body: bytes) -> None:
        self.send_response(status)
        for key, value in headers.items():
            if key.lower() not in _SKIP_RESPONSE_HEADERS:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("content-length") or 0)
        return self.rfile.read(length) if length else b""

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        status, headers, body = self._forward(self.path, None)
        self._respond(status, headers, body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        raw = self._read_body()
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._respond(*self._forward(self.path, raw))
            return
        self._handle_chat(raw)

    def _handle_chat(self, raw: bytes) -> None:
        try:
            request_body = json.loads(raw or b"{}")
        except json.JSONDecodeError as err:
            self._respond(
                400,
                {"content-type": "application/json"},
                json.dumps({"error": {"message": f"unparseable JSON: {err}"}}).encode(),
            )
            return

        if request_body.get("stream"):
            if not self.allow_unrecorded_stream:
                self._respond(
                    400,
                    {"content-type": "application/json"},
                    json.dumps(
                        {
                            "error": {
                                "message": "record-proxy: stream=true cannot be "
                                "recorded; a silently unrecorded turn corrupts the "
                                "trajectory. Disable streaming in the harness, or "
                                "start the proxy with --allow-unrecorded-stream.",
                                "type": "proxy",
                            }
                        }
                    ).encode(),
                )
                return
            self.recorder.counters["unrecorded_stream"] += 1
            self._respond(*self._forward(self.path, raw))
            return

        # The two parameters harbor's own LLM layer adds. Set only if the caller
        # did not: an explicit client value wins, so this cannot silently change
        # what a caller asked for.
        request_body.setdefault("logprobs", True)
        request_body.setdefault("return_token_ids", True)
        forwarded = json.dumps(request_body).encode()

        status, headers, body = self._forward(self.path, forwarded)
        if status != 200:
            self._respond(status, headers, body)
            return

        try:
            response_body = json.loads(body)
            entry = self.recorder.record(request_body, response_body, self.headers)
            print(
                f"  turn {entry['turn']:>3} of {entry['rollout_id']}"
                f"{' FORK' if entry['forked'] else ''}"
                f"  prompt={len(entry['prompt_token_ids'] or [])}"
                f" completion={len(entry['completion_token_ids'] or [])}"
                f" logprobs={len(entry['logprobs'])}",
                flush=True,
            )
        except Exception as err:  # noqa: BLE001 - never fail the trial over recording
            print(f"  WARNING: recording failed ({err}); response relayed", flush=True)

        # Relay upstream's bytes untouched, including the extra fields we asked
        # for: the client sees a normal response plus fields it ignores.
        self._respond(status, headers, body)


def serve(
    upstream: str,
    port: int,
    record_path: Path,
    session_headers: list[str],
    allow_unrecorded_stream: bool,
    timeout_sec: float,
    host: str = "0.0.0.0",  # noqa: S104 - the sandbox has to reach it
) -> ThreadingHTTPServer:
    recorder = Recorder(record_path, session_headers)
    handler = type(
        "BoundHandler",
        (Handler,),
        {
            "upstream": upstream,
            "recorder": recorder,
            "allow_unrecorded_stream": allow_unrecorded_stream,
            "timeout_sec": timeout_sec,
        },
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.recorder = recorder  # ty: ignore[unresolved-attribute]
    return server


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--upstream",
        default="http://127.0.0.1:8000/v1",
        help="The OpenAI-compatible endpoint to forward to (your vLLM)",
    )
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument(
        "--record",
        type=Path,
        required=True,
        help="JSONL file to append recorded turns to",
    )
    parser.add_argument(
        "--session-header",
        action="append",
        default=None,
        help="Header carrying harbor's session id (repeatable). Default X-Session-ID, "
        "which is what mini-swe-agent sends when a session_id is set.",
    )
    parser.add_argument("--allow-unrecorded-stream", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    args = parser.parse_args()

    server = serve(
        args.upstream,
        args.port,
        args.record,
        args.session_header or ["X-Session-ID"],
        args.allow_unrecorded_stream,
        args.timeout_sec,
        args.host,
    )
    print(f"record-proxy on http://{args.host}:{args.port} -> {args.upstream}")
    print(f"recording to {args.record}")
    print("point the harness at this address with OPENAI_BASE_URL / OPENAI_API_BASE")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        counters = server.recorder.counters  # ty: ignore[unresolved-attribute]
        print(f"\nstopping: {counters}")
        if counters["missing_token_ids"]:
            print(
                "  WARNING: some turns came back without token ids -- the upstream "
                "ignored return_token_ids. Those turns cannot be trained on.",
                file=sys.stderr,
            )
        server.server_close()


if __name__ == "__main__":
    main()
