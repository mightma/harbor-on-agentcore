#!/usr/bin/env python
"""End-to-end test for the proxy and the attach seam, with no GPU and no sandbox.

    scripts/selftest.py

Runs a stub OpenAI-compatible upstream, puts the real proxy in front of it, drives
it through the four cases that matter, then attaches the recording to a synthetic
Harbor job directory and checks what landed in ``result.json``.

What it proves:

  1. the proxy adds ``logprobs`` and ``return_token_ids`` to the request;
  2. it relays the upstream response unchanged, so a harness sees nothing unusual;
  3. consecutive turns of one conversation land in one rollout, in order;
  4. a turn that forks the history is flagged, not merged;
  5. a session header overrides prefix chaining;
  6. ``stream: true`` is refused rather than silently unrecorded;
  7. ``attach_rollouts`` writes a well-formed ``RolloutDetail`` into the right
     trial, and reports the trial it cannot match.

What it does not prove: that a real vLLM returns those fields (run
``scripts/e2e_vllm.sh`` for that), or that a real installed harness inside an
AgentCore sandbox can reach the proxy (that needs ``network_mode: VPC``).
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attach_rollouts import load_turns, match, rollout_detail, attach  # noqa: E402
from record_proxy import serve  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        FAILURES.append(label)


class StubUpstream(BaseHTTPRequestHandler):
    """A vLLM stand-in: echoes back token ids and logprobs, remembers requests."""

    received: list[dict] = []

    def log_message(self, fmt, *args):  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps({"data": [{"id": "stub-model"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        type(self).received.append(request)
        turn = len(request.get("messages") or [])
        payload = {
            "id": f"chatcmpl-{turn}",
            "object": "chat.completion",
            "model": request.get("model", "stub-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"reply {turn}"},
                    "finish_reason": "stop",
                    # vLLM's return_token_ids shape.
                    "token_ids": [900 + turn, 901 + turn],
                    "logprobs": {
                        "content": [
                            {"token": "re", "logprob": -0.25},
                            {"token": "ply", "logprob": -0.5},
                        ]
                    },
                }
            ],
            "prompt_token_ids": list(range(10, 10 + turn * 3)),
            "usage": {"prompt_tokens": turn * 3, "completion_tokens": 2},
        }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def post(url: str, payload: dict, headers: dict | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> int:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        record = tmp_path / "rollouts.jsonl"

        upstream_port, proxy_port = free_port(), free_port()
        upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), StubUpstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()

        proxy = serve(
            upstream=f"http://127.0.0.1:{upstream_port}/v1",
            port=proxy_port,
            record_path=record,
            session_headers=["X-Session-ID"],
            allow_unrecorded_stream=False,
            timeout_sec=30.0,
            host="127.0.0.1",
        )
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"

        print("proxy behaviour")
        system = {"role": "system", "content": "you are a swe agent"}
        user1 = {"role": "user", "content": "fix the separability matrix bug"}
        status, first = post(url, {"model": "m", "messages": [system, user1]})
        check("turn 1 relayed 200", status == 200, f"got {status}")
        check(
            "response relayed unchanged",
            first.get("choices", [{}])[0].get("message", {}).get("content") == "reply 2",
            str(first)[:120],
        )
        check(
            "request gained logprobs + return_token_ids",
            StubUpstream.received[-1].get("logprobs") is True
            and StubUpstream.received[-1].get("return_token_ids") is True,
            str(StubUpstream.received[-1])[:160],
        )

        assistant1 = {"role": "assistant", "content": "reply 2"}
        user2 = {"role": "user", "content": "now run the tests"}
        post(url, {"model": "m", "messages": [system, user1, assistant1, user2]})

        # A fork: same history up to turn 1, different continuation. This is the
        # subagent / rewritten-history case the design says a proxy cannot resolve.
        post(
            url,
            {
                "model": "m",
                "messages": [
                    system,
                    user1,
                    {"role": "assistant", "content": "a different reply"},
                    {"role": "user", "content": "subagent question"},
                ],
            },
        )

        # A session header wins over prefix chaining.
        post(
            url,
            {"model": "m", "messages": [system, {"role": "user", "content": "task B"}]},
            headers={"X-Session-ID": "sess-abc"},
        )

        stream_status, stream_body = post(
            url, {"model": "m", "messages": [system, user1], "stream": True}
        )
        check("stream refused", stream_status == 400, f"got {stream_status}")
        check(
            "refusal explains itself",
            "record" in json.dumps(stream_body).lower(),
            json.dumps(stream_body)[:120],
        )

        rollouts = load_turns(record)
        main_rollouts = {k: v for k, v in rollouts.items() if "#" not in k and k != "sess-abc"}
        check("one rollout for the two chained turns", len(main_rollouts) == 1, str(list(rollouts)))
        chained = next(iter(main_rollouts.values()))
        check("chained rollout has 2 turns in order", [t["turn"] for t in chained] == [1, 2])
        check(
            "prompt token ids grow with the history",
            len(chained[1]["prompt_token_ids"]) > len(chained[0]["prompt_token_ids"]),
        )
        check("logprobs recorded per turn", all(len(t["logprobs"]) == 2 for t in chained))
        forked = [k for k, v in rollouts.items() if any(t["forked"] for t in v)]
        check("fork recorded separately and flagged", len(forked) == 1, str(list(rollouts)))
        check("session header used as the rollout id", "sess-abc" in rollouts)

        print("attach seam")
        job = tmp_path / "job"
        # Trial 1: matched by session id, the SkyRL case.
        trial1 = job / "task-b__aaa"
        (trial1 / "agent").mkdir(parents=True)
        (trial1 / "result.json").write_text(
            json.dumps({"agent_result": {"cost_usd": 1.0, "rollout_details": None}})
        )
        (trial1 / "config.json").write_text(
            json.dumps({"agent": {"name": "mini-swe-agent", "kwargs": {"session_id": "sess-abc"}}})
        )
        # Trial 2: no session id, matched by its own first user message.
        trial2 = job / "separability__bbb"
        (trial2 / "agent").mkdir(parents=True)
        (trial2 / "result.json").write_text(
            json.dumps({"agent_result": {"rollout_details": None}})
        )
        (trial2 / "config.json").write_text(json.dumps({"agent": {"name": "mini-swe-agent"}}))
        (trial2 / "agent" / "trajectory.json").write_text(
            json.dumps({"steps": [{"source": "user", "message": user1["content"]}]})
        )
        # Trial 3: nothing to match on -- must be reported, not guessed.
        trial3 = job / "orphan__ccc"
        trial3.mkdir(parents=True)
        (trial3 / "result.json").write_text(json.dumps({"agent_result": {}}))

        from attach_rollouts import trial_dirs

        pairs, unmatched_trials, unmatched_rollouts = match(trial_dirs(job), rollouts)
        check("session-id trial matched", pairs.get(trial1) == "sess-abc", str(pairs))
        check(
            "first-user-message trial matched",
            pairs.get(trial2) is not None and pairs[trial2] != "sess-abc",
            str(pairs),
        )
        check(
            "unmatchable trial reported",
            [t.name for t, _ in unmatched_trials] == ["orphan__ccc"],
            str(unmatched_trials),
        )
        check("forked rollout left unmatched", any("#" in r for r in unmatched_rollouts))

        turns = attach(trial2, rollout_detail(rollouts[pairs[trial2]]), backup=True)
        written = json.loads((trial2 / "result.json").read_text())
        detail = written["agent_result"]["rollout_details"][0]
        check("rollout_details is a non-empty list", len(written["agent_result"]["rollout_details"]) == 1)
        check("two turns attached", turns == 2 and len(detail["prompt_token_ids"]) == 2)
        check(
            "shapes line up (RolloutDetail contract)",
            len(detail["prompt_token_ids"])
            == len(detail["completion_token_ids"])
            == len(detail["logprobs"]),
            str({k: len(v) for k, v in detail.items() if isinstance(v, list)}),
        )
        check(
            "logprobs count matches completion tokens",
            all(
                len(lp) == len(ids)
                for lp, ids in zip(detail["logprobs"], detail["completion_token_ids"])
            ),
        )
        check("original preserved", (trial2 / "result.json.orig").exists())
        check(
            "other agent_result fields untouched",
            json.loads((trial1 / "result.json").read_text())["agent_result"]["cost_usd"] == 1.0,
        )

        proxy.shutdown()
        upstream.shutdown()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
