#!/usr/bin/env python
"""Drive a real vLLM through the proxy and check what actually came back.

    scripts/e2e_probe.py --proxy http://127.0.0.1:8010/v1 \
        --upstream http://127.0.0.1:8000/v1 --record /tmp/rollouts.jsonl

``selftest.py`` proves the proxy's logic against a stub. This proves the
*assumption underneath it*: that a real vLLM, asked for ``return_token_ids`` and
``logprobs``, answers with ``prompt_token_ids`` at the root of the response and
``token_ids`` on the choice -- the two fields harbor's own LLM layer reads
(``llms/lite_llm.py``: ``choice.provider_specific_fields["token_ids"]`` and
``response.prompt_token_ids``). If a vLLM release moved them, this fails loudly
here instead of producing empty rollouts during a training run.

It also checks the counterfactual: the same request sent *directly* to vLLM,
without those two parameters, comes back with no token ids at all. That is what
makes the proxy the thing adding value rather than a passthrough.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok   ' if condition else 'FAIL '} {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def post(base_url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "authorization": "Bearer unused"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    system = {"role": "system", "content": "You are a terse assistant."}
    user1 = {"role": "user", "content": "Reply with exactly: ok"}

    print("through the proxy")
    first = post(
        args.proxy,
        {"model": args.model, "messages": [system, user1], "max_tokens": 16, "temperature": 0},
    )
    content = first["choices"][0]["message"]["content"]
    check("got a completion", isinstance(content, str) and len(content) > 0, repr(content[:40]))
    check(
        "vLLM returned prompt_token_ids at the response root",
        isinstance(first.get("prompt_token_ids"), list) and first["prompt_token_ids"],
        f"n={len(first.get('prompt_token_ids') or [])}",
    )
    check(
        "vLLM returned token_ids on the choice",
        isinstance(first["choices"][0].get("token_ids"), list)
        and first["choices"][0]["token_ids"],
        f"n={len(first['choices'][0].get('token_ids') or [])}",
    )
    logprob_content = ((first["choices"][0].get("logprobs") or {}).get("content")) or []
    check("vLLM returned per-token logprobs", len(logprob_content) > 0, f"n={len(logprob_content)}")
    check(
        "one logprob per completion token",
        len(logprob_content) == len(first["choices"][0].get("token_ids") or []),
        f"{len(logprob_content)} vs {len(first['choices'][0].get('token_ids') or [])}",
    )

    # Turn 2 of the same conversation, so the proxy has to chain it.
    post(
        args.proxy,
        {
            "model": args.model,
            "messages": [
                system,
                user1,
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Now reply with exactly: done"},
            ],
            "max_tokens": 16,
            "temperature": 0,
        },
    )

    print("straight to vLLM, without the proxy's two parameters")
    direct = post(
        args.upstream,
        {"model": args.model, "messages": [system, user1], "max_tokens": 16, "temperature": 0},
    )
    check(
        "no token ids unless asked for",
        direct.get("prompt_token_ids") is None
        and direct["choices"][0].get("token_ids") is None,
        "so the proxy is what supplies them",
    )

    print("the recording")
    entries = [json.loads(line) for line in args.record.read_text().splitlines() if line.strip()]
    check("two turns recorded", len(entries) == 2, f"n={len(entries)}")
    if len(entries) == 2:
        check(
            "both turns in one rollout",
            entries[0]["rollout_id"] == entries[1]["rollout_id"],
            entries[0]["rollout_id"],
        )
        check("turn numbers are 1 then 2", [e["turn"] for e in entries] == [1, 2])
        check(
            "prompt grows with the conversation",
            len(entries[1]["prompt_token_ids"]) > len(entries[0]["prompt_token_ids"]),
            f"{len(entries[0]['prompt_token_ids'])} -> {len(entries[1]['prompt_token_ids'])}",
        )
        check(
            "recorded ids match what the client saw",
            entries[0]["completion_token_ids"] == first["choices"][0]["token_ids"],
        )
        check(
            "logprobs recorded for every turn",
            all(len(e["logprobs"]) == len(e["completion_token_ids"]) for e in entries),
        )
        check("no fork flagged on a linear conversation", not any(e["forked"] for e in entries))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed against a real vLLM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
