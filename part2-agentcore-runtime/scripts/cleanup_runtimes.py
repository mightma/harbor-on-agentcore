#!/usr/bin/env python
"""Delete the AgentCore runtimes Harbor deployed (names starting with ``hb_``).

Harbor deletes a task's runtime when its last trial finishes, unless the run
passed ``--ek delete_runtime=false`` to keep them warm between runs. This removes
the leftovers.

    python part2-agentcore-runtime/scripts/cleanup_runtimes.py --list
    python part2-agentcore-runtime/scripts/cleanup_runtimes.py --delete
"""

from __future__ import annotations

import argparse
import os

import boto3

PREFIX = "hb_"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--region", default=os.environ.get("AWS_REGION", "us-west-2")
    )
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--delete", action="store_true", help="actually delete")
    parser.add_argument("--list", action="store_true", help="list only (default)")
    args = parser.parse_args()

    control = boto3.client("bedrock-agentcore-control", region_name=args.region)
    runtimes: list[dict] = []
    kwargs: dict = {"maxResults": 100}
    while True:
        response = control.list_agent_runtimes(**kwargs)
        runtimes.extend(response.get("agentRuntimes", []))
        token = response.get("nextToken")
        if not token:
            break
        kwargs["nextToken"] = token

    mine = [r for r in runtimes if r["agentRuntimeName"].startswith(args.prefix)]
    for runtime in mine:
        print(f"{runtime['agentRuntimeName']:50} {runtime['status']}")
    print(f"{len(mine)} runtime(s) with prefix {args.prefix!r} of {len(runtimes)} total")

    if not args.delete:
        return 0
    for runtime in mine:
        if runtime["status"] == "DELETING":
            continue
        control.delete_agent_runtime(agentRuntimeId=runtime["agentRuntimeId"])
        print(f"deleting {runtime['agentRuntimeName']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
