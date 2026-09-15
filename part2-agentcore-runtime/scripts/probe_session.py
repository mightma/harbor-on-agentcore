#!/usr/bin/env python
"""Probe a live AgentCore Runtime sandbox and print what it actually looks like.

Answers the questions you need answered before trusting a benchmark run on this
substrate: which architecture, which user, what working directory a command
starts in, how much CPU/memory/disk a session gets, whether the container has an
AWS identity of its own, and how fast a file transfer is.

    python run/probe_sandbox.py <runtime-name-or-arn>

Any runtime deployed by the agentcore environment works; the sandbox is created
and torn down by this script.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

import boto3
from botocore.config import Config

from harbor.environments.agentcore.client import AgentCoreSandbox, new_session_id

REGION = os.environ.get("AWS_REGION", "us-west-2")

PROBES: list[tuple[str, str]] = [
    ("architecture", "uname -m; uname -r"),
    ("working directory", "pwd"),
    ("user", "id"),
    ("pid 1", "tr '\\0' ' ' < /proc/1/cmdline; echo"),
    ("cpus", "nproc; grep -c ^processor /proc/cpuinfo"),
    ("memory", "free -m | head -2"),
    ("disk", "df -h / /tmp | tail -2"),
    ("cgroup limits", "cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max 2>&1"),
    ("aws env", "env | grep -i '^aws' | sed 's/=.*/=<set>/' || echo none"),
    (
        "container credentials",
        "curl -s --max-time 5 "
        '"${AWS_CONTAINER_CREDENTIALS_FULL_URI:-http://169.254.170.2$AWS_CONTAINER_CREDENTIALS_RELATIVE_URI}" '
        "| head -c 300; echo",
    ),
    ("imds", "curl -s --max-time 3 http://169.254.169.254/latest/meta-data/ | head -c 200; echo"),
    ("egress", "curl -s -o /dev/null -w '%{http_code}\\n' --max-time 10 https://registry.npmjs.org/"),
    ("state persists", "echo marker > /tmp/probe-state"),
    ("state readback", "cat /tmp/probe-state"),
    ("process persists", "nohup sleep 300 >/dev/null 2>&1 & sleep 1; pgrep -c sleep"),
]


async def main(target: str) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = boto3.client(
        "bedrock-agentcore",
        region_name=REGION,
        config=Config(read_timeout=900, retries={"max_attempts": 5, "mode": "adaptive"}),
    )
    arn = target
    if not target.startswith("arn:"):
        control = boto3.client("bedrock-agentcore-control", region_name=REGION)
        matches = [
            runtime
            for runtime in control.list_agent_runtimes(maxResults=100)["agentRuntimes"]
            if runtime["agentRuntimeName"] == target
        ]
        if not matches:
            print(f"no runtime named {target}", file=sys.stderr)
            return 1
        arn = matches[0]["agentRuntimeArn"]

    sandbox = AgentCoreSandbox(
        data_client=client, runtime_arn=arn, session_id=new_session_id("probe")
    )
    started = time.monotonic()
    await sandbox.start()
    print(f"session ready in {time.monotonic() - started:.1f}s: {sandbox.session_id}\n")
    try:
        for label, command in PROBES:
            begin = time.monotonic()
            result = await sandbox.exec(command, timeout_sec=60)
            output = (result.stdout or result.stderr or "").strip()
            print(f"[{label}] rc={result.exit_code} ({time.monotonic() - begin:.2f}s)")
            for line in output.splitlines():
                print(f"    {line}")

        for size_mb in (1, 8):
            payload = b"x" * (size_mb << 20)
            begin = time.monotonic()
            await sandbox.write_file(f"/tmp/probe-{size_mb}m", payload)
            up = time.monotonic() - begin
            begin = time.monotonic()
            back = await sandbox.read_file(f"/tmp/probe-{size_mb}m")
            down = time.monotonic() - begin
            ok = back == payload
            print(f"[transfer {size_mb}MB] up {up:.2f}s down {down:.2f}s roundtrip_ok={ok}")
    finally:
        await sandbox.stop()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
