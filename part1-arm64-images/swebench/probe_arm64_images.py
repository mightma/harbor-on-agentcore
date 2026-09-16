#!/usr/bin/env python
"""Find which SWE-bench Verified instances have a published arm64 image.

    python swebench/probe_arm64_images.py --out run/swebv-arm64-instances.txt
    python swebench/probe_arm64_images.py --sizes            # also report layer sizes

AgentCore Runtime only accepts arm64, and the arm64 SWE-bench images cover a
subset of Verified. This is how run/swebv-arm64-instances.txt was produced.

Be aware of the cost: this issues one manifest request per instance, and Docker
Hub counts those against the anonymous pull limit of 100 per rolling hour per IP.
Probing all 500 spends the whole hour's budget, which then makes
`docker pull` (and therefore any Harbor build) fail with 429 for the next hour.
The result is committed for exactly that reason — only re-run this to refresh it.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REGISTRY = "https://registry-1.docker.io/v2"
AUTH = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
    ]
)


def repo_for(instance_id: str, arch: str = "arm64") -> str:
    """swebench names images sweb.eval.<arch>.<instance_id>, `__` spelled `_1776_`."""
    return f"swebench/sweb.eval.{arch}.{instance_id.lower().replace('__', '_1776_', 1)}"


def _token(repo: str) -> str:
    with urllib.request.urlopen(AUTH.format(repo=repo), timeout=30) as response:
        return json.load(response)["token"]


def probe(instance_id: str, want_size: bool) -> tuple[str, bool, int]:
    repo = repo_for(instance_id)
    try:
        request = urllib.request.Request(
            f"{REGISTRY}/{repo}/manifests/latest",
            method="GET" if want_size else "HEAD",
        )
        request.add_header("Authorization", f"Bearer {_token(repo)}")
        request.add_header("Accept", ACCEPT)
        with urllib.request.urlopen(request, timeout=30) as response:
            if not want_size:
                return instance_id, response.status == 200, -1
            manifest = json.load(response)
            size = sum(layer["size"] for layer in manifest.get("layers", []))
            return instance_id, True, size
    except urllib.error.HTTPError as err:
        if err.code == 429:
            raise SystemExit(
                "Docker Hub returned 429: the anonymous pull budget for this hour "
                "is spent. Wait for the window to roll over before retrying."
            ) from err
        return instance_id, False, -1
    except Exception:  # noqa: BLE001 - a probe failure is a "no", not a crash
        return instance_id, False, -1


def verified_instance_ids() -> list[str]:
    import pandas as pd

    url = (
        "https://huggingface.co/api/datasets/princeton-nlp/"
        "SWE-bench_Verified/parquet/default/test/0.parquet"
    )
    frame = pd.read_parquet(url)
    return frame["instance_id"].tolist(), dict(
        zip(frame["instance_id"], frame["repo"], strict=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--sizes",
        action="store_true",
        help="Fetch manifests instead of just checking existence, and report "
        "compressed sizes against AgentCore's 2048 MB image quota",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    instance_ids, repo_of = verified_instance_ids()
    print(f"probing {len(instance_ids)} SWE-bench Verified instances for arm64 images")

    available: list[str] = []
    sizes: dict[str, int] = {}
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        for instance_id, ok, size in pool.map(
            lambda i: probe(i, args.sizes), instance_ids
        ):
            if ok:
                available.append(instance_id)
                if size > 0:
                    sizes[instance_id] = size

    available.sort()
    print(f"arm64 available: {len(available)} / {len(instance_ids)}")
    print("by repo:", dict(Counter(repo_of[i] for i in available).most_common()))

    if sizes:
        values = sorted(sizes.values())
        over = [i for i, s in sizes.items() if s > 2048e6]
        print(
            f"compressed size: min {min(values) / 1e6:.0f} MB, "
            f"median {statistics.median(values) / 1e6:.0f} MB, "
            f"max {max(values) / 1e6:.0f} MB"
        )
        print(f"over the 2048 MB quota: {len(over)} {sorted(over)}")

    if args.out:
        args.out.write_text("\n".join(available) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
