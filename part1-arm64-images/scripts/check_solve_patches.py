#!/usr/bin/env python3
"""Check every task's solution/solve.sh for a truncated golden patch.

A blank context line in a unified diff is a single space, so a `.strip()` on the
patch text silently deletes it and leaves the final hunk one line shorter than
its `@@` header declares. `git apply` rejects that with "corrupt patch at line
N", the oracle scores 0.0, and the repository looks broken when only one task's
patch was mangled. This verifies the arithmetic directly, then confirms a sample
against real `git apply` so the pure-Python check cannot drift from git.

    scripts/check_solve_patches.py $HARBOR_DATASETS/swesmith-arm64
"""

from __future__ import annotations

import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HEREDOC_OPEN = "cat > /testbed/bug_patch.diff << '__SOLUTION__'"
HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")


def extract_patch(script: str) -> str | None:
    lines = script.split("\n")
    try:
        start = lines.index(HEREDOC_OPEN) + 1
        end = lines.index("__SOLUTION__", start)
    except ValueError:
        return None
    return "\n".join(lines[start:end]) + "\n"


def hunk_defects(patch: str) -> list[str]:
    """Return one message per hunk whose body disagrees with its @@ header."""
    lines = patch.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    defects: list[str] = []
    heads = [(i, m) for i, line in enumerate(lines) if (m := HUNK_RE.match(line))]
    for pos, (head, match) in enumerate(heads):
        # A hunk body runs to the next hunk header, the next file's diff header,
        # or EOF -- whichever comes first.
        body_end = heads[pos + 1][0] if pos + 1 < len(heads) else len(lines)
        for j in range(head + 1, body_end):
            if lines[j].startswith("diff --git"):
                body_end = j
                break
        body = lines[head + 1 : body_end]

        want_old = int(match.group(1)) if match.group(1) is not None else 1
        want_new = int(match.group(2)) if match.group(2) is not None else 1

        ctx = sum(1 for b in body if b.startswith(" "))
        dels = sum(1 for b in body if b.startswith("-"))
        adds = sum(1 for b in body if b.startswith("+"))
        got_old, got_new = ctx + dels, ctx + adds
        if (got_old, got_new) != (want_old, want_new):
            defects.append(
                f"line {head + 1}: header says -{want_old} +{want_new}, "
                f"body has -{got_old} +{got_new}"
            )
    return defects


def git_says_corrupt(patch: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.diff"
        path.write_text(patch)
        result = subprocess.run(
            ["git", "apply", "--numstat", str(path)],
            cwd=tmp,
            capture_output=True,
            text=True,
            check=False,
        )
    return "corrupt patch" in result.stderr


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    scripts = sorted(root.glob("*/solution/solve.sh"))
    if not scripts:
        print(f"no solution/solve.sh under {root}", file=sys.stderr)
        return 2

    bad: list[tuple[str, str]] = []
    missing: list[str] = []
    patches: dict[str, str] = {}
    for script in scripts:
        name = script.parents[1].name
        patch = extract_patch(script.read_text())
        if patch is None:
            missing.append(name)
            continue
        patches[name] = patch
        for defect in hunk_defects(patch):
            bad.append((name, defect))

    print(f"tasks checked          : {len(scripts)}")
    print(f"heredoc not found      : {len(missing)}")
    print(f"tasks with bad hunks   : {len(set(n for n, _ in bad))}")
    for name, defect in bad[:10]:
        print(f"  {name}: {defect}")

    # The arithmetic above is a model of git's parser; sample-check it against the
    # real thing so a wrong model cannot pass this script silently.
    sample = random.Random(0).sample(sorted(patches), min(300, len(patches)))
    disagree = [
        n
        for n in sample
        if git_says_corrupt(patches[n]) is not bool(hunk_defects(patches[n]))
    ]
    print(f"sampled against git    : {len(sample)}, disagreements: {len(disagree)}")
    for name in disagree[:10]:
        print(f"  disagreement: {name}")

    return 1 if bad or disagree or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
