## Summary

`SWESmithAdapter._write_solution()` in
`adapters/swesmith/src/swesmith_adapter/adapter.py` inserts the golden patch into
`solve.sh` using `task.patch.strip()`.

A blank context line in a unified diff is represented by a single space. When it appears
at the end of a patch, `strip()` removes that space, leaving the final hunk shorter than
its header declares. `git apply -R` then rejects the patch with:

```text
error: corrupt patch at line N
```

This PR changes the rendering logic to strip newline characters only:

```python
patch.strip("\n")
```

This preserves meaningful whitespace while still keeping the heredoc terminator on its
own line.

## Reproduction

A minimal example showing how a trailing blank context line is encoded. The blank line
has to survive as *context*, so this needs a real change rather than a new file:

```bash
rm -rf /tmp/r && mkdir /tmp/r && cd /tmp/r
git init -q . && git config user.email a@b && git config user.name a
printf 'a\n\n' > f && git add f && git commit -qm base
printf 'b\n\n' > f                  # the change; the blank line stays as context
git diff | tail -1 | xxd | head -1
```

```text
00000000: 200a                                      .
```

The final line contains `20 0a`: a space followed by a newline. `strip()` removes both,
while `strip("\n")` preserves the space. Applying the two renderings in reverse, the way
`solve.sh` does:

```text
strip('\n') -> rc=0
strip()     -> rc=128   error: corrupt patch at line 8
```

The issue affects 8,646 of the 59,136 patches in the current SWE-smith dataset:

```python
from datasets import load_dataset

ds = load_dataset("SWE-bench/SWE-smith", split="train")
print(
    sum(
        1
        for row in ds
        if (lines := row["patch"].splitlines())
        and lines[-1].strip() == ""
    ),
    len(ds),
)
# 8646 59136
```

For an affected task, the oracle solution cannot apply the golden patch and receives a
reward of 0. The failure therefore reflects a broken solution path rather than the
difficulty of the task.

## Changes

- Preserve trailing whitespace-only context lines when rendering `solve.sh`.
- Extract the rendering logic into a small helper function.
- Add regression tests that apply the rendered patch to a real Git repository.
- Add local pytest configuration for the SWE-smith adapter tests.

The pytest configuration prevents these tests from inheriting the repository-level
`testpaths`, which does not include `adapters/`. I can remove or restructure this part if
the maintainers would prefer to handle adapter test configuration separately.

## Testing

```bash
cd adapters/swesmith
uv run pytest tests/ -q
# 5 passed
```

Reverting the one-line fix causes 3 of the 5 tests to fail.

Also checked with:

```bash
uv run ruff check .
uv run ruff format --check .
```

The adapter tests are not currently included in the repository’s main pytest workflow,
so they need to be run from `adapters/swesmith`.

## Context

I found this while working on the AgentCore Runtime environment provider discussed in
#3446, but this bug and its fix are independent of that integration.

## AI assistance disclosure

I used an AI coding agent to help implement and test this change. I reviewed the resulting
code and wrote the final PR description.
