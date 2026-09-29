# Draft: PR 1 — `fix(swesmith): don't truncate golden patches`

**Raw material; rewrite the prose in your own words before posting.** Harbor's
CONTRIBUTING puts the first and last draft of a PR description on the human. The code
blocks below are code and were written and verified by an agent, which is fine by that
rule; the sentences around them are not yours yet. Add the agent disclosure CONTRIBUTING
asks for.

- Branch: `mightma:swesmith-patch-strip-fix` → `harbor-framework/harbor:main`
- Head: `1c2102a`
- Diff: 3 files, +137 / −3

---

## What is broken

`SWESmithAdapter._write_solution()` substitutes the golden patch into `solve.sh` with
`task.patch.strip()`.

A blank context line in a unified diff is **a single space**. `strip()` deletes it, which
leaves the final hunk one line shorter than its `@@` header declares, and `git apply`
rejects the whole patch:

```
error: corrupt patch at line N
```

where `N` is one line past the end of the diff.

## Reproducing it

No Harbor needed — this is `git apply` and 20 lines of shell:

```bash
rm -rf /tmp/r && mkdir /tmp/r && cd /tmp/r
git init -q . && git config user.email a@b && git config user.name a
printf 'fixed\n\n' > f.txt          # the file ends with a blank line
git add -A && git commit -qm base
printf 'broken\n\n' > f.txt         # the injected bug; the blank line is context
patch=$(git diff)                   # the golden patch
                                    # leave the tree broken, as a task ships it

python3 - "$patch" <<'PY'
import subprocess, sys
patch = sys.argv[1]
print("last line of the diff is %r" % patch.splitlines()[-1])
for label, text in (("patch.strip('\\n')", patch.strip("\n") + "\n"),
                    ("patch.strip()     ", patch.strip() + "\n")):
    r = subprocess.run(["git", "apply", "-R", "-"], input=text, text=True, capture_output=True)
    state = "fixed" if open("f.txt").read().startswith("fixed") else "still broken"
    print(f"  git apply -R  {label} -> rc={r.returncode}  {state}  {r.stderr.strip()}")
    open("f.txt", "w").write("broken\n\n")
PY
```

```
last line of the diff is ' '
  git apply -R  patch.strip('\n') -> rc=0    fixed
  git apply -R  patch.strip()      -> rc=128  still broken  error: corrupt patch at line 8
```

## Why it is worth fixing rather than working around

**`solve.sh` applies the golden patch in reverse.** SWE-smith ships each task's repository
already broken, and the solution reverts the injected bug. So a patch that `git apply`
refuses does not produce a loud error in one task — it makes the **oracle score 0.0**, and
a repository whose sampled task happened to hit this looks entirely broken when only that
one patch was mangled. That is why the bug is easy to misattribute to the task images or
the test suites.

How common: counting patches in
[`SWE-bench/SWE-smith`](https://huggingface.co/datasets/SWE-bench/SWE-smith) whose final
line is whitespace-only —

```python
from datasets import load_dataset
ds = load_dataset("SWE-bench/SWE-smith", split="train")
bad = sum(1 for r in ds if (ls := r["patch"].splitlines()) and ls[-1].strip() == "")
print(bad, len(ds))          # 8646 59136
```

**8,646 of 59,136 patches (14.6%), across 216 of the dataset's 222 repositories.**

Worth being precise about the scope, since `strip()` looks harmless: it alters the end of
*every* patch, because every patch ends in a newline. Removing that newline is fine and
`strip("\n")` does it too. The damage is only where a whitespace-only line sits before it.

## The fix

Strip newlines only, in a named function so the reason has somewhere to live:

```python
def render_solve_script(template: str, patch: str) -> str:
    return template.replace("{patch}", patch.strip("\n"))
```

`strip("\n")` rather than no stripping at all, because the template's heredoc terminator
has to land on its own line.

## Verification

Five tests in `adapters/swesmith/tests/test_adapter.py`:

```bash
cd adapters/swesmith && uv run pytest tests/ -q
# 5 passed
```

**Revert the one-line fix and 3 of the 5 fail**, which is the part worth checking —
a regression test that passes either way would be decoration:

```
FAILED tests/test_adapter.py::test_render_preserves_trailing_blank_context_line
FAILED tests/test_adapter.py::test_rendered_patch_reverse_applies[]
FAILED tests/test_adapter.py::test_rendered_patch_reverse_applies[\n\n]
3 failed, 2 passed
```

`test_rendered_patch_reverse_applies` does not compare strings: it builds a real
repository and runs `git apply -R`, so it tests whether `solve.sh` would work rather than
whether the implementation matches my expectations.

Two notes for reviewers:

- **CI does not run these.** `pytest.yml` covers `tests/` and `packages/`, not `adapters/`,
  so the command above is the only way they execute today. Happy to wire them in
  separately if that is wanted — it felt out of scope here.
- The `[tool.pytest.ini_options]` added to `adapters/swesmith/pyproject.toml` exists
  because without it pytest inherits the repo-root config, whose `testpaths` excludes
  `adapters` and whose `asyncio_mode` needs `pytest-asyncio`. `programbench` is the only
  other adapter with tests and does not declare it; tell me if you would rather it did not.

Also checked, with the repo's own pinned ruff: `uv run ruff check .` → all checks passed,
`uv run ruff format --check .` → 1434 files already formatted. No `CHANGELOG.md` entry,
since CONTRIBUTING reserves it for major features and breaking changes.

## Wider context, deliberately not in this PR

Found while building an AgentCore Runtime sandbox provider
([#3446](https://github.com/harbor-framework/harbor/issues/3446)). This fix is independent
of that and stands on its own; I am not asking for them to be considered together.
