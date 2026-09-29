# Draft: PR 2 — `feat(swebench): generate runnable arm64 tasks`

**Raw material; rewrite the prose before posting**, same as PR 1. Code blocks were written
and executed by an agent. Add the AI-assistance disclosure.

- Branch: `mightma:swebench-arm64-tasks` → `harbor-framework/harbor:main`
- Head: `04e1a37`, two commits
- Diff: 6 files, +185 / −3

---

## Summary

The SWE-bench adapter cannot currently generate tasks for an arm64 sandbox. Two things
stand in the way, and fixing only the first produces tasks that look right and score 0, so
they are in one PR.

**1. The image architecture is decided by whichever machine runs the adapter.**
`make_test_spec` derives it from the host, so `get_image_names()` pins it with an
unconditional

```python
spec.instance_image_key.replace("arm64", "x86_64")
```

That is correct for the published amd64 images and leaves no way to reference
`swebench/sweb.eval.arm64.*`. Adds `--arch {x86_64,arm64}`, defaulting to the current
behaviour.

**2. `import numpy` dies with SIGILL inside those arm64 images**, on some hosts. The
prebuilt images ship an OpenBLAS that picks its kernel from the CPU's MIDR rather than
from the advertised feature bits. A vCPU that reports itself as Neoverse-V1 (part `0xd40`)
while masking SVE out of `/proc/cpuinfo` Features — Bedrock AgentCore Runtime does exactly
this — gets an SVE kernel and a crash. So arm64 tasks get

```dockerfile
ENV OPENBLAS_CORETYPE=ARMV8
```

appended to their generated Dockerfile.

## Why the second half is not a separate concern

The crash is quiet where it matters. The test command still exits, so the SWE-bench parser
finds no `PASSED` lines and reports 0 — **with the golden patch applied**. That is
indistinguishable from a hard task, and as an RL reward it is silently dead.

Measured on `astropy__astropy-12907` through AgentCore Runtime: the oracle scored **0.0
before this change and 1.0 after**. Shipping `--arch` alone would hand people a flag that
generates tasks nothing can solve.

`ARMV8` rather than `NEOVERSEN1`, which also works and is faster: `ARMV8` stays correct if
the host CPU changes.

## Changes

- `--arch {x86_64,arm64}` on the CLI, threaded through to `get_image_names()`.
- `get_image_names(arch=…)` replaces in whichever direction is needed, so the generated
  name is the same whether the adapter ran on x86 or arm64. Unknown values raise.
- arm64 Dockerfiles get the `OPENBLAS_CORETYPE` pin, with the reasoning in a comment in the
  generated file — whoever debugs a task should not have to find this PR.
- Tests, and the local pytest configuration they need.

Default behaviour is unchanged: omit `--arch` and the output is byte-identical to today's.

## Testing

`adapters/swebench` had no tests. Added five, offline — they construct a SWE-bench row by
hand rather than downloading the dataset:

```bash
cd adapters/swebench
uv run pytest tests/ -q
# 5 passed
```

The one worth looking at is `test_arch_selects_the_image_namespace`. It monkeypatches
`make_test_spec` to pretend the adapter is running on x86 *and* on arm64, and asserts the
generated name follows `--arch` in both cases. That asymmetry is what the old hardcoded
`replace("arm64", "x86_64")` got wrong, and it is invisible if you only ever test on one
architecture.

Reverting either half fails a test:

```
# --arch restored to the unconditional replace
FAILED tests/test_utils.py::test_arch_selects_the_image_namespace[arm64-arm64-x86_64]

# OPENBLAS_CORETYPE changed to NEOVERSEV1
FAILED tests/test_utils.py::test_arm64_suffix_pins_the_openblas_kernel
```

Also checked:

```bash
uv run ruff check .                      # all checks passed
uv run ruff format --check adapters/swebench/
cd adapters/swebench && uv sync --locked  # rc=0
```

Same two caveats as PR 1: CI's `pytest.yml` does not cover `adapters/`, so these run only
from the adapter directory; and `[tool.pytest.ini_options]` is there because the repo-root
config's `testpaths` excludes `adapters` and its `asyncio_mode` needs `pytest-asyncio`.
Happy to drop or restructure that if you would rather handle adapter test config centrally.

## Context

Found while building an AgentCore Runtime sandbox provider,
[#3446](https://github.com/harbor-framework/harbor/issues/3446). The `--arch` flag is
useful for any arm64-only sandbox, not just that one; the OpenBLAS pin was diagnosed there
but the same masked-SVE microVM shape is not unique to it.

## AI assistance disclosure

_(write your own, as in PR 1)_
