## Summary

The SWE-bench adapter currently cannot reliably generate tasks for arm64 sandboxes. This PR addresses the two issues involved:

1. Image names are always rewritten to the `x86_64` namespace.
2. NumPy can crash with `SIGILL` inside the generated arm64 images on hosts where OpenBLAS incorrectly selects an SVE kernel.

The adapter now accepts:

```text
--arch {x86_64,arm64}
```

The default remains `x86_64`, so existing behavior is unchanged.

## Why both changes are included

`make_test_spec()` derives the image architecture from the machine running the adapter. `get_image_names()` then currently forces the result back to x86 with:

```python
spec.instance_image_key.replace("arm64", "x86_64")
```

This prevents users from selecting the published `swebench/sweb.eval.arm64.*` images. The rewrite itself is deliberate — it is what keeps the generated name the same no matter which machine ran the adapter — so this change keeps that property and applies it to both values rather than removing it.

Adding `--arch` fixes image selection, but the resulting tasks can still fail on some arm64 runtimes. The prebuilt images contain an OpenBLAS version that selects its kernel using the CPU’s MIDR. On a vCPU reported as Neoverse V1 while SVE is masked from the advertised feature set, OpenBLAS selects an unsupported SVE kernel and `import numpy` exits with `SIGILL`.

To avoid this, generated arm64 Dockerfiles now include:

```dockerfile
ENV OPENBLAS_CORETYPE=ARMV8
```

`ARMV8` is used instead of `NEOVERSEN1` because it remains valid if the underlying host CPU changes.

This failure is otherwise easy to miss: the test process exits without producing any `PASSED` lines, so the SWE-bench parser reports a score of 0 even when the golden patch was applied. With `astropy__astropy-12907` on AgentCore Runtime, the oracle score changed from 0.0 to 1.0 after applying the OpenBLAS pin.

## Changes

- Add `--arch {x86_64,arm64}` to the SWE-bench adapter CLI.
- Pass the selected architecture through to `get_image_names()`.
- Normalize the image namespace in either direction, independent of the host running the adapter.
- Reject unsupported architecture values.
- Add `OPENBLAS_CORETYPE=ARMV8` to generated arm64 Dockerfiles.
- Refuse `--arch arm64` over the whole dataset, since published arm64 coverage is partial; the caller passes `--task-ids`/`--instance-id`, or `--allow-unpublished-images` if they build the missing images themselves.
- Add offline tests for architecture selection, Dockerfile generation, and that refusal.

When `--arch` is omitted, the generated x86 task output is unchanged.

## Testing

The tests construct SWE-bench rows locally and do not download the dataset:

```bash
cd adapters/swebench
uv run pytest tests/ -q
# 6 passed
```

The architecture test simulates the adapter running on both x86 and arm64 hosts and verifies that the generated image namespace follows `--arch` regardless of the host, for both values of the flag. That is the property the unconditional rewrite already provided for x86, and the half most easily lost when making it configurable.

Regression coverage includes all three parts of the change: restoring the unconditional x86 replacement, removing the arm64 OpenBLAS pin, or dropping the whole-dataset refusal each causes the corresponding test to fail.

Also checked with:

```bash
uv run ruff check .
uv run ruff format --check adapters/swebench/
cd adapters/swebench && uv sync --locked
```

These tests are not currently included in the repository’s main pytest workflow because it does not cover `adapters/`. The local pytest configuration prevents them from inheriting the repository-level `testpaths`, which excludes this directory. I can remove or restructure that configuration if adapter tests should be handled centrally instead.

## Context

I found these issues while developing the AgentCore Runtime environment provider discussed in #3446. The architecture option is useful for any arm64-only sandbox, and the masked-SVE CPU configuration is not specific to AgentCore.

## AI assistance disclosure

I used an AI coding agent to help implement and test this change. I reviewed the resulting code and wrote the final PR description.