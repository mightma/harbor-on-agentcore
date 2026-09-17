# SWE-bench Verified · arm64

One directory per dataset, the same shape as Harbor's `adapters/<dataset>/`. This one
holds everything specific to SWE-bench Verified; `../shared/` holds what is not.

| File | What it is |
|---|---|
| `tasks.sh` | drives Harbor's `swebench` adapter CLI with `--arch arm64` |
| `probe_arm64_images.py` | which instances have a published `swebench/sweb.eval.arm64.*` image |
| `build_images.py` | builds the ones that do not, through swebench's own base → env → instance chain |
| `data/*.txt` | the instance lists, all of them outputs of the two scripts above |

## The shape of this dataset

**One image per instance, deliberately.** Instances of the same repository span years and
different dependency versions, so the coarsest grouping with an identical installed
environment is `env_image_key` — 40 groups for 500 instances — but *within* a group the
commit still changes, which makes the per-task delta a clone, a history scrub and an
editable install rather than a checkout. That is minutes on every trial forever against a
measured 3 s warm start, so this dataset does **not** use `../shared/task_sharing.py`.
The comparison is worked through in the part 1 README; SWE-smith is the case where
sharing wins.

## Coverage

**\[measured\]** 281 of 500 instances have a published arm64 image. Of the other 219,
**160 have now been built** and are in ECR, putting SWE-bench Verified at **441/500 (88%)**
on arm64. The 59 that do not build all have identified causes — 51 of them hard arm64
package-availability walls (scipy, `cdms2`, an ancient `setuptools`, conda `python=3.5`),
the rest dependency or upstream drift. The part 1 README has the table.

| `data/` file | Instances |
|---|---|
| `swebv-arm64-instances.txt` | 281 published |
| `swebv-arm64-train.txt` / `swebv-arm64-eval.txt` | 208 / 73, split by repository |
| `swebv-arm64-eval-verified.txt` | 70 — eval minus 3 whose oracle ceiling is 0 |
| `swebv-arm64-selfbuilt.txt` | **160 built here**, read back from ECR |
| `swebv-arm64-selfbuilt-gated.txt` | **130** of those gate-clean |

## Usage

```bash
uv run swebench/probe_arm64_images.py --out swebench/data/swebv-arm64-instances.txt
swebench/tasks.sh swebench/data/swebv-arm64-eval.txt "$HARBOR_DATASETS/swebv-arm64/eval"
shared/prepull_arm64.sh "$HARBOR_DATASETS/swebv-arm64/eval"

uv run swebench/build_images.py --list      # the 219 not published
uv run swebench/build_images.py --dry-run   # render the Dockerfiles, no docker needed
uv run swebench/build_images.py --instance-list <ids> --concurrency 3 \
    --push --prune-after-push
```

Everything else — why the arch override is needed, the timings, the task-facing tag, the
`--pin` lever for dependency drift, and what the gate found — is in the part 1 README and
in `build_images.py`'s own docstring.
