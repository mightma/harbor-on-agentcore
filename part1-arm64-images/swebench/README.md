# SWE-bench Verified · arm64

One directory per dataset, the same shape as Harbor's `adapters/<dataset>/`. This one
holds everything specific to SWE-bench Verified; `../shared/` holds what is not.

| File | What it is |
|---|---|
| `tasks.sh` | drives Harbor's `swebench` adapter CLI with `--arch arm64` |
| `probe_arm64_images.py` | which instances have a published `swebench/sweb.eval.arm64.*` image |
| `build_images.py` | builds the ones that do not, through swebench's own base → env → instance chain |
| `make_lists.py` | derives the rest of `data/` from ECR and from gate jobs |
| `data/` | **empty in git.** Every list is an output; see the table in the part 1 README |

## The shape of this dataset

**It is a test set, not a training set.** Verified ships a single `test` split of 500
instances and this kit no longer carves a train half out of it; training data comes from
SWE-smith next door, which shares neither instances nor repositories. (The official
SWE-bench *does* have a 19,008-instance train split over 35 other repositories, and it is
unusable here for a more fundamental reason than missing images: **\[measured\]**
swebench's `MAP_REPO_VERSION_TO_SPECS` has environment specs for **0 of those 35 repos**,
so `make_test_spec` cannot even construct a spec, let alone build an image.)

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

| `data/` file (generated, not committed) | What it lists | Last measured |
|---|---|---|
| **`swebv-arm64-runnable.txt`** | has an arm64 image ACR can deploy — the benchmark, since Verified is never split | **414 / 500** |
| **`swebv-arm64-gated.txt`** | of those, oracle ceiling verified 1.0. Report from this one | **200** |
| `swebv-arm64-instances.txt` | has a *published* image | 281 |
| `swebv-arm64-selfbuilt.txt` | was built here and pushed to ECR | 160 |
| `swebv-arm64-selfbuilt-gated.txt` | `gated` ∩ `selfbuilt` | 130 |
| `swebv-arm64-undeployable.txt` | built, but CreateAgentRuntime refused it as over 2048 MB | 27 |

The "last measured" column is history, not content: the files are outputs, so what you
get depends on what you have built and gated. The numbers are what this kit measured, and
they are what the READMEs quote.

## Usage

In dependency order -- images first, because a task dir names its image in `FROM` and an
unresolvable `FROM` only fails once a trial is minutes in:

```bash
# 1. which instances already have a published image
docker login
uv run swebench/probe_arm64_images.py --out swebench/data/swebv-arm64-instances.txt

# 2. build the rest (inspect first; --dry-run needs no docker)
uv run swebench/build_images.py --list
uv run swebench/build_images.py --dry-run
uv run swebench/build_images.py --concurrency 8 --push       # add --prune-after-push on a small disk

# 3. derive what exists and what can run
uv run swebench/make_lists.py --selfbuilt --runnable

# 4. task dirs, then pull the published bases
swebench/tasks.sh swebench/data/swebv-arm64-runnable.txt "$HARBOR_DATASETS/swebv-arm64"
shared/prepull_arm64.sh "$HARBOR_DATASETS/swebv-arm64"
```

Part 2 gates them; `make_lists.py --from-gate <job>...` turns that into
`swebv-arm64-gated.txt`, and re-running `--runnable` afterwards drops anything the gate
found undeployable.

Everything else — why the arch override is needed, the timings, the task-facing tag, the
`--pin` lever for dependency drift, and what the gate found — is in the part 1 README and
in `build_images.py`'s own docstring.
