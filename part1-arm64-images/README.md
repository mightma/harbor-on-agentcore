# Part 1 · Build arm64 task images for a Harbor dataset

ACR only runs arm64. This part produces, for two datasets, the arm64 images and the
Harbor task directories that reference them.

The two datasets need opposite treatment, and that asymmetry is the whole content of
this part:

| | SWE-bench Verified | SWE-smith |
|---|---|---|
| arm64 images exist? | yes, 281 of 500 | none |
| what you do | generate task dirs, pull the published bases | **build 119 images under qemu** |
| time | ~10 min | **~3 h** (2.5 h build + 25 min wrap) |
| images | one per instance | one per *repository*, shared by all its tasks |
| runtimes needed | 73 (eval) or 208 (train) | **119 for 44,489 tasks** |

## Setup

```bash
cd part1-arm64-images
uv sync
```

Python 3.13 is required — the SWE-bench adapter declares `>=3.13`. The two `swesmith`
/ `swebench` pins in `pyproject.toml` are load-bearing; the comments there say why.

## 1a · SWE-bench Verified

### Check what actually exists first

```bash
uv run scripts/probe_arm64_images.py --dataset princeton-nlp/SWE-bench_Verified
```

**\[measured\]** 281 of 500 instances have a `swebench/sweb.eval.arm64.*` image. The
Docker Hub search for `sweb.eval.arm64` returns exactly 281 repositories, and they
coincide exactly with the per-instance probe. Nothing else in the SWE ecosystem
publishes arm64 images.

A shortcut that **does not work**: reusing a Verified arm64 image as the base for a
same-repo instance from the train split. SWE-bench's train split draws on 35 entirely
different repositories (pandas, qiskit, transformers, …) with zero overlap with the 12
test repositories. Reusable instances: 0.

### Generate task dirs

`data/` already holds the instance lists, so you do not have to re-probe:

| File | Instances | Repositories |
|---|---|---|
| `swebv-arm64-instances.txt` | 281 | all 12 |
| `swebv-arm64-train.txt` | 208 | django, sympy |
| `swebv-arm64-eval.txt` | 73 | the other nine |
| `swebv-arm64-eval-verified.txt` | 70 | eval minus 3 with a broken ceiling |

```bash
scripts/swebench_tasks.sh data/swebv-arm64-eval.txt  "$HARBOR_DATASETS/swebv-arm64/eval"
scripts/swebench_tasks.sh data/swebv-arm64-train.txt "$HARBOR_DATASETS/swebv-arm64/train"
```

The split is **by repository, not random**, and that is a deliberate compromise you
have to state whenever you quote a number from it. Training on a benchmark's own
instances is nobody's preferred design; the table above forced it. Splitting by repo
at least makes the eval a held-out-*repository* generalisation measure (train on
django/sympy, evaluate on sphinx / pytest / astropy / requests / pylint / sklearn /
matplotlib / seaborn / flask) rather than held-out-instance. Two caveats travel with
the number:

- It is **not "SWE-bench Verified"**. It is a 73-instance held-out-repo subset whose
  repository mix is nothing like the full 500 (46% django).
- The training data still comes from the eval benchmark. To separate them properly,
  use SWE-smith below — that path leaves the eval set untouched and only changes
  `RL_TRAIN_DATA` in part 4.

`swebv-arm64-eval-verified.txt` is the 70 of 73 that score 1.0 with the oracle agent.
Two sphinx instances have PASS_TO_PASS tests that fail independently of their own
patch, and `psf__requests-2317`'s verifier hangs on network calls. Scoring a policy
against tasks whose ceiling is 0 just depresses the number by a fixed, uninteresting
amount — part 3 defaults to this list.

### Pull the bases before you evaluate

```bash
scripts/prepull_arm64.sh "$HARBOR_DATASETS/swebv-arm64/eval"
```

Not optional. Each trial otherwise resolves its `FROM` against Docker Hub and dies on
the anonymous rate limit as an `ImageBuildError`, and a trial only gets 600 s for its
build. If you only pulled some, `select_pulled_tasks.py` copies the ready subset into
a separate task dir so a run is not half failures.

## 1b · SWE-smith

No arm64 images exist, so this builds them. The payoff is a training set with no
overlap with the eval benchmark, and ~59k tasks over 222 repositories to draw from.

### Build

```bash
scripts/build_swesmith.sh --list                     # what would be built
scripts/build_swesmith.sh --limit 4 --concurrency 4  # prove the path on four repos
scripts/build_swesmith.sh --concurrency 12 --push    # the real run
```

Two stages, both in that script:

1. **Build** (`build_swesmith_images.py`) — recreate each repo's conda environment on
   arm64 from the published amd64 image, commit as `swesmith.arm64.<repo>`.
   **\[measured\]** 119 of 134 succeeded, median 843 s each at concurrency 12, ~2.5 h
   total.
2. **Wrap** (`prepare_swesmith_images.py`) — bake in git, uv, `/logs`, and **every task
   branch** (`git fetch --all`), producing `<repo>-prepared-arm64`. **\[measured\]** ~148 s
   per image, ~25 min total; 46,448 branches across 121 images (min 10, max 2,391 per
   repo).

Baking the branches is what makes the per-task delta a *local* `git checkout` with no
network, which is what makes a warm rollout ~3 s instead of ~37 s.

**\[measured\]** sizes, against the two limits that matter:

| | Value | Limit |
|---|---|---|
| Prepared image, compressed | ~907 MB | 2048 MB (ACR image quota) |
| Prepared image, uncompressed | ~3.45 GB | 8.8 GB (microVM disk) |

Note the two scripts report different units — `summary.json` records compressed,
`prepared.json` records `docker image inspect .Size` (uncompressed). Comparing them
directly looks like 4× bloat. It is not; the wrap adds 17 MB compressed.

### The 15 build failures

Four distinct causes, only one mechanically fixable:

| Cause | Example | Fixable |
|---|---|---|
| `nvidia-*` CUDA wheels have no aarch64 build | `fvcore` needs `nvidia-cudnn-cu12` | **yes**, filtered now |
| conda name in the `pip:` section | `dask` needs `python-graphviz`, PyPI calls it `graphviz` | maybe, with an alias map |
| genuinely no aarch64 build | `MONAI` needs `nni==2.10.1` | no |
| PEP 517 time drift | `cloudpickle` uses `[tool.flit.metadata]`; build isolation fetches a `flit_core` that dropped it | maybe, via `PIP_CONSTRAINT` |

The last one is worth understanding because it is **not** an architecture problem: the
published amd64 image was built when an older `flit_core` was current, and rebuilding
today pulls a newer one that rejects the repo's config.

To retry, delete the cached specs first or the fix will not apply —
`ensure_env_yml()` returns early when the file exists:

```bash
find "$SWESMITH_BUILD_ROOT/env" -name 'sweenv_*.yml' -delete
scripts/build_swesmith.sh --repos <failed-keys> --concurrency 8 --push
```

### Generate task dirs

```bash
scripts/swesmith_tasks.sh
```

**\[measured\]** 44,489 task dirs over **119 distinct images**, ~9 minutes.

The task dirs are **disposable** — that command rebuilds them. What is durable is
`data/swesmith-manifests/`:

| File | What it is |
|---|---|
| `prepared.json` | the 121 prepared images; reconstructable from ECR, but slowly |
| `prepared_profile_keys.txt` | the 121 profile keys |
| `gate_passing_images.txt` | the 108 images whose oracle scores 1.0 (see part 2) |
| `gate_passing_tasks.txt` | the 38,815 task names on those images |

Keep that directory on durable storage. The scratch disk holding the task dirs was
wiped twice during this work; each time the cost was one 9-minute regeneration
*because* the manifests were elsewhere.

### Verify the golden patches

```bash
uv run scripts/check_solve_patches.py "$HARBOR_DATASETS/swesmith-arm64"
```

Expect `tasks with bad hunks : 0`. If it reports non-zero, your Harbor build predates
the SWE-smith patch fix.

This check exists because of a bug that cost 17 repositories before it was found. A
blank context line in a unified diff is a single space; the adapter's `.strip()` on the
patch text deleted it, leaving the final hunk one line shorter than its `@@` header
declares, and `git apply` rejects the whole patch as `corrupt patch at line N`.
**\[measured\]** 8,658 of 59,136 upstream patches (14.6%) end in such a line, across
216 of 222 repositories. Because the oracle gate samples one task per repository,
~15% of repositories failed on an unlucky draw and looked *entirely broken*.

The script checks each hunk header against its body, then sample-verifies its own
arithmetic against real `git apply` so the model cannot drift from git.

## Two mistakes that cost hours

- **Judge build progress by `docker image ls`, not by the log.** Python block-buffers
  stdout when redirected, so a build log can sit at 0 bytes for an hour while
  subprocess output (docker push) lands immediately — the log looks like it starts
  mid-run. The scripts use `python -u`; if you write your own wrapper, do too.
- **`pkill -f <pattern>` matches your own SSH command line** and will kill your
  session. Count with `ps -eo cmd --no-headers | grep -c '[b]uild_swesmith_images'`.

One non-issue worth knowing so you do not chase it: grepping build logs for rate
limiting finds `429` everywhere. Those are **sha256 digests containing "429"**. There
was no throttling — 134 pulls spread over 2.5 h stayed under 100/hour.

## What part 2 expects

- `$HARBOR_DATASETS/swesmith-arm64` and/or `$HARBOR_DATASETS/swebv-arm64/{eval,train}`
- images pushed to ECR (`--push`)
- `data/swesmith-manifests/prepared.json`
