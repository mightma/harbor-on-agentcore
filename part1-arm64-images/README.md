# Part 1 · Build arm64 task images for a Harbor dataset

ACR only runs arm64. This part produces the arm64 images and the Harbor task
directories that reference them — for the two datasets this kit measures, and for
whichever one you bring instead.

## Step 0 · does your dataset already have an adapter?

Turning a dataset into Harbor tasks is not this kit's job, and you should not write
that code. Harbor's own [`adapters/`](https://github.com/harbor-framework/harbor/tree/main/adapters)
directory holds **85 of them** — including several SWE-shaped ones this kit does not
use: `swegym`, `multi-swe-bench`, `swebenchpro`, `swebench_multilingual`, `swelancer`,
`swtbench`. Part 1 depends on exactly two of them as packages:

```toml
"harbor-swesmith-adapter @ git+.../harbor@acr-kit#subdirectory=adapters/swesmith",
"harbor-swebench-adapter @ git+.../harbor@acr-kit#subdirectory=adapters/swebench",
```

`swebench/tasks.sh` drives one adapter's CLI and `swesmith/tasks.py` drives the other's
class directly. Neither reimplements the conversion, and `swesmith/tasks.py`
deliberately does not patch the adapter either — it flips `arch` on SWE-smith's profile
registry *from outside*, so the adapter stays exactly as upstream ships it.

```
你的数据集有 adapter 吗?
├── 有         -> generate task dirs with it, then continue below
└── 没有       -> write one: NAME + generate_task() + run() (adapters/<any> is the model)
```

The one change this kit needed *inside* an adapter is `--arch` on the SWE-bench one:
upstream hardcodes `spec.instance_image_key.replace("arm64", "x86_64")` because
`make_test_spec` infers the architecture from whichever machine runs the adapter. That
lives on the fork branch and belongs upstream (see the top-level README's commit table).

**What this part adds is downstream of the adapter**, and it is all ACR-shaped: the
adapter emits one task dir per instance with its own `environment/Dockerfile`, and on a
1,000-runtime quota that is the problem, not the solution. So this part supplies arm64
images the adapter's Dockerfiles can reference, and `shared/task_sharing.py` rewrites
generated tasks so tasks with an identical environment share an image and a runtime.

### The layout mirrors that split

```
part1-arm64-images/
├── shared/     dataset-independent: task_sharing.py, prepull, pulled-subset selection
├── swebench/   build_images.py, probe_arm64_images.py, tasks.sh, data/
└── swesmith/   build.sh, build_images.py, prepare_images.py, tasks.py, tasks.sh, data/
```

One directory per dataset, the same shape as `adapters/<dataset>/`, so **adding a
dataset is adding a directory** — its own builder, its own task generator, its own
durable manifests — with `shared/` holding only what is genuinely general. Both
directories share one `uv` environment (`pyproject.toml` at this level), because both
adapters and both upstream packages have to coexist anyway.

## The one question that decides the cost

**How many distinct *environments* does your dataset have?** Not how many tasks — how
many byte-identical installed environments those tasks resolve to. That number is your
image count and your runtime count, and on AgentCore the runtime count is the binding
constraint (1,000 per account, default). Everything else in this part follows from it.

The method, in one line: **find the coarsest grouping over tasks whose installed
environment is identical, then make the per-task difference a cheap in-session step.**

```
Does your dataset publish arm64 images?
├── yes, for all instances    -> generate tasks, prepull, done  (~10 min)
├── yes, for some             -> split by coverage; state the subset in every result
└── no                        -> build. Then:
    What is the coarsest grouping with an identical installed environment?
    ├── one commit per group (synthetic bugs)  -> per-task = git checkout       (~0 s)
    ├── same deps, different commits           -> per-task = checkout + install (minutes)
    └── nothing shared                         -> one image per task; check the 1,000 quota
```

The mechanism that implements the middle branch is dataset-independent and lives in
`shared/task_sharing.py`; `share_tasks()` takes your grouping, your image-per-group
and your per-task setup command as three functions. The SWE-smith adapter over it
(`share_swesmith_tasks()` in `swesmith/tasks.py`) is 13 lines of policy.
**The rule for whether to use it at all:**

> Share when the per-task delta is *local and constant* (SWE-smith: `git checkout`,
> ~0 s). Do not share when it is *work* (SWE-bench Verified: an editable install,
> minutes). The grouping that minimises runtime count is not automatically the
> grouping you want — see the worked comparison in 1a.

**Three hard ceilings bound every grouping choice**, and they pull in opposite
directions:

| Ceiling | Value | Pushes you toward |
|---|---|---|
| ACR image size | 2048 MB **compressed** | fewer things baked in |
| microVM disk | 8.8 GB | fewer things baked in |
| Runtimes per account | 1,000 | coarser grouping |

### One trap that is a validity failure, not a performance one

**Baking full git history leaks the future.** SWE-bench's own instance script
deliberately deletes tags newer than the base commit, expires the reflog and runs
`git gc --prune=now --aggressive`. If you bake one clone with all history into a shared
image and only `git checkout` per task, **the agent can read future commits and tags**
— including, sometimes, the fix. SWE-smith is immune because its bug branches are
synthetic and every branch is a sibling of the same commit. Any dataset with *real*
history has to keep the scrub in the per-task step, and pay for it every session.

The second trap is mechanical: sharing only works if `environment/` ends up **empty**,
because that is when Harbor's `environment_content_hash()` falls back to
`sha256(docker_image)`. A dataset whose generated tasks ship anything else in that
directory keeps a per-task hash and deploys a runtime per task anyway.
`share_tasks()` warns when it sees leftovers; do not ignore that line.

### The two datasets here are the two answers

| | SWE-bench Verified | SWE-smith |
|---|---|---|
| arm64 images exist? | yes, 281 of 500 | none |
| what you do | generate task dirs, pull the published bases | **build 119 images under qemu** |
| time | ~10 min | **~3 h** (2.5 h build + 25 min wrap) |
| distinct environments | one per instance (**deliberately**, see 1a) | one per *repository* |
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
uv run swebench/probe_arm64_images.py --dataset princeton-nlp/SWE-bench_Verified
```

**\[measured\]** 281 of 500 instances have a `swebench/sweb.eval.arm64.*` image. The
Docker Hub search for `sweb.eval.arm64` returns exactly 281 repositories, and they
coincide exactly with the per-instance probe. Nothing else in the SWE ecosystem
publishes arm64 images.

A shortcut that **does not work**: reusing a Verified arm64 image as the base for a
same-repo instance from the train split. SWE-bench's train split draws on 35 entirely
different repositories (pandas, qiskit, transformers, …) with zero overlap with the 12
test repositories. Reusable instances: 0.

### Why this one stays per-task: the grouping worked through

This is the worked example of the method above, and it is the case where the method
says **do not share** — worth reading before you group your own dataset, because the
arithmetic is not the obvious one.

SWE-bench Verified does have a natural grouping. **\[measured\]** over all 500
instances via `make_test_spec`: **1** distinct `base_image_key`, **40** distinct
`env_image_key`, 12 repositories. `env_image_key` is precisely SWE-bench's own
"dependencies are identical" boundary, and group sizes are min 1, max 75, median 4. So
grouping by it would cut 500 images and 500 runtimes to **40**.

It is still the wrong trade:

| | per-task images (**what this kit ships**) | group by `env_image_key` |
|---|---|---|
| Images to build | 500 | 40 |
| Runtimes | 500 | **40** |
| Build cost | one-off, parallel, cacheable | one-off, 12× smaller |
| **Session start** | **~3 s** | clone + reset + history scrub + `pip install -e .[test]` — **minutes, every trial, forever** |
| Large groups (75/44/43 instances) | fine | one image cannot hold every tree anyway → hybrid needed |
| Benchmark validity | scrub baked in, done once | scrub must be redone per session, or the agent sees future commits |

Instances of the same repository span years and different dependency versions, and
*within* an `env_image_key` group the commit still changes — so the per-task delta is
not a checkout, it is an editable install. An eval pass is 70–500 trials and gets
repeated across models and checkpoints; RL multiplies that by epochs. Paying minutes
per trial to save a one-time build inverts the ratio, and it erodes the one property
this whole substrate was chosen for: the measured 3 s warm start.

So the grouping is documented here and **deliberately not implemented**.
`shared/task_sharing.py` stays general for the datasets where the delta *is* cheap;
SWE-smith below is one.

### Generate task dirs

**SWE-bench Verified is the test set. All of it. It is never split.** Verified ships
one split upstream — `test`, 500 instances — and the only reason this kit ever carved a
"train" half out of it was that just 281 instances had arm64 images, which left nothing
else to train on. Building the missing images removed that excuse: training data now
comes from SWE-smith, which shares neither instances nor repositories with Verified.

`data/` holds the lists, so you do not have to re-probe:

| File | Instances | What it is |
|---|---|---|
| **`swebv-arm64-runnable.txt`** | **414 / 500** | every Verified instance with an arm64 image ACR can deploy. This is the benchmark |
| **`swebv-arm64-gated.txt`** | **200** | of those, the ones whose oracle ceiling is verified 1.0. **Report numbers from this list** |
| `swebv-arm64-instances.txt` | 281 | have a *published* arm64 image (an output of `probe_arm64_images.py`) |
| `swebv-arm64-selfbuilt.txt` | 160 | built here (an output of `build_images.py`, read back from ECR) |
| `swebv-arm64-selfbuilt-gated.txt` | 130 | of those, gate-clean — `gated` ∩ `selfbuilt`, kept because the 44.6% Sonnet 5 number is on exactly these |
| `swebv-arm64-undeployable.txt` | 27 | built but over ACR's 2048 MB ceiling — matplotlib, see below |
| `swebv-arm64-eval.txt` / `-eval-verified.txt` | 73 / 70 | **historical.** The old held-out-repo split's eval half; kept only because the kit's earlier measured numbers (Sonnet 5 40.0%, Qwen3.5-4B 10.0%) are on the 70 |

```bash
swebench/tasks.sh swebench/data/swebv-arm64-runnable.txt "$HARBOR_DATASETS/swebv-arm64"
```

Why the test set is 414 and not 500: 86 instances have no arm64 image that can run
here — 59 that cannot be built at all (see below) and 27 built but too large for ACR.
Say **"SWE-bench Verified, 414 of 500 arm64-runnable"** when you quote a number, and
say which of the two lists you used. What you must *not* do is call any subset of this
"SWE-bench Verified" without qualification.

Of the 414, **200 have a verified oracle ceiling** so far (70 published + 130
self-built). The rest are runnable but ungated: an instance whose own golden patch does
not score 1.0 cannot reward a correct answer, and scoring a policy against it only
subtracts a constant. Gate them (part 2) before adding them to a reported number.

### Building the 219 that are not published

`swebench/build_images.py` drives swebench's own base → env → instance chain
with the architecture forced to arm64:

```bash
uv run swebench/build_images.py --list      # the 219, and their image tags
uv run swebench/build_images.py --dry-run   # render every Dockerfile, no docker
uv run swebench/build_images.py --instances psf__requests-5414   # prove the path
uv run swebench/build_images.py --concurrency 3 --push --prune-after-push
```

`--prune-after-push` is not an optimisation, it is what makes a full run possible: each
instance image is ~3.3 GB on disk, so 219 of them need ~700 GB. Pushing each one to ECR
and dropping the local copy as it finishes keeps the working set at the 34 env images
they are built from — and those stay cached, so the next instance is still a ~1 min
build rather than a ~16 min one.

The reason this needs a script at all: `TestSpec.arch` defaults to `x86_64` and every
public entry point in `swebench` builds its specs *without* passing `arch`, so no flag
makes the harness build arm64. The script constructs the specs itself and hands them
back to upstream's builders, which pass `TestSpec` objects through untouched.

**\[measured\]** the 219 resolve to **1 base image and 34 env images** — the env stage
is 34 builds, not 219 — and every rendered base fetches
`Miniconda3-...-Linux-aarch64.sh` (`--dry-run` asserts this; an x86 Miniconda inside a
`--platform arm64` build is exactly the silent failure this replaces).

**\[measured\]** end to end for `psf__requests-5414`, on 8 vCPU under qemu:

| Stage | Time | Size |
|---|---|---|
| base (`sweb.base.py.arm64`) | ~7 min | 573 MB compressed |
| env (`sweb.env.py.arm64.<hash>`) | ~9 min | 856 MB |
| instance (`sweb.eval.arm64.<id>`) | **65 s** | 873 MB compressed / 3.28 GB on disk |
| Harbor wrap + push + runtime deploy + **oracle gate** | 71 s | **reward 1.000** |

So a self-built arm64 image is a working task image, reward path included. The env stage
is where the qemu time goes and it scales with the *dependency tree*, not the instance
count: repos with published aarch64 wheels (requests, pytest, pylint) install binaries;
matplotlib, scikit-learn and xarray compile. Order batches cheapest-first.

**And build on native arm64 if you possibly can.** **\[measured\]** on this host, same
container image, same moment: a CPU-bound Python workload runs **15.5–19× slower** under
qemu-user emulation than natively (sha256 ×200k: 0.2 s amd64 vs 3.1 s arm64; an integer
loop ×2M: 0.4 s vs 7.7 s). That factor is the whole story of this step's cost — django's
instance stage is **\[measured\]** ~35 min each here and is dominated by
`git gc --prune=now --aggressive`, which is exactly that kind of work. On a Graviton
instance the same builds should land in the low single-digit minutes:

```bash
DOCKER_HOST=ssh://ubuntu@<graviton> uv run swebench/build_images.py \
    --instance-list <ids> --concurrency 8 --push --prune-after-push
```

Two honest caveats: that 15–19× is the *emulation* factor on this machine, not Graviton
hardware performance, and the part of a build that is pip downloads will not speed up at
all. The end-to-end numbers below say what it was actually worth: **16× per instance**.

**What to launch**, from doing it:

| | Choice | Why |
|---|---|---|
| Instance | **c7gd.8xlarge** (32 vCPU / 64 GB / 950 GB NVMe) | the cost is `git gc --aggressive` — CPU and memory-bandwidth bound, multithreaded |
| AMI | Ubuntu 24.04 arm64, or AL2023 arm64 | **not** a DLAMI: no GPU is needed and its docker-29-plus-containerd-store config is what broke the other host |
| Pricing | **Spot** | the builder is idempotent and pushes per image, so an interruption costs only the in-flight ones |
| Disk | docker on the instance-store NVMe | with docker's containerd image store, moving `data-root` is **not enough** — the content store is `/var/lib/containerd`, so plan for both, or keep a large root volume |
| Concurrency | **≈ vCPU / 4** | `git gc` opens threads per core; 8 vCPU at concurrency 4 hit load 32 and stretched a 2-minute build to 35 |
| IAM | ECR only | a build box needs no AgentCore and no S3 |
| Emulation | **none** | SWE-bench's chain is arm64 all the way down. (SWE-smith's builder is the exception: it runs the published *amd64* image to export a conda env, so that path needs `binfmt --install amd64`.) |

**Some instances cannot be built on arm64, full stop.** **\[measured\]** two of the 219 —
`django__django-10097` and `django__django-7530` — pin conda `python=3.5`, and linux-aarch64
has no python 3.5 package in `defaults` or `conda-forge`, so `conda create` ends in
`PackagesNotFoundError` after a 458 s solve. The real ceiling is **217 of 219**. Every
other python version the unpublished set needs (3.6 through 3.11) does have aarch64
packages, so this is the only hard exclusion.

**\[measured\]** then the whole set, on a **c7gd.8xlarge** (32 vCPU, native arm64, no
emulation) at concurrency 8 — the same builder, the same day:

| | x86 + qemu (8 vCPU, conc. 4) | **Graviton c7gd.8xlarge (32 vCPU, conc. 8)** |
|---|---|---|
| django instance, mean | 2112 s (35 min) | **131 s** — 16× faster |
| throughput | ~6.9 images/h | **~220 images/h** — 32× |
| django's 6 env images | hours | **~4 min** |
| all 80 django instances | ~12 h projected | **26 min** |

The whole remaining set — 185 instances across six repos — took **~1 h 40 min** of wall
clock instead of the day-plus this would have been under emulation. Result:

| | |
|---|---|
| Self-built images now in ECR | **160** |
| SWE-bench Verified runnable on arm64 | **441 / 500 (88%)** = 281 published + 160 built |
| Of the 219 unpublished | 160 built, **59 not** — every one for a reason below |

`swebench/data/swebv-arm64-selfbuilt.txt` is the list of 160, read back from ECR.

#### Then the gate, which is what says they are usable

**\[measured\]** the oracle gate over all 160, 19 min at concurrency 8:

| | |
|---|---|
| **Gate-clean (reward 1.0)** | **130** — `swebench/data/swebv-arm64-selfbuilt-gated.txt` |
| reward 0.0 | 3 (pylint; undiagnosed, oracle applies and 20/21 FAIL_TO_PASS pass) |
| **could not deploy at all** | **27 (matplotlib)** — see the ceiling below |

Two operational lessons came out of that run:

- **A cold runtime's first invoke can time out.** One django trial died with
  `RuntimeClientError: Runtime initialization time exceeded`; the identical trial re-run
  scored 1.000. Retry before investigating.
- **The docutils pin generalises.** All six sphinx instances that gated 0.000 were fixed by
  `--pin 'sphinx-doc/sphinx=docutils<0.17'` and re-gated **1.000** — 1 verified earlier,
  5 in a batch here.

#### The 2048 MB image ceiling is real, and matplotlib does not fit

All 27 deployment failures were the same error, and none of them is about the runtime quota:

```
ServiceQuotaExceededException ... CreateAgentRuntime:
maxImageSizeMb limit exceeded for account
```

**\[measured\]** those images were **3264 MB compressed** against a django instance's 949 MB.
The fat was the conda package cache: `/opt/miniconda3/pkgs` was **4.4 GB uncompressed**,
against a 2.4 GB installed environment. `conda clean -afy` in the same RUN as the install
(this is the default now; `--no-slim` turns it off) took them to **2230 MB** —
**still over the 2048 MB limit**, and that limit is `Adjustable: False` in Service Quotas.

What is left cannot be removed: 2.4 GB of declared dependencies (pandas, wx, PyQt5, numpy,
matplotlib) plus 330 MB of `/testbed/.git`, which the verifier needs because it produces the
patch with `git diff`. So **matplotlib's 29 unpublished instances can be built but not run on
AgentCore Runtime**, and that is a property of the service, not of this kit. Slimming stays
on regardless: it is a free ~1 GB per image and it keeps other repos clear of the line.

Note the shape of that bug, because it is the kind this kit keeps warning about: the first
slim rebuild appeared to succeed while changing nothing. `--force-rebuild` rebuilt the *env*
images and then reused the cached *instance* images on top of them, pushing an unchanged
image under the same tag. It now drops the instance tags first.

#### The 59 that do not build, and why

Not one of them is a defect in the builder, and only two are really *about* arm64:

| Count | Repo | Cause | Class |
|---|---|---|---|
| 25 | scikit-learn | pinned old scipy has no aarch64 wheel and fails to compile from source | **arm64 hard** |
| 22 | xarray | `cdms2` has no linux-aarch64 build in conda-forge | **arm64 hard** |
| 4 | astropy | `setuptools==38.2.4` (2017) has no aarch64 build in `defaults` | **arm64 hard** |
| 2 | django | conda `python=3.5` does not exist for linux-aarch64 | **arm64 hard** |
| 3 | matplotlib | python 3.8 env, pip subprocess failure during dependency install | not diagnosed |
| 2 | astropy | `pip install -e .` fails at "getting requirements to build editable" | **drift**, pin-fixable |
| 1 | sympy | `setup_repo.sh` clones branch `1.7`, which sympy has since **deleted** upstream | **drift**, unfixable here |

The drift classes are worth internalising, because they are what a *fresh build of an old
benchmark* costs and they are not architecture-specific — they would hit an amd64 rebuild
today just as hard:

- **runtime dependency drift.** sphinx: docutils 0.23 against Sphinx 3.1.0. Fixed with
  `--pin 'sphinx-doc/sphinx=docutils<0.17'`, **\[measured\]** 0 → 14 tests passing, gate
  0.000 → 1.000.
- **tooling drift.** scikit-learn's install command passes `--no-use-pep517`, which modern
  pip has **removed** (`no such option`). Fixed with
  `--pin 'scikit-learn/scikit-learn=pip==22.3.1'` — **\[measured\]** 3 of 3 instances then
  built and pushed. This is what `--pin` is for: it constrains the env image, so anything
  pip-installable including pip itself can be held back.
- **upstream drift.** A deleted branch cannot be pinned around; that instance is simply
  gone unless the clone strategy changes.

Earlier, smaller batch for the record: 24 instances (requests, seaborn, pylint ×3, pytest
×9, sphinx ×10) on the x86 host, 10 env images, 23/23 built, ~1 h 50 min at concurrency 3
— pylint ~140 s, pytest ~215 s, seaborn 273 s, sphinx 470–740 s each.

Then the gate, which is the number that matters:

| | |
|---|---|
| Built | **24 / 24** |
| Oracle gate clean | **15 / 24** on the first pass, **16 / 24** after one dependency pin |
| Cause of every failure | dependency drift, **not** architecture (below) |

`swebench/data/swebv-arm64-selfbuilt-gated.txt` is the resulting allowlist, the same idea as
SWE-smith's `gate_passing_images.txt`: build produces images, the gate says which ones
can actually reward a correct patch.

#### The failures are drift, and the remedy is a pin

Six of ten sphinx instances gated 0.000 with *every* test erroring:

```
ModuleNotFoundError: No module named 'roman'
```

**\[measured\]** the env had resolved **docutils 0.23** against **Sphinx 3.1.0** — and
docutils removed `utils.roman` in 0.18. Pinning `docutils<0.17` in the same container
turned 0 passing into **14 passing**, exactly the 2 FAIL_TO_PASS and 12 PASS_TO_PASS
tests that had failed. Rebuilding the image with the pin and re-gating gives **1.000**.

This is *temporal* drift, not an arm64 problem: an instance's requirements file is years
old and leaves transitive deps unpinned, so today's resolver installs versions the code
never saw. It is the same failure class as the `flit_core`/`cloudpickle` case in 1b, and
it would happen identically on amd64 — which is exactly why the published images work and
a fresh build does not.

```bash
uv run swebench/build_images.py --instances sphinx-doc__sphinx-7748 \
    --pin 'sphinx-doc/sphinx=docutils<0.17'
```

`--pin` adds a pip-constraint layer to that repo's env image before the instance build.
It is opt-in on purpose: the right pin depends on the repo *version* (sphinx 5 wants a
newer docutils than sphinx 3), so a built-in table would silently change builds. The
workflow the kit teaches is **build → gate → find the drift → pin → re-gate**.

The three pylint failures are a different, undiagnosed case: the oracle patch applies and
20 of 21 FAIL_TO_PASS tests pass, with one assertion failing on empty linter output.
Likely `astroid` drift; recorded as open rather than guessed at.

#### Two things that will waste your afternoon

- **A re-pushed image does not reach an existing runtime.** Rebuild a task image, push it
  to the same ECR tag, re-run — and the sandbox still runs the old code. An AgentCore
  runtime resolves its `containerUri` when it is *created*, so `delete_runtime: false`
  (which the whole kit relies on for speed) also means the deployed runtime is frozen.
  `--force-build` is not enough: it rebuilds and re-pushes, then Harbor reuses the same
  runtime. Delete the runtime, or change the task's environment content hash so a new one
  is created. **\[measured\]** the same pinned image gated 0.000 through the old runtime
  and 1.000 through a fresh one.
- **The build's own tag is not the tag the task asks for** — see below.

#### The tag that makes a build usable, and how it bites

swebench tags a local build `sweb.eval.arm64.<instance_id>`. A **generated task dir**
names something else:

```
FROM swebench/sweb.eval.arm64.psf_1776_requests-5414:latest
```

— namespace added, `__` respelled `_1776_` (`TestSpec.instance_image_key` does both when
a namespace is set). A freshly built image therefore does *not* satisfy the task that
needs it, and the mismatch surfaces only as an `ImageBuildError` minutes into a trial.
The script now applies that second local tag itself (`--task-namespace`, default
`swebench`), so a generated task resolves from the local store with no editing. This was
found by running the end-to-end, not by reading the code.

`swebench/data/swebv-arm64-instances.txt` is the coverage list the default selection complements;
it is an *output* of `probe_arm64_images.py --out`, so refresh it rather than trusting a
stale copy.

### Pull the bases before you evaluate

```bash
shared/prepull_arm64.sh "$HARBOR_DATASETS/swebv-arm64"
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
swesmith/build.sh --list                     # what would be built
swesmith/build.sh --limit 4 --concurrency 4  # prove the path on four repos
swesmith/build.sh --concurrency 12 --push    # the real run
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

### Optionally bake the harness in too

Harbor sets the agent up inside the sandbox on **every** trial, and that cost recurs
forever: **\[measured\]** on the 70-task Verified pass, `agent_setup` was a **12.0 s**
median (min 9.0, max 53.6) for `terminus-2` and **51.3 s** for `claude-code`. Baking
makes Harbor's own idempotence checks short-circuit:

```bash
uv run swesmith/prepare_images.py --bake-harness terminus-2 --render-only
uv run swesmith/prepare_images.py --bake-harness claude-code --harness-version 2.1.272
```

| `--bake-harness` | Installs | What Harbor then skips |
|---|---|---|
| `terminus-2` | `tmux`, `asciinema` | `TmuxSession._install_recording_tools()` |
| `claude-code` | the bootstrap installer, on the PATH the check uses | `_installed_claude_satisfies_version()` |
| `mini-swe-agent` | it as a **uv tool** | its `install()` |

**`terminus-2` is host-internal and installs no agent** — it runs in the harbor
process and only shell commands cross into the sandbox. Its 12.0 s is `apt-get install
-y tmux asciinema`, run because `record_terminal_session` defaults to true. So baking
reclaims *that*, not an agent install, and not the tmux session start that follows it:
treat 12.0 s as a ceiling, and `record_terminal_session: false` as the other way to cut
the asciinema half.

Three things to know before using it:

- **the version has to match.** With `version` set in the eval config, Harbor compares
  exactly and reinstalls on a mismatch — silently undoing the saving. `mini-swe-agent`
  is checked with `uv tool list | grep mini-swe-agent`, so a plain `pip install`
  satisfies nothing;
- **a baked image is harness-specific**, so it gets its own tag
  (`<key>-prepared-<harness>-arm64`) and does not replace the tag parts 2–4 point at;
- **each layer costs image size** against the 2048 MB compressed ceiling.

**\[shipped, not run\]** the recipes render and are reviewed (`--render-only` needs no
docker), but no image has been built with them on this host. Validate one before a
batch: build a single image, then run the harness's own version command inside it.

### Where to substitute your own dataset

`swesmith/tasks.sh` applies the sharing as part of generating the tasks
(`--shared-images`), and that step is the only dataset-specific part of it:

```python
# swesmith/tasks.py -- the whole adapter, over shared/task_sharing.py
share_tasks(
    output_dir,
    group_of=...,    # TaskDir -> "mewwts__addict.75284f95", off the task's FROM tag
    image_of=prepared.get,                  # repo key -> its prepared image in ECR
    setup_cmd=lambda t: f"cd /testbed && git checkout {t.instance_id}",
)
```

Write those three functions for your dataset and the rest — deleting the per-task
Dockerfile, moving the setup into `[environment.healthcheck]`, reporting what did and
did not collapse — is already done. `task_sharing.py`'s module docstring is the
reference for *why* each of the three edits is needed; this README's opening section is
the reference for whether you should be sharing at all.

**\[measured\]** sizes, against the two limits that matter:

| | Value | Limit |
|---|---|---|
| Prepared image, compressed | ~907 MB | 2048 MB (ACR image quota) |
| Prepared image, uncompressed | ~3.45 GB | 8.8 GB (microVM disk) |

Note the two scripts report different units — `summary.json` records compressed,
`prepared.json` records `docker image inspect .Size` (uncompressed). Comparing them
directly looks like 4× bloat. It is not; the wrap adds 17 MB compressed.

**And `.Size` itself is not a stable unit.** **\[measured\]** on docker 29 with the
containerd image store, `docker image inspect --format '{{.Size}}'` returns the
**compressed** content size — for the self-built `psf__requests-5414` image it reports
873 MB, while `docker image ls --tree` shows `DISK USAGE 3.28GB / CONTENT SIZE 873MB`.
Under the older overlay2 store the same field was the uncompressed size, which is what
the paragraph above assumed. So when reading any size out of this kit, check which store
produced it: compare against 2048 MB with the *compressed* number and against 8.8 GB
with `DISK USAGE`.

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
swesmith/build.sh --repos <failed-keys> --concurrency 8 --push
```

### Generate task dirs

```bash
swesmith/tasks.sh
```

**\[measured\]** 44,489 task dirs over **119 distinct images**, ~9 minutes.

The task dirs are **disposable** — that command rebuilds them. What is durable is
`swesmith/data/`:

| File | What it is |
|---|---|
| `prepared.json` | the 121 prepared images; reconstructable from ECR, but slowly |
| `prepared_profile_keys.txt` | the 121 profile keys |
| `gate_passing_images.txt` | the 108 images whose oracle scores 1.0 (see part 2) |
| `gate_passing_tasks.txt` | the 38,815 task names on those images |

Image references in those files are stored **without a registry host** —
`swesmith-arm64:<key>-prepared-arm64`, not
`<account>.dkr.ecr.<region>.amazonaws.com/swesmith-arm64:...`. They are committed
artifacts, and a fully qualified URI would pin them to the account that happened to
build the images. The registry is reattached when tasks are generated, from
`$ECR_REGISTRY` or from `aws sts get-caller-identity` plus `$AWS_REGION`. Already
qualified references pass through untouched, so a manifest produced before this
change still works.

Consequence worth knowing: **the images have to exist in *your* registry.** These
manifests describe what to build, not a public dataset you can pull. Run
`swesmith/build.sh --push` first, or set `ECR_REGISTRY` to a registry you can
actually read.

Keep that directory on durable storage. The scratch disk holding the task dirs was
wiped twice during this work; each time the cost was one 9-minute regeneration
*because* the manifests were elsewhere.

### Verify the golden patches

```bash
uv run swesmith/check_solve_patches.py "$HARBOR_DATASETS/swesmith-arm64"
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
- `swesmith/data/prepared.json`
