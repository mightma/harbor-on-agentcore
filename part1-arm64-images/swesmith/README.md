# SWE-smith · arm64

One directory per dataset, the same shape as Harbor's `adapters/<dataset>/`. This one
holds everything specific to SWE-smith; `../shared/` holds what is not.

| File | What it is |
|---|---|
| `build.sh` | the two build stages in order, with the flags that matter |
| `build_images.py` | recreates each repo's conda environment on arm64 from the published amd64 image |
| `prepare_images.py` | wraps a built image into a task-ready one (git, uv, `/logs`, every task branch), optionally with `--bake-harness` |
| `tasks.py` / `tasks.sh` | drives Harbor's `swesmith` adapter, then applies the sharing rewrite |
| `check_solve_patches.py` | verifies the golden patches survive the adapter |
| `data/` | the durable manifests: which images exist, which ones the oracle gate passed |

## The shape of this dataset

**No arm64 images exist at all**, so this builds them: 119 repository images under qemu,
~2.5 h at concurrency 12. The payoff is a training set disjoint from the eval benchmark —
no shared instances *and* no shared repositories.

**One image per repository, shared by all its tasks.** Every task of a repository is a
branch off the same commit, so one installed conda environment serves all of them and the
per-task delta is a local `git checkout` against branches already baked into the image:
~0 s, no network. That is what makes `../shared/task_sharing.py` the right tool here —
**\[measured\]** 44,489 tasks resolve to 119 runtimes instead of 44,489. `tasks.py` is the
13-line policy over it (group = repository, setup = `git checkout`), and it is the model
to copy for a dataset whose per-task delta is also cheap.

## Prerequisite: emulation, in whichever direction you are missing

This dataset's builder needs **both** architectures, which neither host has natively:

```bash
# on an arm64 host (Graviton): the builder runs the published *amd64* image to export
# its conda environment, so x86 emulation is required
docker run --privileged --rm tonistiigi/binfmt --install amd64

# on an x86 host: the images being built are arm64
docker run --privileged --rm tonistiigi/binfmt --install arm64
```

Why: `build_images.py` recovers each repository's dependency set by running
`conda env export` **inside the published amd64 image** — upstream never published the
spec, only the image. Everything after that (`conda env create`, the repo install, baking
the branches) targets arm64. So an arm64 host emulates one short step and runs the
expensive ones natively, which is the better half of the trade.

**\[measured\]** on a c7gd.8xlarge, per repository: pulling the amd64 image **23 s** (no
emulation, just download), the emulated `conda env export` **38 s** (for scale: `python -c
pass` is 2.5 s emulated against 1.3 s native). Across 134 repositories at concurrency 8
that is ~11 minutes of emulation against a build stage measured in hours — the emulator is
not the bottleneck on this path. Docker Hub's pull limit is the one to plan around: 134
images, 100/hour anonymous, so `docker login` first.

## Usage

```bash
build.sh --list                      # what would be built
build.sh --limit 4 --concurrency 4   # prove the path on four repos
build.sh --concurrency 12 --push     # the real run

tasks.sh                             # task dirs for every repo in the manifest
uv run check_solve_patches.py "$HARBOR_DATASETS/swesmith-arm64"
```

`data/` is the part worth protecting: the task dirs are disposable (9 minutes to
regenerate) but the manifests are what say which images exist and which of them can
actually reward a correct patch. `KIT_STATE_DIR` points here for that reason. Image
references in them are stored **without a registry host**, so they do not pin the account
that happened to build the images.

The rest — the 15 build failures and their four causes, the size ceilings, and why
`--bake-harness` reclaims tmux and asciinema rather than an agent install — is in the
part 1 README.
