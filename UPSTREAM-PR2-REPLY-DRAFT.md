# Drafts: replies to the review comments on PR 2

**Edit before posting** — CONTRIBUTING puts comments on the human too.

---

## Comment 1 — full arm64 generation includes missing images

Fixed in 4233c93: `--arch arm64` now refuses the whole dataset unless you pass
`--task-ids`/`--instance-id` or `--allow-unpublished-images`. I skipped the availability
check because probing 500 manifests exhausts Docker Hub's 100/hour anonymous limit and
would make the `docker pull` that follows fail with 429.

## Comment 2 — limited arm64 runs rejected as full dataset

Right on both counts, fixed in 74e91b1. `--limit 0` is now exempt — it slices
`instance_ids[:0]`, so nothing is generated and no task can reference a missing image. A
positive `--limit` is still refused, since it bounds how many instances convert and not
which; the error now says that outright instead of leaving `--limit` looking like a
workaround. I also reworded `--allow-unpublished-images` so it no longer claims you built
the images yourself.
