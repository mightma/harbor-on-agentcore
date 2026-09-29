# Draft: reply to the review comment on PR 2

**Raw material — edit before posting.** CONTRIBUTING puts comments on the human too.

---

Good catch, and it is specific to this change: with the default `--arch x86_64` every
Verified instance has a published image, so the adapter never needed an availability
check. `--arch arm64` is what opens the gap — I checked three instances that have no arm64
image and all three have an x86_64 one, so the asymmetry is real.

Fixed in 4233c93 by refusing the combination rather than guessing at it:

```
$ swebench --arch arm64
swebench: error: --arch arm64 over the whole dataset: published arm64 images cover a
subset of SWEBench Verified, so some generated tasks would reference images that do
not exist.
  Pass --task-ids/--instance-id with the ids you can run, or
  --allow-unpublished-images if you build the missing images yourself.
```

I did not take either of the two suggested routes, and I think both would be worse than
the problem:

**Probing the registry before generating** costs one manifest request per instance, and
Docker Hub counts those against the anonymous limit of 100 per rolling hour per IP.
Checking all 500 spends the caller's whole hourly budget, which then makes `docker pull`
— the very next step this adapter is preparing for — fail with 429 for the following
hour. I measured that while building the tooling this came from. It could be an opt-in
flag, but it should not be on the path by default.

**Shipping a list of published arm64 ids** would go stale in the direction that hurts: as
soon as upstream publishes more arm64 images, a pinned list starts refusing instances that
do work, and the caller has no way to override it.

So the check is local and free: the caller states which ids they mean, or explicitly
accepts the gap. `--allow-unpublished-images` exists because that case is real — I build
the missing arm64 images myself and do want all 500 named `arm64`. Everything that already
passes an id list is unaffected.

Covered by a test that asserts the refusal and that each of the three escapes gets past it;
reverting the guard fails it.
