#!/usr/bin/env bash
# Generate Harbor task directories for the arm64 slice of SWE-bench Verified.
#
#   swebench/tasks.sh swebench/data/swebv-arm64-runnable.txt "$HARBOR_DATASETS/swebv-arm64"
#
# Verified is a test set and this kit does not split it; training data comes from
# SWE-smith (../swesmith/). swebv-arm64-runnable.txt is every instance with an arm64
# image ACR can deploy -- 414 of the 500.
#
# Unlike SWE-smith, nothing is built here: these 281 instances already have
# published `swebench/sweb.eval.arm64.*` images, so this only writes task dirs
# that point at them. Pull the bases with shared/prepull_arm64.sh afterwards.
#
# The --arch flag is the reason this needs the kit's Harbor build. Upstream the
# adapter hardcodes
#
#     spec.instance_image_key.replace("arm64", "x86_64")
#
# because `make_test_spec` infers the architecture from whichever machine runs
# the adapter -- so on an x86 host it always emitted amd64 image names, which ACR
# cannot run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

LIST="${1:?usage: swebench_tasks.sh <instance-id-list> <output-task-dir>}"
OUT="${2:?usage: swebench_tasks.sh <instance-id-list> <output-task-dir>}"

mapfile -t ids < <(grep -v '^[[:space:]]*$' "$LIST")
echo "generating ${#ids[@]} arm64 task dirs from $(basename "$LIST") into $OUT"
mkdir -p "$OUT"

"$PART/.venv/bin/swebench" \
  --no-all \
  --task-ids "${ids[@]}" \
  --arch arm64 \
  --task-dir "$OUT" \
  --overwrite 2>&1 | grep -vE "^20[0-9-]+ .* (INFO|WARNING) -" | tail -20

echo "task dirs: $(find "$OUT" -maxdepth 1 -mindepth 1 -type d | wc -l)"
