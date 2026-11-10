#!/usr/bin/env bash
# E0.1 driver: run each candidate in its OWN `timeout 1800` (30-min) subprocess so
# a hang/download-stall on one model is killed and we move to the next (author
# directive: never wait > 30 min on a single model). Results accumulate
# incrementally into results/E0/arch_manifest.json + e01_smoke.csv (the python
# script merges per-model), so partial progress is never lost.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
# Order: gate-critical NEW families first (granite=new sequential, hymba=new
# parallel) to lock the minimum lineup fast; then alternates; then guaranteed
# public scale points; then higher-risk (recurrentgemma gated, samba id unsure).
ORDER="granite hymba lfm2 zamba2 falcon15 qwen35_2b recurrentgemma samba"
for key in $ORDER; do
  echo "==================== E0.1 $key ($(date -u +%H:%M:%S)) ===================="
  timeout 1800 python experiments/e01_smoke.py "$key" 2>&1 \
    | grep -avE "it/s\]|it\]|Loading weights|Loading checkpoint shards|Downloading" \
    | tail -25
  rc=${PIPESTATUS[0]}
  if [ "$rc" = "124" ]; then
    echo "[E0.1] $key TIMED OUT at 30min -> skipping (logged as blocker)"
    printf '\n## B_E01_%s — [SKIPPED] E0.1 smoke timed out >30min\n%s\n' \
      "$key" "Model $key exceeded the 30-min per-model box during E0.1 (load/download/patch). Skipped per author directive; use its PLAN fallback." \
      >> results/E0/blockers.md
  fi
done
echo "==================== E0.1 ALL DONE ($(date -u +%H:%M:%S)) ===================="
