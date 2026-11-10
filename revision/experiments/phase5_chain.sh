#!/usr/bin/env bash
# Phase-5 robustness chain: waits for E4, then E5.1 (n=1000) + E5.2 (held-out
# corpus). Sequential (single GPU). Per-step wall-clock caps honour the 30-min
# rule; each driver is checkpoint-safe (merges by key).
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
FILT='it/s\]|Loading checkpoint|Loading weights|Warning|deprecated|fast path|not available|torch_dtype|Downloading|Resolving|Generating'

echo "### wait for E4 ($(date -u +%H:%M:%S))"
while pgrep -f "e4_long_context.py" >/dev/null; do sleep 20; done
echo "### E4 done ($(date -u +%H:%M:%S))"

echo "### E5.1 n=1000 (Qwen3.5-0.8b + lfm2) ($(date -u +%H:%M:%S))"
timeout 2700 python experiments/e5_1_n1000.py --models qwen3.5-0.8b,lfm2 --n 1000 --ks 2,4,8 2>&1 | grep -avE "$FILT"

echo "### E5.2 held-out corpus (Falcon-0.5b + Qwen3.5-0.8b) ($(date -u +%H:%M:%S))"
timeout 1500 python experiments/e5_2_redpajama.py --n 200 --ks 2,4 2>&1 | grep -avE "$FILT"

echo "### PHASE5_CHAIN DONE ($(date -u +%H:%M:%S))"
