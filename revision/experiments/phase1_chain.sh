#!/usr/bin/env bash
# Autonomous Phase-1 GPU chain. Waits for the in-flight e1_acceptance to finish,
# then runs the remaining GPU blocks sequentially (single GPU -> no concurrency).
# Each block has a wall-clock cap so a hang cannot stall the whole night
# (author directive: never wait > 30 min on one model). Every block writes its
# own results file; the driver Python is checkpoint-safe (merges by key), so a
# killed block loses only its current model.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
FILT='it/s\]|Loading checkpoint|Loading weights|Warning|deprecated|fast path|not available|torch_dtype'

echo "### wait for in-flight e1_acceptance ($(date -u +%H:%M:%S))"
while pgrep -f "e1_acceptance.py lfm2 granite" >/dev/null; do sleep 20; done
echo "### e1_acceptance done ($(date -u +%H:%M:%S))"

echo "### BLOCK 1 positive control: falcon-0.5b + qwen3.5-0.8b acceptance ($(date -u +%H:%M:%S))"
timeout 2400 python experiments/e1_acceptance.py falcon-h1-0.5b qwen3.5-0.8b --k 2,4 --n 150 2>&1 | grep -avE "$FILT"

echo "### BLOCK 2 zamba2 acceptance (new parallel, patched controller) ($(date -u +%H:%M:%S))"
timeout 1800 python experiments/e1_acceptance.py zamba2 --k 2,4 --n 150 2>&1 | grep -avE "$FILT"

echo "### BLOCK 3 divergence (new families lfm2 granite zamba2) ($(date -u +%H:%M:%S))"
timeout 2400 python experiments/e1_divergence.py lfm2 granite zamba2 --n 100 --pos 50 2>&1 | grep -avE "$FILT"

echo "### BLOCK 4 E6 PPL ablation (all 8 models) ($(date -u +%H:%M:%S))"
# originals first (small), then new/scale, then the 3B ones last
timeout 3000 python experiments/e6_ppl.py qwen2.5-0.5b falcon-h1-0.5b qwen3.5-0.8b \
    lfm2 falcon15 qwen35_2b zamba2 falcon-h1-3b granite --n_docs 200 2>&1 | grep -avE "$FILT"

echo "### PHASE1_CHAIN DONE ($(date -u +%H:%M:%S))"
