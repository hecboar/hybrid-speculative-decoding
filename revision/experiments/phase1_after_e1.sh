#!/usr/bin/env bash
# Post-E1 GPU batch (run after e1_acceptance finishes). Sequential to avoid GPU
# contention. Each block writes its own results file; commit per block.
set -u
cd "$(dirname "$0")/.."
echo "### 1. positive control: reproduce v1 falcon-0.5b/qwen3.5-0.8b acceptance"
python experiments/e1_acceptance.py falcon-h1-0.5b qwen3.5-0.8b --k 2,4 --n 150 2>&1 \
  | grep -avE "it/s\]|Loading checkpoint|Warning|deprecated|fast path|not available|torch_dtype"
echo "### 2. zamba2 acceptance (new parallel family, patched controller)"
python experiments/e1_acceptance.py zamba2 --k 2,4 --n 150 2>&1 \
  | grep -avE "it/s\]|Loading checkpoint|Warning|deprecated|fast path|not available|torch_dtype"
echo "### DONE phase1_after_e1"
