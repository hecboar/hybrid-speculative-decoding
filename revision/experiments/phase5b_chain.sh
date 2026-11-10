#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."
FILT='it/s\]|Loading checkpoint|Loading weights|Warning|deprecated|fast path|not available|torch_dtype|Downloading|Resolving|Generating'
echo "### E5.1 Qwen n=1000 k=2,4 ($(date -u +%H:%M:%S))"
timeout 1500 python experiments/e5_1_n1000.py --models qwen3.5-0.8b --n 1000 --ks 2,4 2>&1 | grep -avE "$FILT"
echo "### E5.2 held-out corpus C4 ($(date -u +%H:%M:%S))"
timeout 1500 python experiments/e5_2_redpajama.py --n 200 --ks 2,4 2>&1 | grep -avE "$FILT"
echo "### PHASE5B_DONE ($(date -u +%H:%M:%S))"
