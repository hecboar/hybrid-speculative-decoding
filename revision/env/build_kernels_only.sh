#!/usr/bin/env bash
# =============================================================================
# build_kernels_only.sh — reinstall the official mamba-ssm/causal-conv1d fast
# path on a fresh (ephemeral) pod, using the EXACT recipe proven in
# rerun_kernels4.sh (source build, no cache, no deps, torch protected).
#
# torch is assumed already canonical (2.4.1+cu124); this script only rebuilds
# the kernels. It does NOT run any experiment — the caller does that after the
# import + fast-path gates pass. Hard-fails on a non-L4 GPU. 45-min box.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
TS="$(date +%F_%H%M%S)"
BUILD_LOG="logs/kernels_build_attnonly_${TS}.log"
mkdir -p logs
CANON="2.4.1+cu124"
TORCH_IDX="https://download.pytorch.org/whl/cu124"
BOX=2700
DEADLINE=$(( $(date +%s) + BOX ))
remaining() { echo $(( DEADLINE - $(date +%s) )); }
say() { echo "[build_kernels $(date -u +%H:%M:%S)] $*"; }
torch_is_canonical() { python -c "import torch,sys; sys.exit(0 if torch.__version__=='${CANON}' else 1)" 2>/dev/null; }

# GPU assert (L4) — kernels re-measure is latency-sensitive
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
say "GPU: ${GPU_NAME:-<none>}"
case "$GPU_NAME" in
  *L4*) say "L4 confirmed." ;;
  *) echo "FATAL: requires NVIDIA L4 (got '${GPU_NAME:-none}'). Aborting." >&2; exit 1 ;;
esac

if ! torch_is_canonical; then
  say "FATAL: torch is $(python -c 'import torch;print(torch.__version__)' 2>&1), expected $CANON. Aborting."
  exit 1
fi
say "torch canonical ($CANON) OK."

# Step 1: clean slate
say "STEP 1: pip cache purge + uninstall mamba/causal"
pip cache purge >/dev/null 2>&1 || true
pip uninstall -y mamba-ssm causal-conv1d >/dev/null 2>&1 || true
pip install -q ninja >/dev/null 2>&1 || true

# Step 2: pinned SOURCE build (no cache, no deps) — proven recipe
R=$(remaining); [ "$R" -le 0 ] && { say "box expired before build"; exit 1; }
say "STEP 2: source build (budget ${R}s); log -> $BUILD_LOG"
set +e
timeout "$R" bash -c '
  set -e
  export MAX_JOBS=4 MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE
  echo "[build] causal-conv1d==1.6.2.post1 (source, no cache, no deps) ..."
  pip install --no-build-isolation --no-cache-dir --no-deps causal-conv1d==1.6.2.post1
  echo "[build] mamba-ssm==2.3.2.post1 (source, no cache, no deps) ..."
  pip install --no-build-isolation --no-cache-dir --no-deps mamba-ssm==2.3.2.post1
' >"$BUILD_LOG" 2>&1
BUILD_RC=$?
set -e
[ "$BUILD_RC" = "124" ] && { say "source build exceeded box -> STOP"; exit 124; }
say "source build rc=$BUILD_RC (tail: $(tail -1 "$BUILD_LOG"))"

# torch must be UNCHANGED
if ! torch_is_canonical; then
  say "CRITICAL: torch changed to $(python -c 'import torch;print(torch.__version__)' 2>&1) — restoring."
  pip install --no-deps --force-reinstall "torch==2.4.1" --index-url "$TORCH_IDX" >/dev/null 2>&1 || true
  torch_is_canonical || { say "FATAL: could not restore torch"; exit 1; }
fi
say "torch STILL canonical after build."

# Import gate
if python -c "import mamba_ssm, causal_conv1d; print('IMPORT OK', mamba_ssm.__version__)"; then
  say "IMPORT GATE PASSED."
else
  say "IMPORT GATE FAILED -> STOP"; exit 1
fi
say "ALL_DONE_BUILD_KERNELS"
