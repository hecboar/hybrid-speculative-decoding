#!/usr/bin/env bash
# =============================================================================
# run_L4.sh — re-run ONLY the latency-sensitive experiments (E3.1, E3.2, E4) on
# a CANONICAL NVIDIA L4, at full scale, and reconcile the data-freeze decision.
#
# Hard-fails on a non-L4 GPU. Idempotent install. Checkpoint-safe at BLOCK
# granularity (completed blocks auto-skip via results/.l4markers/*.done), so a
# dead pod resumes with the SAME command. RTX PRO 4500 rows are PRESERVED (each
# CSV gains a `gpu` column; L4 rows are appended, not substituted). Commits +
# pushes after each block.
#
# Launch on a fresh pod (after bootstrap.sh):
#   tmux new -d -s run 'bash revision/run_L4.sh 2>&1 | tee logs/L4_$(date +%F).log'
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
export PYTHONUNBUFFERED=1
mkdir -p logs results/.l4markers
MARK=results/.l4markers

say() { echo "[run_L4 $(date -u +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- 1. GPU assert
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
say "GPU: ${GPU_NAME:-<none>}"
case "$GPU_NAME" in
  *L4*) say "L4 confirmed — canonical run proceeding." ;;
  *) echo "FATAL: canonical latency run REQUIRES an NVIDIA L4 (got '${GPU_NAME:-none}'). Aborting." >&2
     exit 1 ;;
esac

# --------------------------------------------------- 2. idempotent pinned stack
STACK_OK=0
python - <<'PY' >/dev/null 2>&1 && STACK_OK=1
import torch
assert torch.cuda.is_available()
x = torch.randn(512, 512, device="cuda"); float((x @ x).sum())   # real kernel launch
import transformers, datasets, accelerate, scipy   # noqa
PY
if [ "$STACK_OK" = "1" ]; then
  say "GPU stack already functional — skipping install."
else
  say "installing pinned stack (revision/requirements_revision.txt)..."
  pip install -q -r revision/requirements_revision.txt || { echo "FATAL: dep install failed" >&2; exit 1; }
  python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    say "torch missing/CPU-only — installing a default (sm_89-capable) torch wheel..."
    pip install -q torch || { echo "FATAL: torch install failed" >&2; exit 1; }
  }
fi
# NOTE: the main E3.1/E3.2/E4 re-runs are the **L4-naive** scenario, so we do NOT
# install the mamba kernels here. A fresh L4 pod has no mamba-ssm -> naive path.
# (If the base image already ships mamba-ssm, the "naive" label is inaccurate; we
# record the actual state so l4_reconcile can note it.)
if python -c "import mamba_ssm, causal_conv1d" 2>/dev/null; then
  say "WARN: mamba-ssm already present -> main L4 runs will use the fast path (not truly naive)."
  echo "main_stack=mamba_ssm_present" > "$MARK/kernels_main.txt"
else
  say "mamba kernels absent -> main L4 runs are NAIVE (as intended). Kernels installed later."
  echo "main_stack=naive" > "$MARK/kernels_main.txt"
fi
python - <<'PY'
import torch
print("[run_L4] torch", torch.__version__, "| device", torch.cuda.get_device_name(0))
PY

# ---------------------------------------------------- 3. checkpoint-safe blocks
run_block() {   # run_block <name> <cmd...>
  local name="$1"; shift
  local marker="$MARK/${name}.done"
  if [ -f "$marker" ]; then say "$name already complete (marker) — skip."; return 0; fi
  say "=== $name START ==="
  if "$@"; then
    touch "$marker"
    git add -A
    git commit -q -m "L4 canonical: $name re-run on $GPU_NAME" || true
    git push origin HEAD 2>&1 | tail -1 || say "push failed (will retry next block)"
    say "=== $name DONE + committed ==="
  else
    local rc=$?; say "=== $name FAILED (rc=$rc) — marker NOT written; rerun to resume ==="
    return $rc
  fi
}

# E3.1 — partial-drafter frontier, full scale (n=150, k=1,2,4)
run_block E31 python experiments/e3_partial.py --n 150 --ks 1,2,4

# E3.2 — benchmark audit, full scale (max_new=64, reps=3, n_v1=30, n_opt=150)
run_block E32 python experiments/e3_2_benchmark.py --ks 1,2,4 --n_v1 30 --n_opt 150 --reps 3 --max_new 64

# E4 — long context to 8192 (VRAM-checked; caps itself if 24GB can't hold 8192),
#      50 prompts/length, canonical timing reps.
run_block E4 env E4_TIMING_REPS=8 python experiments/e4_long_context.py \
    --lengths 512,1024,2048,4096,8192 --n_per_len 50 --ks 1,2

# --------------------------- 3b. FINAL time-boxed block: official mamba stack ---
# Install the OFF-THE-SHELF mamba-ssm + causal-conv1d wheels (a released library,
# NOT kernel work under constraint 1) and re-measure c_eff/S_theory for
# component_only on Falcon-H1-0.5B only, k in {1,2}, E3.1-style + one E4 sweep.
# Whole block time-boxed to 3h; wheel install time-boxed to 30 min -> skip (no
# build debugging) on failure. On skip we STILL mark done so re-runs don't retry.
KMARK="$MARK/E_KERNELS.done"
if [ -f "$KMARK" ]; then
  say "E_KERNELS already complete (marker) — skip."
else
  say "=== E_KERNELS START (official mamba stack, 3h box) ==="
  ( set +e
    timeout 10800 bash -c '
      set -e
      echo "[E_KERNELS] pip installing causal-conv1d + mamba-ssm (30-min box; wheels preferred)..."
      if ! timeout 1800 pip install -q causal-conv1d mamba-ssm >>logs/L4_kernels_install.log 2>&1; then
        echo "[E_KERNELS] wheel install failed/timed out within 30 min -> SKIP (not debugging builds)."
        echo "kernels_block=skipped_install_failed" > results/.l4markers/kernels_result.txt
        exit 42
      fi
      if ! python -c "import mamba_ssm, causal_conv1d" 2>/dev/null; then
        echo "[E_KERNELS] mamba-ssm/causal-conv1d not importable after install -> SKIP."
        echo "kernels_block=skipped_import_failed" > results/.l4markers/kernels_result.txt
        exit 42
      fi
      echo "[E_KERNELS] kernels active -> re-measuring component_only (k=1,2) + E4 sweep..."
      python experiments/e_with_kernels.py --n 150 --ks 1,2 --lengths 512,1024,2048,4096 --n_per_len 30
      echo "kernels_block=done" > results/.l4markers/kernels_result.txt
    ' )
  krc=$?
  # 0 = ran; 42 = cleanly skipped; anything else = error (still record + continue)
  [ "$krc" = "42" ] && say "E_KERNELS SKIPPED (see logs/L4_kernels_install.log)." \
                     || say "E_KERNELS finished (rc=$krc)."
  touch "$KMARK"
  git add -A
  git commit -q -m "L4 canonical: E_KERNELS (official mamba stack) on $GPU_NAME (rc=$krc)" || true
  git push origin HEAD 2>&1 | tail -1 || true
  say "=== E_KERNELS DONE ==="
fi

# ------------------------------------------------ 4. reconcile decision + figs
say "=== RECONCILE START ==="
python experiments/l4_reconcile.py
python experiments/make_figures.py 2>&1 | grep -E "wrote|skip" || true
git add -A
git commit -q -m "L4 canonical: reconcile decision memo + regenerate figures ($GPU_NAME)" || true
git push origin HEAD 2>&1 | tail -1 || say "final push failed — re-run to retry"
say "=== ALL DONE. See report/DATA_FREEZE_DECISION.md (L4 RECONCILIATION section). ==="
grep -A3 "### " report/DATA_FREEZE_DECISION.md | tail -6 || true
