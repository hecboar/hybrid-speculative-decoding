# L4 canonical re-run — launch instructions

`revision/run_L4.sh` re-runs ONLY the latency-sensitive experiments — **E3.1**
(partial-drafter frontier + measured c_eff), **E3.2** (v1-vs-optimized benchmark),
**E4** (long-context c_eff sweep) — at full scale on a **canonical NVIDIA L4**,
then reconciles the data-freeze decision. Everything else (acceptance / DTV / PPL
/ stats) is GPU-agnostic and already final from the RTX PRO 4500 pod.

## One-command launch (fresh L4 pod)

```bash
# 0. clone + bootstrap (installs base deps, sets HF_HOME, etc.)
bash bootstrap.sh

# 1. launch the canonical re-run in tmux, teeing a dated log
tmux new -d -s run 'bash revision/run_L4.sh 2>&1 | tee logs/L4_$(date +%F).log'

# 2. watch it
tmux attach -t run          # detach: Ctrl-b d
tail -f logs/L4_$(date +%F).log
```

That's it. The script is fully headless and self-contained.

## What it does / guarantees

- **Hard-fails on a non-L4 GPU** (checks `nvidia-smi` name; aborts if not `*L4*`).
  The GPU name is written into every `meta.json` and into a new `gpu` column in
  every CSV it touches.
- **Idempotent install**: verifies torch+CUDA with a real kernel launch and the
  pinned `transformers/datasets/accelerate/scipy` (`revision/requirements_revision.txt`);
  installs only if missing. Also **best-effort installs `mamba-ssm` +
  `causal-conv1d`** (they have sm_89 wheels) which activate Falcon-H1's fast path
  on L4 — this is the main reason the canonical `c_eff` can differ from the RTX
  naive-path number. Whether the kernels activated is recorded in
  `results/.l4markers/kernels.txt`.
- **Checkpoint-safe at block granularity**: each block writes
  `results/.l4markers/<BLOCK>.done` only after it commits. Re-running the exact
  same command **skips completed blocks**, so a pod that dies mid-run resumes
  cleanly. (Markers are pod-local / git-ignored so a *fresh* pod re-runs.)
- **Non-destructive**: RTX PRO 4500 rows are preserved. Each results CSV gains a
  `gpu` column; L4 rows are **appended** (old rows are back-filled with
  `NVIDIA_RTX_PRO_4500_Blackwell`). Both sets coexist so the paper can cite
  canonical L4 numbers and the summary can show the RTX→L4 delta.
- **Commits + pushes after each block**; logs under `logs/`.
- **Final step** runs `experiments/l4_reconcile.py`, which recomputes
  c_eff/S_theory from the L4 rows and rewrites the `L4 CANONICAL RECONCILIATION`
  section of `report/DATA_FREEZE_DECISION.md`. **If any credible (non-diagnostic)
  drafter reaches S_theory > 1.0 on the L4, it flags the change LOUDLY and says
  to switch the manuscript to Ending A**; otherwise it annotates that Ending B
  holds with the canonical numbers. Figures are regenerated (they will contain
  both RTX and L4 points — that IS the delta visualisation).

## Three scenarios (the point of the run)

The main E3.1/E3.2/E4 blocks run **L4-naive** (no mamba kernels — a fresh L4 pod
has none), so they compare like-for-like against the RTX **v1-naive** rows. Then
a FINAL time-boxed block installs the **official mamba-ssm + causal-conv1d wheels**
(an off-the-shelf library — NOT kernel work under constraint 1) and re-measures
c_eff/S_theory for `component_only` on Falcon-H1-0.5B only, k∈{1,2}, E3.1-style
plus one E4 length sweep → **L4-kernels**. `l4_reconcile.py` then reports all three
(**v1-naive / L4-naive / L4-kernels**) and states explicitly whether any credible
regime reaches S_theory > 1.0 **with the official stack**.

## Full-scale parameters used (restored from the RTX wall-clock trims)

| block | command | scenario |
|---|---|---|
| E3.1 | `e3_partial.py --n 150 --ks 1,2,4` | L4-naive |
| E3.2 | `e3_2_benchmark.py --ks 1,2,4 --n_v1 30 --n_opt 150 --reps 3 --max_new 64` | L4-naive |
| E4   | `E4_TIMING_REPS=8 e4_long_context.py --lengths 512,1024,2048,4096,8192 --n_per_len 50 --ks 1,2` | L4-naive |
| E_KERNELS | 3h box: `pip install causal-conv1d mamba-ssm` (30-min box, skip on fail — no build debugging) → `e_with_kernels.py --n 150 --ks 1,2 --lengths 512,1024,2048,4096 --n_per_len 30` | **L4-kernels** |

E4 runs the VRAM check from 8192 downward and caps itself if 24 GB cannot hold a
length (documented in `results/E4/long_context.meta.json`). The E_KERNELS block
writes to `results/E3.1/with_kernels_{e31,e4}.csv` (marked `stack=mamba_ssm`,
`gpu=NVIDIA_L4`); it is skipped-but-marked-done if the wheels don't install
within 30 min (state recorded in `results/.l4markers/kernels_result.txt`), so the
run never blocks on a source build.

## To force a clean re-run of a block

```bash
rm results/.l4markers/E31.done      # (or E32.done / E4.done / E_KERNELS.done)
bash revision/run_L4.sh
```

## Expected outputs

- `results/E3/partial_drafter_ablation.csv`, `results/E3/benchmark_audit.csv`,
  `results/E4/long_context.csv` — now with a `gpu` column and L4 rows alongside RTX.
- `results/E3.1/with_kernels_e31.csv`, `results/E3.1/with_kernels_e4.csv` —
  the L4-kernels scenario (if the wheels installed).
- Updated `report/DATA_FREEZE_DECISION.md` (3-scenario reconciliation section).
- Regenerated `figures/fig_e3_frontier.*`, `fig_e3_2_breakdown.*`,
  `fig_e4_ceff_vs_len.*`.
