# results/v1 — immutable v1 baseline snapshot

This directory is a **frozen, read-only snapshot** of the eleven raw v1 result
CSVs that back the submitted manuscript
(`CAEE_Manuscript_Component_Aware_Self_Speculative_Decoding.pdf` /
`paper4_CEE_final.tex`), copied verbatim from the `results/` root on
2026-07-07 (Phase 0, PLAN §3 first action).

**Do not modify anything in this directory** (CLAUDE.md standing instruction).
All new outputs go to `results/<experiment_id>/`; nothing is ever written here
or to `results/` root. These files are the input to E0.6 (metric normalization,
`results/E0/v1_metrics_normalized.csv`) and E0.2 (per-prompt data audit).

## Files (11)

| File | Rows | Backs paper table (approx.) |
|---|---|---|
| `acceptance_rate.csv` | 18 | Acceptance A_k (α) grid, 3 models × k{2,4,8} × T{0,0.6} |
| `divergence_summary.csv` | 3 | D_TV / top-1 agreement / KL per model |
| `optimal_k.csv` | 3 | Optimal k*, theoretical vs empirical speedup |
| `paper2_correlation.csv` | 2 | PPL-ablation ↔ acceptance correlation (only 2 pts — R2.8 attack) |
| `quality_check.csv` | 3 | Lossless-ness match-rate — **superseded, see below** |
| `quality_check_v2.csv` | 3 | Lossless-ness match-rate — **authoritative** |
| `speedup.csv` | 9 | Wall-clock speedup — **superseded, see below** |
| `speedup_v2.csv` | 9 | Wall-clock speedup — **authoritative** |
| `strategy_comparison.csv` | 2 | Qwen3.5 early_exit vs layer_skip (k=4) |
| `task_acceptance.csv` | 9 | Task-suite acceptance (mmlu/gsm8k/alpaca) |
| `theoretical_speedup.csv` | 18 | S_theory grid = (1+τ̄)/(1+k·c) |

## Duplicate `*_v2` vs non-`v2` files — authoritative version

Per PLAN §3: where a `*_v2.csv` and its non-`v2` sibling coexist, **the `_v2`
file is authoritative** (it matches the submitted paper). Both are kept here;
the discrepancy is recorded below. These two generations plausibly correspond
to the `*_v3` vs `*_v3fix` pickle generations on Drive (to be confirmed in E0.2
once `checkpoints/v1_drive/` is populated).

### `quality_check.csv` (superseded) vs `quality_check_v2.csv` (authoritative)

Same schema (`model,strategy,match_rate,mismatches`). Match rates differ
substantially — the "fix" materially improved measured losslessness:

| model | strategy | v1 match_rate / mism. | **v2 (authoritative)** |
|---|---|---|---|
| qwen3.5-0.8b | linear_only | 0.64 / 18 | **0.96 / 2** |
| falcon-h1-0.5b | ssm_only | 0.74 / 13 | **0.96 / 2** |
| qwen2.5-0.5b | layer_skip | 0.50 / 25 | **0.90 / 5** |

The paper quotes the ~0.90–0.96 (v2) match rates. The jump from ~0.5–0.74 to
~0.90–0.96 is exactly what E0.3 (fp32 control) must explain: the residual
mismatches are the hypothesised bf16-nondeterminism cases. The non-v2 numbers
are retained only as provenance evidence, never quoted.

### `speedup.csv` (superseded) vs `speedup_v2.csv` (authoritative)

Schema differs: `speedup.csv` carries an extra `draft_cache` column
(True only for falcon-h1-0.5b/ssm_only) that `speedup_v2.csv` drops. Wall-clock
numbers differ at the ~1–15% level (re-measurement noise / cache handling), e.g.
falcon-h1-0.5b k=2 speedup 0.3483 (v1) → **0.3420 (v2)**;
qwen2.5-0.5b k=2 0.5753 (v1) → **0.4959 (v2)**. The `_v2` values back the
submitted Table 7. **Latency numbers must never be mixed between generations**
(PLAN §4 E0.2 provenance rule).

## Notes carried into Phase 0

- `theoretical_speedup.csv` already uses the τ̄-based formula
  S_theory = (1+τ̄)/(1+k·c) with τ̄ = `mean_accepted_length` and c = `flop_ratio`
  (verified: Falcon-H1-0.5B k=2 → (1+1.36)/(1+2·0.7843137) = 0.918779, matching
  the stored 0.9187786). The paper's textual "geometric-α Eq. 1" is being
  removed in the revision (PLAN §6); the CSV itself needs no numeric change,
  only re-labelling. This is finalised in `results/E0/v1_metrics_normalized.csv`.
- These CSVs are **aggregates only** — no per-prompt vectors. Whether per-prompt
  records exist for the R2.3 permutation test (E5.3) depends on the Drive
  pickles; see `results/E0/checkpoint_inventory.md`.
- `paper2_correlation.csv` has only **2 points** (the R2.8 weakness); E6
  rebuilds this table to 7–8 points.
