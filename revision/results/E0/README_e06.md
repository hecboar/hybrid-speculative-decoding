# E0.6 — v1 metric normalization: notes & verdict

**Generator:** `scripts/e06_normalize_v1_metrics.py` (stdlib only, no GPU).
**Output:** `results/E0/v1_metrics_normalized.csv` (18 rows) — the immutable basis
for every revised v1 acceptance/theoretical-speedup table.

## What was verified (all PASS)

1. **S_theory formula reproduction — exact.** Recomputing
   `S_theory = (1 + τ̄) / (1 + k·c)` from the v1 `mean_accepted_length` (τ̄) and
   `flop_ratio` (c_param) reproduces the stored `theoretical_speedup.csv` values
   with **max abs error 0.0 over all 18 rows**, including the headline
   Falcon-H1-0.5B, k=2, T=0 → **0.918779** (PLAN E0.6 target 0.919, τ̄=1.36,
   c=0.784). The v1 CSV column already used this τ̄-based formula; only the
   paper's *textual* "geometric-α Eq. 1" is being removed (PLAN §6). No v1
   number changes — this is a re-labelling, not a re-computation.

## Definitional finding (must go into the revised methods section)

**v1's `acceptance_rate` (α) is numerically `τ̄ / k`, exactly, for all 18
conditions** (verified, 0 error). i.e. v1's estimator is the **mean per-token
accepted fraction** `E[na]/k` (from `bootstrap_ci(accepted_lengths / k)`), *not*
literally "the fraction of rounds in which all k draft tokens are accepted" as
PLAN §6's prose phrases the renamed `A_k`. These two quantities differ in general
(mean-fraction ≥ all-k-rate). The revised manuscript should either:
- keep reporting the mean-fraction estimator and define `A_k := E[na]/k`
  precisely (recommended — it is what all v1 tables and figures already use), or
- switch to the strict all-k-accepted rate, which would require re-deriving every
  acceptance number from per-prompt `na` vectors (blocked on E0.2 pickles).

The `A_k` column in `v1_metrics_normalized.csv` carries the v1 mean-fraction
values verbatim; `tau_bar` is reported alongside so downstream tables never have
to reconstruct one from the other.

## Break-even verdict (feeds E3.1 / the Day-5 ending decision)

Break-even is `S_theory > 1 ⟺ τ̄/k > c ⟺ A_k > c_param`. Under the **parameter-count**
cost `c_param`:

> **0 of 18 v1 conditions reach S_theory > 1.**

The closest is Falcon-H1-0.5B ssm_only, k=2 (S_theory = 0.919; A_k = 0.68 vs
c_param = 0.784). This quantifies exactly why v1's all-layer internal drafter
cannot be net-positive on parameter-count cost, and sets the bar for E3.1: only a
**smaller measured cost** (`c_eff`, e.g. via partial drafters / long context) can
push any point above the τ̄/k = c diagonal. This normalized baseline is what the
E3.1 acceptance–cost frontier figure is plotted against.

_Caveat:_ `c_param` here is the v1 FLOP/parameter-count ratio. The revision's Eq. 1
replacement uses **measured** `c_eff`; whether any regime crosses 1.0 is decided
in E3.1/E4, not here.
