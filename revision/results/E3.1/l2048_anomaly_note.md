# L=2048 τ̄ anomaly in the L4-kernels long-context sweep (with_kernels_e4.csv)

Analysis-only note (no data re-run, no row edited). Added during the pre-writing
audit to characterise a non-monotone dip so a reviewer question can be answered
honestly.

## The observation

`results/E3.1/with_kernels_e4.csv` (L4-kernels, component_only, k=1, n=30/len):

| L | τ̄ | c_eff | S_theory |
|---|----|-------|----------|
| 512  | 0.733 | 0.776 | 0.976 |
| 1024 | 0.733 | 0.786 | 0.970 |
| 2048 | **0.400** | 0.842 | **0.760** |
| 4096 | 0.700 | 0.830 | 0.929 |

τ̄ dips sharply at L=2048 (0.400 vs ~0.7 elsewhere) and recovers at 4096. The
cost side is *not* anomalous: c_eff at 2048 (0.842) sits in line with 4096
(0.830), so the anomaly is entirely on the **acceptance** side (τ̄), not the cost
side. Because S_theory=(1+τ̄)/(1+k·c_eff), the τ̄ dip alone drags S_theory down
to 0.760 at that single length.

## (a) Is per-prompt data available for this run? — NO

No per-prompt records were saved for the L4-kernels E4 sweep. There is **no
`results/E4/perprompt/` directory and no kernels-run per-prompt pickle anywhere in
the tree** (the only per-prompt pkls are for E1/E2/E3/E5). `with_kernels_e4.csv`
stores a single aggregate mean per length over n=30. We therefore **cannot**
decompose the 2048 dip into "few outliers vs broad shift" from saved data, and we
do not re-run anything (data freeze). With k=1, τ̄ is the mean accepted-draft-token
count per round over 30 prompts, so 0.400 ≈ 12/30 vs 0.733 ≈ 22/30 — a coarse,
high-variance point estimate.

## (b) Does the anomaly appear in the L4-naive run too? — YES (reproduces)

The L4-naive E4 sweep (`results/E4/long_context.csv`, component_only, k=1) uses the
**identical WikiText-2 concatenated prompt construction**. Its τ̄ sequence:

| L | τ̄ (L4-naive, n=50) | τ̄ (RTX-naive, n=5) | τ̄ (L4-kernels, n=30) |
|---|----|----|----|
| 512  | 0.68 | 0.60 | 0.733 |
| 1024 | 0.72 | 0.60 | 0.733 |
| 2048 | **0.50** | **0.40** | **0.400** |
| 4096 | 0.62 | 0.40* | 0.700 |

(*RTX 4096 is the partial single-row point.) **All three independent runs — L4-naive
(n=50), RTX-naive (n=5) and L4-kernels (n=30) — show L=2048 as the lowest-τ̄ length.**
The dip is therefore *not* specific to the kernels run and *not* a kernels artifact:
it is a property of the prompt set. Same construction → both draft stacks dip →
prompt-set-specific / content-driven.

## (c) Honest one-sentence explanation for a reviewer

> With only n=30–50 prompts per length, τ̄ at any single context length is a
> high-variance point estimate; the dip at L=2048 reproduces across the
> independent naive-L4, RTX and kernels runs on the identical WikiText-2
> concatenated prompt set, indicating it is content-driven (that particular
> 2048-token prompt slice is intrinsically harder for the SSM branch to draft)
> rather than a kernels artifact or a genuine monotone context effect — and it
> does not affect the headline, since the best credible S_theory=0.986 comes from
> the short-context E3.1 run (n=150), while the long-context sweep only
> establishes that c_eff stays roughly flat with L (the recurrent-state advantage
> does not materialise, supporting the infrastructure-bottleneck framing).

The row is left exactly as measured — not smoothed, dropped, or re-run.
