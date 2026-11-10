# E0.4 — T>0 sampling-path audit
**Verdict: the v1 T>0 (T=0.6) correction step is INCORRECT (NOT Leviathan-exact).**
## Static finding (frozen notebook, Cells 12/24/42)
On rejection the v1 code draws the correction/bonus token **directly from the full model distribution P_H** at the reject position (`full_logits.argmax()` for greedy; `multinomial(softmax(full_logits/T))` for T>0). It never forms the normalized residual `P_res = norm(max(0, P_H - P_S))`. Therefore the T>0 path is **not** Leviathan-exact rejection sampling and is **not lossless for T>0**.
## Toy unit test (10 prompts, exact analytic + Monte-Carlo)
For a single draft token with draft x~P_S and accept prob min(1, P_H(x)/P_S(x)):

- correct residual emit(x) = min(P_S,P_H)(x) + max(0,P_H-P_S)(x) = **P_H(x)** (exact)
- v1-direct emit(x) = min(P_S,P_H)(x) + TV(P_S,P_H)·P_H(x) (**biased** unless TV=0)

| prompt | TV(P_H,P_S) | v1 analytic TV→target | residual analytic TV→target | v1 MC TV | residual MC TV |
|---|---|---|---|---|---|
| 0 | 0.273823 | 0.123865 | 0.000e+00 | 0.124218 | 0.000837 |
| 1 | 0.231716 | 0.113250 | 6.939e-18 | 0.113088 | 0.000459 |
| 2 | 0.335548 | 0.094659 | 0.000e+00 | 0.094525 | 0.000777 |
| 3 | 0.224574 | 0.148796 | 0.000e+00 | 0.149135 | 0.000718 |
| 4 | 0.489159 | 0.187742 | 0.000e+00 | 0.187137 | 0.000882 |
| 5 | 0.298994 | 0.132128 | 0.000e+00 | 0.132185 | 0.000816 |
| 6 | 0.394558 | 0.159139 | 0.000e+00 | 0.159433 | 0.000882 |
| 7 | 0.424867 | 0.157240 | 0.000e+00 | 0.157736 | 0.000702 |
| 8 | 0.244010 | 0.095397 | 0.000e+00 | 0.095896 | 0.000725 |
| 9 | 0.155664 | 0.083546 | 0.000e+00 | 0.083255 | 0.000992 |

Max TV-to-target: v1-direct = **0.1877** (biased), residual = 6.9e-18 (exact, numerical zero). Monte-Carlo (2,000,000 draws/prompt) confirms the analytic values.

## Consequence for the manuscript (PLAN E0.4 rule)
This is a **writing fix, not a code fix** (DO-NOT list item: do not re-implement lossless residual sampling for T=0.6). All **T=0.6 tables are relabelled as draft-quality diagnostics**; the losslessness claim is retained only for the greedy (T=0) results, whose residual mismatch is a separate bf16-numerics question handled by E0.3. The Leviathan-exact `residual_correction_token` is added to `specdiag.specloop` for reference but is deliberately NOT wired into the v1-reproduction path.
