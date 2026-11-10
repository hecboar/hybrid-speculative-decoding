# E0.3 — FP32 match-rate control (R2.5, R3)

Greedy, 30 prompts, 64 new tokens, prefix<= 256 tok. Metric: batched-verify argmax vs incremental-AR argmax, scored by the audited v1 `accept_reject` (greedy). Divergences are drafter-independent under greedy decoding (see script docstring).
GPU: RTX PRO 4500 (non-L4; GPU-agnostic metric — see blockers.md B1).

| model | dtype | token_agreement | exact_match_rate | mean_matched_prefix | median_margin_at_div |
|---|---|---|---|---|---|
| falcon-h1-0.5b | fp32 | 1.0000 | 1.000 | 64.0 | n/a |
| falcon-h1-0.5b | bf16 | 0.9901 | 0.433 | 43.4 | 0.0156 |
| qwen3.5-0.8b | fp32 | 1.0000 | 1.000 | 64.0 | n/a |
| qwen3.5-0.8b | bf16 | 0.9891 | 0.600 | 45.4 | 0.0000 |

**Verdict.** fp32 match_rate near 1.0 confirms v1's 90-96% agreement as bf16 non-determinism: greedy spec-decoding is lossless, and the residual mismatches occur exactly where the top1-top2 logit margin is below bf16 resolution (~1e-2..1e-1). If fp32 mismatches PERSIST, the rejection path has a real bug — STOP and debug before other GPU work (PLAN E0.3).
