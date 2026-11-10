# E4 — long-context sweep: interpretation (R2.9, R3, tests §3.4 claim)

Falcon-H1-0.5B, greedy, k in {1,2}, n_per_len=5. Lengths 512/1024/2048 complete;
4096 partial (component_only k=1); **8192 cut for wall-clock** (naive-mamba
forward ~4-8 s at 8192; VRAM was NOT the limiter — 4096 fit in ~4 GB; the 30-min
box was). Latency non-canonical (RTX PRO 4500); c_eff / tau are GPU-agnostic.

## Key result: c_eff is FLAT with context length
| L | c_eff (component_only) | AR tok/s |
|---|---|---|
| 512  | 0.987 | 4.03 |
| 1024 | 0.992 | 2.05 |
| 2048 | 0.992 | 1.02 |
| 4096 | 0.993 | 0.50 |

**c_eff does NOT drop as context grows** — it is pinned at ~0.99 across an 8x
range. The §3.4 hypothesis (the SSM drafter's recurrent state should make it
relatively cheaper at long context, lowering c_eff and enabling speedup) does
**not** materialize in the current HuggingFace execution path. Both draft and
full forwards scale the SAME way with L (AR tok/s falls ~linearly: 4.0->0.5),
because the naive-mamba scan dominates BOTH and attention is a small add-on at
these lengths. Reported as measured, this SUPPORTS the
infrastructure-bottleneck framing (Ending B): the recurrent-state advantage is
real in principle but is masked by the un-accelerated execution path. With
optimized SSM kernels the draft forward would get cheaper faster than the full
forward as L grows, which is where a long-context speedup would appear — again a
kernel/infrastructure matter, out of scope under HARD CONSTRAINT 1.
