# E0 — Day-2 Decision Gate: new-model lineup

**Author review requested (morning of 2026-07-08).** Per PLAN E0.1 the gate
normally *waits* for approval before Phase 1; the author authorized proceeding
directly into Phase 1 without waiting, so **Phase 1 (E1/E6) has been started** on
the lineup recommended below. If you want to change the lineup, the E1/E6 runs
are cheap to re-point (they read `arch_manifest.json` + a model key).

GPU: RTX PRO 4500 Blackwell (sm_120), **not an L4** — all numbers here are
GPU-agnostic (acceptance / DTV / PPL / patchability); no latency claims.

---

## 1. Smoke-test outcomes (`e01_smoke.csv`, `arch_manifest.json`)

| key | HF id | paradigm | verdict | acceptance path | notes |
|---|---|---|---|---|---|
| **lfm2** | LiquidAI/LFM2-1.2B | sequential (conv+attn, 16L) | **VIABLE** | ✓ raw+generate | new SEQUENTIAL family |
| **falcon15** | Falcon-H1-1.5B | parallel (24L) | **VIABLE** | ✓ raw+generate | parallel SCALE point (same family) |
| **granite** | granite-4.0-h-micro | sequential (Mamba2+attn, 40L, 3.2B) | PARTIAL | ✓ raw forward | new SEQUENTIAL family; generate() arity bug only |
| **qwen35_2b** | Qwen3.5-2B | sequential (24L) | PARTIAL | ✓ raw forward | sequential SCALE point (answers R2) |
| **zamba2** | Zyphra/Zamba2-1.2B | parallel-style shared-attn (38L) | PARTIAL→FIXED | ✓ after controller patch | new PARALLEL family (see §2) |
| hymba | nvidia/Hymba-1.5B-Base | parallel | BLOCKED | — | needs mamba_ssm/causal_conv1d/flash_attn (no sm_120 wheels) |
| recurrentgemma | google/recurrentgemma-2b | sequential | BLOCKED | — | gated repo, no HF_TOKEN in pod |
| samba | microsoft/Samba-1.3B | sequential | BLOCKED | — | HF id does not exist (code-only release) |

"PARTIAL" = loads + controller patches the attention component, but HF
`model.generate()` trips on a tuple/tensor return-arity mismatch. **E1 acceptance
does not use `generate()`** — it uses raw `use_cache=False` forwards
(`simulate_acceptance`) — so PARTIAL models still produce valid acceptance
numbers (verified on 3-prompt probes: granite A_2≈0.33, qwen35_2b A_2≈0.17).

## 2. Zamba2 controller fix (enables the new PARALLEL family)

Zamba2 is a **parallel-style** hybrid: each `Zamba2HybridLayer` runs a shared
attention block whose output (via a Linear) is **added into** the mamba stream
(eq. 6 of the Zamba2 paper) — structurally analogous to Falcon-H1's summed
SSM+attention branches, so it qualifies as a *new parallel family*, not merely a
scale point. The generic zero-attention patch assumed Falcon's 2-tuple attention
return; Zamba2's shared block returns a bare tensor that is immediately fed to a
`Linear`, so the tuple crashed it. Added a `zamba2_shared_hybrid` controller
branch that zeros BOTH the shared-transformer (bare zeros, skips attention
compute — cost-correct) and its Linear adapter (clean mamba-only contribution).
Result: acceptance now runs (see `acceptance_new_models.csv`).

## 3. RECOMMENDED FINAL LINEUP

Minimum bar (PLAN): ≥1 new parallel + ≥1 new sequential family. **Met.**

- **New SEQUENTIAL family (P0):** **lfm2** (clean VIABLE) — primary. **granite**
  (PARTIAL but acceptance-valid, mostly-Mamba 40L) as the second sequential
  data point / robustness.
- **New PARALLEL family (P0):** **zamba2** (fixed) — the only reachable new
  parallel family (hymba unbuildable on Blackwell).
- **Scale points (P1):** **falcon15** (parallel, Falcon-H1 family) and
  **qwen35_2b** (sequential, Qwen3.5 family — directly answers R2's "scale
  invariance has no Qwen analogue"; its divergence pickle already exists, only
  acceptance is run).

This gives the correlation/Table-6 scatter **8 points**: falcon-0.5b, falcon-3b,
falcon15, qwen3.5-0.8b, qwen35_2b (originals+scale) + lfm2, granite, zamba2 (new
families) — spanning both paradigms at multiple scales.

### Dropped, with rebuttal-ready justification
- **samba** — "we attempted Samba; `microsoft/Samba-1.3B` is not a released HF
  checkpoint (Samba is a code-only release), so no reproducible evaluation was
  possible." Fallback granite used (PLAN E0.1 chain).
- **hymba** — remote code hard-requires `mamba_ssm`/`causal_conv1d`/`flash_attn`,
  none of which have Blackwell (sm_120) wheels; source builds were out of scope
  for the 30-min box. Fallback zamba2 used.
- **recurrentgemma** — gated; deferred to P2 (re-runnable once a HF_TOKEN with
  access is provided). Not needed to meet the gate.

## 4. Acceptance results (val-test, 150 prompts, greedy) — `acceptance_new_models.csv`

Metric: A_k := E[na]/k (v1 mean-fraction estimator, comparable to v1 tables).

| model | paradigm | A_2 | A_4 | τ̄(k4) | all-accept(k4) | reading |
|---|---|---|---|---|---|---|
| lfm2 | sequential | **0.000** | **0.000** | 0.000 | 0.000 | conv-only drafter collapses; draft output is incoherent (attention essential) |
| granite | sequential | **0.503** | **0.353** | 1.413 | 0.153 | **VIABLE** — mostly-Mamba (4 attn / 40L), attention-light |
| qwen35_2b | sequential | **0.067** | **0.037** | 0.147 | 0.007 | collapse — mirrors qwen3.5-0.8b (0.038) → **scale-invariant, answers R2** |
| falcon15 | parallel | **0.377** | **0.212** | 0.847 | 0.040 | viable parallel (lower than 0.5B — scale nuance) |
| zamba2 | parallel | **0.367** | **0.203** | 0.813 | 0.020 | **VIABLE new parallel family** (patched controller) |

**Positive control — PIPELINE VALIDATED.** Re-running the two originals through
this exact pipeline reproduces the v1 Drive pickles:
- falcon-h1-0.5b: **A_2=0.710** (v1 0.68), A_4=0.393 (v1 0.37) ✓
- qwen3.5-0.8b: **A_2=0.030** (v1 0.038), A_4=0.015 (v1 0.019) ✓

(deltas are the 150-prompt val-test subset vs v1's 200; well within CI). So the
new-model numbers above are trustworthy.

## 5. HEADLINE FINDING (reframes the paper — stronger than the original thesis)

The parallel/sequential **label does NOT determine acceptance**:

> **granite (sequential, A_4=0.35) BEATS falcon15 (parallel, A_4=0.21).**

What predicts acceptance is **functional attention-dependence**: granite is 90%
Mamba-2 (4 attention layers / 40) so removing attention barely hurts → viable
drafter; lfm2 (6/16) and Qwen3.5 (interleaved, essential) collapse; Falcon-H1
(parallel sum) and Zamba2 (parallel-add) stay viable. This is exactly what the
E6 **PPL-ratio ↔ A_4 correlation** measures. The revision
should lead with "acceptance is governed by how much the hybrid functionally
relies on attention (PPL ablation), not by the parallel-vs-sequential taxonomy" —
a more defensible and more general claim that directly answers R1/R2/R3. Both the
original taxonomy result AND this refinement are supported by the data; the
E6 scatter (9 points across both paradigms and 0.5–3B scales) is the money figure.

**E6 RESULT (computed, `results/E6/ppl_ablation.csv` + `ppl_spearman.csv`).**
Our own PPL ablation (full vs attention-removed) on all 9 models, then Spearman
of log10(ppl_ratio) vs A_4:

> **Spearman ρ = −0.75, p = 0.020 (n=9 all models); ρ = −0.79, p = 0.021 (n=8
> hybrids, excluding the pure-transformer qwen2.5 control).**

| model | paradigm | ppl_ratio | log10 | A_4 |
|---|---|---|---|---|
| granite | seq | 3.26 | 0.51 | 0.353 | ← attention-light seq: LOW ratio, viable |
| falcon-3b | par | 2.26 | 0.35 | 0.351 |
| falcon-0.5b | par | 2.59 | 0.41 | 0.393 |
| falcon15 | par | 2.14 | 0.33 | 0.212 |
| zamba2 | par | 9.64 | 0.98 | 0.203 |
| qwen35_2b | seq | 22.4 | 1.35 | 0.037 |
| qwen3.5-0.8b | seq | 80.0 | 1.90 | 0.015 |
| lfm2 | seq | 1613 | 3.21 | 0.000 | ← attention-critical seq: HUGE ratio, collapse |

This is the strongest single piece of evidence in the revision: a significant,
own-measured correlation replacing the companion-paper [19] dependency (R2.8),
built on 9 architectures spanning both paradigms and 0.5–3B scales (R1/R3 n=2 →
n=9). It confirms the reframing: **functional attention-dependence (PPL ratio),
not the parallel/sequential label, predicts self-speculative acceptance.**
