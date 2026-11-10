# src/specdiag — package provenance & verification status

> **Read this first.** The document below was written mid-revision and is kept
> as the provenance record of how the package was extracted. Two things in it
> are out of date for the released version:
>
> 1. **The `UNVERIFIED (blocker B1)` markers are historical.** They date from
>    the weeks when the available pod carried an sm_120 GPU that the pinned
>    torch build could not drive, so no GPU path could be exercised. Every path
>    listed below as unverified was subsequently run on the canonical NVIDIA L4
>    and on the RTX pod, and produced the results published with the article:
>    `_attention_only` → `results/E3.1/attention_only_kernels.csv`,
>    `_component_only` and `_component_only_partial` →
>    `results/E3/partial_drafter_ablation.csv`, `specloop` →
>    `results/E1/acceptance_new_models.csv`, `metrics.perplexity` →
>    `results/E6/ppl_ablation.csv`, `timing` →
>    `results/E3/benchmark_audit.csv`, `data.build_long_context_prompts` →
>    `results/E4/long_context.csv`, `data.load_recent_corpus` →
>    `results/E5/acceptance_recent_corpus.csv`. The markers were left in the
>    source rather than edited out, so that the released code is byte-identical
>    to the code that produced those numbers.
> 2. **References to `PLAN.md`, `CLAUDE.md`, numbered constraints,
>    `report/DATA_FREEZE_DECISION.md` and `checkpoints/v1_drive/`** point to the
>    authors' internal working repository, which is not part of this release.
>    They are left in place for traceability; nothing in the reproduction path
>    depends on them.

Refactor of the frozen v1 notebook `hybrid_self_speculative_decoding_v4.ipynb`
(constraint 8: read-only) into a reusable package (PLAN §3). Extraction was done
from a text dump of the notebook cells, never by executing or editing it.

## Module → notebook-cell provenance

| Module | Extracted from | Notes |
|---|---|---|
| `models.py` | Cell 7 (registry, loaders, `get_decoder_layers`, `get_max_inference_seq_length`, `unload_model`); Cell 9 (`estimate_flop_ratio`, `estimate_flops_gflops`); Cell 14 (falcon-h1-3b spec) | HF token read from env (Cell 4's hardcoded token dropped). torch/transformers imported lazily. |
| `controller.py` | Cell 9 `DraftModelManager` + **Cell 10 HOTFIX** (the LIVE patch fns) | Superset: adds `component_only_partial`, `attention_only`, `attn_subset`, policy-based layer selection per PLAN §3.1. Legacy strategy strings aliased. |
| `specloop.py` | Cell 19 (`simulate_acceptance_rate`), Cell 12/42 (`measure_speedup_v2`) | v1 correction-token behaviour preserved; separate Leviathan-exact `residual_correction_token` added but NOT wired in. |
| `metrics.py` | Cell 16 (D_TV, top-1, KL), Cell 19 (acceptance) | `dtv_top100` (PLAN §6) and `perplexity` are NEW (no live PPL in notebook). |
| `timing.py` | Cells 11/12/24/42 CUDA-event pattern | `RoundBreakdown` (draft/verify/accept split) is NEW for E3.2. |
| `stats.py` | Cell 5 (`bootstrap_ci`) | permutation / Wilcoxon / Friedman+Holm / Spearman / McNemar / power_note all NEW (E5.x). scipy lazy. |
| `data.py` | Cells 12/16/19/28 (WikiText + task prompts) | `val_tune_test_split`, `build_long_context_prompts`, `load_recent_corpus` are NEW (constraints 6, E4, E5.2). |
| `io.py` | Cell 5 (`CheckpointManager`, save helpers) | `write_meta_json`, `validate_schema`, pandas-free `write_csv` are NEW (constraint 5). |

## Design for blocker B1 (torch has no sm_120 kernels)

Every module imports on a CPU-only / broken-CUDA host: `torch`, `transformers`,
`datasets`, `scipy` are imported lazily inside the functions that need them.
Verified: `python -c "import specdiag.*"` for all 8 modules succeeds, and
`python tests/test_pure.py` passes 22 CPU-only assertions (math + IO paths).

## GPU-dependent paths still UNVERIFIED (grep `# UNVERIFIED (blocker B1)`)

Validate each against a real checkpoint once the torch stack is fixed:

- `models.load_model_and_tokenizer` — real load path (flash-attn fallback, arch_info side-effects).
- `controller._find_ssm_submodule` / `_make_zero_submodule_forward` / `_attention_only` — **SSM/mixer submodule name is a guess**; confirm via `named_children()` on a Falcon-H1 layer before trusting `attention_only`.
- `controller._component_only` (parallel zero-attn return arity) / `_component_only_partial` — validate patched `model.generate(use_cache=True)` still runs.
- `specloop.simulate_acceptance` / `spec_generate_v1` — full draft/verify loops.
- `metrics.perplexity` — new sliding-window PPL (E6); validate against a known WikiText PPL.
- `timing.*` — all CUDA-event timers and `RoundBreakdown` end-to-end.
- `data.build_long_context_prompts` / `load_recent_corpus` — need tokenizer + network.

## Fidelity deviations from v1 (decisions the revision must acknowledge)

1. **No residual rejection sampling in v1.** For T>0 the v1 correction token is
   drawn straight from the full model's logits at the reject position, not from
   `norm(max(0, P_H − P_S))` → NOT Leviathan-exact (E0.4 finding). Preserved in
   `spec_generate_v1`; correct version in `residual_correction_token`.
2. **D_TV was full-vocab in v1**, not top-100. Both `dtv_full` (v1 repro) and
   `dtv_top100` (PLAN §6) are provided.
3. **KL is KL(P_full‖P_draft)** despite the notebook comment saying "draft‖full".
4. **`A_k` (v1 α) == τ̄/k exactly** (mean per-token accepted fraction), not the
   all-k-accepted-round rate — see `results/E0/README_e06.md`.
5. No `meta.json` and no live PPL existed in v1 — both written from scratch.
