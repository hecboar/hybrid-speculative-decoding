# specdiag

The revision experiments were refactored out of the original submission's
notebook (`notebook/hybrid_self_speculative_decoding.ipynb` at the root of this
repository) into a small package, so that each experiment script is a thin
driver over shared code rather than a copy of a notebook cell.

The notebook was treated as read-only: extraction was done from a text dump of
its cells, never by executing or editing it. The table below records where each
module came from, so any number in the paper can be traced back to the cell that
first produced it.

Modules import on a CPU-only machine — `torch`, `transformers`, `datasets` and
`scipy` are imported lazily inside the functions that need them — so the
pure-math paths (`metrics`, `stats`, `io`) run anywhere. `python tests/test_pure.py`
exercises those paths with 22 assertions and needs no GPU.

## Module provenance

| Module | Extracted from | Notes |
|---|---|---|
| `models.py` | Cell 7 (registry, loaders, `get_decoder_layers`, `get_max_inference_seq_length`, `unload_model`); Cell 9 (`estimate_flop_ratio`, `estimate_flops_gflops`); Cell 14 (falcon-h1-3b spec) | The HF token is read from the environment; Cell 4's hardcoded token was dropped. |
| `controller.py` | Cell 9 `DraftModelManager` + the Cell 10 hotfix (the live patch functions) | Superset: adds `component_only_partial`, `attention_only`, `attn_subset` and policy-based layer selection. Legacy strategy strings are aliased. |
| `specloop.py` | Cell 19 (`simulate_acceptance_rate`), Cells 12 and 42 (`measure_speedup_v2`) | v1 correction-token behaviour preserved; a separate Leviathan-exact `residual_correction_token` is provided but deliberately not wired in. |
| `metrics.py` | Cell 16 (D_TV, top-1, KL), Cell 19 (acceptance) | `dtv_top100` and `perplexity` are new; the notebook had no live PPL. |
| `timing.py` | Cells 11, 12, 24, 42 (CUDA-event pattern) | `RoundBreakdown` (draft/verify/accept split) is new, for E3.2. |
| `stats.py` | Cell 5 (`bootstrap_ci`) | Permutation, Wilcoxon, Friedman + Holm, Spearman, McNemar and the power note are all new, for E5. |
| `data.py` | Cells 12, 16, 19, 28 (WikiText and task prompts) | `val_tune_test_split`, `build_long_context_prompts` and `load_recent_corpus` are new, for E4 and E5.2. |
| `io.py` | Cell 5 (`CheckpointManager`, save helpers) | `write_meta_json`, `validate_schema` and the pandas-free `write_csv` are new. |

## Where this code departs from the notebook

These are deliberate, and each one is acknowledged in the article or its
supplementary material.

1. **No residual rejection sampling in v1.** For `T > 0` the v1 correction token
   is drawn straight from the full model's logits at the reject position, not
   from `norm(max(0, P_H - P_S))`, so it is not Leviathan-exact. That behaviour
   is preserved in `spec_generate_v1`; the correct draw is implemented
   separately in `residual_correction_token`. This is why every `T = 0.6` number
   in the paper is labelled a draft-quality diagnostic rather than a lossless
   speculative-decoding result. The greedy results, which carry the paper's
   claims, are unaffected.
2. **D_TV was full-vocabulary in v1**, not top-100. Both `dtv_full` (the v1
   reproduction) and `dtv_top100` are provided.
3. **KL is KL(P_full || P_draft)**, despite the notebook comment saying the
   reverse.
4. **`A_k` equals tau_bar / k exactly** — the mean per-token accepted fraction,
   not the rate of rounds in which all k tokens are accepted. See
   `results/E0/README_e06.md`.
5. **No `meta.json` and no live perplexity existed in v1.** Both were written
   from scratch for the revision.
