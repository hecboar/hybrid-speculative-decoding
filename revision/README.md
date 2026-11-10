# Revision materials

This directory holds the experiments added during peer review, for the published
version of the study:

> Borobia, H., Seguí-Mas, E., Tormo-Carbó, G. (2026). Component-aware
> self-speculative decoding for hybrid language models: An architectural
> viability study. *Computers and Electrical Engineering*, 140, 111505.
> <https://doi.org/10.1016/j.compeleceng.2026.111505>

The top level of this repository holds the pipeline of the original submission
(notebook, checkpoints, CSVs and figures). Everything added in revision is here.

## What was added

| Addition | Code | Data |
|---|---|---|
| Nine-model evaluation: eight hybrids across five families and both integration paradigms, plus a pure-Transformer control | `experiments/e1_acceptance.py`, `e1_divergence.py`, `e2_qwen_variants.py`, `e_task_acceptance_granite.py` | `results/E1`, `results/E2` |
| Hardware-measured draft/verify cost ratio `c_eff` under the three execution scenarios | `experiments/e3_partial.py`, `e3_2_benchmark.py`, `e_with_kernels.py`, `e_with_kernels_attention.py` | `results/E3`, `results/E3.1` |
| Long-context `c_eff` sweep | `experiments/e4_long_context.py` | `results/E4` |
| Statistical tests: permutation, Wilcoxon, Friedman + Holm, Spearman | `experiments/e5_3_permutation.py`, `e7_stats.py`, `src/specdiag/stats.py` | `results/E5`, `results/E7` |
| Held-out corpus replications (n = 1000; RedPajama) | `experiments/e5_1_n1000.py`, `e5_2_redpajama.py`, `e5_2_redpajama_real.py` | `results/E5` |
| fp32 losslessness control | `experiments/e03_fp32_match.py` | `results/E0/fp32_match.{csv,md,meta.json}` |
| Ablation-perplexity / acceptance correlation | `experiments/e6_ppl.py`, `e6_spearman.py` | `results/E6` |
| Early-exit forensics and the sampling audit | `experiments/e05_early_exit_forensics.py`, `scripts/e04_sampling_audit.py` | `results/E0` |
| Environment lock and kernel provenance | `env/` | `results/E0/env/` |

## Headline results

- Under greedy decoding, the parallel hybrid Falcon-H1-0.5B reaches a mean
  acceptance rate of A₂ = 0.680; the sequential hybrid Qwen3.5-0.8B reaches
  0.038.
- The governing factor is functional attention-dependence rather than the
  parallel/sequential label: acceptance falls as the perplexity increase under
  attention suppression grows (Spearman ρ = −0.79, p = 0.028 over the eight
  hybrids; ρ = −0.75 with the control included).
- The separation is stable at the 0.5B and 3B scales tested (permutation test,
  p = 0.962).
- Acceptance viability does not by itself yield speedup. With hardware-measured
  `c_eff`, and even with the official accelerated state-space kernels active, no
  drafter reaches a theoretical speedup above one (best S_theory = 0.986). With
  the kernels active the binding constraint is acceptance, not drafting cost,
  which has fallen to its parameter-count floor.

## Layout

```
revision/
├── README.md                ← You are here
├── env/                     ← Dependencies, kernel build, canonical stack
├── src/specdiag/            ← The library the experiment scripts drive
├── experiments/             ← One script per experiment (E0–E7)
├── scripts/                 ← Sampling audit and v1-metric normalisation
├── tests/test_pure.py       ← CPU-only assertions for the math and IO paths
├── results/                 ← Raw CSV / JSON / pickle output, by experiment
│   └── E0/env/              ← Environment lock and L4 kernel provenance
└── figures/                 ← The six figures of the published article
```

## Reproducing

The latency-sensitive experiments (E3, E3.1, E4) are hardware-bound and are
canonical only on an NVIDIA L4 24GB. The rest — acceptance, divergence,
perplexity, the statistical tests — are hardware independent. See
[`env/README.md`](env/README.md) for the exact stack and why the kernel build
is part of the result.

```bash
pip install -r env/requirements.txt
bash env/install_kernels.sh        # only for the kernels scenario
export HF_TOKEN="your_token_here"

python experiments/e1_acceptance.py
```

The pipeline is checkpoint-safe: completed experiments auto-skip on re-run.

```bash
python tests/test_pure.py          # 22 CPU-only assertions, no GPU needed
```

## Inspecting the data without a GPU

```python
import pandas as pd

# Acceptance across the models added in revision
acc = pd.read_csv("results/E1/acceptance_new_models.csv")

# Measured draft/verify cost ratio c_eff and theoretical speedup
frontier = pd.read_csv("results/E3/partial_drafter_ablation.csv")

# The same frontier with the accelerated state-space kernels active
kernels = pd.read_csv("results/E3.1/with_kernels_e31.csv")

# Ablation perplexity vs acceptance, and the statistical tests
corr = pd.read_csv("results/E6/ppl_spearman.csv")
tests = pd.read_csv("results/E7/stat_tests.csv")
```

The latency-sensitive CSVs carry a `gpu` column, and each experiment writes a
sibling `*.meta.json` recording the GPU, the library versions and whether the
state-space kernels were active. Per-prompt raw output is under the `perprompt/`
subdirectories.
