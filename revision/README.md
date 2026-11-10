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
| Nine-model evaluation: eight hybrids across five families and both integration paradigms, plus a pure-Transformer control | `experiments/e1_acceptance.py`, `e2_qwen_variants.py`, `e_task_acceptance_granite.py` | `results/E1`, `results/E2` |
| Hardware-measured draft/verify cost ratio `c_eff` under the three execution scenarios (L4-naive, official SSM kernels, attention-only) | `experiments/e3_partial.py`, `e3_2_benchmark.py`, `e_with_kernels.py`, `e_with_kernels_attention.py`, `l4_reconcile.py` | `results/E3`, `results/E3.1` |
| Long-context `c_eff` sweep | `experiments/e4_long_context.py` | `results/E4` |
| Statistical tests: permutation, Wilcoxon, Friedman + Holm, Spearman | `experiments/e5_3_permutation.py`, `e7_stats.py`, `src/specdiag/stats.py` | `results/E5`, `results/E7` |
| Held-out corpus replications (n = 1000; RedPajama) | `experiments/e5_1_n1000.py`, `e5_2_redpajama.py`, `e5_2_redpajama_real.py` | `results/E5` |
| fp32 losslessness control | `experiments/e03_fp32_match.py` | `results/E0/fp32_match.{csv,md,meta.json}` |
| Ablation-perplexity / acceptance correlation | `experiments/e6_ppl.py`, `e6_spearman.py` | `results/E6` |
| Early-exit forensics and sampling audit | `experiments/e05_early_exit_forensics.py`, `scripts/e04_sampling_audit.py` | `results/E0` |
| Environment lock and kernel provenance for the canonical NVIDIA L4 runs | `env/` | `results/E0/env/`, `results/E0/arch_manifest.json` |

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
├── bootstrap.sh             ← Pod bootstrap (base deps, HF_HOME)
├── env/                     ← Environment lock and launch kit for the L4 re-run
│   ├── L4_README.md         ← Launch instructions and the three-scenario design
│   ├── requirements_canonical_L4.txt   ← Full pip freeze of the canonical pod
│   ├── requirements_revision.txt       ← Pinned direct dependencies
│   ├── run_L4.sh            ← Headless canonical re-run (hard-fails off an L4)
│   ├── build_kernels_only.sh           ← Installs mamba-ssm + causal-conv1d
│   └── kernels_shim/sitecustomize.py   ← Forces the kernel path on/off
├── src/specdiag/            ← Library extracted from the v1 notebook (see its README)
├── experiments/             ← One script per experiment (E0–E7)
├── scripts/                 ← Sampling audit and v1-metric normalisation
├── tests/test_pure.py       ← CPU-only assertions for the math and IO paths
├── results/                 ← Raw CSV / JSON / pickle output, by experiment
│   └── E0/env/              ← Pod environment lock + canonical L4 kernel provenance
└── figures/                 ← The six figures of the published article
```

`env/L4_README.md` was written for the working repository, where this directory
was the root and `env/` was called `revision/`. Read its paths accordingly.

## Environment and kernel provenance

The `c_eff` figures of the kernels scenario hold only when the accelerated
state-space kernels are actually active, so the stack that produced them is
recorded rather than described:

- `results/E0/env/key_versions.txt` — the canonical L4 stack behind
  `c_eff = 0.7843`: `torch 2.4.1+cu124`, `transformers 5.13.0`,
  `mamba-ssm 2.3.2.post1` and `causal-conv1d 1.6.2.post1`, both source-built
  with the fastpath active. The same file also records the Blackwell and
  CPU-only support pods, on which the kernels were *not* installed and the
  kernel result is therefore not reproducible.
- `results/E0/env/requirements_lock.txt` — full `pip freeze` of the pod.
- `results/E0/env/nvidia_smi.txt` — the GPU as the driver reported it.
- `results/E0/env/l4_provenance/` — verbatim copies of the pod-local kernel
  status files. The originals lived in `results/.l4markers/`, which is
  gitignored and pod-local, so these copies are what survived teardown.

Every latency run also carries its own `*.meta.json` with the GPU name and a
`gpu_is_l4` flag, so a reader can tell which rows are canonical without
trusting the directory name.

## Reproducing

The latency-sensitive experiments (E3.1, E3.2, E4) are hardware-bound and are
canonical only on an NVIDIA L4 24GB; `run_L4.sh` aborts on any other GPU. The
remaining experiments (acceptance, divergence, perplexity, statistics) are
GPU-agnostic.

```bash
# Base environment
pip install -r env/requirements_revision.txt
# Or the exact canonical pod
pip install -r env/requirements_canonical_L4.txt

# Optional: the accelerated state-space kernels. The c_eff numbers of the
# kernels scenario depend on these being active.
bash env/build_kernels_only.sh

export HF_TOKEN="your_token_here"

# Full canonical re-run on an L4 (checkpoint-safe, resumes on re-run)
bash env/run_L4.sh

# Or a single experiment
python experiments/e1_acceptance.py
```

The pipeline is checkpoint-safe: completed experiments auto-skip on re-run.

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
