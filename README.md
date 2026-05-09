# Component-Aware Self-Speculative Decoding in Hybrid Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2504.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2504.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TL;DR** — Self-speculative decoding viability in hybrid LMs is determined by topology, not scale. Parallel hybrids (Falcon-H1) reach α = 0.68 with SSM-only drafts; sequential hybrids (Qwen3.5) collapse to α = 0.038 — an 18× gap. The perplexity ratio from attention ablation (established in Paper 2) directly predicts which architecture will succeed.

## Key Findings

| # | Finding | Qwen3.5-0.8B (sequential) | Falcon-H1-0.5B (parallel) |
|---|---------|--------------------------|--------------------------|
| 1 | Acceptance rate at k=2 | α = 0.038 (linear-only) | α = 0.68 (SSM-only) |
| 2 | Token-level divergence (D_TV) | 0.803 — distributions diverge | 0.302 — distributions align |
| 3 | PPL ratio (attention ablation) | 81.96× — structural damage | 3.15× — SSM absorbs load |
| 4 | Scale invariance | — | Falcon-H1 3B ≈ 0.5B acceptance |
| 5 | Best alternative strategy | LayerSkip α = 0.233 (6× better than linear-only) | Component-aware is already optimal |
| 6 | PPL ratio predicts viability | Confirmed: < 5× viable; > 20× avoid component-aware | Confirmed cross-model |

## Repository Structure

```
├── README.md                               ← You are here
├── LICENSE                                 ← MIT License
├── requirements.txt                        ← Python dependencies
├── notebook/
│   └── hybrid_self_speculative_decoding.ipynb  ← Full reproducible pipeline
├── figures/                                ← Publication figures (PDF + PNG)
│   ├── paper_acceptance_vs_k.{pdf,png}
│   ├── paper_divergence.{pdf,png}
│   ├── paper_ppl_vs_acceptance.{pdf,png}
│   ├── paper_speedup_theo_vs_empirical.{pdf,png}
│   ├── paper_strategy_comparison.{pdf,png}
│   └── paper_task_acceptance_heatmap.{pdf,png}
├── results/
│   ├── acceptance_rate.csv                 ← α by model × strategy × k × temperature
│   ├── divergence_summary.csv             ← D_TV and KL divergence per model
│   ├── speedup_v2.csv                     ← Wall-clock and theoretical speedup
│   ├── theoretical_speedup.csv            ← Speedup formula sweep over k and α
│   ├── optimal_k.csv                      ← Optimal k per model with speedup
│   ├── strategy_comparison.csv            ← Component-aware vs LayerSkip vs early-exit
│   ├── task_acceptance.csv                ← α per model × task (MMLU, GSM8K, Alpaca)
│   ├── quality_check_v2.csv               ← Output match rates (correctness verification)
│   └── paper2_correlation.csv             ← PPL ratio → acceptance rate correlation
└── checkpoints/                           ← Intermediate experiment state (.pkl)
```

## Quick Start

```bash
# Clone
git clone https://github.com/hecboar/hybrid-speculative-decoding.git
cd hybrid-speculative-decoding
```

**Reproduce from scratch — GPU machine (≥16GB VRAM, CUDA 11.8+, Python 3.10+)**

```bash
pip install -r requirements.txt

# Optional: for Falcon-H1 full sequence length support
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation

# Set HuggingFace token if needed
export HF_TOKEN="your_token_here"

jupyter notebook notebook/hybrid_self_speculative_decoding.ipynb
```

In the notebook, set `ROOT_DIR` to your preferred output path and follow the execution order. The pipeline is checkpoint-safe — completed experiments auto-skip on re-run.

**Inspect results only — load tables directly:**

```python
import pandas as pd

# Acceptance rates across all models and strategies
acc = pd.read_csv("results/acceptance_rate.csv")
k2 = acc[acc.k == 2][["model", "strategy", "acceptance_rate", "ci_lo", "ci_hi"]]
print(k2.to_string())

# PPL ratio → acceptance correlation
corr = pd.read_csv("results/paper2_correlation.csv")
print(corr[["display_name", "ppl_ratio", "alpha_k4"]].to_string())
```

## Models and Strategies

| Model | Topology | Strategies tested | α at k=2 |
|-------|----------|------------------|----------|
| Qwen3.5-0.8B-Base | Sequential (18 GDN + 6 softmax attn) | `linear_only`, `early_exit`, `layer_skip` | 0.038 / 0.000 / 0.233 |
| Falcon-H1-0.5B-Base | Parallel (attn ∥ Mamba-2 per block) | `ssm_only`, `attention_only`, `hybrid` | 0.68 |
| Falcon-H1-3B-Base | Parallel | `ssm_only` | ~0.68 (scale-invariant) |
| Qwen2.5-0.5B | Homogeneous Transformer (baseline) | `layer_skip` | 0.52 |

All experiments: k ∈ {2, 4, 8}, temperature ∈ {0.0, 0.6}, n = 200 trials per condition.

## Evaluation Tasks

| Task | Dataset | Metric |
|------|---------|--------|
| Mathematics | GSM8K | Acceptance rate α, output match rate |
| General instruction | Alpaca | Acceptance rate α, output match rate |
| Knowledge | MMLU | Acceptance rate α, output match rate |

Task-dependent results are in `results/task_acceptance.csv`. GSM8K yields the highest acceptance rates across all models; MMLU the lowest. Output correctness is verified in `results/quality_check_v2.csv` (90–96% match rates across all conditions).

## Practitioner Decision Guide

| Architecture | PPL ratio (attention ablation) | Recommended strategy | Expected α at k=2 |
|-------------|-------------------------------|---------------------|------------------|
| Parallel hybrid (Falcon-H1 family) | < 5× | Component-aware, SSM-only | ~0.68 |
| Homogeneous Transformer | — | LayerSkip | ~0.45–0.52 |
| Sequential hybrid (Qwen3.5 family) | > 20× | LayerSkip (not component-aware) | ~0.23 |
| Unknown hybrid | Compute PPL ratio first | < 5× → component-aware; > 20× → LayerSkip | — |

The PPL ratio is computed by ablating the attention pathway and measuring perplexity — see `results/paper2_correlation.csv` and Paper 2 below.

## Related Papers

This work is the fourth in a series studying hybrid language model internals:

- **Paper 1** — *How Pruning Reshapes Features: Sparse Autoencoder Analysis of Weight-Pruned Language Models* (arXiv 2026)
- **Paper 2** — *Functional Component Ablation Reveals Specialization Patterns in Hybrid Language Model Architectures* (arXiv 2026)
- **Paper 3** — *Where Should LoRA Go? Component-Type Placement in Hybrid Language Models* (arXiv 2026)
- **Paper 4** — This paper: *Component-Aware Self-Speculative Decoding in Hybrid Language Models*

Paper 2 established that in parallel hybrids the SSM absorbs most representational load (low PPL ratio on attention removal), while in sequential hybrids the recurrent backbone dominates (high PPL ratio). Paper 4 shows this same metric — computed identically — directly predicts whether component-aware speculative decoding will succeed or fail.

## Citation

```bibtex
@article{borobia2026speculative,
  title={Component-Aware Self-Speculative Decoding in Hybrid Language Models},
  author={Borobia, Hector and Segu{\'i}-Mas, Elies and Tormo-Carb{\'o}, Guillermina},
  journal={arXiv preprint arXiv:XXXX.XXXXX},  % update with real arXiv ID once assigned
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
