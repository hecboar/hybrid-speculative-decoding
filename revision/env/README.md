# Environment

The acceptance, divergence, perplexity and statistical results are hardware
independent. The cost-ratio results are not: `c_eff` is a measured
draft/verify latency ratio, so it only means what the paper says it means on
the hardware and software stack recorded here.

## Files

| File | What it is |
|---|---|
| `requirements.txt` | The pinned direct dependencies. |
| `requirements_lock_L4.txt` | Full `pip freeze` of the canonical pod. |
| `install_kernels.sh` | Source-builds `mamba-ssm` and `causal-conv1d` for the kernels scenario. |
| `kernels_shim/sitecustomize.py` | Compatibility shim, only needed if `import mamba_ssm` fails on renamed `transformers.generation` symbols. |

The machine-readable provenance of the canonical run lives next to the data, in
[`../results/E0/env/`](../results/E0/env/): `key_versions.txt`,
`requirements_lock.txt`, `nvidia_smi.txt` and `l4_provenance/`. Those files are
verbatim snapshots taken before the pod was torn down, not edited for release.

## The canonical stack

The latency numbers in the paper were produced on a single **NVIDIA L4 24GB**
with:

```
python           3.11.10
torch            2.4.1+cu124   (CUDA 12.4)
transformers     5.13.0
mamba-ssm        2.3.2.post1   (source-built, fastpath active)
causal-conv1d    1.6.2.post1   (source-built)
```

`c_eff = 0.7843` for `component_only` on Falcon-H1-0.5B comes from that stack.
Without `mamba-ssm` and `causal-conv1d` installed and active, the same code
measures a materially different ratio, so the two packages are part of the
result rather than an optional speed-up.

## The three execution scenarios

`c_eff` was measured three ways, and the paper reports all three rather than the
most favourable one:

1. **Naive** — no accelerated state-space kernels, which is what a fresh pod
   gives you. Comparable like-for-like against the original submission's
   measurements. → `results/E3/partial_drafter_ablation.csv`
2. **Kernels** — the official `mamba-ssm` and `causal-conv1d` wheels installed
   and the fastpath confirmed active, re-measuring `component_only` on
   Falcon-H1-0.5B. → `results/E3.1/with_kernels_e31.csv`
3. **Attention-only** — the complementary drafter, under the same kernel stack.
   → `results/E3.1/attention_only_kernels.csv`

With the kernels active, drafting cost falls to its parameter-count floor and
the binding constraint becomes acceptance. That is the result, and it is why the
best theoretical speedup is still below one.

## Installing

```bash
pip install -r env/requirements.txt          # or requirements_lock_L4.txt
bash env/install_kernels.sh                  # only for the kernels scenario
export HF_TOKEN="your_token_here"            # gated checkpoints
```

`install_kernels.sh` aborts on a non-L4 GPU, because a kernel build timed on
other hardware would not be comparable to the published numbers. Every latency
run writes a `*.meta.json` beside its CSV carrying the GPU name and a
`gpu_is_l4` flag, so canonical rows can be told apart from exploratory ones
without trusting a directory name.
