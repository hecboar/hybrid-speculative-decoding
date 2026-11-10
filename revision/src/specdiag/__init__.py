"""specdiag — Component-Aware Self-Speculative Decoding diagnostics package.

Refactor of the frozen v1 notebook `hybrid_self_speculative_decoding_v4.ipynb`
into a reusable package (PLAN §3). Modules are designed to IMPORT on CPU without
a working CUDA stack (torch/datasets/scipy are imported lazily inside the
functions that need them) so that the pure-math paths (metrics, stats, io) run
even while blocker B1 (torch has no sm_120 kernels) is open.

See src/specdiag/README.md for the module -> notebook-cell provenance map and
the list of GPU-dependent paths still marked UNVERIFIED.
"""

__all__ = [
    "models", "controller", "specloop", "metrics",
    "timing", "stats", "data", "io",
]

# Global constants ported from notebook Cell 4 (no torch needed).
SEED = 42
MAX_NEW_TOKENS = 64
SPEEDUP_TRIALS = 100
SPEEDUP_WARMUP_RUNS = 5
DRAFT_LENGTHS = [2, 4, 8]
TEMPERATURES = [0.0, 0.6]
BOOTSTRAP_N = 10_000
NUM_GENERATIONS = 200
