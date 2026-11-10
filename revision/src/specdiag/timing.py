"""timing.py — CUDA-event timing helpers + draft/verify/accept breakdown.

Ported from the notebook CUDA-event pattern (Cells 11, 12, 24, 42). The notebook
timed only the WHOLE draft->verify->accept round; E3.2 needs a finer per-phase
breakdown, added here. All timing needs a GPU.
"""
from contextlib import contextmanager

SPEEDUP_WARMUP_RUNS = 5


@contextmanager
def cuda_timer():
    """Yield a dict; on exit fills ['ms'] with elapsed GPU time.

        with cuda_timer() as t:
            ...work...
        print(t['ms'])
    """
    import torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    out = {}
    start.record()
    try:
        yield out
    finally:
        end.record()
        torch.cuda.synchronize()
        out["ms"] = start.elapsed_time(end)


def warmup(fn, n=SPEEDUP_WARMUP_RUNS):
    """Run fn() n times then synchronize (Cell 11/12 pattern)."""
    import torch
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_calls(fn, n=30):
    """Mean ms/call over n timed iterations after the caller has warmed up."""
    with cuda_timer() as t:
        for _ in range(n):
            fn()
    return t["ms"] / n


class RoundBreakdown:
    """NEW (PLAN E3.2): accumulate per-phase timings across speculative rounds.

    Records draft_ms / verify_ms / accept_ms / overhead_ms separately (the v1 code
    only had one enclosing event pair per prompt). Time each phase with cuda_timer
    and feed the ms into `add`.
    """
    def __init__(self):
        self.draft_ms = 0.0
        self.verify_ms = 0.0
        self.accept_ms = 0.0
        self.overhead_ms = 0.0
        self.tokens = 0
        self.rounds = 0

    def add(self, draft_ms=0.0, verify_ms=0.0, accept_ms=0.0, overhead_ms=0.0, tokens=0):
        self.draft_ms += draft_ms
        self.verify_ms += verify_ms
        self.accept_ms += accept_ms
        self.overhead_ms += overhead_ms
        self.tokens += tokens
        self.rounds += 1

    def per_token(self):
        t = max(self.tokens, 1)
        return {"draft_ms_tok": self.draft_ms / t, "verify_ms_tok": self.verify_ms / t,
                "accept_ms_tok": self.accept_ms / t, "overhead_ms_tok": self.overhead_ms / t}

    def c_eff(self):
        """Measured cost ratio = mean draft ms/token / mean verify ms/token."""
        if self.verify_ms <= 0:
            return float("nan")
        return self.draft_ms / self.verify_ms
