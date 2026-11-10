#!/usr/bin/env python3
"""L4 + official mamba-ssm/causal-conv1d re-measure — attention_only drafter.

Companion to e_with_kernels.py (which does component_only). Same protocol, same
reused infra (E3.1 measure_c_eff / acceptance / c_param_for), only the draft mode
changes: attention_only = drafts with the ATTENTION branch, the SSM branch is
zeroed (parallel-arch diagnostic, PLAN controller spec 3.1). With the official
mamba-ssm + causal-conv1d wheels active, Falcon-H1's fast SSM path is used inside
the verify (full) forward, so the c_eff ratio (draft/full) must be re-measured on
the kernel stack rather than reused from the naive-path E3.1 row.

Installing a released library is NOT kernel work under HARD CONSTRAINT 1.
Latency ratios are canonical on L4 -> this script hard-fails on a non-L4 GPU and
only runs if mamba_ssm + causal_conv1d import.

Headline: c_eff(attention_only, kernels) and
  S_theory(k=1) = (1 + tau_bar_1) / (1 + 1 * c_eff)
using the ALREADY-MEASURED tau_bar_1 of attention_only (from the frozen L4 E3.1
row in results/E3/partial_drafter_ablation.csv). Acceptance is also re-measured
here purely as a consistency check (kernels do not change which tokens accept).

Output (new file; frozen component_only files untouched):
  results/E3.1/attention_only_kernels.csv
    model,strategy,k,A_k,ci_lo,ci_hi,tau_bar_frozen,tau_bar_measured,
    c_param,c_eff,s_theory,n_prompts,gpu,stack
  results/E3.1/attention_only_kernels.meta.json
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))   # import sibling drivers

import torch  # noqa: E402
from specdiag import models as M  # noqa: E402
from specdiag.models import load_model_and_tokenizer  # noqa: E402
from specdiag.controller import DraftModeController  # noqa: E402
from specdiag.data import load_wikitext, val_tune_test_split  # noqa: E402
from specdiag.io import write_csv, write_meta_json, gpu_tag  # noqa: E402
import e3_partial as E31   # noqa: E402  (reuse measure_c_eff / acceptance / c_param_for)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "results", "E3.1")
OUT_CSV = os.path.join(OUTDIR, "attention_only_kernels.csv")
META = os.path.join(OUTDIR, "attention_only_kernels.meta.json")
FROZEN_E3 = os.path.join(REPO, "results", "E3", "partial_drafter_ablation.csv")
MODEL = "falcon-h1-0.5b"
STRAT = "attention_only"


def kernels_present():
    try:
        import mamba_ssm  # noqa: F401
        import causal_conv1d  # noqa: F401
        return True
    except Exception:
        return False


def frozen_tau_bar(k, gpu_pref="L4"):
    """Read the already-measured tau_bar for attention_only at the given k from
    the frozen E3.1 ablation CSV. Prefer the L4 row (canonical), fall back to any."""
    best = None
    with open(FROZEN_E3) as fh:
        for row in csv.DictReader(fh):
            if row["strategy"] == STRAT and int(row["k"]) == k:
                if gpu_pref in row.get("gpu", ""):
                    return float(row["tau_bar"]), row.get("gpu", "")
                best = (float(row["tau_bar"]), row.get("gpu", ""))
    if best is None:
        raise RuntimeError(f"no frozen attention_only tau_bar for k={k} in {FROZEN_E3}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--maxlen", type=int, default=256)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    if not kernels_present():
        print("mamba_ssm / causal_conv1d NOT importable — L4-kernels scenario SKIPPED.")
        return 2
    try:
        import mamba_ssm
        stack = f"mamba_ssm-{getattr(mamba_ssm, '__version__', '?')}+causal_conv1d"
    except Exception:
        stack = "mamba_ssm+causal_conv1d"

    GPU = gpu_tag()
    if "L4" not in GPU:
        print(f"FATAL: attention_only kernels re-measure is latency-sensitive and "
              f"requires an L4 (got {GPU}). Aborting.")
        return 1
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"L4-kernels re-measure | strategy={STRAT} | "
          f"GPU={torch.cuda.get_device_name(0)} | stack={stack}")

    model, tok = load_model_and_tokenizer(MODEL)
    spec = M.MODEL_SPECS[MODEL]
    ctrl = DraftModeController(model, MODEL)
    _, test = val_tune_test_split(load_wikitext(split="validation", threshold=100), 50)
    test = test[:args.n]
    tids = tok(test[0], return_tensors="pt", truncation=True,
               max_length=args.maxlen)["input_ids"].to("cuda")

    set_fn = lambda: ctrl.set_mode("attention_only")   # noqa: E731
    cparam = E31.c_param_for(spec, STRAT, 0.5)
    c_eff, t_draft, t_full = E31.measure_c_eff(model, ctrl, set_fn, tids)
    print(f"  {STRAT} c_eff (kernels) = {c_eff:.5f}  "
          f"[t_draft={t_draft:.2f}ms t_full={t_full:.2f}ms]  c_param={cparam:.5f}")

    rows = []
    for k in ks:
        tau_frozen, tau_gpu = frozen_tau_bar(k)
        r = E31.acceptance(model, tok, ctrl, set_fn, k, test, args.maxlen)
        # S_theory uses the ALREADY-MEASURED tau_bar of attention_only (frozen L4 row)
        s_theory = (1 + tau_frozen) / (1 + k * c_eff)
        rows.append({"model": MODEL, "strategy": STRAT, "k": k,
                     "A_k": round(r["A_k"], 5), "ci_lo": round(r["ci_lo"], 5),
                     "ci_hi": round(r["ci_hi"], 5),
                     "tau_bar_frozen": round(tau_frozen, 4),
                     "tau_bar_measured": round(r["tau"], 4),
                     "c_param": round(cparam, 5), "c_eff": round(c_eff, 5),
                     "s_theory": round(s_theory, 5), "n_prompts": len(test),
                     "gpu": GPU, "stack": stack})
        flag = "  <-- S_theory>1!" if s_theory > 1.0 else ""
        print(f"  k={k} A_k={r['A_k']:.4f} tau_frozen={tau_frozen:.4f} "
              f"(from {tau_gpu}) tau_meas={r['tau']:.4f} c_eff={c_eff:.5f} "
              f"S_theory={s_theory:.5f}{flag}")
        write_csv(rows, OUT_CSV, fieldnames=list(rows[0].keys()))

    write_meta_json(META, cwd=REPO, experiment="L4-kernels-attention_only", model=MODEL,
                    strategy=STRAT, stack=stack, gpu=GPU, ks=ks, n_prompts=len(test),
                    c_eff_attention_only=c_eff, c_param_attention_only=cparam,
                    t_draft_ms=t_draft, t_full_ms=t_full, gpu_is_l4=("L4" in GPU),
                    s_theory_uses_frozen_tau_bar=True,
                    note="attention_only drafter (SSM branch zeroed); official "
                    "mamba-ssm/causal-conv1d wheels (off-the-shelf lib, not kernel work); "
                    "canonical L4 latency ratios; S_theory(k=1) uses the already-measured "
                    "tau_bar_1 of attention_only from the frozen E3.1 L4 row",
                    metric_is_gpu_agnostic=True)
    print("wrote:", OUT_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
