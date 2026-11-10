#!/usr/bin/env python3
"""L4 + official mamba-ssm/causal-conv1d re-measure (scenario 'L4-kernels').

Re-measures c_eff and S_theory for the component_only SSM drafter on
Falcon-H1-0.5B ONLY, k in {1,2}, E3.1-style (measured c_eff + acceptance) plus
one E4-style length sweep — but with the OFF-THE-SHELF mamba-ssm + causal-conv1d
wheels installed, which activate Falcon-H1's fast SSM path. Installing a released
library is NOT kernel work under HARD CONSTRAINT 1.

Runs ONLY if `mamba_ssm` and `causal_conv1d` import (the run_L4.sh block gates on
this and time-boxes the install). Latency ratios are canonical on L4.

Outputs (separate files so the naive L4 rows are untouched):
  results/E3.1/with_kernels_e31.csv
    model,strategy,k,A_k,ci_lo,ci_hi,tau_bar,c_param,c_eff,s_theory,n_prompts,gpu,stack
  results/E3.1/with_kernels_e4.csv
    prompt_len,strategy,k,ar_tok_s,draft_ms_tok,verify_ms_tok,c_eff,tau_bar,s_theory,n_prompts,gpu,stack
  results/E3.1/with_kernels.meta.json
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))   # import sibling drivers

import torch  # noqa: E402
from specdiag import models as M  # noqa: E402
from specdiag.models import load_model_and_tokenizer  # noqa: E402
from specdiag.controller import DraftModeController  # noqa: E402
from specdiag.data import load_wikitext, val_tune_test_split, build_long_context_prompts  # noqa: E402
from specdiag.io import write_csv, write_meta_json, gpu_tag  # noqa: E402
import e3_partial as E31   # noqa: E402  (reuse measure_c_eff / acceptance / c_param_for)
import e4_long_context as E4mod  # noqa: E402  (reuse sweep_length)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "results", "E3.1")
E31_CSV = os.path.join(OUTDIR, "with_kernels_e31.csv")
E4_CSV = os.path.join(OUTDIR, "with_kernels_e4.csv")
META = os.path.join(OUTDIR, "with_kernels.meta.json")
MODEL = "falcon-h1-0.5b"
POLICY = "keep_first_last_stride"


def kernels_present():
    try:
        import mamba_ssm  # noqa: F401
        import causal_conv1d  # noqa: F401
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--lengths", default="512,1024,2048,4096")
    ap.add_argument("--n_per_len", type=int, default=30)
    ap.add_argument("--maxlen", type=int, default=256)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]

    if not kernels_present():
        print("mamba_ssm / causal_conv1d NOT importable — L4-kernels scenario SKIPPED.")
        return 2
    try:
        import mamba_ssm
        stack = f"mamba_ssm-{getattr(mamba_ssm, '__version__', '?')}+causal_conv1d"
    except Exception:
        stack = "mamba_ssm+causal_conv1d"

    GPU = gpu_tag()
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"L4-kernels re-measure | GPU={torch.cuda.get_device_name(0)} | stack={stack}")

    model, tok = load_model_and_tokenizer(MODEL)
    spec = M.MODEL_SPECS[MODEL]
    ctrl = DraftModeController(model, MODEL)
    _, test = val_tune_test_split(load_wikitext(split="validation", threshold=100), 50)
    test = test[:args.n]
    tids = tok(test[0], return_tensors="pt", truncation=True,
               max_length=args.maxlen)["input_ids"].to("cuda")

    set_fn = lambda: ctrl.set_mode("component_only")   # noqa: E731
    cparam = E31.c_param_for(spec, "component_only", 1.0)
    c_eff, t_draft, t_full = E31.measure_c_eff(model, ctrl, set_fn, tids)
    print(f"  component_only c_eff (kernels) = {c_eff:.3f}  "
          f"[t_draft={t_draft:.2f}ms t_full={t_full:.2f}ms]")

    # ---- E3.1-style: acceptance + S_theory, k in {1,2} ----
    rows31 = []
    for k in ks:
        r = E31.acceptance(model, tok, ctrl, set_fn, k, test, args.maxlen)
        s_theory = (1 + r["tau"]) / (1 + k * c_eff)
        rows31.append({"model": MODEL, "strategy": "component_only", "k": k,
                       "A_k": round(r["A_k"], 5), "ci_lo": round(r["ci_lo"], 5),
                       "ci_hi": round(r["ci_hi"], 5), "tau_bar": round(r["tau"], 4),
                       "c_param": round(cparam, 5), "c_eff": round(c_eff, 5),
                       "s_theory": round(s_theory, 5), "n_prompts": len(test),
                       "gpu": GPU, "stack": stack})
        flag = "  <-- S_theory>1!" if s_theory > 1.0 else ""
        print(f"  k={k} A_k={r['A_k']:.4f} tau={r['tau']:.3f} "
              f"c_eff={c_eff:.3f} S_theory={s_theory:.3f}{flag}")
        write_csv(rows31, E31_CSV, fieldnames=list(rows31[0].keys()))

    # ---- E4-style: one length sweep, component_only, k=1 ----
    rows4 = []
    for L in lengths:
        prompts = build_long_context_prompts(tok, target_lens=(L,),
                                             n_per_len=args.n_per_len)[L]
        try:
            r = E4mod.sweep_length(model, tok, ctrl, set_fn, prompts, 1, L)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  L={L}: OOM -> skip"); continue
        rows4.append({"prompt_len": L, "strategy": "component_only", "k": 1,
                      "ar_tok_s": round(r["ar_tok_s"], 3),
                      "draft_ms_tok": round(r["draft_ms_tok"], 4),
                      "verify_ms_tok": round(r["verify_ms_tok"], 4),
                      "c_eff": round(r["c_eff"], 5), "tau_bar": round(r["tau"], 4),
                      "s_theory": round(r["s_theory"], 5), "n_prompts": len(prompts),
                      "gpu": GPU, "stack": stack})
        print(f"  L={L:5d} c_eff={r['c_eff']:.3f} tau={r['tau']:.3f} "
              f"S_theory={r['s_theory']:.3f}")
        write_csv(rows4, E4_CSV, fieldnames=list(rows4[0].keys()))

    write_meta_json(META, cwd=REPO, experiment="L4-kernels", model=MODEL, stack=stack,
                    gpu=GPU, ks=ks, lengths=lengths, n_prompts=len(test),
                    c_eff_component_only=c_eff, gpu_is_l4=("L4" in GPU),
                    note="official mamba-ssm/causal-conv1d wheels (off-the-shelf lib, not "
                    "kernel work); canonical L4 latency ratios", metric_is_gpu_agnostic=True)
    print("wrote:", E31_CSV, E4_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
