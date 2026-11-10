#!/usr/bin/env python3
"""E1 / E1.b — all-token acceptance + tau_bar for new / scale-point models.

PLAN Phase 1. On val-test (default 150 prompts), greedy, k in {2,4}: all-token
acceptance A_k with bootstrap CI and mean accepted tokens tau_bar. Per-prompt
accepted_lengths are ALWAYS saved (constraint 4).

Metric convention (locked by E0.6): A_k := E[na]/k = tau_bar/k (v1's
mean-fraction estimator, `bootstrap_ci(accepted_lengths/k)`), so new-model
numbers are directly comparable to the v1 tables. We ALSO record the strict
all-k-accepted rate `all_accept_rate = mean(na==k)` as a transparency column
(the literal PLAN sec6 prose); the paper's primary A_k stays the v1 estimator.

Models already in MODEL_SPECS are used directly; new models (lfm2, zamba2, ...)
are registered from results/E0/arch_manifest.json (arch paradigm, layer count).

Usage:
  python experiments/e1_acceptance.py <key> [<key> ...] [--k 2,4] [--n 150]
Output: results/E1/acceptance_new_models.csv (appends/merges by row key) +
        results/E1/perprompt/<model>__<strategy>__k<k>__T<T>.pkl + meta.json
GPU: non-L4 OK — acceptance is GPU-agnostic (blockers.md B1).
"""
import os
import sys
import json
import argparse
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from specdiag import models as M  # noqa: E402
from specdiag.models import load_model_and_tokenizer  # noqa: E402
from specdiag.controller import DraftModeController  # noqa: E402
from specdiag.specloop import simulate_acceptance  # noqa: E402
from specdiag.data import load_wikitext, val_tune_test_split  # noqa: E402
from specdiag.io import write_csv, write_meta_json  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E1 = os.path.join(REPO, "results", "E1")
PP = os.path.join(E1, "perprompt")
MANIFEST = os.path.join(REPO, "results", "E0", "arch_manifest.json")
OUT_CSV = os.path.join(E1, "acceptance_new_models.csv")
OUT_META = os.path.join(E1, "acceptance_new_models.meta.json")

FIELDS = ["model", "arch_paradigm", "strategy", "k", "temperature", "A_k",
          "ci_lo", "ci_hi", "tau_bar", "all_accept_rate", "n_prompts",
          "per_prompt_file"]


def register_from_manifest(key):
    """Add a MODEL_SPECS entry for a new model using its E0.1 manifest record."""
    if key in M.MODEL_SPECS:
        return M.MODEL_SPECS[key]
    man = json.load(open(MANIFEST)).get(key)
    if not man:
        raise SystemExit(f"{key}: not in MODEL_SPECS and no manifest record")
    paradigm = man.get("requested_paradigm", "sequential")
    cfg_class = man.get("config_class", "")
    if "Zamba2" in cfg_class:
        fam = "zamba2_shared_hybrid"   # parallel-style shared-attention (see controller)
    else:
        fam = {"parallel": "falcon_parallel_hybrid",
               "sequential": "qwen_sequential_hybrid"}.get(paradigm, "transformer_baseline")
    n = man.get("n_layers", 0)
    cfg = man.get("config", {})
    M.MODEL_SPECS[key] = {
        "display_name": key, "model_id": man["model_id"], "arch_family": fam,
        "hybrid_type": paradigm, "n_layers": n, "n_alt_layers": n,
        "n_attn_layers": n, "hidden_dim": cfg.get("hidden_size", 0),
    }
    return M.MODEL_SPECS[key]


def run_model(key, ks, n_prompts, temperature, val_test):
    spec = register_from_manifest(key)
    paradigm = spec["hybrid_type"]
    strategy = "component_only"     # ssm_only(par)/linear_only(seq) canonical drafter
    model, tok = load_model_and_tokenizer(key)
    ctrl = DraftModeController(model, key)
    prompts = val_test[:n_prompts]
    rows = []
    for k in ks:
        res = simulate_acceptance(model, tok, ctrl, strategy, k,
                                  temperature=temperature, prompts=prompts,
                                  n_trials=len(prompts), max_len=512)
        na = np.asarray(res["accepted_lengths"])
        all_accept = float(np.mean(na == k))
        ppfile = os.path.join(PP, f"{key}__{strategy}__k{k}__T{temperature}.pkl")
        os.makedirs(PP, exist_ok=True)
        with open(ppfile, "wb") as f:
            pickle.dump({"model": key, "strategy": strategy, "k": k,
                         "temperature": temperature,
                         "accepted_lengths": res["accepted_lengths"]}, f)
        rows.append({
            "model": key, "arch_paradigm": paradigm, "strategy": strategy,
            "k": k, "temperature": temperature,
            "A_k": round(res["acceptance_rate"], 5),
            "ci_lo": round(res["acceptance_rate_ci_lo"], 5),
            "ci_hi": round(res["acceptance_rate_ci_hi"], 5),
            "tau_bar": round(res["mean_accepted_length"], 4),
            "all_accept_rate": round(all_accept, 5),
            "n_prompts": len(prompts),
            "per_prompt_file": os.path.relpath(ppfile, REPO),
        })
        print(f"  {key} k={k}: A_k={res['acceptance_rate']:.4f} "
              f"tau_bar={res['mean_accepted_length']:.3f} "
              f"all_accept={all_accept:.4f} (n={len(prompts)})")
    M.unload_model(key)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--k", default="2,4")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",")]

    print(f"E1 acceptance | GPU={torch.cuda.get_device_name(0)} | keys={args.keys}")
    all_texts = load_wikitext(split="validation", threshold=100)
    _, val_test = val_tune_test_split(all_texts, n_tune=50)
    print(f"val-test pool: {len(val_test)} prompts (using first {args.n})")

    # merge with existing CSV (checkpoint-safe accumulation)
    existing = {}
    if os.path.exists(OUT_CSV):
        import csv as _csv
        for r in _csv.DictReader(open(OUT_CSV)):
            existing[(r["model"], r["strategy"], r["k"], r["temperature"])] = r

    for key in args.keys:
        try:
            rows = run_model(key, ks, args.n, args.temperature, val_test)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:200]} — skipping")
            continue
        for r in rows:
            existing[(r["model"], r["strategy"], str(r["k"]),
                      str(r["temperature"]))] = r
        write_csv(list(existing.values()), OUT_CSV, fieldnames=FIELDS)

    write_meta_json(OUT_META, cwd=REPO, experiment="E1", keys=args.keys,
                    k=ks, n_prompts=args.n, temperature=args.temperature,
                    metric="A_k := E[na]/k (v1 mean-fraction); all_accept_rate=mean(na==k)",
                    gpu_is_l4=False, metric_is_gpu_agnostic=True)
    print("wrote:", OUT_CSV)


if __name__ == "__main__":
    main()
