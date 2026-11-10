#!/usr/bin/env python3
"""E5.1 — enlarged n for low-acceptance conditions (PLAN Phase 5; R2.2).

Qwen3.5-0.8B (linear-only) and a new sequential model (lfm2) at n=1000 prompts,
k in {2,4,8}, greedy. Additionally `stats.power_note`: given observed alpha and
n, report the bootstrap CI width to justify n=200 for high-alpha conditions.

Output: results/E5/acceptance_n1000.csv (schema as E1) + per-prompt pkls +
power_note.csv + meta.json. GPU-agnostic (acceptance).
"""
import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from specdiag import models as M, stats  # noqa: E402
from specdiag.models import load_model_and_tokenizer  # noqa: E402
from specdiag.controller import DraftModeController  # noqa: E402
from specdiag.specloop import simulate_acceptance  # noqa: E402
from specdiag.data import load_wikitext, val_tune_test_split  # noqa: E402
from specdiag.io import write_csv, write_meta_json  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E5 = os.path.join(REPO, "results", "E5")
PP = os.path.join(E5, "perprompt")
OUT_CSV = os.path.join(E5, "acceptance_n1000.csv")
POWER_CSV = os.path.join(E5, "power_note.csv")
OUT_META = os.path.join(E5, "acceptance_n1000.meta.json")
FIELDS = ["model", "arch_paradigm", "strategy", "k", "temperature", "A_k",
          "ci_lo", "ci_hi", "tau_bar", "all_accept_rate", "n_prompts", "per_prompt_file"]


def register(key):
    if key in M.MODEL_SPECS:
        return
    man = __import__("json").load(open(os.path.join(REPO, "results/E0/arch_manifest.json"))).get(key)
    par = man["requested_paradigm"]
    fam = {"parallel": "falcon_parallel_hybrid", "sequential": "qwen_sequential_hybrid"}[par]
    n = man["n_layers"]
    M.MODEL_SPECS[key] = {"display_name": key, "model_id": man["model_id"], "arch_family": fam,
                          "hybrid_type": par, "n_layers": n, "n_alt_layers": n,
                          "n_attn_layers": n, "hidden_dim": man.get("config", {}).get("hidden_size", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3.5-0.8b,lfm2")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ks", default="2,4,8")
    args = ap.parse_args()
    keys = args.models.split(","); ks = [int(x) for x in args.ks.split(",")]
    print(f"E5.1 n={args.n} | GPU={torch.cuda.get_device_name(0)} | {keys}")

    # n=1000 needs prompts beyond val-test; draw the extra pool with no val-tune overlap
    texts = load_wikitext(split="validation", threshold=100)
    tune, test = val_tune_test_split(texts, n_tune=50)
    pool = test  # already excludes the 50 val-tune; use as many as available
    if len(pool) < args.n:
        # extend with the train split (never overlaps val-tune/test)
        extra = load_wikitext(split="train", threshold=100, n=args.n - len(pool) + 10)
        pool = list(pool) + list(extra)
    prompts = pool[:args.n]
    print(f"  prompt pool size used: {len(prompts)}")
    os.makedirs(PP, exist_ok=True)

    rows, power_rows = [], []
    for key in keys:
        register(key)
        model, tok = load_model_and_tokenizer(key)
        ctrl = DraftModeController(model, key)
        par = M.MODEL_SPECS[key]["hybrid_type"]
        for k in ks:
            res = simulate_acceptance(model, tok, ctrl, "component_only", k,
                                      temperature=0.0, prompts=prompts,
                                      n_trials=len(prompts), max_len=256)
            na = np.asarray(res["accepted_lengths"])
            ppf = os.path.join(PP, f"{key}__component_only__k{k}__n{args.n}.pkl")
            pickle.dump({"model": key, "k": k, "accepted_lengths": res["accepted_lengths"]},
                        open(ppf, "wb"))
            rows.append({"model": key, "arch_paradigm": par, "strategy": "component_only",
                         "k": k, "temperature": 0.0, "A_k": round(res["acceptance_rate"], 5),
                         "ci_lo": round(res["acceptance_rate_ci_lo"], 5),
                         "ci_hi": round(res["acceptance_rate_ci_hi"], 5),
                         "tau_bar": round(res["mean_accepted_length"], 4),
                         "all_accept_rate": round(float(np.mean(na == k)), 5),
                         "n_prompts": len(prompts),
                         "per_prompt_file": os.path.relpath(ppf, REPO)})
            ciw = res["acceptance_rate_ci_hi"] - res["acceptance_rate_ci_lo"]
            print(f"  {key} k={k}: A_k={res['acceptance_rate']:.5f} "
                  f"CI=[{res['acceptance_rate_ci_lo']:.4f},{res['acceptance_rate_ci_hi']:.4f}] "
                  f"width={ciw:.4f} (n={len(prompts)})")
            # power note: CI width at n=200 vs n=1000 for this observed alpha
            for nn in (200, args.n):
                pn = stats.power_note(res["acceptance_rate"], nn)
                power_rows.append({"model": key, "k": k, "observed_A_k": round(res["acceptance_rate"], 5),
                                   "n": nn, **{kk: round(vv, 6) if isinstance(vv, float) else vv
                                               for kk, vv in pn.items()}})
            write_csv(rows, OUT_CSV, fieldnames=FIELDS)
        M.unload_model(key)
    write_csv(power_rows, POWER_CSV)
    write_meta_json(OUT_META, cwd=REPO, experiment="E5.1", models=keys, n=args.n, ks=ks,
                    gpu_is_l4=False, metric_is_gpu_agnostic=True)
    print("wrote:", OUT_CSV, POWER_CSV)


if __name__ == "__main__":
    main()
