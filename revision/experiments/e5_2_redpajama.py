#!/usr/bin/env python3
"""E5.2 — recent/held-out corpus check (PLAN Phase 5; R2.4).

Repeat the headline result on a corpus OUTSIDE WikiText-2: Falcon-H1-0.5B
(ssm_only) and Qwen3.5-0.8B (linear-only), k in {2,4}, greedy, 200 prompts, on a
held-out slice of RedPajama-Data-1T-Sample (R2 names it verbatim, so the concern
is answered on the reviewer's own terms). Documents source + date in meta.json.

Output: results/E5/acceptance_recent_corpus.csv (schema as E1, + corpus,
corpus_date) + per-prompt pkls. GPU-agnostic (acceptance).
"""
import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from specdiag import models as M  # noqa: E402
from specdiag.models import load_model_and_tokenizer  # noqa: E402
from specdiag.controller import DraftModeController  # noqa: E402
from specdiag.specloop import simulate_acceptance  # noqa: E402
from specdiag.data import load_recent_corpus  # noqa: E402
from specdiag.io import write_csv, write_meta_json  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E5 = os.path.join(REPO, "results", "E5")
PP = os.path.join(E5, "perprompt")
OUT_CSV = os.path.join(E5, "acceptance_recent_corpus.csv")
OUT_META = os.path.join(E5, "acceptance_recent_corpus.meta.json")
MODELS = ["falcon-h1-0.5b", "qwen3.5-0.8b"]
FIELDS = ["model", "arch_paradigm", "strategy", "k", "temperature", "A_k",
          "ci_lo", "ci_hi", "tau_bar", "all_accept_rate", "n_prompts",
          "per_prompt_file", "corpus", "corpus_date"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ks", default="2,4")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    print(f"E5.2 RedPajama | GPU={torch.cuda.get_device_name(0)}")
    try:
        texts, cmeta = load_recent_corpus(n=args.n, threshold=100, name="redpajama")
    except Exception as e:
        print(f"  RedPajama load FAILED: {type(e).__name__}: {str(e)[:200]}")
        write_meta_json(OUT_META, cwd=REPO, experiment="E5.2", error=str(e)[:300],
                        note="RedPajama unavailable; log blocker and retry",
                        gpu_is_l4=False)
        raise SystemExit(1)
    print(f"  corpus: {cmeta['corpus']} ({cmeta['corpus_date']}) n={len(texts)}")
    os.makedirs(PP, exist_ok=True)

    rows = []
    for key in MODELS:
        model, tok = load_model_and_tokenizer(key)
        ctrl = DraftModeController(model, key)
        par = M.MODEL_SPECS[key]["hybrid_type"]
        for k in ks:
            res = simulate_acceptance(model, tok, ctrl, "component_only", k,
                                      temperature=0.0, prompts=texts,
                                      n_trials=len(texts), max_len=256)
            na = np.asarray(res["accepted_lengths"])
            ppf = os.path.join(PP, f"redpajama__{key}__k{k}.pkl")
            pickle.dump({"model": key, "k": k, "corpus": cmeta["corpus"],
                         "accepted_lengths": res["accepted_lengths"]}, open(ppf, "wb"))
            rows.append({"model": key, "arch_paradigm": par, "strategy": "component_only",
                         "k": k, "temperature": 0.0, "A_k": round(res["acceptance_rate"], 5),
                         "ci_lo": round(res["acceptance_rate_ci_lo"], 5),
                         "ci_hi": round(res["acceptance_rate_ci_hi"], 5),
                         "tau_bar": round(res["mean_accepted_length"], 4),
                         "all_accept_rate": round(float(np.mean(na == k)), 5),
                         "n_prompts": len(texts), "per_prompt_file": os.path.relpath(ppf, REPO),
                         "corpus": cmeta["corpus"], "corpus_date": cmeta["corpus_date"]})
            print(f"  {key} k={k}: A_k={res['acceptance_rate']:.5f} "
                  f"tau={res['mean_accepted_length']:.3f} (RedPajama)")
            write_csv(rows, OUT_CSV, fieldnames=FIELDS)
        M.unload_model(key)
    write_meta_json(OUT_META, cwd=REPO, experiment="E5.2", corpus=cmeta,
                    models=MODELS, ks=ks, gpu_is_l4=False, metric_is_gpu_agnostic=True)
    print("wrote:", OUT_CSV)


if __name__ == "__main__":
    main()
