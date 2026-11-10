#!/usr/bin/env python3
"""E6 — Functional PPL ablation IN THIS PAPER (PLAN Phase 1; R2.8 critical).

For every model (4 original + all viable new): perplexity on WikiText-2 val-test
in `full` mode and in `component_only` mode (attention branch removed). This
replaces reliance on companion paper [19]; Table 6 becomes our own measurement.

Then compute Spearman rho between log10(ppl_ratio) and A_4 across all models.

Output: results/E6/ppl_ablation.csv
  (model, arch_paradigm, ppl_full, ppl_no_attn, ppl_ratio, log10_ratio, n_tokens)
+ results/E6/ppl_spearman.csv (rho, p, n) once A_4 is available.
GPU: PPL is GPU-agnostic (blockers.md B1); non-L4 OK.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from specdiag import models as M  # noqa: E402
from specdiag.models import load_model_and_tokenizer  # noqa: E402
from specdiag.controller import DraftModeController  # noqa: E402
from specdiag.metrics import perplexity  # noqa: E402
from specdiag.data import load_wikitext, val_tune_test_split  # noqa: E402
from specdiag.io import write_csv, write_meta_json  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E6 = os.path.join(REPO, "results", "E6")
MANIFEST = os.path.join(REPO, "results", "E0", "arch_manifest.json")
OUT_CSV = os.path.join(E6, "ppl_ablation.csv")
OUT_META = os.path.join(E6, "ppl_ablation.meta.json")
FIELDS = ["model", "arch_paradigm", "ppl_full", "ppl_no_attn", "ppl_ratio",
          "log10_ratio", "n_tokens"]


def register_from_manifest(key):
    if key in M.MODEL_SPECS:
        return M.MODEL_SPECS[key]
    man = json.load(open(MANIFEST)).get(key)
    if not man:
        raise SystemExit(f"{key}: no MODEL_SPECS/manifest")
    paradigm = man.get("requested_paradigm", "sequential")
    cfg_class = man.get("config_class", "")
    if "Zamba2" in cfg_class:
        fam = "zamba2_shared_hybrid"
    else:
        fam = {"parallel": "falcon_parallel_hybrid",
               "sequential": "qwen_sequential_hybrid"}.get(paradigm, "transformer_baseline")
    n = man.get("n_layers", 0)
    M.MODEL_SPECS[key] = {"display_name": key, "model_id": man["model_id"],
                          "arch_family": fam, "hybrid_type": paradigm, "n_layers": n,
                          "n_alt_layers": n, "n_attn_layers": n,
                          "hidden_dim": man.get("config", {}).get("hidden_size", 0)}
    return M.MODEL_SPECS[key]


def run_model(key, texts):
    spec = register_from_manifest(key)
    model, tok = load_model_and_tokenizer(key)
    ctrl = DraftModeController(model, key)

    ctrl.deactivate()
    r_full = perplexity(model, tok, texts)
    ctrl.set_mode("component_only")
    r_abl = perplexity(model, tok, texts)
    ctrl.deactivate()
    M.unload_model(key)

    ppl_full, ppl_abl = r_full["ppl"], r_abl["ppl"]
    ratio = ppl_abl / ppl_full if ppl_full > 0 else float("nan")
    row = {"model": key, "arch_paradigm": spec["hybrid_type"],
           "ppl_full": round(ppl_full, 4), "ppl_no_attn": round(ppl_abl, 4),
           "ppl_ratio": round(ratio, 4),
           "log10_ratio": round(float(np.log10(ratio)), 4) if ratio > 0 else "",
           "n_tokens": r_full["n_tokens"]}
    print(f"  {key}: ppl_full={ppl_full:.2f} ppl_no_attn={ppl_abl:.2f} "
          f"ratio={ratio:.3f} log10={row['log10_ratio']}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--n_docs", type=int, default=200,
                    help="number of val-test docs to concatenate for PPL")
    args = ap.parse_args()
    print(f"E6 PPL ablation | GPU={torch.cuda.get_device_name(0)} | keys={args.keys}")
    all_texts = load_wikitext(split="validation", threshold=100)
    _, val_test = val_tune_test_split(all_texts, n_tune=50)
    texts = val_test[:args.n_docs]

    existing = {}
    if os.path.exists(OUT_CSV):
        import csv as _csv
        for r in _csv.DictReader(open(OUT_CSV)):
            existing[r["model"]] = r
    for key in args.keys:
        try:
            row = run_model(key, texts)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:200]}")
            continue
        existing[key] = row
        write_csv(list(existing.values()), OUT_CSV, fieldnames=FIELDS)
    write_meta_json(OUT_META, cwd=REPO, experiment="E6", keys=args.keys,
                    n_docs=args.n_docs, gpu_is_l4=False, metric_is_gpu_agnostic=True)
    print("wrote:", OUT_CSV)


if __name__ == "__main__":
    main()
