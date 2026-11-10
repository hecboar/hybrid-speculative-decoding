#!/usr/bin/env python3
"""E1 (divergence half) — DTV(top-100) + top-1 agreement for new models.

PLAN Phase 1: DTV (total variation over top-100 logits, 100 prompts) + top-1
agreement, full vs component_only drafter. Schema mirrors the v1
`divergence_v3__*.pkl` dicts so new-model rows drop straight into the same table
(the v1 originals + falcon15 + qwen35_2b divergences are REUSED from Drive; this
computes the genuinely-new families lfm2, granite, zamba2).

For each prompt we teacher-force the prefix through the model in `full` mode and
in `component_only` mode and compare the two next-token distributions at each
prefix position (top-100 TV, top-1 agreement, KL). Aggregated over positions.

Output: results/E1/divergence_new_models.csv + per-model pkl + meta.json.
GPU-agnostic; non-L4 OK.
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
from specdiag.metrics import dtv_top100, top1_agreement, kl_full_draft, _softmax_np  # noqa: E402
from specdiag.data import load_wikitext, val_tune_test_split  # noqa: E402
from specdiag.io import write_csv, write_meta_json  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E1 = os.path.join(REPO, "results", "E1")
MANIFEST = os.path.join(REPO, "results", "E0", "arch_manifest.json")
OUT_CSV = os.path.join(E1, "divergence_new_models.csv")
OUT_META = os.path.join(E1, "divergence_new_models.meta.json")
FIELDS = ["model", "arch_paradigm", "strategy", "d_tv_mean", "d_tv_std",
          "d_tv_median", "d_tv_p90", "top1_agreement", "kl_mean", "n_positions",
          "n_seqs"]


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


@torch.inference_mode()
def logits_for(model, ids, use_cache=False):
    return model(input_ids=ids, use_cache=use_cache).logits[0].float().cpu().numpy()


def run_model(key, prompts, tok_maxlen, pos_per_seq):
    spec = register_from_manifest(key)
    model, tok = load_model_and_tokenizer(key)
    ctrl = DraftModeController(model, key)
    dtvs, kls, top1s = [], [], []
    n_seqs = 0
    for text in prompts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=tok_maxlen).to("cuda")
        ids = enc["input_ids"]
        if ids.shape[1] < 8:
            continue
        ctrl.deactivate()
        lf = logits_for(model, ids)          # (seq, vocab) full
        ctrl.set_mode("component_only")
        ld = logits_for(model, ids)          # (seq, vocab) draft
        ctrl.deactivate()
        seq = lf.shape[0]
        # sample the last pos_per_seq positions (most-context predictions)
        idxs = list(range(max(0, seq - pos_per_seq), seq))
        for p in idxs:
            dtvs.append(dtv_top100(lf[p], ld[p]))
            top1s.append(int(np.argmax(lf[p]) == np.argmax(ld[p])))
            kls.append(kl_full_draft(_softmax_np(lf[p]), _softmax_np(ld[p])))
        n_seqs += 1
    M.unload_model(key)
    dtv = np.asarray(dtvs)
    row = {"model": key, "arch_paradigm": spec["hybrid_type"],
           "strategy": "component_only",
           "d_tv_mean": round(float(dtv.mean()), 5),
           "d_tv_std": round(float(dtv.std()), 5),
           "d_tv_median": round(float(np.median(dtv)), 5),
           "d_tv_p90": round(float(np.percentile(dtv, 90)), 5),
           "top1_agreement": round(float(np.mean(top1s)), 5),
           "kl_mean": round(float(np.mean(kls)), 5),
           "n_positions": int(dtv.size), "n_seqs": n_seqs}
    print(f"  {key}: D_TV={row['d_tv_mean']:.4f} top1={row['top1_agreement']:.4f} "
          f"KL={row['kl_mean']:.3f} (n_pos={row['n_positions']})")
    # save per-model pkl (per-position DTV for stats reuse)
    with open(os.path.join(E1, f"divergence_new__{key}.pkl"), "wb") as f:
        pickle.dump({**row, "dtv_per_position": dtvs}, f)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--pos", type=int, default=50, help="positions sampled per seq")
    args = ap.parse_args()
    print(f"E1 divergence | GPU={torch.cuda.get_device_name(0)} | keys={args.keys}")
    all_texts = load_wikitext(split="validation", threshold=200)
    _, val_test = val_tune_test_split(all_texts, n_tune=50)
    prompts = val_test[:args.n]

    existing = {}
    if os.path.exists(OUT_CSV):
        import csv as _csv
        for r in _csv.DictReader(open(OUT_CSV)):
            existing[r["model"]] = r
    for key in args.keys:
        try:
            row = run_model(key, prompts, args.maxlen, args.pos)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {key}: FAILED {type(e).__name__}: {str(e)[:200]}")
            continue
        existing[key] = row
        write_csv(list(existing.values()), OUT_CSV, fieldnames=FIELDS)
    write_meta_json(OUT_META, cwd=REPO, experiment="E1-divergence", keys=args.keys,
                    n_prompts=args.n, pos_per_seq=args.pos,
                    gpu_is_l4=False, metric_is_gpu_agnostic=True)
    print("wrote:", OUT_CSV)


if __name__ == "__main__":
    main()
