#!/usr/bin/env python3
"""E5.2b — REAL RedPajama held-out corpus check (PLAN Phase 5; R2.4).

Same acceptance loop as e5_2_redpajama.py (Falcon-H1-0.5B ssm_only, Qwen3.5-0.8B
linear-only, component_only, k in {2,4}, greedy, bf16, 200 prompts), but loads
the corpus the reviewer named *verbatim* — RedPajama — instead of the C4
substitute that e5_2_redpajama.py fell back to.

Why a new loader: `togethercomputer/RedPajama-Data-1T-Sample` is 404 on the Hub,
and the parent `togethercomputer/RedPajama-Data-1T` ships a dataset *loading
script* that datasets>=5 no longer executes (this is exactly why
`load_recent_corpus` degraded to C4). We bypass the script entirely: the repo's
`urls/common_crawl.txt` lists the raw `.jsonl.zst` shards on data.together.xyz,
and we stream them with the plain `json` loader (no remote code), reading the
`text` field. CommonCrawl is the dominant slice of RedPajama-Data-1T-Sample and
the closest analogue to the C4 web-text run, so the numbers are comparable.

CRITICAL: this loader ONLY loads RedPajama. There is no C4 fallback. If RedPajama
cannot be streamed it raises; and before any result is written we assert the
corpus label contains "RedPajama".

Output (all NEW files — nothing frozen is touched):
  results/E5/acceptance_redpajama.csv            (schema as e5_2_redpajama.py)
  results/E5/acceptance_redpajama.meta.json
  results/E5/perprompt/redpajama_real__<model>__k<k>.pkl
GPU-agnostic (acceptance metric); the GPU is only recorded in meta.
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
from specdiag.io import write_csv, write_meta_json  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E5 = os.path.join(REPO, "results", "E5")
PP = os.path.join(E5, "perprompt")
OUT_CSV = os.path.join(E5, "acceptance_redpajama.csv")
OUT_META = os.path.join(E5, "acceptance_redpajama.meta.json")
MODELS = ["falcon-h1-0.5b", "qwen3.5-0.8b"]
FIELDS = ["model", "arch_paradigm", "strategy", "k", "temperature", "A_k",
          "ci_lo", "ci_hi", "tau_bar", "all_accept_rate", "n_prompts",
          "per_prompt_file", "corpus", "corpus_date"]

# RedPajama-Data-1T-Sample is defunct; stream the CommonCrawl slice of its live
# parent repo, whose raw shard URLs are listed in urls/common_crawl.txt.
RP_PARENT = "togethercomputer/RedPajama-Data-1T"
RP_SLICE = "urls/common_crawl.txt"
RP_LABEL = "RedPajama-Data-1T-Sample (CommonCrawl slice, streamed from parent 1T)"
RP_DATE = "2023"


def load_redpajama_real(n=200, threshold=100, max_chars=2000, max_shards=8):
    """Stream real RedPajama CommonCrawl docs, bypassing the defunct load script.

    Reads the `urls/common_crawl.txt` shard list from the parent 1T repo and
    streams `.jsonl.zst` shards with the plain `json` loader (no remote code),
    collecting the first `n` docs with len(text)>threshold. Char-truncates each
    doc to `max_chars` (a prefix well above the 256-token cap simulate_acceptance
    applies, so results are unaffected) to bound memory. Returns (texts, meta).
    Raises if nothing loads — never falls back to C4.
    """
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset

    tok = os.environ.get("HF_TOKEN")
    urls_path = hf_hub_download(RP_PARENT, RP_SLICE, repo_type="dataset", token=tok)
    shard_urls = [u.strip() for u in open(urls_path) if u.strip()]
    if not shard_urls:
        raise RuntimeError(f"{RP_SLICE} listed no shards")

    texts, used_shards, last_err = [], [], None
    for url in shard_urls[:max_shards]:
        try:
            ds = load_dataset("json", data_files=url, split="train", streaming=True)
            used_shards.append(url)
            for e in ds:
                t = (e.get("text") or "").strip()
                if len(t) > threshold:
                    texts.append(t[:max_chars])
                if len(texts) >= n:
                    break
        except Exception as ex:  # noqa: BLE001
            last_err = ex
            continue
        if len(texts) >= n:
            break

    if len(texts) < n:
        raise RuntimeError(f"RedPajama: only {len(texts)}/{n} docs streamed; "
                           f"last error: {type(last_err).__name__ if last_err else None}")

    meta = {"corpus": RP_LABEL, "hf_id": RP_PARENT, "config": "common_crawl",
            "corpus_date": RP_DATE, "split": "train", "n": len(texts),
            "shard_list_file": RP_SLICE, "shards_used": used_shards,
            "max_chars_prefix": max_chars,
            "note": ("RedPajama-Data-1T-Sample id is 404; streamed the CommonCrawl "
                     "slice of the live parent RedPajama-Data-1T, bypassing the "
                     "defunct dataset loading script (raw .jsonl.zst via json loader).")}
    return texts, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ks", default="2,4")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    print(f"E5.2b RedPajama(real) | GPU={torch.cuda.get_device_name(0)}")

    texts, cmeta = load_redpajama_real(n=args.n, threshold=100)
    # CRITICAL contamination-control guard: must be RedPajama, never C4.
    if "redpajama" not in cmeta["corpus"].lower():
        raise SystemExit(f"ABORT: loaded corpus is not RedPajama: {cmeta['corpus']!r}")
    if "c4" in cmeta["hf_id"].lower():
        raise SystemExit(f"ABORT: hf_id looks like C4: {cmeta['hf_id']!r}")
    print(f"  corpus: {cmeta['corpus']} ({cmeta['corpus_date']}) "
          f"hf_id={cmeta['hf_id']} n={len(texts)} shards={len(cmeta['shards_used'])}")
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
            ppf = os.path.join(PP, f"redpajama_real__{key}__k{k}.pkl")
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
                  f"tau={res['mean_accepted_length']:.3f} (RedPajama-real)")
            write_csv(rows, OUT_CSV, fieldnames=FIELDS)
        M.unload_model(key)
    write_meta_json(OUT_META, cwd=REPO, experiment="E5.2b", corpus=cmeta,
                    models=MODELS, ks=ks, gpu_is_l4=False, metric_is_gpu_agnostic=True)
    print("wrote:", OUT_CSV)


if __name__ == "__main__":
    main()
