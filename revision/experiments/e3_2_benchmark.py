#!/usr/bin/env python3
"""E3.2 — benchmark audit: optimized vs v1 path (PLAN Phase 3; R2 §2, R2.6, EiC-6).

Falcon-H1-0.5B, greedy, k in {1,2,4}, val-test prompts, max_new_tokens=64.
  * v1 path: `spec_generate_v1` (full-sequence recompute each round, per-round
    hook churn absent but no prefix-KV reuse) timed on 30 prompts.
  * optimized path: prefix past_key_values REUSE at the HF API level (verify
    processes only the k draft tokens, not prefix+k), cached incremental draft in
    a second cache, DynamicCache.crop to drop rejected-draft KV, torch.inference_mode,
    warmup, no tokenization/tqdm in the timed region, no torch.cat in the hot loop.
    Timed per-phase (draft/verify/accept/overhead) via RoundBreakdown, on 150
    prompts x 3 reps.

Target narrative: v1 ~0.34x -> optimized ~0.7-0.85x, i.e. approaching the
tau-based bound; residual gap = HF software stack, not the drafting principle.

LATENCY IS NON-CANONICAL (RTX PRO 4500, not L4) — speedups here are feasibility
figures; c_eff and the v1->optimized RATIO are the transferable results.
Output: results/E3/benchmark_audit.csv + meta.json.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from specdiag import models as M  # noqa: E402
from specdiag.models import load_model_and_tokenizer  # noqa: E402
from specdiag.controller import DraftModeController  # noqa: E402
from specdiag.specloop import spec_generate_v1, accept_reject  # noqa: E402
from specdiag.data import load_wikitext, val_tune_test_split  # noqa: E402
from specdiag.timing import cuda_timer, warmup, RoundBreakdown  # noqa: E402
from specdiag.io import write_csv, write_meta_json, gpu_tag, load_other_gpu_rows  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E3 = os.path.join(REPO, "results", "E3")
OUT_CSV = os.path.join(E3, "benchmark_audit.csv")
OUT_META = os.path.join(E3, "benchmark_audit.meta.json")
MODEL = "falcon-h1-0.5b"
STRATEGY = "component_only"
DEFAULT_PRE_L4_GPU = "NVIDIA_RTX_PRO_4500_Blackwell"
FIELDS = ["impl", "k", "rep", "ar_tok_s", "spec_tok_s", "speedup", "draft_ms_tok",
          "verify_ms_tok", "accept_ms_tok", "overhead_ms_tok", "c_eff", "n_prompts", "gpu"]


@torch.inference_mode()
def ar_tok_s(model, tok, prompt, max_new=40, maxlen=256):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=maxlen)["input_ids"].to("cuda")
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    torch.cuda.synchronize()
    with cuda_timer() as t:
        for _ in range(max_new):
            nxt = logits.argmax(dim=-1, keepdim=True)
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
    return max_new / (t["ms"] / 1000.0)


@torch.inference_mode()
def optimized_spec(model, tok, ctrl, prompt, k, max_new=40, maxlen=256):
    """Greedy spec with prefix-KV reuse + cropped dual caches + phase timing."""
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=maxlen)["input_ids"].to("cuda")
    dev = ids.device
    br = RoundBreakdown()
    # seed full + draft caches over the prefix (let the model build its NATIVE
    # hybrid Mamba+attn cache — a plain DynamicCache lacks the conv/ssm layers)
    ctrl.deactivate()
    fo = model(input_ids=ids, use_cache=True)
    full_cache = fo.past_key_values
    full_last = fo.logits[:, -1, :]
    ctrl.set_mode(STRATEGY)
    do = model(input_ids=ids, use_cache=True)
    draft_cache = do.past_key_values
    draft_last = do.logits[:, -1, :]
    ctrl.deactivate()

    committed = ids.shape[1]        # tokens whose KV is committed in both caches
    gen = 0
    while gen < max_new:
        # --- DRAFT (draft mode, incremental) ---
        with cuda_timer() as td:
            ctrl.set_mode(STRATEGY)
            dtok, dprob = [], []
            cur = draft_last
            for _ in range(k):
                t = int(cur.argmax(dim=-1).item())
                dtok.append(t); dprob.append(1.0)
                o = model(input_ids=torch.tensor([[t]], device=dev),
                          past_key_values=draft_cache, use_cache=True)
                cur = o.logits[:, -1, :]
            ctrl.deactivate()
        # --- VERIFY (full mode) reusing prefix cache, only k tokens ---
        with cuda_timer() as tv:
            vo = model(input_ids=torch.tensor([dtok], device=dev),
                       past_key_values=full_cache, use_cache=True)
            vlog = vo.logits[0].float()      # (k, vocab)
        # --- ACCEPT (cpu) ---
        with cuda_timer() as ta:
            na = accept_reject(dtok, dprob, vlog.cpu().numpy(), 0.0)
            if na < k:
                corr = int(vlog[na].argmax().item())
                accepted = dtok[:na] + [corr]
            else:
                corr = int(vo.logits[:, -1, :].argmax().item())
                accepted = dtok + [corr]
        # --- OVERHEAD: roll caches to committed+na, then commit accepted tokens ---
        with cuda_timer() as to:
            n_acc = len(accepted)
            # verify wrote k tokens into full_cache; keep only na, drop the rest
            full_cache.crop(committed + na)
            # draft wrote k tokens into draft_cache; keep only na
            draft_cache.crop(committed + na)
            # now advance BOTH caches by feeding the accepted tail (na..n_acc) tokens
            tail = torch.tensor([accepted[na:]], device=dev)
            ctrl.deactivate()
            fo = model(input_ids=tail, past_key_values=full_cache, use_cache=True)
            full_cache = fo.past_key_values; full_last = fo.logits[:, -1, :]
            ctrl.set_mode(STRATEGY)
            do = model(input_ids=tail, past_key_values=draft_cache, use_cache=True)
            draft_cache = do.past_key_values; draft_last = do.logits[:, -1, :]
            ctrl.deactivate()
            committed += n_acc
        br.add(draft_ms=td["ms"], verify_ms=tv["ms"], accept_ms=ta["ms"],
               overhead_ms=to["ms"], tokens=n_acc)
        gen += n_acc
    return br, gen


@torch.inference_mode()
def time_v1(model, tok, ctrl, prompts, k, max_new=40):
    total_tok, total_ms = 0, 0.0
    for p in prompts:
        torch.cuda.synchronize()
        with cuda_timer() as t:
            r = spec_generate_v1(model, tok, ctrl, STRATEGY, p, k,
                                 temperature=0.0, max_new_tokens=max_new, max_len=256)
        total_ms += t["ms"]; total_tok += r["n_generated"]
    return total_tok / (total_ms / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--n_opt", type=int, default=150)
    ap.add_argument("--n_v1", type=int, default=30)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max_new", type=int, default=64,
                    help="tokens generated per timed run (64 = canonical; reduce only for feasibility)")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    GPU = gpu_tag(); is_l4 = "L4" in GPU
    tag = "CANONICAL" if is_l4 else "NON-L4 feasibility"
    mn = args.max_new
    print(f"E3.2 benchmark | GPU={torch.cuda.get_device_name(0)} ({tag}, max_new={mn})")

    model, tok = load_model_and_tokenizer(MODEL)
    ctrl = DraftModeController(model, MODEL)
    _, test = val_tune_test_split(load_wikitext(split="validation", threshold=100), 50)
    opt_prompts = test[:args.n_opt]; v1_prompts = test[:args.n_v1]
    OTHER = load_other_gpu_rows(OUT_CSV, GPU, DEFAULT_PRE_L4_GPU)

    # AR baseline tok/s (mean over a few prompts, reps)
    ar_rates = []
    for _ in range(args.reps):
        ar_rates.append(np.mean([ar_tok_s(model, tok, p, max_new=mn) for p in test[:10]]))
    ar = float(np.mean(ar_rates))
    print(f"  AR tok/s (mean of {args.reps} reps) = {ar:.2f}")

    rows = []
    for k in ks:
        # v1 path
        for rep in range(args.reps):
            sv1 = time_v1(model, tok, ctrl, v1_prompts, k, max_new=mn)
            rows.append({"impl": "v1", "k": k, "rep": rep, "ar_tok_s": round(ar, 3),
                         "spec_tok_s": round(sv1, 3), "speedup": round(sv1 / ar, 4),
                         "draft_ms_tok": "", "verify_ms_tok": "", "accept_ms_tok": "",
                         "overhead_ms_tok": "", "c_eff": "", "n_prompts": len(v1_prompts),
                         "gpu": GPU})
            print(f"  v1        k={k} rep={rep} spec_tok/s={sv1:.2f} speedup={sv1/ar:.3f}x")
            write_csv(OTHER + rows, OUT_CSV, fieldnames=FIELDS)
        # optimized path
        for rep in range(args.reps):
            agg = RoundBreakdown(); tot_tok = 0
            with cuda_timer() as tall:
                for p in opt_prompts:
                    br, g = optimized_spec(model, tok, ctrl, p, k, max_new=mn)
                    for a in ("draft_ms", "verify_ms", "accept_ms", "overhead_ms"):
                        setattr(agg, a, getattr(agg, a) + getattr(br, a))
                    agg.tokens += br.tokens; tot_tok += g
            spec_opt = tot_tok / (tall["ms"] / 1000.0)
            pt = agg.per_token()
            rows.append({"impl": "optimized", "k": k, "rep": rep, "ar_tok_s": round(ar, 3),
                         "spec_tok_s": round(spec_opt, 3), "speedup": round(spec_opt / ar, 4),
                         "draft_ms_tok": round(pt["draft_ms_tok"], 4),
                         "verify_ms_tok": round(pt["verify_ms_tok"], 4),
                         "accept_ms_tok": round(pt["accept_ms_tok"], 4),
                         "overhead_ms_tok": round(pt["overhead_ms_tok"], 4),
                         "c_eff": round(agg.c_eff(), 5), "n_prompts": len(opt_prompts),
                         "gpu": GPU})
            print(f"  optimized k={k} rep={rep} spec_tok/s={spec_opt:.2f} "
                  f"speedup={spec_opt/ar:.3f}x c_eff={agg.c_eff():.3f} "
                  f"[d={pt['draft_ms_tok']:.2f} v={pt['verify_ms_tok']:.2f} "
                  f"a={pt['accept_ms_tok']:.2f} o={pt['overhead_ms_tok']:.2f}]ms/tok")
            write_csv(OTHER + rows, OUT_CSV, fieldnames=FIELDS)

    write_meta_json(OUT_META, cwd=REPO, experiment="E3.2", model=MODEL, strategy=STRATEGY,
                    ks=ks, reps=args.reps, ar_tok_s=ar, max_new=mn, gpu=GPU,
                    note=("CANONICAL L4 latency" if is_l4 else
                          "LATENCY NON-CANONICAL (non-L4); v1->optimized ratio + c_eff are the "
                          "transferable results"),
                    gpu_is_l4=is_l4, metric_is_gpu_agnostic=False)
    print("wrote:", OUT_CSV)


if __name__ == "__main__":
    main()
