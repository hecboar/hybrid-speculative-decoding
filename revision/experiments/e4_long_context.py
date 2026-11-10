#!/usr/bin/env python3
"""E4 — long-context sweep (PLAN Phase 4; R2.9, R3, tests the §3.4 claim).

Falcon-H1-0.5B, greedy, k in {1,2}, drafters: component_only(100%) + best partial
from E3.1. Prompt lengths {512,1024,2048,4096,8192}, 50 prompts/length (built by
concatenating WikiText docs), max_new_tokens=64. Before the sweep: single-prompt
VRAM check at 8192 (bf16); if OOM, cap at 4096 and say so.

Per length: c_eff(L) (draft/full fwd-time ratio), tau_bar, draft_ms_tok,
verify_ms_tok, AR tok/s, S_wall (optional), S_theory. NB: without mamba-ssm
kernels the Falcon no-kernel context cap is 1024 (models.get_max_inference_seq_length),
so lengths > 1024 run the naive path and may be slow / OOM — handled + documented.

Honest-reporting rule (PLAN): if c_eff does NOT drop with L, report it as "the
recurrent-state advantage does not materialize in the current HuggingFace
execution path" — which SUPPORTS the infrastructure-bottleneck framing.

Latency numbers are non-canonical (RTX PRO 4500, not L4); c_eff/tau are
GPU-agnostic. Output: results/E4/long_context.csv + meta.json.
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
from specdiag.specloop import accept_reject  # noqa: E402
from specdiag.data import build_long_context_prompts  # noqa: E402
from specdiag.timing import cuda_timer, warmup  # noqa: E402
from specdiag.io import write_csv, write_meta_json, gpu_tag, load_other_gpu_rows  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E4 = os.path.join(REPO, "results", "E4")
OUT_CSV = os.path.join(E4, "long_context.csv")
OUT_META = os.path.join(E4, "long_context.meta.json")
MODEL = "falcon-h1-0.5b"
DEFAULT_PRE_L4_GPU = "NVIDIA_RTX_PRO_4500_Blackwell"
FIELDS = ["prompt_len", "strategy", "k", "ar_tok_s", "draft_ms_tok",
          "verify_ms_tok", "c_eff", "tau_bar", "s_wall", "s_theory", "n_prompts", "gpu"]
TIMING_REPS = int(os.environ.get("E4_TIMING_REPS", "8"))   # 8 canonical; lower = feasibility


@torch.inference_mode()
def time_fwd(model, ids, n=TIMING_REPS):
    def fwd():
        model(input_ids=ids, use_cache=False)
    warmup(fwd, 3)
    with cuda_timer() as t:
        for _ in range(n):
            model(input_ids=ids, use_cache=False)
    return t["ms"] / n


@torch.inference_mode()
def sweep_length(model, tok, ctrl, set_fn, prompts, k, L):
    """Measure c_eff(L), tau_bar, per-token draft/verify ms, AR tok/s at length L."""
    taus, draft_ms, verify_ms = [], [], []
    for text in prompts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=L).to("cuda")
        ids = enc["input_ids"]
        seqlen = ids.shape[1]
        # full (verify-style) forward time over the L-context
        ctrl.deactivate()
        t_full = time_fwd(model, ids)
        # draft forward time in draft mode
        set_fn()
        t_draft = time_fwd(model, ids)
        # one speculative round for tau: draft k tokens, verify
        dt, dp = [], []
        cids = ids.clone()
        for _ in range(k):
            lg = model(input_ids=cids, use_cache=False).logits[:, -1, :].float()
            pr = torch.softmax(lg, dim=-1); tkn = lg.argmax(dim=-1, keepdim=True)
            dt.append(tkn.item()); dp.append(pr[0, tkn.item()].item())
            cids = torch.cat([cids, tkn], dim=-1)
        ctrl.deactivate()
        vid = torch.cat([ids, torch.tensor([dt], device="cuda")], dim=-1)
        fl = model(input_ids=vid, use_cache=False).logits[:, seqlen - 1: seqlen - 1 + k, :].float()
        na = accept_reject(dt, dp, fl[0].cpu().numpy(), 0.0)
        taus.append(na)
        draft_ms.append(t_draft / max(1, k))   # per drafted token (k tokens/draft pass amortised)
        verify_ms.append(t_full / max(1, k))
    ctrl.deactivate()
    tau = float(np.mean(taus))
    d_ms = float(np.mean(draft_ms)); v_ms = float(np.mean(verify_ms))
    # AR tok/s: single-token full forward throughput proxy at this context
    t_full1 = time_fwd(model, tok(prompts[0], return_tensors="pt", truncation=True,
                                  max_length=L)["input_ids"].to("cuda"))
    ar_tok_s = 1000.0 / t_full1 if t_full1 > 0 else float("nan")
    c_eff = d_ms / v_ms if v_ms > 0 else float("nan")
    s_theory = (1 + tau) / (1 + k * c_eff) if c_eff == c_eff else float("nan")
    return {"tau": tau, "draft_ms_tok": d_ms, "verify_ms_tok": v_ms,
            "ar_tok_s": ar_tok_s, "c_eff": c_eff, "s_theory": s_theory}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--n_per_len", type=int, default=50)
    ap.add_argument("--ks", default="1,2")
    ap.add_argument("--partial_frac", type=float, default=0.75,
                    help="best partial from E3.1 (default 0.75)")
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    GPU = gpu_tag(); is_l4 = "L4" in GPU
    print(f"E4 long context | GPU={torch.cuda.get_device_name(0)} "
          f"({'CANONICAL' if is_l4 else 'non-L4 feasibility'}) | lengths={lengths}")

    model, tok = load_model_and_tokenizer(MODEL)
    ctrl = DraftModeController(model, MODEL)
    OTHER = load_other_gpu_rows(OUT_CSV, GPU, DEFAULT_PRE_L4_GPU)

    # VRAM check at the max length before the sweep
    max_len_ok = max(lengths)
    for L in sorted(lengths, reverse=True):
        try:
            torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(0, 1000, (1, L), device="cuda")
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f"  VRAM check L={L}: OK ({vram:.1f} GB)")
            max_len_ok = L
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  VRAM check L={L}: OOM -> capping below")
            continue
    lengths = [L for L in lengths if L <= max_len_ok]
    print(f"  running lengths: {lengths}")

    strategies = [("component_only", lambda: ctrl.set_mode("component_only")),
                  (f"component_only_partial_{args.partial_frac}",
                   lambda: ctrl.set_mode("component_only_partial", frac=args.partial_frac,
                                         policy="keep_first_last_stride"))]
    rows = []
    for L in lengths:
        prompts = build_long_context_prompts(tok, target_lens=(L,),
                                             n_per_len=args.n_per_len)[L]
        for sname, set_fn in strategies:
            for k in ks:
                try:
                    r = sweep_length(model, tok, ctrl, set_fn, prompts, k, L)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  L={L} {sname} k={k}: OOM -> skip")
                    continue
                rows.append({"prompt_len": L, "strategy": sname, "k": k,
                             "ar_tok_s": round(r["ar_tok_s"], 3),
                             "draft_ms_tok": round(r["draft_ms_tok"], 4),
                             "verify_ms_tok": round(r["verify_ms_tok"], 4),
                             "c_eff": round(r["c_eff"], 5), "tau_bar": round(r["tau"], 4),
                             "s_wall": "", "s_theory": round(r["s_theory"], 5),
                             "n_prompts": len(prompts), "gpu": GPU})
                print(f"  L={L:5d} {sname:32s} k={k} c_eff={r['c_eff']:.3f} "
                      f"tau={r['tau']:.3f} S_theory={r['s_theory']:.3f}")
                write_csv(OTHER + rows, OUT_CSV, fieldnames=FIELDS)

    write_meta_json(OUT_META, cwd=REPO, experiment="E4", model=MODEL, gpu=GPU,
                    lengths=lengths, capped_at=max_len_ok, n_per_len=args.n_per_len,
                    no_kernel_seq_cap=1024,
                    note=("CANONICAL L4 (mamba kernels if installed)" if is_l4 else
                          "latency non-canonical (non-L4); c_eff/tau GPU-agnostic; naive-mamba "
                          "path is the 'current HF execution path'"),
                    gpu_is_l4=is_l4, metric_is_gpu_agnostic=True)
    print("wrote:", OUT_CSV)


if __name__ == "__main__":
    main()
