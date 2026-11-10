#!/usr/bin/env python3
"""E3.1 — partial-component drafters on Falcon-H1-0.5B (PLAN Phase 3; EiC, R2 §2,
R2.6, R2.7).

Drafter variants: component_only(100%), component_only_partial 75/50/25%
(policy on val-tune), attention_only, layerskip(0.33), early_exit(0.5). The last
two are the generic baselines R2.7 wants run on the PARALLEL arch too (not only
the failing sequential one). Per variant, val-test, greedy, k in {1,2,4}:
  A_k, tau_bar, DTV(top-100), top1, c_param (FLOP), c_eff (MEASURED time ratio),
  S_theory = (1+tau)/(1+k*c_eff).

Success criterion: any variant with S_theory > 1.0 at any k => a credible path to
net positive speedup (break-even S_theory>1 <=> tau/k > c_eff).

c_eff = mean(draft forward ms) / mean(verify/full forward ms) over a fixed
256-token prefix, use_cache=False, CUDA events, after warmup. On this RTX PRO
4500 (non-L4) absolute latency is non-canonical, but the RATIO c_eff is
GPU-agnostic (per author) and is what S_theory needs.

Output: results/E3/partial_drafter_ablation.csv + per-prompt pkls + meta.json.
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
from specdiag.controller import DraftModeController, _kept_layer_ids  # noqa: E402
from specdiag.specloop import accept_reject  # noqa: E402
from specdiag.metrics import dtv_top100  # noqa: E402
from specdiag.stats import bootstrap_ci  # noqa: E402
from specdiag.data import load_wikitext, val_tune_test_split  # noqa: E402
from specdiag.timing import cuda_timer, warmup  # noqa: E402
from specdiag.io import write_csv, write_meta_json, gpu_tag, load_other_gpu_rows  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E3 = os.path.join(REPO, "results", "E3")
PP = os.path.join(E3, "perprompt")
OUT_CSV = os.path.join(E3, "partial_drafter_ablation.csv")
OUT_META = os.path.join(E3, "partial_drafter_ablation.meta.json")
MODEL = "falcon-h1-0.5b"
DEFAULT_PRE_L4_GPU = "NVIDIA_RTX_PRO_4500_Blackwell"   # producer of the pre-L4 rows
FIELDS = ["model", "strategy", "frac", "policy", "k", "A_k", "ci_lo", "ci_hi",
          "tau_bar", "DTV", "top1", "c_param", "c_eff", "s_theory",
          "s_wall_optional", "n_prompts", "gpu"]


def flop_parts(spec):
    d = spec["hidden_dim"]; d_ff = spec.get("_intermediate_size", d * 4)
    n_heads = spec.get("_num_attention_heads", 16)
    n_kv = spec.get("_num_kv_heads", n_heads); head_d = spec.get("_head_dim", d // n_heads)
    sl = 256
    f_attn = 2 * d * (n_heads * head_d + 2 * n_kv * head_d) + 2 * n_heads * sl * head_d + 2 * d * d
    f_ssm = 8 * d * d; f_mlp = 3 * 2 * d * d_ff
    return f_ssm, f_attn, f_mlp


def c_param_for(spec, strategy, frac):
    """Parallel Falcon FLOP ratio c_draft/c_full per variant."""
    L = spec["n_layers"]
    f_ssm, f_attn, f_mlp = flop_parts(spec)
    c_full = L * (f_ssm + f_attn + f_mlp)
    if strategy == "component_only":
        c_draft = L * (f_ssm + f_mlp)
    elif strategy == "component_only_partial":
        n_keep = len(_kept_layer_ids(L, frac, "keep_first_last_stride"))
        c_draft = n_keep * (f_ssm + f_mlp)
    elif strategy == "attention_only":
        c_draft = L * (f_attn + f_mlp)
    elif strategy == "layerskip":
        n_keep = L - max(1, int(L * frac))
        c_draft = n_keep * (f_ssm + f_attn + f_mlp)
    elif strategy == "early_exit":
        c_draft = max(1, int(L * frac)) * (f_ssm + f_attn + f_mlp)
    else:
        c_draft = c_full
    return c_draft / c_full


@torch.inference_mode()
def measure_c_eff(model, ctrl, set_fn, ids, n=20):
    """draft forward ms / full forward ms over the same prefix (use_cache=False)."""
    def full_fwd():
        ctrl.deactivate(); model(input_ids=ids, use_cache=False)
    def draft_fwd():
        set_fn(); model(input_ids=ids, use_cache=False)
    warmup(full_fwd, 5);
    with cuda_timer() as tf:
        for _ in range(n):
            ctrl.deactivate(); model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    warmup(draft_fwd, 5)
    with cuda_timer() as td:
        for _ in range(n):
            set_fn(); model(input_ids=ids, use_cache=False)
    ctrl.deactivate()
    t_full = tf["ms"] / n; t_draft = td["ms"] / n
    return t_draft / t_full if t_full > 0 else float("nan"), t_draft, t_full


@torch.inference_mode()
def divergence(model, tok, ctrl, set_fn, test, maxlen=256, div_np=40):
    """DTV(top-100) + top-1, full vs variant, last-position, div_np prompts. Once
    per variant (k-independent)."""
    dtvs, top1s = [], []
    for text in test[:div_np]:
        iids = tok(text, return_tensors="pt", truncation=True, max_length=maxlen).to("cuda")["input_ids"]
        ctrl.deactivate()
        lf = model(input_ids=iids, use_cache=False).logits[0, -1].float().cpu().numpy()
        set_fn()
        ld = model(input_ids=iids, use_cache=False).logits[0, -1].float().cpu().numpy()
        ctrl.deactivate()
        dtvs.append(dtv_top100(lf, ld))
        top1s.append(int(np.argmax(lf) == np.argmax(ld)))
    return (float(np.mean(dtvs)) if dtvs else float("nan"),
            float(np.mean(top1s)) if top1s else float("nan"))


@torch.inference_mode()
def acceptance(model, tok, ctrl, set_fn, k, test, maxlen=256):
    accepted = []
    for text in test:
        iids = tok(text, return_tensors="pt", truncation=True, max_length=maxlen).to("cuda")["input_ids"]
        set_fn()
        dt, dp = [], []
        cids = iids.clone()
        for _ in range(k):
            lg = model(input_ids=cids, use_cache=False).logits[:, -1, :].float()
            pr = torch.softmax(lg, dim=-1); tk = lg.argmax(dim=-1, keepdim=True)
            dt.append(tk.item()); dp.append(pr[0, tk.item()].item())
            cids = torch.cat([cids, tk], dim=-1)
        ctrl.deactivate()
        vid = torch.cat([iids, torch.tensor([dt], device="cuda")], dim=-1)
        fl = model(input_ids=vid, use_cache=False).logits[:, iids.shape[1] - 1: iids.shape[1] - 1 + k, :].float()
        accepted.append(accept_reject(dt, dp, fl[0].cpu().numpy(), 0.0))
    ptr = np.array(accepted) / k
    mr, cl, ch = bootstrap_ci(ptr)
    return {"A_k": mr, "ci_lo": cl, "ci_hi": ch, "tau": float(np.mean(accepted)),
            "accepted": accepted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--maxlen", type=int, default=256)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    print(f"E3.1 partial drafters | GPU={torch.cuda.get_device_name(0)} | n={args.n}")

    model, tok = load_model_and_tokenizer(MODEL)
    spec = M.MODEL_SPECS[MODEL]
    ctrl = DraftModeController(model, MODEL)
    texts = load_wikitext(split="validation", threshold=100)
    _, test = val_tune_test_split(texts, n_tune=50)
    test = test[:args.n]
    os.makedirs(PP, exist_ok=True)
    # a fixed prefix for c_eff timing
    tids = tok(test[0], return_tensors="pt", truncation=True, max_length=args.maxlen)["input_ids"].to("cuda")

    POLICY = "keep_first_last_stride"   # chosen default (keeps layer 0 & L-1; see PLAN 3.1)
    variants = [
        ("component_only", 1.0, "-"),
        ("component_only_partial", 0.75, POLICY),
        ("component_only_partial", 0.50, POLICY),
        ("component_only_partial", 0.25, POLICY),
        ("attention_only", None, "-"),
        ("layerskip", 0.33, "-"),
        ("early_exit", 0.5, "-"),
    ]
    GPU = gpu_tag()
    OTHER = load_other_gpu_rows(OUT_CSV, GPU, DEFAULT_PRE_L4_GPU)   # keep other-GPU rows
    rows = []
    for strat, frac, policy in variants:
        if strat == "component_only":
            set_fn = lambda: ctrl.set_mode("component_only")
        elif strat == "component_only_partial":
            set_fn = (lambda f=frac: ctrl.set_mode("component_only_partial", frac=f, policy=POLICY))
        elif strat == "attention_only":
            set_fn = lambda: ctrl.set_mode("attention_only")
        elif strat == "layerskip":
            set_fn = lambda: ctrl.set_mode("layerskip", frac=0.33)
        elif strat == "early_exit":
            set_fn = lambda: ctrl.set_mode("early_exit", frac=0.5)
        cparam = c_param_for(spec, strat, frac if frac is not None else 0.5)
        try:
            c_eff, t_draft, t_full = measure_c_eff(model, ctrl, set_fn, tids)
        except Exception as e:
            print(f"  {strat}: c_eff FAILED {type(e).__name__}: {str(e)[:100]} -> using c_param")
            c_eff = cparam
        label = f"{strat}" + (f"_{frac}" if frac is not None else "")
        dtv, top1 = divergence(model, tok, ctrl, set_fn, test, args.maxlen)
        for k in ks:
            r = acceptance(model, tok, ctrl, set_fn, k, test, args.maxlen)
            r["DTV"], r["top1"] = dtv, top1
            s_theory = (1 + r["tau"]) / (1 + k * c_eff) if c_eff == c_eff else float("nan")
            with open(os.path.join(PP, f"{label}__k{k}.pkl"), "wb") as fh:
                pickle.dump({"strategy": label, "k": k, "accepted_lengths": r["accepted"]}, fh)
            rows.append({"model": MODEL, "strategy": strat,
                         "frac": frac if frac is not None else "", "policy": policy,
                         "k": k, "A_k": round(r["A_k"], 5), "ci_lo": round(r["ci_lo"], 5),
                         "ci_hi": round(r["ci_hi"], 5), "tau_bar": round(r["tau"], 4),
                         "DTV": round(r["DTV"], 5), "top1": round(r["top1"], 5),
                         "c_param": round(cparam, 5), "c_eff": round(c_eff, 5),
                         "s_theory": round(s_theory, 5), "s_wall_optional": "",
                         "n_prompts": len(test), "gpu": GPU})
            flag = "  <-- S_theory>1!" if s_theory > 1.0 else ""
            print(f"  {label:28s} k={k} A_k={r['A_k']:.4f} tau={r['tau']:.3f} "
                  f"c_param={cparam:.3f} c_eff={c_eff:.3f} S_theory={s_theory:.3f}{flag}")
            write_csv(OTHER + rows, OUT_CSV, fieldnames=FIELDS)

    any_win = any(r["s_theory"] != "" and float(r["s_theory"]) > 1.0 for r in rows)
    is_l4 = "L4" in GPU
    write_meta_json(OUT_META, cwd=REPO, experiment="E3.1", model=MODEL, policy=POLICY,
                    ks=ks, n_prompts=args.n, any_s_theory_gt_1=any_win, gpu=GPU,
                    c_eff_note="draft/full forward-time ratio, use_cache=False, CUDA events; "
                    "the ratio is GPU-agnostic (canonical on L4, feasibility elsewhere)",
                    gpu_is_l4=is_l4, metric_is_gpu_agnostic=True)
    print(f"\nANY S_theory>1: {any_win}")
    print("wrote:", OUT_CSV)


if __name__ == "__main__":
    main()
