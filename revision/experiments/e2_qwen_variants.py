#!/usr/bin/env python3
"""E2 — alternative draft constructions for Qwen3.5 (PLAN Phase 2; R2.1).

The claim "sequential hybrids are non-viable" must survive draft constructions
LESS destructive than full identity pass-through. We sweep attn_subset: keep
n_kept in {0,2,4,5} of the 6 attention layers (suppress {6,4,2,1}), under two
keep-policies chosen on val-tune:
  (a) keep_first_last : always keep the first & last attention layers.
  (b) top_ppl         : keep the attention layers whose INDIVIDUAL removal
                        degrades val-tune PPL most (measured here first).
Reference points: linear-only (= n_kept 0 = component_only), layerskip(0.33),
early_exit(0.5).

Metrics per variant on val-test (greedy, n=500 default): A_2, A_4, tau_bar, DTV
(top-100), c_param. Success = monotone collapse once >=2-3 attention layers are
suppressed (turns R2.1 into our strongest figure). If keeping 2 retains decent
alpha AND c_param<1, that is a NEW positive result — reported either way.

Output: results/E2/qwen_draft_variants.csv + per-prompt pkls + meta.json.
GPU-agnostic (acceptance/DTV/c_param); non-L4 OK.
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
from specdiag.controller import DraftModeController, normalize_qwen_layer_type  # noqa: E402
from specdiag.specloop import simulate_acceptance  # noqa: E402
from specdiag.metrics import perplexity, dtv_top100  # noqa: E402
from specdiag.data import load_wikitext, val_tune_test_split  # noqa: E402
from specdiag.io import write_csv, write_meta_json  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E2 = os.path.join(REPO, "results", "E2")
PP = os.path.join(E2, "perprompt")
OUT_CSV = os.path.join(E2, "qwen_draft_variants.csv")
OUT_META = os.path.join(E2, "qwen_draft_variants.meta.json")
MODEL = "qwen3.5-0.8b"
FIELDS = ["strategy", "n_attn_kept", "keep_policy", "k", "A_k", "ci_lo", "ci_hi",
          "tau_bar", "DTV", "c_param", "n_prompts"]


def flop_components(spec):
    d = spec["hidden_dim"]
    d_ff = spec.get("_intermediate_size", d * 4)
    n_heads = spec.get("_num_attention_heads", 16)
    n_kv = spec.get("_num_kv_heads", n_heads)
    head_d = spec.get("_head_dim", d // n_heads)
    seq_len = 256
    f_attn = 2 * d * (n_heads * head_d + 2 * n_kv * head_d) + 2 * n_heads * seq_len * head_d + 2 * d * d
    f_ssm = 8 * d * d
    f_mlp = 3 * 2 * d * d_ff
    return f_ssm, f_attn, f_mlp


def c_param_attn_subset(spec, n_linear, n_attn_total, n_attn_kept):
    """Sequential drafter keeps all linear layers + n_attn_kept attention layers;
    suppressed attention layers are identity (skip attn AND their mlp)."""
    f_ssm, f_attn, f_mlp = flop_components(spec)
    c_full = n_linear * (f_ssm + f_mlp) + n_attn_total * (f_attn + f_mlp)
    c_draft = n_linear * (f_ssm + f_mlp) + n_attn_kept * (f_attn + f_mlp)
    return c_draft / c_full if c_full > 0 else 1.0


@torch.inference_mode()
def variant_dtv(model, tok, ctrl, set_mode_fn, prompts, maxlen=192, pos=20, npr=40):
    """Mean DTV(top-100) full vs variant over sampled positions."""
    dtvs = []
    for text in prompts[:npr]:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=maxlen).to("cuda")
        ids = enc["input_ids"]
        if ids.shape[1] < 8:
            continue
        ctrl.deactivate()
        lf = model(input_ids=ids, use_cache=False).logits[0].float().cpu().numpy()
        set_mode_fn()
        ld = model(input_ids=ids, use_cache=False).logits[0].float().cpu().numpy()
        ctrl.deactivate()
        seq = lf.shape[0]
        for p in range(max(0, seq - pos), seq):
            dtvs.append(dtv_top100(lf[p], ld[p]))
    return float(np.mean(dtvs)) if dtvs else float("nan")


def per_layer_ppl_ranking(model, tok, ctrl, attn_idx, tune_texts):
    """Policy (b) prep: rank attention layers by individual-removal val-tune PPL."""
    ctrl.deactivate()
    base = perplexity(model, tok, tune_texts)["ppl"]
    impacts = {}
    for i in attn_idx:
        keep = [j for j in attn_idx if j != i]     # suppress ONLY layer i
        ctrl.set_mode("attn_subset", keep_ids=keep)
        ppl_i = perplexity(model, tok, tune_texts)["ppl"]
        ctrl.deactivate()
        impacts[i] = ppl_i - base
    ranked = sorted(attn_idx, key=lambda i: impacts[i], reverse=True)
    return ranked, impacts, base


def keep_set(policy, ranked, first_last, n_kept):
    if n_kept <= 0:
        return []
    if policy == "keep_first_last":
        s = list(first_last)                       # first & last first
        for i in ranked:                           # fill remainder deterministically
            if len(s) >= n_kept:
                break
            if i not in s:
                s.append(i)
        return sorted(s[:n_kept])
    else:  # top_ppl
        return sorted(ranked[:n_kept])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--ks", default="2,4")
    ap.add_argument("--maxlen", type=int, default=256)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    print(f"E2 Qwen variants | GPU={torch.cuda.get_device_name(0)} | n={args.n}")

    model, tok = load_model_and_tokenizer(MODEL)
    spec = M.MODEL_SPECS[MODEL]
    ctrl = DraftModeController(model, MODEL)
    lt = [normalize_qwen_layer_type(t) for t in model.config.layer_types]
    attn_idx = [i for i, t in enumerate(lt) if t == "attention"]
    n_attn, n_lin = len(attn_idx), sum(1 for t in lt if t == "linear")
    first_last = [attn_idx[0], attn_idx[-1]]

    texts = load_wikitext(split="validation", threshold=100)
    tune, test = val_tune_test_split(texts, n_tune=50)
    test = test[:args.n]

    ranked, impacts, base_ppl = per_layer_ppl_ranking(model, tok, ctrl, attn_idx, tune[:15])
    print(f"  val-tune base ppl={base_ppl:.2f}; per-layer removal impact (ranked): "
          f"{[(i, round(impacts[i],2)) for i in ranked]}")

    os.makedirs(PP, exist_ok=True)
    rows = []

    # build the variant list: (strategy_label, n_kept, policy, set_mode_fn)
    variants = []
    for n_kept in (0, 2, 4, 5):
        policies = ["keep_first_last", "top_ppl"] if 0 < n_kept < n_attn else ["-"]
        for pol in policies:
            kids = keep_set(pol, ranked, first_last, n_kept)
            label = f"attn_subset_keep{n_kept}" + ("" if pol == "-" else f"_{pol}")
            variants.append((label, n_kept, pol, kids))
    # references
    variants.append(("layerskip_0.33", None, "-", "layerskip"))
    variants.append(("early_exit_0.5", None, "-", "early_exit"))

    for label, n_kept, pol, spec_arg in variants:
        if spec_arg == "layerskip":
            set_fn = lambda: ctrl.set_mode("layerskip", frac=0.33)
            cparam = M.estimate_flop_ratio(MODEL)  # approximate ref
            nk_report = ""
        elif spec_arg == "early_exit":
            set_fn = lambda: ctrl.set_mode("early_exit", frac=0.5)
            cparam = M.estimate_flop_ratio(MODEL)
            nk_report = ""
        else:
            kids = spec_arg
            set_fn = (lambda kk=kids: ctrl.set_mode("attn_subset", keep_ids=kk))
            cparam = c_param_attn_subset(spec, n_lin, n_attn, n_kept)
            nk_report = n_kept
        # DTV (shared across k)
        dtv = variant_dtv(model, tok, ctrl, set_fn, test)
        for k in ks:
            # acceptance: wrap strategy via a temp mode set inside simulate? simulate
            # calls ctrl.set_mode(strategy). We instead pre-set and pass a sentinel by
            # using a custom strategy path: emulate by monkey-setting before each draft.
            res = simulate_acceptance_custom(model, tok, ctrl, set_fn, k, test, args.maxlen)
            na = np.asarray(res["accepted_lengths"])
            ppf = os.path.join(PP, f"{label}__k{k}.pkl")
            pickle.dump({"strategy": label, "k": k, "keep_policy": pol,
                         "accepted_lengths": res["accepted_lengths"]}, open(ppf, "wb"))
            rows.append({"strategy": label, "n_attn_kept": nk_report, "keep_policy": pol,
                         "k": k, "A_k": round(res["acceptance_rate"], 5),
                         "ci_lo": round(res["acceptance_rate_ci_lo"], 5),
                         "ci_hi": round(res["acceptance_rate_ci_hi"], 5),
                         "tau_bar": round(res["mean_accepted_length"], 4),
                         "DTV": round(dtv, 5), "c_param": round(cparam, 5),
                         "n_prompts": len(test)})
            print(f"  {label:34s} k={k} A_k={res['acceptance_rate']:.4f} "
                  f"tau={res['mean_accepted_length']:.3f} DTV={dtv:.3f} c_param={cparam:.3f}")
            write_csv(rows, OUT_CSV, fieldnames=FIELDS)

    write_meta_json(OUT_META, cwd=REPO, experiment="E2", model=MODEL,
                    attn_idx=attn_idx, ranked_by_ppl=ranked,
                    per_layer_impact={str(k): round(v, 3) for k, v in impacts.items()},
                    n_prompts=args.n, ks=ks, gpu_is_l4=False, metric_is_gpu_agnostic=True)
    print("wrote:", OUT_CSV)


@torch.inference_mode()
def simulate_acceptance_custom(model, tok, ctrl, set_fn, k, prompts, maxlen):
    """simulate_acceptance but the draft mode is installed by set_fn() (supports
    attn_subset with keep_ids, which the string-based simulate_acceptance can't)."""
    from specdiag.stats import bootstrap_ci
    from specdiag.specloop import accept_reject
    device = "cuda"
    accepted = []
    for text in prompts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=maxlen).to(device)
        iids = enc["input_ids"]
        set_fn()
        dt, dp = [], []
        cids = iids.clone()
        for _ in range(k):
            o = model(input_ids=cids, use_cache=False)
            lg = o.logits[:, -1, :].float()
            pr = torch.softmax(lg, dim=-1)
            tk = lg.argmax(dim=-1, keepdim=True)
            dt.append(tk.item()); dp.append(pr[0, tk.item()].item())
            cids = torch.cat([cids, tk], dim=-1)
        ctrl.deactivate()
        vid = torch.cat([iids, torch.tensor([dt], device=device)], dim=-1)
        fo = model(input_ids=vid, use_cache=False)
        fl = fo.logits[:, iids.shape[1] - 1: iids.shape[1] - 1 + k, :].float()
        na = accept_reject(dt, dp, fl[0].cpu().numpy(), 0.0)
        accepted.append(na)
    ptr = np.array(accepted) / k
    mr, cl, ch = bootstrap_ci(ptr)
    return {"acceptance_rate": mr, "acceptance_rate_ci_lo": cl,
            "acceptance_rate_ci_hi": ch,
            "mean_accepted_length": float(np.mean(accepted)),
            "accepted_lengths": accepted}


if __name__ == "__main__":
    main()
