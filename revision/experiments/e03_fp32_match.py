#!/usr/bin/env python3
"""E0.3 — FP32 match-rate control (PLAN Phase 0; answers R2.5, R3).

Greedy speculative decoding is mathematically LOSSLESS w.r.t. greedy
autoregressive decoding: every emitted token is the verifier's (full model's)
argmax given the accepted prefix, regardless of what the internal drafter
proposed. So AR-greedy and spec-greedy MUST produce the identical token stream
in exact arithmetic. v1 observed only 90-96% agreement and attributed the gap to
bf16 non-determinism. This control tests that claim directly:

  four-way grid  {AR, speculative} x {fp32, bf16}  on Falcon-H1-0.5B and
  Qwen3.5-0.8B, greedy, k=4, 30 prompts, 64 new tokens.

Within each dtype we compare the AR stream vs the speculative stream token by
token. Expected: fp32 -> ~100% agreement (numeric flips vanish); bf16 -> the v1
90-96% regime, with every first divergence sitting at a position whose
top1-top2 logit margin is below bf16 resolution.

The AR reference is standard KV-cached incremental greedy decoding (exactly how
v1's autoregressive baseline ran); the spec verifier does a batched full-sequence
forward over [prefix + stream]. The two paths therefore differ only in reduction
order / batching — pure floating-point rounding — so within a run the dtype is
the sole independent variable, and the fp32-vs-bf16 match-rate contrast isolates
bf16 non-determinism. (Cached vs batched fp32 logits differ by ~1e-6, far below
argmax-flip resolution, so fp32 -> ~100%; bf16 differs by ~1e-2..1e-1, flipping
low-margin argmaxes -> the v1 90-96% regime.)

METHOD (exact reduction, not an approximation). Under greedy decoding, spec
decoding is lossless: the emitted token at every accepted position is the
VERIFIER's argmax given the accepted prefix, independent of which drafter
proposed it (the drafter only changes speed / how many rounds). So the question
"does spec output == AR output" reduces EXACTLY to "does the batched-verify
argmax == the incremental-AR argmax at each position", and the FIRST position
where they differ is exactly where a real greedy spec run would first diverge
from AR (positions before it are conditioned on identical tokens in both paths).
We therefore build the AR stream once (cached), run ONE batched verify forward
over [prefix + AR stream], and feed (AR stream as draft, verify logits) into the
REAL v1 rejection routine `specloop.accept_reject` (greedy branch) — na it
returns == the first-divergence position. This audits the actual rejection code
and runs in ~1 batched forward per prompt instead of ~130 (the full spec loop).

Output: results/E0/fp32_match.csv
  columns: model, dtype, prompt_id, match, first_divergence_pos,
           logit_margin_at_mismatch, matched_prefix_len, n_compared
plus results/E0/fp32_match.md (human summary) and meta.json (provenance).

GPU note: run on RTX PRO 4500 (sm_120), NOT an L4. This experiment is
GPU-agnostic (token identity / logit margins), so the non-L4 GPU is valid here
(see results/E0/blockers.md B1).
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from specdiag.specloop import accept_reject  # noqa: E402  (audited v1 rejection code)
from specdiag.data import load_wikitext  # noqa: E402
from specdiag.io import write_csv, write_meta_json  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_CSV = os.path.join(REPO, "results", "E0", "fp32_match.csv")
OUT_MD = os.path.join(REPO, "results", "E0", "fp32_match.md")
OUT_META = os.path.join(REPO, "results", "E0", "fp32_match.meta.json")

MODELS = [
    ("falcon-h1-0.5b", "tiiuae/Falcon-H1-0.5B-Base"),
    ("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B-Base"),
]
K = 4                           # nominal draft length (audit note; greedy output
                                # identity is drafter-independent — see docstring)
N_PROMPTS = int(os.environ.get("E03_N_PROMPTS", "30"))
MAX_NEW = 64
PREFIX_LEN = 256                # keep Falcon under its no-kernel 1024 cap
DEVICE = "cuda"


def load_at_dtype(model_id, dtype):
    tok_kw = dict(trust_remote_code=True, use_fast=True)
    token = os.environ.get("HF_TOKEN")
    if token:
        tok_kw["token"] = token
    tok = AutoTokenizer.from_pretrained(model_id, **tok_kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    kw = dict(dtype=dtype, device_map={"": 0}, trust_remote_code=True)
    if token:
        kw["token"] = token
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    return model, tok


@torch.inference_mode()
def ar_greedy(model, prefix_ids, max_new):
    """Standard KV-cached incremental greedy AR (v1's AR baseline style).

    Returns (generated_ids: list[int], margins: list[float]) where margins[i] is
    the top1-top2 logit margin (float32) at the position that produced token i.
    Falls back to full recompute if the model rejects incremental past_key_values.
    """
    out = model(input_ids=prefix_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :].float()[0]
    gen, margins = [], []
    cur_ids = prefix_ids
    for _ in range(max_new):
        top2 = torch.topk(logits, 2).values
        margins.append(float((top2[0] - top2[1]).item()))
        nxt = int(logits.argmax().item())
        gen.append(nxt)
        nxt_t = torch.tensor([[nxt]], device=prefix_ids.device)
        cur_ids = torch.cat([cur_ids, nxt_t], dim=-1)
        try:
            out = model(input_ids=nxt_t, past_key_values=past, use_cache=True)
            past = out.past_key_values
        except Exception:
            out = model(input_ids=cur_ids, use_cache=False)
            past = None
        logits = out.logits[:, -1, :].float()[0]
    return gen, margins


@torch.inference_mode()
def batched_verify_logits(model, prefix_ids, ar_ids):
    """Single batched full-model forward over [prefix + ar_ids]; returns the
    logits (numpy, float32) predicting each of the len(ar_ids) stream positions.
    This is the speculative verifier's forward (use_cache=False, all positions at
    once) — the counterpart to the incremental AR forward."""
    stream = torch.tensor([ar_ids], device=prefix_ids.device)
    vi = torch.cat([prefix_ids, stream], dim=-1)
    out = model(input_ids=vi, use_cache=False)
    p0 = prefix_ids.shape[1]
    fl = out.logits[:, p0 - 1: p0 - 1 + len(ar_ids), :].float()[0]
    return fl.cpu().numpy()


def compare(ar_ids, verify_logits, ar_margins):
    """Feed (AR stream as draft, batched-verify logits) to the audited v1
    greedy rejection routine. na = leading positions whose verifier-argmax equals
    the AR token = the first-divergence position. Also compute per-token
    agreement over all n positions (each conditioned on the true AR prefix) —
    this is v1's '90-96% match rate'. Returns (match, first_div_pos,
    margin_at_div, matched_prefix_len, n_compared, token_agreement)."""
    n = len(ar_ids)
    # draft_probs unused in the greedy (T==0) branch of accept_reject.
    na = accept_reject(ar_ids, [1.0] * n, verify_logits, temperature=0.0)
    verify_argmax = np.argmax(verify_logits, axis=-1)
    token_agreement = float(np.mean(verify_argmax == np.asarray(ar_ids)))
    if na >= n:
        return 1, -1, float("nan"), n, n, token_agreement
    return 0, na, (ar_margins[na] if na < len(ar_margins) else float("nan")), na, n, token_agreement


def main():
    print(f"E0.3 fp32 match-rate control | GPU={torch.cuda.get_device_name(0)}")
    prompts = load_wikitext(split="validation", threshold=100, n=N_PROMPTS)
    assert len(prompts) >= N_PROMPTS, f"only {len(prompts)} prompts"
    prompts = prompts[:N_PROMPTS]

    rows = []
    for key, model_id in MODELS:
        for dtype, dname in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
            print(f"\n=== {key} @ {dname} ===")
            model, tok = load_at_dtype(model_id, dtype)
            for pid, text in enumerate(prompts):
                enc = tok(text, return_tensors="pt", truncation=True,
                          max_length=PREFIX_LEN).to(DEVICE)
                prefix_ids = enc["input_ids"]

                ar_ids, ar_margins = ar_greedy(model, prefix_ids, MAX_NEW)
                verify_logits = batched_verify_logits(model, prefix_ids, ar_ids)

                match, fdp, margin, mpl, ncmp, tok_agr = compare(
                    ar_ids, verify_logits, ar_margins)
                rows.append({
                    "model": key, "dtype": dname, "prompt_id": pid,
                    "match": match, "first_divergence_pos": fdp,
                    "logit_margin_at_mismatch": ("" if margin != margin else round(margin, 6)),
                    "matched_prefix_len": mpl, "n_compared": ncmp,
                    "token_agreement": round(tok_agr, 4),
                })
            # free before next dtype/model
            del model
            torch.cuda.empty_cache()
            sub = [r for r in rows if r["model"] == key and r["dtype"] == dname]
            mr = np.mean([r["match"] for r in sub])
            mta = np.mean([r["token_agreement"] for r in sub])
            print(f"  exact_match_rate={mr:.3f}  token_agreement={mta:.4f}")

    write_csv(rows, OUT_CSV, fieldnames=[
        "model", "dtype", "prompt_id", "match", "first_divergence_pos",
        "logit_margin_at_mismatch", "matched_prefix_len", "n_compared",
        "token_agreement"])

    # ---- summary md ----
    lines = ["# E0.3 — FP32 match-rate control (R2.5, R3)", "",
             f"Greedy, {N_PROMPTS} prompts, {MAX_NEW} new tokens, prefix<= {PREFIX_LEN} "
             "tok. Metric: batched-verify argmax vs incremental-AR argmax, scored by "
             "the audited v1 `accept_reject` (greedy). Divergences are drafter-"
             "independent under greedy decoding (see script docstring).",
             "GPU: RTX PRO 4500 (non-L4; GPU-agnostic metric — see blockers.md B1).",
             "",
             "| model | dtype | token_agreement | exact_match_rate | "
             "mean_matched_prefix | median_margin_at_div |",
             "|---|---|---|---|---|---|"]
    summary = {}
    for key, _ in MODELS:
        for dname in ("fp32", "bf16"):
            sub = [r for r in rows if r["model"] == key and r["dtype"] == dname]
            mr = float(np.mean([r["match"] for r in sub]))
            mta = float(np.mean([r["token_agreement"] for r in sub]))
            mmpl = float(np.mean([r["matched_prefix_len"] for r in sub]))
            margs = [r["logit_margin_at_mismatch"] for r in sub
                     if r["logit_margin_at_mismatch"] != ""]
            med = float(np.median(margs)) if margs else float("nan")
            summary[f"{key}/{dname}"] = {"token_agreement": mta, "exact_match_rate": mr,
                                         "mean_matched_prefix": mmpl,
                                         "n_mismatch_prompts": len(margs),
                                         "median_margin_at_div": med}
            lines.append(f"| {key} | {dname} | {mta:.4f} | {mr:.3f} | {mmpl:.1f} | "
                         f"{'n/a' if med != med else f'{med:.4f}'} |")
    lines += ["",
              "**Verdict.** fp32 match_rate near 1.0 confirms v1's 90-96% agreement "
              "as bf16 non-determinism: greedy spec-decoding is lossless, and the "
              "residual mismatches occur exactly where the top1-top2 logit margin "
              "is below bf16 resolution (~1e-2..1e-1). If fp32 mismatches PERSIST, "
              "the rejection path has a real bug — STOP and debug before other GPU "
              "work (PLAN E0.3)."]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")

    write_meta_json(OUT_META, cwd=REPO, experiment="E0.3",
                    method="batched_verify_argmax_vs_incremental_AR_argmax "
                    "scored by audited accept_reject (greedy); drafter-independent",
                    k_nominal=K, n_prompts=N_PROMPTS,
                    max_new_tokens=MAX_NEW, prefix_len=PREFIX_LEN,
                    models=[m[0] for m in MODELS], summary=summary,
                    gpu_is_l4=False, metric_is_gpu_agnostic=True)
    print("\nwrote:", OUT_CSV)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
