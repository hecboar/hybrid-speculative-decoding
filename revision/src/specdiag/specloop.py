"""specloop.py — draft/verify speculative loop (greedy + sampling).

FIDELITY SPLIT (important — do not merge these two paths):

  * `simulate_acceptance` / `spec_generate_v1` reproduce the v1 notebook EXACTLY
    (Cell 19 acceptance simulator; Cell 12/42 `measure_speedup_v2`). In v1 the
    correction / bonus token after a rejection is drawn DIRECTLY from the full
    model's logits at the reject position (`fl[:, na, :]`), NOT from the
    normalized residual norm(max(0, P_H − P_S)). This is NOT Leviathan-exact and
    breaks the lossless guarantee for T>0. We KEEP it to reproduce v1 numbers.

  * `residual_correction_token(P_H, P_S)` implements the correct Leviathan
    residual draw for reference / E0.4. It is deliberately NOT wired into the v1
    reproduction path.

The pure helpers (`accept_reject`, `residual_correction_token`) run on CPU with
numpy; the generation loops need a GPU.
"""
import random
import numpy as np


# ------------------------------- pure helpers (CPU-testable) --------------------

def residual_correction_token(P_H, P_S, rng=None):
    """Leviathan-exact residual draw: sample from norm(max(0, P_H − P_S)).

    P_H, P_S: 1-D probability vectors (numpy) of the full and draft models. Used
    for E0.4 reference; NOT part of the v1 reproduction path.
    """
    P_H = np.asarray(P_H, dtype=np.float64)
    P_S = np.asarray(P_S, dtype=np.float64)
    resid = np.maximum(0.0, P_H - P_S)
    s = resid.sum()
    if s <= 0:
        resid = P_H / P_H.sum()          # degenerate: fall back to target
    else:
        resid = resid / s
    rng = rng or random
    r = rng.random()
    c = np.cumsum(resid)
    return int(np.searchsorted(c, r, side="right"))


def accept_reject(draft_tokens, draft_probs, full_logits, temperature, rng=None):
    """v1-faithful accept/reject over one draft round; returns n_accepted (na).

    full_logits: array shape (k, vocab). Greedy (T==0): accept while the full
    model's argmax matches the draft token. Sampling (T>0): accept token i with
    prob min(1, P_full(i)/P_draft(i)), stop at first rejection. Ported from
    Cell 19 / Cell 12.
    """
    rng = rng or random
    k = len(draft_tokens)
    fl = np.asarray(full_logits, dtype=np.float64)
    na = 0
    for i in range(k):
        row = fl[i]
        if temperature == 0.0:
            if int(np.argmax(row)) == draft_tokens[i]:
                na += 1
            else:
                break
        else:
            z = row / temperature
            z = z - z.max()
            p = np.exp(z); p = p / p.sum()
            pf = p[draft_tokens[i]]
            pd = draft_probs[i]
            if pd == 0:
                break
            if rng.random() < min(1.0, pf / pd):
                na += 1
            else:
                break
    return na


# ------------------------------- GPU loops -------------------------------------

def simulate_acceptance(model, tokenizer, controller, strategy, k,
                        temperature=0.0, prompts=None, n_trials=None,
                        device=None, max_len=None):
    """v1 acceptance simulator (Cell 19). Always returns the per-prompt
    accepted_lengths alongside the aggregates.

    Draft uses use_cache=False full-seq recompute (acceptance, not timing). Verify
    is a single full forward. No correction token is drawn here (acceptance only).
    """
    import torch
    from .stats import bootstrap_ci
    from .models import get_max_inference_seq_length, MODEL_SPECS  # noqa: F401
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    accepted_lengths = []
    n = min(n_trials or len(prompts), len(prompts))
    for ti in range(n):
        enc = tokenizer(prompts[ti], return_tensors="pt", truncation=True,
                        max_length=max_len or 512).to(device)
        iids = enc["input_ids"]

        controller.set_mode(strategy)
        draft_tokens, draft_probs = [], []
        cids = iids.clone()
        with torch.inference_mode():
            for _ in range(k):
                o = model(input_ids=cids, use_cache=False)
                lg = o.logits[:, -1, :].float()
                if temperature == 0.0:
                    pr = torch.softmax(lg, dim=-1)
                    tk = lg.argmax(dim=-1, keepdim=True)
                else:
                    pr = torch.softmax(lg / temperature, dim=-1)
                    tk = torch.multinomial(pr, 1)
                draft_tokens.append(tk.item())
                draft_probs.append(pr[0, tk.item()].item())
                cids = torch.cat([cids, tk], dim=-1)
        controller.deactivate()

        dt = torch.tensor([draft_tokens], device=device)
        vid = torch.cat([iids, dt], dim=-1)
        with torch.inference_mode():
            fo = model(input_ids=vid, use_cache=False)
            fl = fo.logits[:, iids.shape[1] - 1: iids.shape[1] - 1 + k, :].float()
        na = accept_reject(draft_tokens, draft_probs,
                           fl[0].cpu().numpy(), temperature)
        accepted_lengths.append(na)

    ptr = np.array(accepted_lengths) / k if k > 0 else np.zeros(len(accepted_lengths))
    mr, cl, ch = bootstrap_ci(ptr)
    return {"strategy": strategy, "k": k, "temperature": temperature,
            "n_trials": len(accepted_lengths), "acceptance_rate": mr,
            "acceptance_rate_ci_lo": cl, "acceptance_rate_ci_hi": ch,
            "mean_accepted_length": float(np.mean(accepted_lengths)),
            "std_accepted_length": float(np.std(accepted_lengths)),
            "accepted_lengths": accepted_lengths}


def spec_generate_v1(model, tokenizer, controller, strategy, prompt, k,
                     temperature=0.0, max_new_tokens=64, device=None, max_len=256):
    """v1 speculative generation with incremental-KV draft (Cell 12
    measure_speedup_v2 inner loop). Correction token drawn DIRECTLY from the full
    model's logits at the reject position (v1 behaviour — NOT Leviathan-exact).

    Returns the generated token ids and the per-round accepted lengths.
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_len).to(device)
    cids = enc["input_ids"]
    gen = 0
    round_accepts = []
    with torch.inference_mode():
        while gen < max_new_tokens:
            controller.set_mode(strategy)
            draft_tokens, draft_probs = [], []
            draft_past, draft_input = None, cids
            for _ in range(k):
                try:
                    out = model(input_ids=draft_input, past_key_values=draft_past, use_cache=True)
                    draft_past = out.past_key_values
                except Exception:
                    seq = torch.cat([cids] + [torch.tensor([[t]], device=device)
                                              for t in draft_tokens], dim=-1) if draft_tokens else cids
                    out = model(input_ids=seq, use_cache=False)
                    draft_past = None
                lg = out.logits[:, -1, :].float()
                pr = torch.softmax(lg, dim=-1)
                tk = lg.argmax(dim=-1, keepdim=True) if temperature == 0.0 \
                    else torch.multinomial(torch.softmax(lg / temperature, dim=-1), 1)
                draft_tokens.append(tk.item())
                draft_probs.append(pr[0, tk.item()].item())
                draft_input = tk
            controller.deactivate()

            actual_k = len(draft_tokens)
            if actual_k == 0:
                break
            vi = torch.cat([cids, torch.tensor([draft_tokens], device=device)], dim=-1)
            vo = model(input_ids=vi, use_cache=False)
            fl = vo.logits[:, cids.shape[1] - 1: cids.shape[1] - 1 + actual_k, :].float()
            na = accept_reject(draft_tokens, draft_probs, fl[0].cpu().numpy(), temperature)
            round_accepts.append(na)

            if na < actual_k:  # rejection: v1 draws bonus from FULL logits at pos na
                bl = fl[:, na, :]
                bonus = bl.argmax(dim=-1, keepdim=True) if temperature == 0.0 \
                    else torch.multinomial(torch.softmax(bl / temperature, dim=-1), 1)
                acc = draft_tokens[:na] + [bonus.item()]
            else:              # all accepted: bonus from next-position full logits
                ll = vo.logits[:, -1, :].float()
                bonus = ll.argmax(dim=-1, keepdim=True) if temperature == 0.0 \
                    else torch.multinomial(torch.softmax(ll / temperature, dim=-1), 1)
                acc = draft_tokens + [bonus.item()]
            cids = torch.cat([cids, torch.tensor([acc], device=device)], dim=-1)
            gen += len(acc)
    return {"sequences": cids, "n_generated": gen, "round_accepts": round_accepts}
