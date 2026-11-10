"""metrics.py — divergence, acceptance, and PPL metrics (PLAN §6).

Pure-math parts (dtv_full, dtv_top100, top1_agreement, kl_full_draft,
acceptance_rate_from_lengths, tau_bar) operate on numpy arrays and are CPU-
testable. `perplexity` needs a live model (UNVERIFIED, blocker B1) — it did NOT
exist in the notebook (PPL values were hardcoded in Cell 30); it is new here.

Metric conventions:
  A_k        = mean(accepted_lengths / k)               (v1's alpha; == tau_bar/k)
  tau_bar    = mean(accepted_lengths)
  D_TV       = 0.5 * sum_v |P_full - P_draft|           (v1 = FULL vocab)
  dtv_top100 = same but restricted to the top-100 index union (PLAN §6 default)
  top-1      = mean( argmax P_full == argmax P_draft )
  KL         = sum_v P_full * (log P_full - log P_draft)  (KL(P_full || P_draft))
"""
import numpy as np


def _softmax_np(logits, axis=-1):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ------------------------------- acceptance ------------------------------------

def acceptance_rate_from_lengths(accepted_lengths, k):
    """A_k = mean(na/k). Matches v1 `bootstrap_ci(accepted_lengths/k)` point est."""
    a = np.asarray(accepted_lengths, dtype=np.float64)
    if k <= 0:
        return 0.0
    return float(np.mean(a / k))


def tau_bar(accepted_lengths):
    return float(np.mean(np.asarray(accepted_lengths, dtype=np.float64)))


def s_theory(tau, k, c):
    """(1 + tau) / (1 + k*c)  — PLAN §6 replacement for the geometric-alpha Eq.1."""
    return (1.0 + tau) / (1.0 + k * c)


# ------------------------------- divergence ------------------------------------

def dtv_full(p_full, p_draft):
    """Full-vocab total variation, 0.5*sum|p-q|. Accepts probs (1-D or 2-D)."""
    p = np.asarray(p_full, dtype=np.float64)
    q = np.asarray(p_draft, dtype=np.float64)
    return 0.5 * np.abs(p - q).sum(axis=-1)


def dtv_top100(logits_full, logits_draft, topk=100):
    """Top-k TV over the union of each distribution's top-k indices (PLAN §6).

    Softmaxes the logits (full vocab), then sums 0.5*|p-q| over the union of the
    top-k indices of the two distributions. This is a lower bound on the full TV;
    documented choice (renormalization intentionally NOT applied so values stay
    comparable to dtv_full magnitude). For 1-D inputs only.
    """
    pf = _softmax_np(logits_full)
    pd = _softmax_np(logits_draft)
    idx = np.union1d(np.argsort(pf)[-topk:], np.argsort(pd)[-topk:])
    return float(0.5 * np.abs(pf[idx] - pd[idx]).sum())


def top1_agreement(logits_full, logits_draft):
    f = np.asarray(logits_full); d = np.asarray(logits_draft)
    return float(np.mean(np.argmax(f, axis=-1) == np.argmax(d, axis=-1)))


def kl_full_draft(p_full, p_draft, eps=1e-10):
    """KL(P_full || P_draft) weighted by P_full (matches v1 Cell 16)."""
    p = np.asarray(p_full, dtype=np.float64)
    q = np.asarray(p_draft, dtype=np.float64)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1))


def divergence_summary(dtv_vals, top1_vals, kl_vals):
    dtv = np.asarray(dtv_vals, dtype=np.float64)
    return {
        "d_tv_mean": float(np.mean(dtv)), "d_tv_std": float(np.std(dtv)),
        "d_tv_median": float(np.median(dtv)), "d_tv_p90": float(np.percentile(dtv, 90)),
        "top1_agreement": float(np.mean(top1_vals)),
        "kl_mean": float(np.mean(kl_vals)), "kl_std": float(np.std(kl_vals)),
        "n_positions": int(dtv.size),
    }


# ------------------------------- perplexity (NEW, UNVERIFIED) -------------------

def perplexity(model, tokenizer, texts, device=None, stride=512, max_length=1024):
    """Sliding-window perplexity over concatenated texts (E6).

    NEW: no live PPL eval existed in the notebook. Standard HF sliding-window NLL.
    UNVERIFIED (blocker B1): needs a working GPU/model.
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    seq_len = input_ids.size(1)
    nlls, n_tokens = [], 0
    prev_end = 0
    with torch.inference_mode():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            trg_len = end - prev_end
            ids = input_ids[:, begin:end]
            target = ids.clone()
            target[:, :-trg_len] = -100
            # use_cache=False so component_only identity/zero patches return the
            # bare-tensor arity these decoder loops expect (the cached path trips
            # 'tuple' has no attribute dtype/float on Qwen-family layers). No cache
            # is reused across windows anyway, so this is also marginally faster.
            out = model(ids, labels=target, use_cache=False)
            # out.loss is mean over trg_len-1 valid positions
            valid = trg_len - 1
            nlls.append(out.loss.float() * valid)
            n_tokens += valid
            prev_end = end
            if end == seq_len:
                break
    ppl = torch.exp(torch.stack(nlls).sum() / n_tokens)
    return {"ppl": float(ppl), "n_tokens": int(n_tokens)}
