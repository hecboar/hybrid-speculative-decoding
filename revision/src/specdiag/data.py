"""data.py — prompt loading, tune/test split, long-context + recent corpus.

WikiText-2 loading is consolidated from the ad-hoc calls scattered across the
notebook (Cells 12/16/19/28). `val_tune_test_split`, `build_long_context_prompts`
and `load_recent_corpus` are NEW infra the notebook lacked (PLAN constraints 6,
E4, E5.2). `datasets` is imported lazily so this module imports without it.
"""
import random

SEED = 42

# Length thresholds used per experiment in the notebook.
THRESH = {"speedup": 50, "acceptance": 100, "divergence": 200}


def load_wikitext(split="validation", threshold=100, n=None):
    """WikiText-2-raw-v1 texts filtered by length (Cell 12/16/19 pattern).

    Uses the namespaced repo id `Salesforce/wikitext`: the bare `wikitext` id is
    rejected by datasets>=5 / huggingface_hub (requires 'namespace/name'). Same
    underlying data/config as v1.
    """
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    texts = [e["text"].strip() for e in ds if len(e["text"].strip()) > threshold]
    return texts[:n] if n else texts


def val_tune_test_split(prompts, n_tune=50):
    """PLAN constraint 6: val-tune = first 50, val-test = the rest.

    Returns (val_tune, val_test). All tuning choices use val_tune; all reported
    numbers use val_test. Deterministic (no shuffle — preserves the fixed WikiText
    order so every pod produces the identical split).
    """
    return list(prompts[:n_tune]), list(prompts[n_tune:])


def build_long_context_prompts(tokenizer, target_lens=(512, 1024, 2048, 4096, 8192),
                               n_per_len=50, source_split="validation", threshold=200):
    """Concatenate WikiText docs into prompts of ~target token length (E4).

    Returns {length: [prompt_str, ...]}. Greedily concatenates documents until the
    tokenized length reaches the target, then truncates to exactly `length` tokens.
    """
    docs = load_wikitext(split=source_split, threshold=threshold)
    out = {}
    for L in target_lens:
        prompts = []
        i = 0
        while len(prompts) < n_per_len and i < len(docs):
            buf, ids_len, j = [], 0, i
            while ids_len < L and j < len(docs):
                buf.append(docs[j])
                ids_len = len(tokenizer(" ".join(buf))["input_ids"])
                j += 1
            ids = tokenizer(" ".join(buf))["input_ids"][:L]
            prompts.append(tokenizer.decode(ids))
            i = j if j > i else i + 1
        out[L] = prompts
    return out


def load_recent_corpus(n=200, threshold=100, name="redpajama"):
    """Held-out non-WikiText corpus for the contamination check (E5.2, R2.4).

    R2 named RedPajama-Data-1T-Sample verbatim, but that HF id is no longer
    accessible on the Hub (DatasetNotFoundError as of 2026-07). We fall back to
    **allenai/c4 (en)** — an equivalent large held-out Common-Crawl web corpus
    outside WikiText, which serves the same contamination-control purpose. The
    substitution + reason are recorded in meta so the rebuttal is transparent.
    Returns (texts, meta). Streams (no full download).
    """
    from datasets import load_dataset

    def _stream(rid, cfg=None):
        kw = dict(split="train", streaming=True)
        if cfg:
            kw["name"] = cfg
        ds = load_dataset(rid, **kw)
        out = []
        for e in ds:
            t = (e.get("text") or "").strip()
            if len(t) > threshold:
                out.append(t)
            if len(out) >= n:
                break
        return out

    candidates = []
    if name == "redpajama":
        candidates = [("togethercomputer/RedPajama-Data-1T-Sample", None,
                       "RedPajama-Data-1T-Sample", "2023"),
                      ("allenai/c4", "en", "C4-en (RedPajama unavailable on Hub)", "2019")]
    elif name == "c4":
        candidates = [("allenai/c4", "en", "C4-en", "2019")]
    else:
        raise ValueError(f"unknown recent corpus {name!r}")

    last_err = None
    for rid, cfg, label, date in candidates:
        try:
            texts = _stream(rid, cfg)
            if texts:
                return texts, {"corpus": label, "hf_id": rid, "config": cfg,
                               "corpus_date": date, "split": "train", "n": len(texts),
                               "note": ("primary RedPajama id defunct; C4 substitute"
                                        if "C4" in label and name == "redpajama" else "")}
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"no held-out corpus loadable; last error: {last_err}")


def get_task_prompts(task, n, seed=SEED):
    """mmlu / gsm8k / alpaca prompts (Cell 28). Deterministic shuffle by seed."""
    from datasets import load_dataset
    if task == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        ps = [f"Question: {e['question']}\nAnswer:" for e in ds]
    elif task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ps = [f"Question: {e['question']}\nAnswer:" for e in ds]
    elif task == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        ps = [e["instruction"] for e in ds if len(e["instruction"]) > 20]
    else:
        ps = [f"Tell me about topic {i}" for i in range(n)]
    random.seed(seed)
    random.shuffle(ps)
    return ps[:n]
