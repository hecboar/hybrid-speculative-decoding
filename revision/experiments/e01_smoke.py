#!/usr/bin/env python3
"""E0.1 — Smoke tests for candidate new architectures (PLAN Phase 0).

For each candidate model: load it, dump the REAL architecture from config.json
(layer count, interleave pattern, attn/ssm/conv submodule naming), verify the
DraftModeController can patch it (component_only for the arch family), generate
20 tokens in both full and draft modes, and record peak VRAM. Anything that
fails to load or patch within the time box is logged as a blocker and its
fallback is used (PLAN E0.1 table).

Outputs:
  results/E0/arch_manifest.json   (feeds new Appendix A, answers R2.11)
  results/E0/e01_smoke.csv        (per-model load/patch/VRAM verdict)
  results/E0/e01_smoke.meta.json

Run ONE model:   python experiments/e01_smoke.py <key>
Run the default lineup: python experiments/e01_smoke.py

GPU-agnostic de-risking (loadability/patchability/VRAM) — non-L4 GPU is valid
VRAM figures are indicative on 32 GB Blackwell, not an L4 SLA.
"""
import os
import sys
import json
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # noqa: E402

from specdiag.io import write_csv, write_meta_json  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
E0 = os.path.join(REPO, "results", "E0")
MANIFEST = os.path.join(E0, "arch_manifest.json")
OUT_CSV = os.path.join(E0, "e01_smoke.csv")
OUT_META = os.path.join(E0, "e01_smoke.meta.json")

# Candidate lineup (PLAN E0.1 table). id = best-guess HF id, confirmed at runtime.
# family: how the controller should try to patch it.
CANDIDATES = {
    "granite":        {"id": "ibm-granite/granite-4.0-h-micro", "paradigm": "sequential",
                       "fallback": "lfm2", "priority": "P0-seq", "risk": "LOW"},
    "lfm2":           {"id": "LiquidAI/LFM2-1.2B", "paradigm": "sequential",
                       "fallback": "recurrentgemma", "priority": "P0-seq-alt", "risk": "LOW"},
    "hymba":          {"id": "nvidia/Hymba-1.5B-Base", "paradigm": "parallel",
                       "fallback": "zamba2", "priority": "P0-par", "risk": "MEDIUM"},
    "zamba2":         {"id": "Zyphra/Zamba2-1.2B", "paradigm": "parallel",
                       "fallback": None, "priority": "P0-par-alt", "risk": "MEDIUM"},
    "recurrentgemma": {"id": "google/recurrentgemma-2b", "paradigm": "sequential",
                       "fallback": None, "priority": "P1-seq3", "risk": "LOW"},
    "falcon15":       {"id": "tiiuae/Falcon-H1-1.5B-Base", "paradigm": "parallel",
                       "fallback": None, "priority": "P1-scale", "risk": "LOW"},
    "qwen35_2b":      {"id": "Qwen/Qwen3.5-2B-Base", "paradigm": "sequential",
                       "fallback": None, "priority": "P1-scale", "risk": "LOW"},
    "samba":          {"id": "microsoft/Samba-1.3B", "paradigm": "sequential",
                       "fallback": "granite", "priority": "P0-seq-preferred", "risk": "HIGH",
                       "note": "R2/R3 named; 24h time-box; also check microsoft/Samba GitHub"},
}
DEFAULT_LINEUP = ["granite", "lfm2", "hymba", "zamba2", "falcon15", "qwen35_2b",
                  "recurrentgemma", "samba"]

PROMPT = "The theory of general relativity, published by Albert Einstein, describes"


def find_layers(model):
    from torch import nn
    for path, fn in [("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
                     ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None))]:
        L = fn(model)
        if isinstance(L, (nn.ModuleList, list)) and len(L) > 0:
            return L, path
    best = (None, None)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
            if best[1] is None or len(mod) > len(best[1]):
                best = (name, mod)
    return best[1], best[0]


def describe_layer0(layer):
    """Child submodule names + classes of a decoder layer (for patch planning)."""
    kids = {}
    for name, mod in layer.named_children():
        kids[name] = mod.__class__.__name__
    return kids


def interleave_pattern(config, layers):
    """Best-effort per-layer type list from config, else from submodule presence."""
    for attr in ("layer_types", "layers_block_type", "hybrid_override_pattern",
                 "attn_layer_indices", "block_types"):
        v = getattr(config, attr, None)
        if v is not None:
            return {"source": attr, "pattern": list(v) if hasattr(v, "__iter__")
                    and not isinstance(v, str) else v}
    # fallback: probe each layer for an attention-ish child
    pat = []
    for lyr in layers:
        has_attn = any(("attn" in n.lower() or "attention" in m.__class__.__name__.lower())
                       for n, m in lyr.named_children())
        has_ssm = any((k in n.lower() or k in m.__class__.__name__.lower())
                      for n, m in lyr.named_children()
                      for k in ("mamba", "ssm", "mixer", "conv", "recurrent"))
        pat.append("attn" if has_attn else ("ssm" if has_ssm else "other"))
    return {"source": "submodule_probe", "pattern": pat}


def try_patch(model, key, paradigm, layers):
    """Register the model in a temporary spec and try component_only patching.
    Returns (ok, n_patched, err)."""
    from specdiag import models as M
    from specdiag.controller import DraftModeController
    cfg = model.config
    lt = getattr(cfg, "layer_types", None)
    n = len(layers)
    if paradigm == "parallel":
        fam = "falcon_parallel_hybrid"
    elif paradigm == "sequential":
        fam = "qwen_sequential_hybrid"
    else:
        fam = "transformer_baseline"
    M.MODEL_SPECS[key] = {
        "display_name": key, "model_id": key, "arch_family": fam,
        "hybrid_type": paradigm, "n_layers": n,
        "n_alt_layers": n, "n_attn_layers": n, "hidden_dim":
        getattr(cfg, "hidden_size", 0),
    }
    try:
        ctrl = DraftModeController(model, key)
        ctrl.set_mode("component_only")
        n_patched = len(ctrl.patched)
        ctrl.deactivate()
        return (n_patched > 0), n_patched, None
    except Exception as e:
        return False, 0, f"{type(e).__name__}: {e}"


@torch.inference_mode()
def gen20(model, tok, patch_fn=None):
    enc = tok(PROMPT, return_tensors="pt").to("cuda")
    out = model.generate(**enc, max_new_tokens=20, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def smoke_one(key):
    spec = CANDIDATES[key]
    model_id = spec["id"]
    rec = {"key": key, "model_id": model_id, "paradigm": spec["paradigm"],
           "priority": spec["priority"], "risk": spec["risk"],
           "loaded": 0, "patched": 0, "gen_full_ok": 0, "gen_draft_ok": 0,
           "n_layers": "", "n_patched": "", "peak_vram_gb": "", "load_s": "",
           "verdict": "", "error": ""}
    manifest = {"key": key, "model_id": model_id, "requested_paradigm": spec["paradigm"]}
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    token = os.environ.get("HF_TOKEN")
    tkw = {"trust_remote_code": True}
    if token:
        tkw["token"] = token
    try:
        cfg = AutoConfig.from_pretrained(model_id, **tkw)
        manifest["config_class"] = cfg.__class__.__name__
        manifest["config"] = {k: v for k, v in cfg.to_dict().items()
                              if isinstance(v, (int, float, str, bool, list))
                              and k in ("model_type", "num_hidden_layers", "hidden_size",
                                        "num_attention_heads", "num_key_value_heads",
                                        "intermediate_size", "vocab_size", "layer_types",
                                        "layers_block_type", "attn_layer_indices",
                                        "mamba_d_state", "mamba_expand", "sliding_window",
                                        "hybrid_override_pattern")}
    except Exception as e:
        rec["verdict"] = "CONFIG_FAIL"; rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        manifest["error"] = rec["error"]
        return rec, manifest

    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, **tkw)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map={"": 0}, **tkw)
        model.eval()
        rec["loaded"] = 1
        rec["load_s"] = round(time.time() - t0, 1)
    except Exception as e:
        rec["verdict"] = "LOAD_FAIL"; rec["error"] = f"{type(e).__name__}: {str(e)[:250]}"
        manifest["error"] = rec["error"]
        torch.cuda.empty_cache()
        return rec, manifest

    layers, path = find_layers(model)
    rec["n_layers"] = len(layers) if layers is not None else 0
    manifest["decoder_layers_path"] = path
    manifest["n_layers"] = rec["n_layers"]
    manifest["layer0_submodules"] = describe_layer0(layers[0]) if layers else {}
    manifest["interleave"] = interleave_pattern(model.config, layers) if layers else None
    manifest["n_params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)

    try:
        rec_full = gen20(model, tok)
        rec["gen_full_ok"] = 1
        manifest["gen_full_sample"] = rec_full[:120]
    except Exception as e:
        manifest["gen_full_error"] = f"{type(e).__name__}: {str(e)[:150]}"

    ok, n_patched, perr = try_patch(model, key, spec["paradigm"], layers)
    rec["patched"] = int(ok); rec["n_patched"] = n_patched
    if perr:
        manifest["patch_error"] = perr

    if ok:
        try:
            from specdiag.controller import DraftModeController
            ctrl = DraftModeController(model, key)
            ctrl.set_mode("component_only")
            draft_sample = gen20(model, tok)
            ctrl.deactivate()
            rec["gen_draft_ok"] = 1
            manifest["gen_draft_sample"] = draft_sample[:120]
        except Exception as e:
            manifest["gen_draft_error"] = f"{type(e).__name__}: {str(e)[:150]}"

    rec["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    manifest["peak_vram_gb"] = rec["peak_vram_gb"]
    if rec["loaded"] and rec["patched"] and rec["gen_draft_ok"]:
        rec["verdict"] = "VIABLE"
    elif rec["loaded"] and not rec["patched"]:
        rec["verdict"] = "LOADS_NOT_PATCHABLE"
    elif rec["loaded"]:
        rec["verdict"] = "LOADS_PATCH_PARTIAL"
    manifest["verdict"] = rec["verdict"]

    del model
    torch.cuda.empty_cache()
    return rec, manifest


def main():
    keys = sys.argv[1:] or DEFAULT_LINEUP
    keys = [k for k in keys if k in CANDIDATES]
    print(f"E0.1 smoke tests | GPU={torch.cuda.get_device_name(0)} | lineup={keys}")

    # merge into existing manifest/csv so per-model runs accumulate
    manifests = {}
    if os.path.exists(MANIFEST):
        try:
            manifests = json.load(open(MANIFEST))
        except Exception:
            manifests = {}
    rows_by_key = {}
    if os.path.exists(OUT_CSV):
        import csv as _csv
        for r in _csv.DictReader(open(OUT_CSV)):
            rows_by_key[r["key"]] = r

    for key in keys:
        print(f"\n===== {key} ({CANDIDATES[key]['id']}) =====")
        try:
            rec, manifest = smoke_one(key)
        except Exception:
            rec = {"key": key, "model_id": CANDIDATES[key]["id"],
                   "paradigm": CANDIDATES[key]["paradigm"],
                   "priority": CANDIDATES[key]["priority"], "risk": CANDIDATES[key]["risk"],
                   "loaded": 0, "patched": 0, "gen_full_ok": 0, "gen_draft_ok": 0,
                   "n_layers": "", "n_patched": "", "peak_vram_gb": "", "load_s": "",
                   "verdict": "CRASH", "error": traceback.format_exc().splitlines()[-1][:200]}
            manifest = {"key": key, "error": rec["error"]}
        rows_by_key[key] = rec
        manifests[key] = manifest
        print(f"  verdict={rec['verdict']}  loaded={rec['loaded']} patched={rec['patched']}"
              f"({rec['n_patched']}) gen_draft={rec['gen_draft_ok']} "
              f"vram={rec['peak_vram_gb']}GB  err={rec['error'][:80]}")
        # write incrementally (checkpoint-safe): dump after every model
        json.dump(manifests, open(MANIFEST, "w"), indent=2)
        fieldnames = ["key", "model_id", "paradigm", "priority", "risk", "loaded",
                      "patched", "n_patched", "gen_full_ok", "gen_draft_ok",
                      "n_layers", "peak_vram_gb", "load_s", "verdict", "error"]
        write_csv([rows_by_key[k] for k in rows_by_key], OUT_CSV, fieldnames=fieldnames)

    write_meta_json(OUT_META, cwd=REPO, experiment="E0.1",
                    lineup=keys, gpu_is_l4=False, metric_is_gpu_agnostic=True,
                    note="VRAM indicative on 32GB Blackwell, not an L4 SLA")
    print("\nwrote:", MANIFEST, OUT_CSV)


if __name__ == "__main__":
    main()
