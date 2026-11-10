"""models.py — model registry + loading + FLOP model.

Extracted from notebook Cell 7 (Cell 1A) and the FLOP estimators in Cell 9.
torch/transformers are imported lazily inside functions so this module imports
on a CPU-only / broken-CUDA host (blocker B1).

Key exports:
  MODEL_SPECS                       registry keyed by model_key
  get_decoder_layers(model)         -> (layers, path)
  get_max_inference_seq_length(key) -> int
  load_model_and_tokenizer(key)     -> (model, tokenizer)   [UNVERIFIED: needs GPU]
  unload_model(key)
  estimate_flop_ratio(key, seq_len) -> c_draft/c_full
  estimate_flops_gflops(key, seq_len) -> (draft_gf, full_gf)
"""
import os

# Arch families / hybrid types drive ALL controller branching.
MODEL_SPECS = {
    "qwen3.5-0.8b": {
        "display_name": "Qwen3.5-0.8B",
        "model_id": "Qwen/Qwen3.5-0.8B-Base",
        "arch_family": "qwen_sequential_hybrid", "hybrid_type": "sequential",
        "n_layers": 24, "n_alt_layers": 18, "n_attn_layers": 6, "hidden_dim": 1536,
    },
    "falcon-h1-0.5b": {
        "display_name": "Falcon-H1-0.5B",
        "model_id": "tiiuae/Falcon-H1-0.5B-Base",
        "arch_family": "falcon_parallel_hybrid", "hybrid_type": "parallel",
        "max_seq_length_no_kernel": 1024,
        "n_layers": 36, "n_alt_layers": 36, "n_attn_layers": 36, "hidden_dim": 1024,
    },
    "qwen2.5-0.5b": {
        "display_name": "Qwen2.5-0.5B", "model_id": "Qwen/Qwen2.5-0.5B",
        "arch_family": "transformer_baseline", "hybrid_type": "transformer",
        "n_layers": 24, "n_alt_layers": 0, "n_attn_layers": 24, "hidden_dim": 896,
    },
    "falcon-h1-1.5b": {
        "display_name": "Falcon-H1-1.5B",
        "model_id": "tiiuae/Falcon-H1-1.5B-Base",
        "arch_family": "falcon_parallel_hybrid", "hybrid_type": "parallel",
        "max_seq_length_no_kernel": 1024,
        "n_layers": 36, "n_alt_layers": 36, "n_attn_layers": 36, "hidden_dim": 1536,
    },
    "qwen3.5-2b": {
        "display_name": "Qwen3.5-2B", "model_id": "Qwen/Qwen3.5-2B-Base",
        "arch_family": "qwen_sequential_hybrid", "hybrid_type": "sequential",
        "n_layers": 28, "n_alt_layers": 21, "n_attn_layers": 7, "hidden_dim": 1536,
    },
    # Injected in notebook Cell 14; folded into the static registry here.
    "falcon-h1-3b": {
        "display_name": "Falcon-H1-3B",
        "model_id": "tiiuae/Falcon-H1-3B-Base",
        "arch_family": "falcon_parallel_hybrid", "hybrid_type": "parallel",
        "max_seq_length_no_kernel": 1024,
        "n_layers": 32, "n_alt_layers": 32, "n_attn_layers": 32, "hidden_dim": 2048,
    },
}

HF_TOKEN = os.environ.get("HF_TOKEN")  # never hardcode (notebook Cell 4 leaked one)

_LOADED_MODELS = {}


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype():
    import torch
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        else torch.float16


def get_decoder_layers(model):
    """Return (layers, path). Ported verbatim from Cell 7."""
    from torch import nn
    for path, fn in [("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
                     ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None))]:
        layers = fn(model)
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            return layers, path
    best_name, best_mod = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4:
            if best_mod is None or len(module) > len(best_mod):
                best_name, best_mod = name, module
    if best_mod is None:
        raise ValueError("Could not locate decoder layers")
    return best_mod, best_name


def get_max_inference_seq_length(model_key):
    spec = MODEL_SPECS[model_key]
    if "max_seq_length_no_kernel" in spec:
        try:
            import mamba_ssm  # noqa: F401
            return 4096
        except ImportError:
            return spec["max_seq_length_no_kernel"]
    return 4096


def load_model_and_tokenizer(model_key, log=print):
    """Load (model, tokenizer); writes arch_info side-effects into MODEL_SPECS.

    UNVERIFIED (blocker B1): needs a working GPU stack. Faithful to Cell 7 except
    the HF token is read from env (never hardcoded) and OOM messages are kept.
    """
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_key in _LOADED_MODELS:
        return _LOADED_MODELS[model_key]
    spec = MODEL_SPECS[model_key]
    log(f"Loading {model_key} from {spec['model_id']}")
    dtype = get_dtype()

    tok_kw = dict(trust_remote_code=True, use_fast=True)
    if HF_TOKEN:
        tok_kw["token"] = HF_TOKEN
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], **tok_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    kw = dict(torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    if HF_TOKEN:
        kw["token"] = HF_TOKEN
    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec["model_id"], attn_implementation="flash_attention_2", **kw)
        log("  Using Flash Attention 2")
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        raise RuntimeError(f"OOM loading {model_key} — unload others or use a larger GPU")
    except Exception:
        try:
            model = AutoModelForCausalLM.from_pretrained(spec["model_id"], **kw)
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            raise RuntimeError(f"OOM loading {model_key} — insufficient VRAM")
    model.eval()

    if spec["arch_family"] == "qwen_sequential_hybrid":
        lt = getattr(model.config, "layer_types", None)
        if lt:
            spec["n_alt_layers"] = sum(1 for t in lt if t in ("linear_attention", "linear"))
            spec["n_attn_layers"] = sum(1 for t in lt if t in ("full_attention", "attention"))
            spec["n_layers"] = spec["n_alt_layers"] + spec["n_attn_layers"]
    cfg = model.config
    spec["_intermediate_size"] = getattr(cfg, "intermediate_size", spec["hidden_dim"] * 4)
    spec["_num_attention_heads"] = getattr(cfg, "num_attention_heads", 16)
    spec["_num_kv_heads"] = getattr(cfg, "num_key_value_heads", spec["_num_attention_heads"])
    spec["_head_dim"] = spec["hidden_dim"] // spec["_num_attention_heads"]
    _LOADED_MODELS[model_key] = (model, tokenizer)
    log(f"  {model_key}: {sum(p.numel() for p in model.parameters()):,} params")
    return model, tokenizer


def unload_model(model_key):
    import gc
    if model_key in _LOADED_MODELS:
        del _LOADED_MODELS[model_key]
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def estimate_flop_ratio(model_key, seq_len=256):
    """c_draft / c_full FLOP ratio (c_param). Ported verbatim from Cell 9."""
    spec = MODEL_SPECS[model_key]
    d = spec["hidden_dim"]; L = spec["n_layers"]
    na = spec["n_alt_layers"]; nat = spec["n_attn_layers"]
    d_ff = spec.get("_intermediate_size", d * 4)
    n_heads = spec.get("_num_attention_heads", 16)
    n_kv = spec.get("_num_kv_heads", n_heads)
    head_d = spec.get("_head_dim", d // n_heads)

    f_qkv = 2 * d * (n_heads * head_d + 2 * n_kv * head_d)
    f_attn_dot = 2 * n_heads * seq_len * head_d
    f_attn_out = 2 * d * d
    f_attn_total = f_qkv + f_attn_dot + f_attn_out
    f_ssm = 8 * d * d
    f_mlp = 3 * 2 * d * d_ff

    if spec["hybrid_type"] == "parallel":
        c_full = L * (f_ssm + f_attn_total + f_mlp)
        c_draft = L * (f_ssm + f_mlp)
    elif spec["hybrid_type"] == "sequential":
        c_full = na * (f_ssm + f_mlp) + nat * (f_attn_total + f_mlp)
        c_draft = na * (f_ssm + f_mlp)
    else:
        c_full = L * (f_attn_total + f_mlp)
        c_draft = c_full * 0.67
    return c_draft / c_full if c_full > 0 else 1.0


def estimate_flops_gflops(model_key, seq_len=256):
    """(draft_gf, full_gf). Ported verbatim from Cell 9 (note f_mlp=6*d*d_ff here)."""
    spec = MODEL_SPECS[model_key]
    d = spec["hidden_dim"]; L = spec["n_layers"]
    na, nat = spec["n_alt_layers"], spec["n_attn_layers"]
    d_ff = spec.get("_intermediate_size", d * 4)
    n_heads = spec.get("_num_attention_heads", 16)
    n_kv = spec.get("_num_kv_heads", n_heads)
    head_d = spec.get("_head_dim", d // n_heads)
    f_attn = 2 * d * (n_heads * head_d + 2 * n_kv * head_d) + 2 * n_heads * seq_len * head_d + 2 * d * d
    f_ssm = 8 * d * d
    f_mlp = 6 * d * d_ff
    if spec["hybrid_type"] == "parallel":
        return L * (f_ssm + f_mlp) / 1e9, L * (f_ssm + f_attn + f_mlp) / 1e9
    elif spec["hybrid_type"] == "sequential":
        return na * (f_ssm + f_mlp) / 1e9, (na * (f_ssm + f_mlp) + nat * (f_attn + f_mlp)) / 1e9
    else:
        c = L * (f_attn + f_mlp) / 1e9
        return c * 0.67, c
