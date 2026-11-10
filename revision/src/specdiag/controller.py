"""controller.py - DraftModeController.

Persistent, flag-based mode switching by forward-monkeypatching decoder layers /
attention submodules. NO hook registration/removal inside the decoding loop
(that was a major v1 overhead source). `set_mode(...)` restores any previous
patches (O(L)) then installs the new ones (O(L)); deactivation restores originals.

Superset of the notebook's `DraftModelManager` (Cell 9 + Cell 10 HOTFIX, which is
the LIVE version and is what is ported here). The notebook only implemented
ssm_only / linear_only / layer_skip / early_exit; this class adds the full PLAN
mode set: full, component_only, component_only_partial(frac, policy),
attention_only, layerskip(frac), early_exit(frac), attn_subset(keep_ids).

Legacy aliases preserved so existing call sites keep working:
  ssm_only   -> component_only (parallel)
  linear_only-> component_only (sequential)
  layer_skip -> layerskip(0.33)
  early_exit -> early_exit(0.5)

torch is imported lazily, so the module imports on a CPU-only host.
"""
from .models import MODEL_SPECS, get_decoder_layers


# ----------------------------- patch helpers (LIVE Cell-10 hotfix versions) ----

def normalize_qwen_layer_type(lt):
    return {"linear_attention": "linear", "full_attention": "attention",
            "linear": "linear", "attention": "attention"}.get(lt, lt)


def _find_attention_submodule(layer):
    """(module, name) of the attention submodule inside a decoder layer (Cell 9)."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn, "self_attn"
    for name, mod in layer.named_children():
        cn = mod.__class__.__name__.lower()
        if "attention" in cn or "attn" in cn:
            return mod, name
    return None, None


def _find_ssm_submodule(layer):
    """(module, name) of the SSM/Mamba/mixer submodule (for attention_only).

    Submodule naming varies across Falcon-H1 releases (mamba / mixer / ssm), so
    children are scanned by class and attribute name. A release that uses another
    name needs it added to the tuple below.
    """
    for attr in ("mamba", "ssm", "mixer"):
        if hasattr(layer, attr):
            return getattr(layer, attr), attr
    for name, mod in layer.named_children():
        cn = mod.__class__.__name__.lower()
        if "mamba" in cn or "ssm" in cn or "mixer" in cn:
            return mod, name
    return None, None


def _make_zero_attn_forward(attn_mod):
    """Attention submodule returns zeros BEFORE computing QKV/attention.

    LIVE Cell-10 hotfix: Falcon self_attn expects exactly 2 return values.
    Never computes QKV — returns (zeros_like(h), None) immediately, so the
    measured cost of the zeroed branch is genuinely ~0: it short-circuits BEFORE
    computing attention, rather than computing and then zeroing.
    """
    import torch
    orig_forward = attn_mod.forward

    def skip_forward(*args, **kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is None:
            for a in args:
                if torch.is_tensor(a) and a.dim() >= 2:
                    h = a; break
        zero_out = torch.zeros_like(h)
        return (zero_out, None)

    skip_forward._is_draft_skip = True
    return skip_forward, orig_forward


def _make_identity_layer_forward(layer):
    """Whole layer becomes identity on the residual stream.

    LIVE Cell-10 hotfix: Qwen3.5 layers return a bare tensor when use_cache=False,
    else (h, None).
    """
    import torch
    orig_forward = layer.forward

    def identity_forward(*args, **kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is None:
            for a in args:
                if torch.is_tensor(a) and a.dim() >= 2:
                    h = a; break
        use_cache = kwargs.get("use_cache", False)
        if use_cache:
            return (h, None)
        return h

    identity_forward._is_draft_skip = True
    return identity_forward, orig_forward


def _make_zero_bare_forward(mod):
    """Submodule returns a BARE zeros_like(hidden_states) (no tuple wrap).

    For Zamba2's shared attention block (Zamba2AttentionDecoderLayer), whose real
    forward returns a bare tensor that the HybridLayer immediately feeds into a
    Linear. Short-circuits BEFORE computing attention (PLAN 3.1 cost rule).
    """
    import torch
    orig_forward = mod.forward

    def zero_forward(*args, **kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is None:
            for a in args:
                if torch.is_tensor(a) and a.dim() >= 2:
                    h = a; break
        return torch.zeros_like(h)

    zero_forward._is_draft_skip = True
    return zero_forward, orig_forward


def _make_zero_linear_forward(lin):
    """Linear returns zeros of shape (*input.shape[:-1], out_features).

    Zeros the attention CONTRIBUTION that Zamba2 adds into the mamba stream
    (transformer_hidden_states), so the drafter is a clean mamba-only path (not
    mamba + linear.bias). Cheap: no matmul, just an allocation.
    """
    import torch
    out_features = getattr(lin, "out_features", None)
    orig_forward = lin.forward

    def zero_forward(*args, **kwargs):
        x = args[0] if args else kwargs.get("input")
        of = out_features if out_features is not None else x.shape[-1]
        return torch.zeros(*x.shape[:-1], of, dtype=x.dtype, device=x.device)

    zero_forward._is_draft_skip = True
    return zero_forward, orig_forward


def _make_identity_layer_forward_tuple(layer):
    """Whole layer becomes identity, returning a (h, None) TUPLE.

    Falcon-H1's decoder loop reads `layer_outputs[0]` (unlike Qwen, which reads a
    bare tensor), so an identitied Falcon block MUST return a tuple or the loop
    slices the hidden_states' batch dim (`layer_outputs[0]` on a 3-D tensor ->
    2-D) and the next mamba mixer crashes on `batch,seq,_ = x.shape`. Used for
    component_only_partial / layerskip / early_exit on parallel (Falcon) models.
    """
    import torch
    orig_forward = layer.forward

    def identity_forward(*args, **kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is None:
            for a in args:
                if torch.is_tensor(a) and a.dim() >= 2:
                    h = a; break
        return (h, None)

    identity_forward._is_draft_skip = True
    return identity_forward, orig_forward


def _make_zero_submodule_forward(mod):
    """Submodule returns zeros_like(hidden_states) (for attention_only SSM-zeroing).

    The return arity of the SSM/mixer submodule varies across releases, so a bare
    zeros tensor is returned, falling back to a 2-tuple when the caller unpacks.
    """
    import torch
    orig_forward = mod.forward

    def zero_forward(*args, **kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        if h is None:
            for a in args:
                if torch.is_tensor(a) and a.dim() >= 2:
                    h = a; break
        return torch.zeros_like(h)

    zero_forward._is_draft_skip = True
    return zero_forward, orig_forward


# ------------------------------- layer-selection policies -----------------------

def _kept_layer_ids(n, frac, policy):
    """Indices of layers to KEEP for component_only_partial / layerskip.

    Always keeps layer 0 and layer n-1. frac in (0,1].
      uniform_stride         : evenly spaced kept indices.
      keep_first_last_stride : keep 0 and n-1 explicitly, fill the middle by stride.
    """
    n_keep = max(2, int(round(frac * n)))
    n_keep = min(n_keep, n)
    if policy == "uniform_stride":
        import numpy as np
        idx = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, n_keep)))
    elif policy == "keep_first_last_stride":
        import numpy as np
        mids = [int(round(x)) for x in np.linspace(1, n - 2, max(0, n_keep - 2))] if n_keep > 2 else []
        idx = sorted(set([0, n - 1] + mids))
    else:
        raise ValueError(f"unknown policy {policy!r}")
    idx = set(idx)
    idx.add(0); idx.add(n - 1)
    return sorted(idx)


class DraftModeController:
    """Flag-based draft-mode controller. Use `set_mode(...)` / `deactivate()`."""

    LEGACY = {"ssm_only", "linear_only", "layer_skip", "early_exit"}

    def __init__(self, model, model_key):
        self.model = model
        self.model_key = model_key
        self.spec = MODEL_SPECS[model_key]
        self.patched = {}          # key -> {"module", "original"}
        self.mode = "full"
        self._params = {}

    # ---- lifecycle -----------------------------------------------------------
    def deactivate(self):
        for _, item in list(self.patched.items()):
            item["module"].forward = item["original"]
        self.patched.clear()
        self.mode = "full"
        self._params = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.deactivate()

    def _patch(self, key, module, maker):
        fwd, orig = maker(module)
        module.forward = fwd
        self.patched[key] = {"module": module, "original": orig}

    def _identity_maker(self):
        """Arch-aware whole-layer identity: Falcon's decoder loop reads
        layer_outputs[0] (needs a tuple); Qwen reads a bare tensor."""
        if self.spec["arch_family"] == "falcon_parallel_hybrid":
            return _make_identity_layer_forward_tuple
        return _make_identity_layer_forward

    # ---- dispatch ------------------------------------------------------------
    def set_mode(self, mode, frac=None, policy="uniform_stride", keep_ids=None):
        """Install `mode`. Restores previous patches first (O(L))."""
        self.deactivate()
        arch = self.spec["arch_family"]

        # legacy aliases
        if mode == "ssm_only":
            mode = "component_only"
        elif mode == "linear_only":
            mode = "component_only"
        elif mode == "layer_skip":
            mode, frac = "layerskip", (frac or 0.33)
        elif mode == "early_exit" and frac is None:
            frac = 0.5

        if mode == "full":
            pass
        elif mode == "component_only":
            self._component_only()
        elif mode == "component_only_partial":
            self._component_only_partial(frac if frac is not None else 0.5, policy)
        elif mode == "attention_only":
            self._attention_only()
        elif mode == "layerskip":
            self._layerskip(frac if frac is not None else 0.33)
        elif mode == "early_exit":
            self._early_exit(frac if frac is not None else 0.5)
        elif mode == "attn_subset":
            self._attn_subset(keep_ids if keep_ids is not None else [])
        else:
            raise ValueError(f"unknown mode {mode!r} for arch {arch}")

        self.mode = mode
        self._params = {"frac": frac, "policy": policy, "keep_ids": keep_ids}
        return self

    # ---- back-compat with notebook DraftModelManager -------------------------
    def activate_draft_mode(self, strategy="ssm_only"):
        return self.set_mode(strategy)

    def deactivate_draft_mode(self):
        self.deactivate()

    @property
    def active(self):
        return self.mode != "full"

    # ---- modes ---------------------------------------------------------------
    def _component_only(self):
        """Parallel: zero the attention branch of every layer (=ssm_only).
        Sequential: identity the attention-type layers (=linear_only)."""
        layers, _ = get_decoder_layers(self.model)
        arch = self.spec["arch_family"]
        if arch == "falcon_parallel_hybrid":
            for idx, layer in enumerate(layers):
                attn_mod, _ = _find_attention_submodule(layer)
                if attn_mod is None:
                    continue
                self._patch(f"comp_attn_{idx}", attn_mod, _make_zero_attn_forward)
        elif arch == "qwen_sequential_hybrid":
            lt_raw = getattr(self.model.config, "layer_types", None)
            if lt_raw is None:
                raise ValueError("Missing layer_types in config")
            lt = [normalize_qwen_layer_type(t) for t in lt_raw]
            for idx, layer in enumerate(layers):
                if idx < len(lt) and lt[idx] == "attention":
                    self._patch(f"comp_lyr_{idx}", layer, _make_identity_layer_forward)
        elif arch == "zamba2_shared_hybrid":
            # Zamba2 (parallel-style): each HybridLayer adds a shared-attention
            # branch (shared_transformer -> linear) into the mamba stream. Zero
            # BOTH: shared_transformer returns bare zeros (skips attention compute,
            # cost-correct) and its Linear returns zeros (clean mamba-only value).
            n_hit = 0
            for idx, layer in enumerate(layers):
                st = getattr(layer, "shared_transformer", None)
                lin = getattr(layer, "linear", None)
                if st is not None:
                    self._patch(f"z2_st_{idx}", st, _make_zero_bare_forward)
                    n_hit += 1
                if lin is not None:
                    self._patch(f"z2_lin_{idx}", lin, _make_zero_linear_forward)
            if n_hit == 0:
                raise ValueError("zamba2_shared_hybrid: no HybridLayer.shared_transformer found")
        else:
            # transformer baseline has no separable component -> fall back to layerskip
            self._layerskip(0.33)

    def _component_only_partial(self, frac, policy):
        """Parallel only: component_only on KEPT layers + identity on dropped blocks.
        Keeps `frac` of layers (always layer 0 and L-1)."""
        arch = self.spec["arch_family"]
        if arch != "falcon_parallel_hybrid":
            raise ValueError("component_only_partial is parallel-only")
        layers, _ = get_decoder_layers(self.model)
        n = len(layers)
        keep = set(_kept_layer_ids(n, frac, policy))
        id_maker = self._identity_maker()
        for idx, layer in enumerate(layers):
            if idx in keep:
                attn_mod, _ = _find_attention_submodule(layer)
                if attn_mod is not None:
                    self._patch(f"cop_attn_{idx}", attn_mod, _make_zero_attn_forward)
            else:
                self._patch(f"cop_skip_{idx}", layer, id_maker)

    def _attention_only(self):
        """Parallel only: zero the SSM branch (diagnostic). NEW — not in notebook."""
        arch = self.spec["arch_family"]
        if arch != "falcon_parallel_hybrid":
            raise ValueError("attention_only is parallel-only")
        layers, _ = get_decoder_layers(self.model)
        for idx, layer in enumerate(layers):
            ssm_mod, _ = _find_ssm_submodule(layer)
            if ssm_mod is not None:
                self._patch(f"attn_only_{idx}", ssm_mod, _make_zero_submodule_forward)
        # If no SSM submodule was found nothing is patched, and the draft falls
        # back to the unmodified forward.

    def _layerskip(self, frac):
        """Skip a contiguous middle block of `frac` layers (Cell 9 _layer_skip)."""
        layers, _ = get_decoder_layers(self.model)
        n = len(layers)
        ns = max(1, int(n * frac))
        mid = n // 2
        skip_start = mid - ns // 2
        id_maker = self._identity_maker()
        for idx in range(skip_start, min(skip_start + ns, n)):
            self._patch(f"skip_{idx}", layers[idx], id_maker)

    def _early_exit(self, frac):
        """Keep first `frac` layers, identity the rest (+ shared norm/LM head)."""
        layers, _ = get_decoder_layers(self.model)
        n = len(layers)
        el = max(1, int(n * frac))
        id_maker = self._identity_maker()
        for idx in range(el, n):
            self._patch(f"ee_{idx}", layers[idx], id_maker)

    def _attn_subset(self, keep_ids):
        """Sequential only: suppress all attention layers EXCEPT keep_ids (E2)."""
        arch = self.spec["arch_family"]
        if arch != "qwen_sequential_hybrid":
            raise ValueError("attn_subset is sequential-only")
        layers, _ = get_decoder_layers(self.model)
        lt_raw = getattr(self.model.config, "layer_types", None)
        if lt_raw is None:
            raise ValueError("Missing layer_types in config")
        lt = [normalize_qwen_layer_type(t) for t in lt_raw]
        keep_ids = set(keep_ids)
        for idx, layer in enumerate(layers):
            if idx < len(lt) and lt[idx] == "attention" and idx not in keep_ids:
                self._patch(f"as_{idx}", layer, _make_identity_layer_forward)


# Back-compat name
DraftModelManager = DraftModeController
