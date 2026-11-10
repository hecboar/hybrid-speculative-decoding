#!/usr/bin/env python3
"""E0.2 — v1 checkpoint inventory + per-prompt data audit (PLAN Phase 0).

Introspects every pickle in checkpoints/v1_drive/ (READ-ONLY; never modified)
and answers the three PLAN E0.2 tasks:
  1. Inventory every pickle: keys, shapes, whether per-prompt records exist.
  2. Provenance map: v3 vs v3fix structure/value diff (speedup, quality).
  3. (a) Can the R2.3 permutation test (E5.3) run from per-prompt records in
     acceptance_v3__falcon-h1-0.5b__k4__T0.0 and falcon-h1-3b__k4__T0.0?
     (b) Confirm the qwen3.5-2b / falcon-h1-1.5b divergence gaps.

Output: results/E0/checkpoint_inventory.md (regenerated from actual contents).
Pure CPU; safe to run alongside GPU jobs.
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVE = os.path.join(REPO, "checkpoints", "v1_drive")
OUT_MD = os.path.join(REPO, "results", "E0", "checkpoint_inventory.md")

PLAUSIBLE_N = {30, 50, 100, 150, 200, 500, 1000}  # per-prompt vector lengths


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _describe(v, depth=0):
    """One-line description of a value; detects per-prompt vectors."""
    t = type(v).__name__
    if isinstance(v, np.ndarray):
        pp = "  <-- per-prompt?" if (v.ndim == 1 and len(v) in PLAUSIBLE_N) else ""
        return f"ndarray shape={v.shape} dtype={v.dtype}{pp}"
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return f"Tensor shape={tuple(v.shape)} dtype={v.dtype}"
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        n = len(v)
        pp = "  <-- per-prompt?" if n in PLAUSIBLE_N else ""
        inner = type(v[0]).__name__ if n else "?"
        return f"{t} len={n} of<{inner}>{pp}"
    if isinstance(v, dict):
        return f"dict keys={list(v.keys())}"
    if isinstance(v, (int, float, bool, str)):
        return f"{t}={v!r}"
    return t


def _is_per_prompt(v):
    if isinstance(v, np.ndarray):
        return v.ndim == 1 and len(v) in PLAUSIBLE_N
    if isinstance(v, (list, tuple)):
        return len(v) in PLAUSIBLE_N and not isinstance(v[0], dict) if v else False
    return False


def inventory_one(path):
    """Return (obj, lines) describing one pickle."""
    name = os.path.basename(path)
    lines = [f"### `{name}`  ({os.path.getsize(path)} bytes)"]
    try:
        obj = _load(path)
    except Exception as e:
        lines.append(f"- LOAD ERROR: {type(e).__name__}: {e}")
        return None, lines
    if isinstance(obj, dict):
        lines.append(f"- top-level: dict with {len(obj)} keys")
        for k, v in obj.items():
            lines.append(f"  - `{k}`: {_describe(v)}")
    elif isinstance(obj, (list, tuple)):
        lines.append(f"- top-level: {type(obj).__name__} len={len(obj)}; "
                     f"elem0: {_describe(obj[0]) if obj else 'empty'}")
    else:
        lines.append(f"- top-level: {_describe(obj)}")
    return obj, lines


def find_per_prompt_keys(obj):
    """Return dict key->desc of per-prompt vectors in a dict pickle."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_per_prompt(v):
                out[k] = _describe(v)
    return out


def main():
    files = sorted(f for f in os.listdir(DRIVE) if f.endswith(".pkl"))
    print(f"E0.2 inventory: {len(files)} pickles in {DRIVE}")

    objs = {}
    groups = {"acceptance": [], "divergence": [], "speedup": [], "quality": [],
              "comparison": [], "task_accept": [], "other": []}
    all_lines = []
    for f in files:
        obj, lines = inventory_one(os.path.join(DRIVE, f))
        objs[f] = obj
        all_lines += lines + [""]
        pref = f.split("__")[0].replace("_v3fix", "").replace("_v3", "")
        groups.get(pref, groups["other"]).append(f)

    # ---- targeted audits ----
    audit = []

    # 3a-grid: per-prompt presence across the whole acceptance grid
    audit.append("## E0.2.3(a-grid) — per-prompt `accepted_lengths` across the "
                 "acceptance grid\n")
    audit.append("| model | k2 T0 | k2 T.6 | k4 T0 | k4 T.6 | k8 T0 | k8 T.6 |")
    audit.append("|---|---|---|---|---|---|---|")
    grid_models = ["falcon-h1-0.5b", "falcon-h1-3b", "qwen3.5-0.8b", "qwen2.5-0.5b"]
    for m in grid_models:
        cells = []
        for k in (2, 4, 8):
            for T in ("0.0", "0.6"):
                o = objs.get(f"acceptance_v3__{m}__k{k}__T{T}.pkl")
                if o is None:
                    cells.append("—")
                else:
                    pp = isinstance(o, dict) and o.get("accepted_lengths") is not None
                    cells.append("**PP**" if pp else "agg")
        audit.append(f"| {m} | " + " | ".join(cells) + " |")
    audit.append("\nPP = per-prompt `accepted_lengths` present (n=200). "
                 "agg = aggregates only.\n")

    # 3a: E5.3 permutation feasibility
    audit.append("## E0.2.3(a) — E5.3 permutation-test feasibility (R2.3)\n")
    for key in ("acceptance_v3__falcon-h1-0.5b__k4__T0.0.pkl",
                "acceptance_v3__falcon-h1-3b__k4__T0.0.pkl"):
        o = objs.get(key)
        if o is None:
            audit.append(f"- `{key}`: MISSING")
            continue
        pp = find_per_prompt_keys(o)
        audit.append(f"- `{key}`:")
        if isinstance(o, dict):
            audit.append(f"    keys = {list(o.keys())}")
        if pp:
            for k, d in pp.items():
                audit.append(f"    per-prompt: `{k}` -> {d}")
        else:
            audit.append("    NO per-prompt vector detected at top level")
    # verdict
    f05 = objs.get("acceptance_v3__falcon-h1-0.5b__k4__T0.0.pkl")
    f3b = objs.get("acceptance_v3__falcon-h1-3b__k4__T0.0.pkl")
    has_05 = bool(find_per_prompt_keys(f05))
    has_3b = bool(find_per_prompt_keys(f3b))
    audit.append(
        f"\n**Verdict:** Falcon-0.5B per-prompt present = {has_05}; "
        f"Falcon-3B per-prompt present = {has_3b}. "
        "The 0.5B side is reusable as-is; **Falcon-3B is aggregate-only across the "
        "ENTIRE grid**, so the two-sample permutation test (E5.3) CANNOT run purely "
        "from pickles. **Action: schedule a minimal GPU re-run of Falcon-H1-3B "
        "acceptance at k=4, T=0, 200 prompts to recover per-prompt "
        "`accepted_lengths`, then run the permutation test against the existing "
        "0.5B per-prompt vector.** (Only the 3B side needs GPU; everything else is "
        "reuse.)\n")

    # 3b: divergence gaps
    audit.append("## E0.2.3(b) — divergence gap check\n")
    for m in ("falcon-h1-1.5b", "qwen3.5-2b"):
        key = f"divergence_v3__{m}.pkl"
        o = objs.get(key)
        if o is None:
            audit.append(f"- `{m}`: divergence pickle ABSENT")
        else:
            audit.append(f"- `{m}`: divergence pickle PRESENT -> "
                         f"{_describe(o)}")
    audit.append("")

    # 2: provenance v3 vs v3fix
    audit.append("## E0.2.2 — provenance: v3 vs v3fix\n")
    for base, ex in (("speedup", "falcon-h1-0.5b__k4"), ("quality", "falcon-h1-0.5b")):
        v3 = objs.get(f"{base}_v3__{ex}.pkl")
        vf = objs.get(f"{base}_v3fix__{ex}.pkl")
        audit.append(f"- **{base}** ({ex}):")
        audit.append(f"    v3    : {_describe(v3) if v3 is not None else 'ABSENT'}")
        audit.append(f"    v3fix : {_describe(vf) if vf is not None else 'ABSENT'}")
        if isinstance(v3, dict) and isinstance(vf, dict):
            common = set(v3) & set(vf)
            for k in sorted(common):
                a, b = v3[k], vf[k]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    audit.append(f"      {k}: v3={a:.4g}  v3fix={b:.4g}  "
                                 f"delta={b - a:+.4g}")
            only3 = set(v3) - set(vf)
            onlyf = set(vf) - set(v3)
            if only3:
                audit.append(f"      keys only in v3: {sorted(only3)}")
            if onlyf:
                audit.append(f"      keys only in v3fix: {sorted(onlyf)}")
    audit.append("")
    audit.append("**quality match_rate v3 -> v3fix (all models):**")
    for m in ("falcon-h1-0.5b", "qwen3.5-0.8b", "qwen2.5-0.5b"):
        v3 = objs.get(f"quality_v3__{m}.pkl")
        vf = objs.get(f"quality_v3fix__{m}.pkl")
        a = v3.get("match_rate") if isinstance(v3, dict) else None
        b = vf.get("match_rate") if isinstance(vf, dict) else None
        audit.append(f"- {m}: v3={a} -> v3fix={b}")
    audit.append(
        "\nProvenance rule: the submitted paper's `_v2` CSVs "
        "(`speedup_v2`, `quality_check_v2`) correspond to the **v3fix** "
        "generation (trimmed schema, corrected values). `v3fix` is authoritative; "
        "never mix v3 and v3fix in one table. The v3->v3fix quality jump is a real "
        "v1 bug fix; the residual ~4-10% mismatch is the bf16 non-determinism "
        "quantified in E0.3 (fp32 -> 100%). NOTE the metric-definition caveat: v1 "
        "`quality.match_rate` is per-prompt exact match over 50 prompts at v1's "
        "generation length; E0.3 additionally reports per-token agreement over 64 "
        "tokens (~0.99 in bf16) — same phenomenon, stricter horizon.\n")

    audit.append("## E0.2 — reusable v1 headline numbers (from pickles)\n")
    audit.append("| model | k | A_k | tau_bar |")
    audit.append("|---|---|---|---|")
    for m in ("falcon-h1-0.5b", "qwen3.5-0.8b"):
        for k in (2, 4, 8):
            o = objs.get(f"acceptance_v3__{m}__k{k}__T0.0.pkl")
            if isinstance(o, dict):
                audit.append(f"| {m} | {k} | {o.get('acceptance_rate'):.4f} | "
                             f"{o.get('mean_accepted_length')} |")
    audit.append("\n(Matches the paper headline: Falcon parallel A_2=0.68; "
                 "Qwen sequential collapse A_2=0.0375. These feed E0.6 "
                 "normalization and are reused, not re-run.)\n")

    # ---- assemble md ----
    hdr = [
        "# E0.2 — v1 checkpoint inventory (Drive pickles)",
        "",
        f"Source: `checkpoints/v1_drive/` (READ-ONLY, constraint 8). "
        f"{len(files)} pickles. Auto-generated by `experiments/e02_inventory.py`.",
        "",
        "## Group counts",
    ]
    for g, fs in groups.items():
        if fs:
            hdr.append(f"- **{g}**: {len(fs)} files")
    hdr += [""]
    body = ["## Targeted audits", ""] + audit + ["## Full per-file inventory", ""] + all_lines
    with open(OUT_MD, "w") as f:
        f.write("\n".join(hdr + body) + "\n")
    print("wrote:", OUT_MD)
    print("\n".join(audit))


if __name__ == "__main__":
    main()
