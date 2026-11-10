"""io.py — checkpoint-safe IO, CSV/meta writers, schema validation.

`CheckpointManager` is ported from notebook Cell 5. The CSV writer uses the stdlib
`csv` module (pandas is NOT a dependency — PLAN wants CSV output that works
without it). `write_meta_json` and `validate_schema` are NEW (PLAN constraint 5:
log torch/transformers/CUDA versions, GPU, dtype, commit hash — the notebook wrote
no meta.json).
"""
import os
import re
import csv
import json
import pickle
import subprocess
from datetime import datetime, timezone


class CheckpointManager:
    """Pickle-based checkpoint store with a run log (Cell 5)."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.ckpt_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.log_file = os.path.join(base_dir, "logs", "experiment.log")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def _path(self, name):
        safe = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
        return os.path.join(self.ckpt_dir, f"{safe}.pkl")

    def save(self, name, data):
        with open(self._path(name), "wb") as f:
            pickle.dump(data, f)
        self.log(f"Saved: {name}")

    def load(self, name):
        p = self._path(name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def exists(self, name):
        return os.path.exists(self._path(name))

    def list_checkpoints(self):
        return [f[:-4] for f in os.listdir(self.ckpt_dir) if f.endswith(".pkl")]

    def log(self, msg):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")


def checkpoint_skip(ckpt, name, key=None):
    """Return cached result if present (and key non-None), else None (Cell 20)."""
    cached = ckpt.load(name)
    if cached is None:
        return None
    if key is not None and cached.get(key) is None:
        return None
    return cached


def write_csv(rows, path, fieldnames=None):
    """Write a list-of-dicts to CSV with the stdlib csv module (no pandas)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        open(path, "w").close()
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


def save_dataframe(rows_or_df, path, index=False):
    """Write CSV from list-of-dicts OR a pandas DataFrame (pandas optional)."""
    if hasattr(rows_or_df, "to_csv"):
        rows_or_df.to_csv(path, index=index)
        return path
    return write_csv(rows_or_df, path)


def gpu_tag():
    """Sanitised current-GPU name for the `gpu` provenance column (e.g. 'NVIDIA_L4',
    'NVIDIA_RTX_PRO_4500_Blackwell'). 'cpu' if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            return re.sub(r"[^A-Za-z0-9]+", "_", torch.cuda.get_device_name(0)).strip("_")
        return "cpu"
    except Exception:
        return "unknown"


def load_other_gpu_rows(path, current_gpu, default_gpu):
    """Read existing CSV rows that belong to a DIFFERENT GPU than `current_gpu`,
    so a re-run on a new GPU ACCUMULATES rows instead of clobbering the others
    (PLAN: keep both RTX-4500 and L4 numbers). Rows missing a 'gpu' column are
    backfilled with `default_gpu` (the GPU that produced the pre-L4 results).
    """
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            g = (r.get("gpu") or "").strip() or default_gpu
            r["gpu"] = g
            if g != current_gpu:
                out.append(r)
    return out


def save_figure(fig, figures_dir, name, dpi=300):
    os.makedirs(figures_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figures_dir, f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")


def _git_commit(cwd=None):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def write_meta_json(path, cwd=None, **fields):
    """Capture provenance (PLAN constraint 5): versions, GPU, dtype, commit, ts.

    Any extra keyword fields are merged in. Degrades gracefully if torch/
    transformers are missing or CUDA is broken.
    """
    meta = {"timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(cwd)}
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["cuda_version"] = torch.version.cuda
        try:
            meta["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception as e:
            meta["gpu_name"] = f"UNAVAILABLE: {type(e).__name__}"
    except Exception:
        meta["torch_version"] = None
    try:
        import transformers
        meta["transformers_version"] = transformers.__version__
    except Exception:
        meta["transformers_version"] = None
    meta.update(fields)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def validate_schema(rows, required_cols):
    """Assert every row has exactly/at least the required columns. Raises on miss."""
    missing_any = []
    req = set(required_cols)
    for i, r in enumerate(rows):
        missing = req - set(r.keys())
        if missing:
            missing_any.append((i, sorted(missing)))
    if missing_any:
        raise ValueError(f"schema validation failed for rows: {missing_any[:5]}"
                         f" (required={sorted(req)})")
    return True
