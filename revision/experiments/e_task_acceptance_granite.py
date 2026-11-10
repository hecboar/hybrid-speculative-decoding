#!/usr/bin/env python3
"""Task-suite acceptance for granite-4.0-h-micro (B-10; extends v1 Table 5).

Runs the SAME task-suite acceptance protocol/estimator as the frozen v1
`results/task_acceptance.csv` (MMLU / GSM8K / Alpaca, greedy T=0, all-token
acceptance A_k with bootstrap CI + mean accepted length tau_bar), but for the new
viable SEQUENTIAL model granite-4.0-h-micro with the canonical component_only
(= linear-only for a sequential hybrid) drafter. Reuses the task-acceptance infra
verbatim: `data.get_task_prompts` (task loader) + `specloop.simulate_acceptance`
(v1 estimator, raw forwards, use_cache=False) + `stats.bootstrap_ci`.

Acceptance is GPU-agnostic (blockers.md B1): nvidia-smi is logged to meta.json but
there is NO L4 hard-fail. Only the DRAFT MODE / prompt SOURCE differ from E1; the
estimator is unchanged, so these rows drop straight into Table 5.

Output (NEW file; frozen task_acceptance.csv untouched):
  results/task_acceptance_granite.csv   (v1 schema, so it extends Table 5 1:1)
    model,task,acceptance_rate,ci_lo,ci_hi,mean_accepted_length
  results/task_acceptance_granite.meta.json
  results/E1/perprompt/granite__task_<task>__component_only__k<k>__T0.0.pkl
"""
import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))   # import sibling driver (register_from_manifest)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from specdiag import models as M  # noqa: E402
from specdiag.models import load_model_and_tokenizer  # noqa: E402
from specdiag.controller import DraftModeController  # noqa: E402
from specdiag.specloop import simulate_acceptance  # noqa: E402
from specdiag.data import get_task_prompts  # noqa: E402
from specdiag.io import write_csv, write_meta_json, gpu_tag  # noqa: E402
from e1_acceptance import register_from_manifest  # noqa: E402  (reuse E1 registration)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PP = os.path.join(REPO, "results", "E1", "perprompt")
OUT_CSV = os.path.join(REPO, "results", "task_acceptance_granite.csv")
OUT_META = os.path.join(REPO, "results", "task_acceptance_granite.meta.json")
MODEL = "granite"
STRAT = "component_only"     # sequential hybrid -> linear-only drafter (attn = identity)
TASKS = ["mmlu", "gsm8k", "alpaca"]
# v1 Table-5 schema (frozen task_acceptance.csv) — matched exactly for drop-in rows
FIELDS = ["model", "task", "acceptance_rate", "ci_lo", "ci_hi", "mean_accepted_length"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--tasks", default=",".join(TASKS))
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    GPU = gpu_tag()
    print(f"Task-suite acceptance (granite) | GPU={torch.cuda.get_device_name(0)} "
          f"| GPU-agnostic (no L4 hard-fail) | strategy={STRAT} k={args.k} "
          f"T={args.temperature} n={args.n}")

    spec = register_from_manifest(MODEL)
    paradigm = spec["hybrid_type"]
    model, tok = load_model_and_tokenizer(MODEL)
    ctrl = DraftModeController(model, MODEL)
    os.makedirs(PP, exist_ok=True)

    rows = []
    for task in tasks:
        prompts = get_task_prompts(task, args.n)
        res = simulate_acceptance(model, tok, ctrl, STRAT, args.k,
                                  temperature=args.temperature, prompts=prompts,
                                  n_trials=len(prompts), max_len=args.max_len)
        ppfile = os.path.join(
            PP, f"{MODEL}__task_{task}__{STRAT}__k{args.k}__T{args.temperature}.pkl")
        with open(ppfile, "wb") as f:
            pickle.dump({"model": MODEL, "task": task, "strategy": STRAT,
                         "k": args.k, "temperature": args.temperature,
                         "accepted_lengths": res["accepted_lengths"]}, f)
        rows.append({"model": MODEL, "task": task,
                     "acceptance_rate": round(res["acceptance_rate"], 5),
                     "ci_lo": round(res["acceptance_rate_ci_lo"], 5),
                     "ci_hi": round(res["acceptance_rate_ci_hi"], 5),
                     "mean_accepted_length": round(res["mean_accepted_length"], 4)})
        na = np.asarray(res["accepted_lengths"])
        print(f"  {task:6s} n={len(prompts)} A_{args.k}={res['acceptance_rate']:.4f} "
              f"CI=[{res['acceptance_rate_ci_lo']:.4f},{res['acceptance_rate_ci_hi']:.4f}] "
              f"tau_bar={res['mean_accepted_length']:.4f} (all_accept={np.mean(na==args.k):.4f})")
        write_csv(rows, OUT_CSV, fieldnames=FIELDS)

    write_meta_json(OUT_META, cwd=REPO, experiment="B-10_task_acceptance_granite",
                    model=MODEL, model_id=spec["model_id"], arch_paradigm=paradigm,
                    strategy=STRAT, k=args.k, temperature=args.temperature,
                    n_prompts=args.n, tasks=tasks, gpu=GPU, gpu_is_l4=("L4" in GPU),
                    metric="A_k := E[na]/k (v1 mean-fraction, bootstrap CI); "
                    "same estimator as frozen task_acceptance.csv",
                    metric_is_gpu_agnostic=True,
                    note="granite-4.0-h-micro (sequential viable model) on task suites; "
                    "component_only = linear-only drafter; extends v1 Table 5")
    M.unload_model(MODEL)
    print("wrote:", OUT_CSV)


if __name__ == "__main__":
    main()
