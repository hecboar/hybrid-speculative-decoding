#!/usr/bin/env python3
"""CPU-only pure-path tests for specdiag (no torch/GPU, no scipy required).

Run: python tests/test_pure.py   (exits nonzero on any failure)
Exercises the math/IO paths that must be correct regardless of blocker B1.
"""
import os
import sys
import math
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from specdiag import stats, metrics, io, data, specloop  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# 1. bootstrap_ci on a known array
m, lo, hi = stats.bootstrap_ci(np.arange(1, 101), n_boot=2000, seed=0)
check("bootstrap_ci mean ~= 50.5", abs(m - 50.5) < 1e-9, f"m={m}")
check("bootstrap_ci lo < mean < hi", lo < m < hi, f"{lo},{m},{hi}")

# 2. permutation_test symmetry / sanity
obs, p_same = stats.permutation_test([1, 1, 1, 1], [1, 1, 1, 1], n=500)
check("permutation_test identical -> obs 0, p==1", obs == 0.0 and p_same == 1.0,
      f"obs={obs} p={p_same}")
obs2, p_diff = stats.permutation_test([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], n=2000, seed=1)
check("permutation_test disjoint -> small p", p_diff < 0.05, f"p={p_diff}")

# 3. dtv on toy distributions (hand-checkable)
p = np.array([0.5, 0.5, 0.0, 0.0])
q = np.array([0.0, 0.5, 0.5, 0.0])
# 0.5*(|0.5-0|+|0.5-0.5|+|0-0.5|+0) = 0.5*(0.5+0+0.5) = 0.5
check("dtv_full toy == 0.5", abs(float(metrics.dtv_full(p, q)) - 0.5) < 1e-12,
      f"{metrics.dtv_full(p, q)}")
check("dtv_full identical == 0", abs(float(metrics.dtv_full(p, p))) < 1e-12)

# dtv_top100 on 2 logits with topk=2 (union of top-2 covers all mass)
lf = np.log(np.array([0.5, 0.5, 1e-9, 1e-9]))
ld = np.log(np.array([1e-9, 0.5, 0.5, 1e-9]))
d100 = metrics.dtv_top100(lf, ld, topk=2)
check("dtv_top100 ~ 0.5 on toy", abs(d100 - 0.5) < 1e-3, f"{d100}")

# 4. acceptance_rate + tau_bar from toy accepted_lengths
lengths = [2, 1, 0, 2]  # k=2 -> na/k = [1,0.5,0,1] mean 0.625 ; tau=1.25
check("acceptance_rate_from_lengths", abs(metrics.acceptance_rate_from_lengths(lengths, 2) - 0.625) < 1e-12)
check("tau_bar", abs(metrics.tau_bar(lengths) - 1.25) < 1e-12)
# S_theory reproduces the Falcon k=2 value
check("s_theory falcon k2 == 0.918779",
      abs(metrics.s_theory(1.36, 2, 0.7843137254901961) - 0.9187786259541987) < 1e-9)

# 5. accept_reject greedy: full argmax matches first token only
full_logits = np.array([[0.0, 5.0, 0.0],   # argmax=1
                        [9.0, 0.0, 0.0],    # argmax=0
                        [0.0, 0.0, 9.0]])   # argmax=2
na = specloop.accept_reject([1, 2, 2], [1.0, 1.0, 1.0], full_logits, temperature=0.0)
check("accept_reject greedy stops at first mismatch", na == 1, f"na={na}")

# residual_correction_token: P_H mass only on idx 0 after subtracting P_S
P_H = np.array([0.7, 0.3, 0.0])
P_S = np.array([0.0, 0.3, 0.7])   # resid = [0.7,0,0] -> always idx 0
tok = specloop.residual_correction_token(P_H, P_S)
check("residual_correction_token picks residual support", tok == 0, f"tok={tok}")

# 6. val_tune_test_split sizes
prompts = list(range(200))
tune, test = data.val_tune_test_split(prompts, n_tune=50)
check("val_tune_test_split sizes", len(tune) == 50 and len(test) == 150 and tune[0] == 0 and test[0] == 50)

# 7. CheckpointManager round-trip + csv writer + schema + meta
with tempfile.TemporaryDirectory() as tmp:
    ck = io.CheckpointManager(tmp)
    ck.save("unit__x", {"acceptance_rate": 0.42, "vec": [1, 2, 3]})
    got = ck.load("unit__x")
    check("CheckpointManager round-trip", got is not None and got["acceptance_rate"] == 0.42)
    check("checkpoint_skip returns cached", io.checkpoint_skip(ck, "unit__x", "acceptance_rate") is not None)
    check("checkpoint_skip missing -> None", io.checkpoint_skip(ck, "nope") is None)

    csv_path = os.path.join(tmp, "out", "t.csv")
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    io.write_csv(rows, csv_path)
    import csv as _csv
    back = list(_csv.DictReader(open(csv_path)))
    check("write_csv round-trip", len(back) == 2 and back[0]["a"] == "1" and back[1]["b"] == "4")
    check("validate_schema ok", io.validate_schema(rows, ["a", "b"]))
    try:
        io.validate_schema(rows, ["a", "b", "c"])
        check("validate_schema raises on missing", False)
    except ValueError:
        check("validate_schema raises on missing", True)

    meta = io.write_meta_json(os.path.join(tmp, "meta.json"), experiment="unit")
    check("write_meta_json has keys", "timestamp_utc" in meta and meta.get("experiment") == "unit")

# 8. power_note
pn = stats.power_note(0.68, 200)
check("power_note ci width positive", pn["ci_width_normal"] > 0)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("ALL PURE TESTS PASSED")
