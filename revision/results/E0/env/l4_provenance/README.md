# Canonical L4 provenance (committed copy)

Verbatim copies of the L4 pod-local status files that document the kernel
stack behind the canonical result **c_eff = 0.7843** (E3.1). The originals
live in `results/.l4markers/`, which is **gitignored and pod-local**, so
these copies exist to make the provenance survive pod teardown.

## Files

- `kernels_result.txt` — fastpath status + exact source-built stack:
  `mamba-ssm==2.3.2.post1`, `causal-conv1d==1.6.2.post1`, on `torch 2.4.1+cu124`.
- `kernels_main.txt` — main-path stack used for the naive baseline (`main_stack=naive`).
- `fastpath_status.txt` — kernel fastpath active flag (`yes`).

Cross-references (also committed): `results/E3.1/with_kernels.meta.json`,
`results/E3.1/attention_only_kernels.meta.json`, and the
`CANONICAL L4 ENVIRONMENT` block in `results/E0/env/key_versions.txt`.

## Deliberately NOT copied

The `*.done` files in `results/.l4markers/` (`E31.done`, `E32.done`, `E4.done`,
`E_KERNELS.done`) are **empty resume markers**. Per `.gitignore` they must NOT
travel to a fresh pod: if present, an L4 rerun would skip those experiments,
believing them already complete. They carry no provenance value (0 bytes), so
only the text status files above are preserved here.
