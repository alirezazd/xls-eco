# xls/contrib/eco

Engineering Change Order (ECO) tooling for XLS IR. Given two versions of an IR package — a *before* and a revised *after* — this computes a small set of node and edge edits that turns one into the other, and replays them on the before IR.

The motivation is **design reuse.** When a small spec change lands after the IR has already been scheduled, codegened, or verified, ECO keeps node identities, stage assignments, and downstream artifacts for the unchanged majority of the design — only the diff has to be re-examined. The patch is a serialized proto, so the diff itself is reviewable and replayable.

Procs are supported. Channels and state elements participate in the diff alongside ordinary nodes, so adding, removing, retyping, or reordering them is applied as an ordinary edit and leaves everything untouched genuinely untouched. Channel support covers proc-scoped (new-style) channels.

## Quick start (recommended: Bazel)

The full diff → apply → verify chain is wrapped as two macros in [eco_build_defs.bzl](eco_build_defs.bzl). `xls_patch_ir` runs the equivalence check automatically, so the target only builds if the patched IR is verified equivalent to the after IR.

```
xls_dslx_ir_diff(
    name = "mydiff",
    srcs = [":before.x", ":after.x"],
    dslx_top = "main",
)

xls_patch_ir(
    name = "mypatched",
    ir_diff = ":mydiff",
)
```

`xls_dslx_ir_diff` compiles both DSLX inputs with `xls_dslx_opt_ir`, runs the diff, and emits `.patch.bin` and `.report` plus an `EcoPatchInfo` provider. `xls_patch_ir` consumes that, applies the patch, then verifies and writes an `.equiv.report`. Optional kwargs: `schedule=`, `top=`, `activation_count=`, `eqv_method=` (for `xls_patch_ir`) and the diff tuning flags below (for `xls_dslx_ir_diff`). [test/BUILD](test/BUILD) uses both as the end-to-end harness.

## Manual

1. Diff the two IRs to produce a patch.

   ```
   ged_main --before_ir=a.ir --after_ir=b.ir --patch=patch.bin
   ```

2. Apply the patch to the before IR.

   ```
   patch_ir_main --input_ir_path=a.ir --input_patch_path=patch.bin --output_ir_path=patched.ir
   ```

3. Verify the patched IR matches the after IR.

   ```
   check_ir_equivalence_main patched.ir b.ir          # logical
   check_ir_struct_equivalence_main patched.ir b.ir   # structural
   ```

Step 3 is the trust boundary: until one of these passes, the patch is unverified. To preserve the original pipeline schedule across step 2, pass `--input_schedule_path=schedule.textproto` (a `PackageScheduleProto`); deleted nodes drop out, substituted nodes keep their stage. For procs, logical checking needs `--activation_count=N`.

## Equivalence checking

Both checkers take positional `IR_A IR_B` and the same `--match_exit_code` / `--mismatch_exit_code` / `--equivalence_report_path` flags. `xls_patch_ir`'s `eqv_method` selects which one it runs:

| `eqv_method` | Tool | Checks |
| --- | --- | --- |
| `structural` (default) | `check_ir_struct_equivalence_main` | Graph isomorphism — cheap, and the practical choice for large designs. |
| `logical` | `//xls/dev_tools:check_ir_equivalence_main` | Functional equivalence (Z3). Thorough but slow on large or floating-point designs. |

`structural` may false-reject equivalent-but-restructured IR; it never false-accepts.

## How it works

The two IR versions are lowered to labeled graphs, where a node's label is its identity: op, types, and the attributes that distinguish it from a same-op sibling. Matching happens in two stages. First, maximum common subgraph (MCS) preprocessing [[1]](#ref-1) finds the largest identical region and takes it out of the search — this is what lets the chain scale, since unchanged logic never reaches the expensive stage. Then graph edit distance (GED) computes an edit path over what remains, using a constructive BRANCH-FAST heuristic [[2]](#ref-2) over a linear assignment [[3]](#ref-3): polynomial time, and no blow-up on large or imbalanced inputs.

The heuristic's edit path is an upper bound on the true edit distance, not necessarily optimal. That is deliberate: a looser match only makes the patch larger, never wrong, because correctness is certified by the [equivalence check](#equivalence-checking) rather than by optimality. Substitutions are permitted only between nodes whose labels are identical, so applying a patch can never silently reshape a node it claims to preserve.

## Diff tuning

`ged_main` flags:

| Flag | Default | Effect |
| --- | --- | --- |
| `--before_ir`, `--after_ir` | — | IR file paths. Positional fallback if both are unset. |
| `--patch` | `""` | Output `IrPatchProto` path. |
| `--report` | `""` | Output stats report: graph sizes, MCS runtime + match counts + prune %, GED runtime, op breakdown, peak RSS. |
| `--use_mcs` | `true` | Run MCS preprocessing before GED. Set `false` to skip it. |
| `--mcs_cutoff` | `-1` | Stop MCS as soon as residual unmatched nodes ≤ N. `-1` runs to completion. |
| `--mcs_optimal` | `true` | If `false`, MCS may terminate on a no-improvement plateau (faster, may leave a smaller MCS). |
| `--mcs_timeout` | `-1` | MCS wall-clock cap in seconds. `-1` disables. |

Some designs plateau late in MCS search and are much faster with `--mcs_optimal=false`; the resulting patch is larger but still verified.

## Logging

Both binaries route through `xls::InitXls`, which honours absl's logging flags:

- `--v=0` (default): top-level progress — start, MCS enabled, GED finished, totals.
- `--v=1`: parsed graph sizes, residual sizes after MCS preprocessing, pinned boundary counts.
- `--v=2`: per-pair MCS pinning, per-edit substitute/insert/delete with node names.
- `--vmodule=mcs=2,ged=1`: scope verbosity per source file.
- `--logtostderr`: emit to stderr (default for CLI use).

The `--report` file is the structured counterpart to log output and is meant for CI / regression dashboards.

## References

<a id="ref-1"></a>[1] Kaiqiang Yu, Kaixin Wang, Cheng Long, Laks V.S. Lakshmanan, and Reynold Cheng. 2025. Fast Maximum Common Subgraph Search: A Redundancy-Reduced Backtracking Approach. *Proc. ACM Manag. Data* 3, 3, Article 160 (2025). https://doi.org/10.1145/3725404

<a id="ref-2"></a>[2] David B. Blumenthal, Nicolas Boria, Johann Gamper, Sébastien Bougleux, and Luc Brun. 2020. Comparing heuristics for graph edit distance computation. *The VLDB Journal* 29 (2020), 419–458. https://doi.org/10.1007/s00778-019-00544-1

<a id="ref-3"></a>[3] David F. Crouse. 2016. On Implementing 2D Rectangular Assignment Algorithms. *IEEE Transactions on Aerospace and Electronic Systems* 52, 4 (2016), 1679–1696.
