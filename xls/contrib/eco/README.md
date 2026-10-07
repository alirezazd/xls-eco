# xls/contrib/eco

Engineering Change Order (ECO) tooling for XLS. Given two versions of a design, a *before* and a revised *after*, it computes a small set of node and edge edits that turns one into the other, replays them on the before IR, keeps the before pipeline schedule, and emits RTL in which everything the ECO did not touch reads exactly as before.

The goal is design reuse. When a small spec change lands after the IR has already been scheduled, codegened, or verified, ECO keeps node identities, stage assignments, register names and RTL text for the unchanged part of the design. Only the diff has to be re-examined, and a downstream netlist ECO sees only the changed cones. The patch is a serialized proto, so the diff itself is reviewable and replayable.

Procs are supported. Channels and state elements participate in the diff alongside ordinary nodes, so adding, removing, retyping, or reordering them is applied as an ordinary edit and leaves the rest of the proc as it was. Channel support covers proc-scoped (new-style) channels.

## Quick start (Bazel)

One macro in [eco_build_defs.bzl](eco_build_defs.bzl) runs the whole chain: diff, patch, verify, then schedule- and RTL-preserving codegen.

```
xls_eco(
    name = "mydesign",
    srcs = [":before.x", ":after.x"],
    dslx_top = "main",
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "3",
        "reset": "rst",
    },
)
```

| Target | What it is |
| --- | --- |
| `mydesign_diff` | `.patch.bin` and `.report` from diffing the two optimized IRs. |
| `mydesign_before_sv` | The before design through `xls_ir_verilog`; its schedule and codegen options are the reference for everything below. |
| `mydesign_patched_opt_ir` | The before IR with the patch applied, verified equivalent to the after IR, then rescheduled to keep the before schedule (`.schedule.textproto`, `.schedule.opt.ir`, `.equiv.report`). |
| `mydesign_patched_sv` | RTL of the patched design emitted against the before RTL: `mydesign_patched.rtl.diff` is the ECO and nothing else, `.rtl.report` counts what was kept. |
| `mydesign_before_rtl_test` | Checks the before RTL round-trips through the ECO emitter bit-for-bit, so the diff above is meaningful. |
| `mydesign_report` | `mydesign.eco.report`, the combined report described in [Reports](#reports). |

Without `codegen_args` only the first and third targets are made. Optional kwargs: `patched_codegen_args` (overrides for the patched side, e.g. `{"pipeline_stages": "4"}` for a stage-count ECO), `greedy_schedule` (the baseline scheduler, see below), `eqv_method`, `activation_count`, `top`, and the MCS flags of [Diff tuning](#diff-tuning). [test/BUILD](test/BUILD) uses it for every design.

The pieces are available on their own: `xls_dslx_ir_diff` (diff), `xls_patch_ir` (patch + verify, `schedule=` to preserve a schedule) and the `xls_eco_verilog` rule (codegen against a reference). The `*_pieces_*` targets at the end of [test/BUILD](test/BUILD) spell them out for crc32 and rle_enc, with the logical equivalence check.

## Manual

1. Diff the two IRs to produce a patch.

   ```
   ged_main --before_ir=a.ir --after_ir=b.ir --patch=patch.bin
   ```

2. Apply the patch to the before IR, keeping its schedule. The schedule inputs are what `codegen_main` writes with `--output_schedule_path` and `--output_schedule_ir_path`; pass the same scheduling flags (`--pipeline_stages`, `--delay_model`, ...).

   ```
   patch_ir_main --input_ir_path=a.ir --input_patch_path=patch.bin --output_ir_path=patched.ir \
       --input_schedule_path=a.schedule.textproto --pipeline_stages=3 --delay_model=unit \
       --output_schedule_path=patched.schedule.textproto --output_schedule_ir_path=patched.schedule.ir
   ```

3. Verify the patched IR matches the after IR.

   ```
   check_ir_equivalence_main patched.ir b.ir          # logical
   check_ir_struct_equivalence_main patched.ir b.ir   # structural
   ```

4. Emit RTL against the before RTL. `a.block.ir` is `codegen_main --output_block_ir_path`; the residual data is captured by running the before design through the same tool first.

   ```
   codegen_scheduled_main a.schedule.ir --input_schedule_path=a.schedule.textproto \
       --reference_block_ir_path=a.block.ir --output_residual_data_path=a.residual.textproto \
       --pipeline_stages=3 --delay_model=unit --reset=rst --output_verilog_path=a.sv
   codegen_scheduled_main patched.schedule.ir --input_schedule_path=patched.schedule.textproto \
       --reference_block_ir_path=a.block.ir --reference_residual_data_path=a.residual.textproto \
       --pipeline_stages=3 --delay_model=unit --reset=rst --output_verilog_path=patched.sv
   ```

The patch is unverified until step 3 passes. For procs, logical checking needs `--activation_count=N`.

## Schedule preservation

`patch_ir_main` reschedules the patched IR under the before options, then solves one SDC LP with a *soft* node-in-cycle constraint per surviving node (`SoftNodeInCycleConstraint` in `xls/scheduling`, weight = bit width): the scheduler first minimizes the total weighted deviation from the before schedule, then optimizes its usual objective within that optimum. Deleted nodes drop out, inserted nodes are free. The passes create nodes of their own (state legalization, mutual exclusion) named after the package watermark, so the patched post-pass IR is first aligned with the before one (name anchors, then GED on what is left) and the schedule maps by id; those nodes keep their stage as well. The target clock is the before schedule's; if the ECO lengthened the critical path it is relaxed to the shortest feasible period, which is logged along with `Schedule assignments kept: X of N`. `--greedy_schedule` (`PatchScheduleGreedy`) instead pins nodes one at a time, one solve each, and is the baseline the LP is compared against.

## RTL preservation

Register and port names come from the scheduled IR and survive the patch. Every other wire is named after a node id, and codegen's block finalization reassigns those from the package watermark, so a one-literal ECO used to rename every wire in the module. `codegen_scheduled_main --reference_block_ir_path` fixes that after codegen has finished: the block is matched against the reference block with the same MCS+GED used for the IR diff (block IR lowers to the same graphs), every matched node takes the reference's id (and its name, if it had none), and the Verilog is emitted again. With `--reference_residual_data_path` (upstream's `CodegenResidualData`, keyed by node id) the reference's emission order and inline/assign choices are replayed too, so new logic lands where its inputs are and untouched lines are byte-identical. A design matched against its own block IR takes the identity fast path; otherwise `--align_mcs_optimal` (default off, the optimal MCS is slow on large blocks) and `--align_mcs_timeout` tune the search. `--output_rtl_report_path` writes the nodes matched, the registers and ports kept, and (with `--reference_verilog_path`) the RTL lines changed, both as a diff would count them and disregarding order: when the ECO makes an early node read a late one, codegen moves the whole cone, which a diff reports and a netlist ECO never sees.

## Reports

Three reports are written along the chain and `<name>_report` concatenates them into `<name>.eco.report`. Every line is `key: value`; counts are `kept of total (original total)`.

| Section | Written by | What it says |
| --- | --- | --- |
| `Diff` | `ged_main --report` | Graph sizes, MCS matches and pruning, GED node and edge edits: how much of the IR the ECO touched. |
| `Schedule` | `patch_ir_main --output_schedule_report_path` | Nodes matched to the original post-pass IR, clock and stage count before and after, original stage assignments kept. |
| `Block` / `Registers` / `Ports` | `codegen_scheduled_main --output_rtl_report_path` | Block nodes matched; registers kept, moved (same node, other stage), retyped, removed, added, in count and bits; how many kept registers have an identical data cone (Merkle hash of the value written, with register reads, ports and literals as leaves) and how many still do once load enable and reset are included, which in a proc hang off shared control; output ports likewise. The data-cone count is what a netlist ECO leaves untouched. |
| `RTL lines` | same | Changed lines as a diff counts them, and disregarding order. |

## Equivalence checking

Both checkers take positional `IR_A IR_B` and the same `--match_exit_code` / `--mismatch_exit_code` / `--equivalence_report_path` flags. `xls_patch_ir`'s `eqv_method` selects which one it runs:

| `eqv_method` | Tool | Checks |
| --- | --- | --- |
| `structural` (default) | `check_ir_struct_equivalence_main` | Graph isomorphism. Cheap, and the practical choice for large designs. |
| `logical` | `//xls/dev_tools:check_ir_equivalence_main` | Functional equivalence (Z3). Thorough but slow on large or floating-point designs. |

`structural` may false-reject equivalent-but-restructured IR; it never false-accepts.

## How it works

The two IR versions are lowered to labeled graphs, where a node's label is its identity: op, types, and the attributes that distinguish it from a same-op sibling. Matching happens in two stages. First, maximum common subgraph (MCS) preprocessing [[1]](#ref-1) finds the largest identical region and takes it out of the search, so unchanged logic never reaches the expensive stage. Then graph edit distance (GED) computes an edit path over what remains, using a constructive BRANCH-FAST heuristic [[2]](#ref-2) over a linear assignment [[3]](#ref-3): polynomial time, and no blow-up on large or imbalanced inputs.

The heuristic's edit path is an upper bound on the true edit distance, not necessarily optimal. A looser match only makes the patch larger, never wrong, because correctness is established by the [equivalence check](#equivalence-checking) rather than by optimality. Substitutions are permitted only between nodes whose labels are identical, so applying a patch can never silently reshape a node it claims to preserve.

## Diff tuning

`ged_main` flags:

| Flag | Default | Effect |
| --- | --- | --- |
| `--before_ir`, `--after_ir` | none | IR file paths. Positional fallback if both are unset. |
| `--patch` | `""` | Output `IrPatchProto` path. |
| `--report` | `""` | Output stats report: graph sizes, MCS runtime + match counts + prune %, GED runtime, op breakdown, peak RSS. |
| `--use_mcs` | `true` | Run MCS preprocessing before GED. Set `false` to skip it. |
| `--mcs_cutoff` | `-1` | Stop MCS once at most N residual nodes are unmatched. `-1` runs to completion. |
| `--mcs_optimal` | `true` | If `false`, MCS may terminate on a no-improvement plateau (faster, may leave a smaller MCS). |
| `--mcs_timeout` | `-1` | MCS wall-clock cap in seconds. `-1` disables. |

Some designs plateau late in MCS search and are much faster with `--mcs_optimal=false`; the resulting patch is larger but still verified.

## Logging

Both binaries route through `xls::InitXls`, which honours absl's logging flags:

- `--v=0` (default): top-level progress: start, MCS enabled, GED finished, totals.
- `--v=1`: parsed graph sizes, residual sizes after MCS preprocessing, pinned boundary counts.
- `--v=2`: per-pair MCS pinning, per-edit substitute/insert/delete with node names.
- `--vmodule=mcs=2,ged=1`: scope verbosity per source file.
- `--logtostderr`: emit to stderr (default for CLI use).

The `--report` file is the structured counterpart to log output and is meant for CI / regression dashboards.

## References

<a id="ref-1"></a>[1] Kaiqiang Yu, Kaixin Wang, Cheng Long, Laks V.S. Lakshmanan, and Reynold Cheng. 2025. Fast Maximum Common Subgraph Search: A Redundancy-Reduced Backtracking Approach. *Proc. ACM Manag. Data* 3, 3, Article 160 (2025). https://doi.org/10.1145/3725404

<a id="ref-2"></a>[2] David B. Blumenthal, Nicolas Boria, Johann Gamper, Sébastien Bougleux, and Luc Brun. 2020. Comparing heuristics for graph edit distance computation. *The VLDB Journal* 29 (2020), 419-458. https://doi.org/10.1007/s00778-019-00544-1

<a id="ref-3"></a>[3] David F. Crouse. 2016. On Implementing 2D Rectangular Assignment Algorithms. *IEEE Transactions on Aerospace and Electronic Systems* 52, 4 (2016), 1679-1696.
