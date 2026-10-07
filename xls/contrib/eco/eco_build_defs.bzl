# Copyright 2025 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Build rules for generating and patching XLS IR to support ECO workflows.
"""

load("@bazel_skylib//rules:diff_test.bzl", "diff_test")
load("//xls/build_rules:xls_build_defs.bzl", "xls_dslx_opt_ir", "xls_ir_verilog")
load("//xls/build_rules:xls_providers.bzl", "CODEGEN_FIELDS", "CodegenInfo", "IrFileInfo", "SCHEDULING_FIELDS")

EcoPatchInfo = provider(
    doc = "Carries the generated patch file.",
    fields = {
        "patch": "File containing the IrPatchProto.",
        "before_ir": "Input opt IR file (before).",
        "after_ir": "Input opt IR file (after).",
        "report": "Diff statistics.",
    },
)

EcoScheduleInfo = provider(
    doc = "Patched IR in scheduling form and its schedule-preserving schedule.",
    fields = {
        "schedule_ir_file": "IR file the schedule names.",
        "schedule_file": "PackageScheduleProto textproto.",
        "report_file": "What the patch did to the schedule.",
    },
)

EcoRtlInfo = provider(
    doc = "RTL of a design together with what a later ECO needs to reproduce its naming.",
    fields = {
        "verilog_file": "The generated (System)Verilog.",
        "block_ir_file": "Block IR the RTL was emitted from.",
        "residual_file": "CodegenResidualData textproto: the emission order and inlining used.",
        "report_file": "Registers, ports, cones and RTL lines against the reference, or None.",
    },
)

_SCHEDULING_FLAGS = SCHEDULING_FIELDS.keys()
_CODEGEN_FLAGS = CODEGEN_FIELDS.keys() + _SCHEDULING_FLAGS

# CodegenInfo fields that are not codegen_main flags.
_NON_FLAG_FIELDS = ["input_ir", "top"]

def _codegen_flags(codegen, fields, overrides):
    """Flags for a codegen binary: those an xls_ir_verilog target ran with, then `overrides`."""
    args = {}
    if CodegenInfo in codegen:
        info = codegen[CodegenInfo]
        for key in fields:
            value = getattr(info, key, None)
            if value != None and key not in _NON_FLAG_FIELDS:
                args[key] = value
    for key in overrides:
        if key not in fields:
            fail("unknown codegen arg '%s'" % key)
    args.update(overrides)
    return ["--%s=%s" % (key, value) for key, value in args.items()]

def _dslx_ir_diff_impl(ctx):
    before_ir = ctx.attr.before[IrFileInfo].ir_file
    after_ir = ctx.attr.after[IrFileInfo].ir_file

    patch_out = ctx.actions.declare_file(ctx.label.name + ".patch.bin")
    report_out = ctx.actions.declare_file(ctx.label.name + ".report")

    ged_args = []
    if ctx.attr.mcs == False:
        ged_args.append("--use_mcs=false")
    if ctx.attr.mcs_cutoff >= 0:
        ged_args.append("--mcs_cutoff=" + str(ctx.attr.mcs_cutoff))
    if ctx.attr.mcs_optimal == False:
        ged_args.append("--mcs_optimal=false")
    if ctx.attr.mcs_timeout >= 0:
        ged_args.append("--mcs_timeout=" + str(ctx.attr.mcs_timeout))
    ged_args.append("--report=" + report_out.path)

    ged_runfiles = ctx.attr.ged_main[DefaultInfo].default_runfiles.files.to_list()
    outputs = [patch_out, report_out]

    ged_args += [
        "--before_ir=" + before_ir.path,
        "--after_ir=" + after_ir.path,
        "--patch=" + patch_out.path,
    ]
    ctx.actions.run(
        inputs = [before_ir, after_ir] + ctx.files.ged_main + ged_runfiles,
        outputs = outputs,
        tools = [ctx.executable.ged_main],
        mnemonic = "GedMain",
        arguments = ged_args,
        executable = ctx.executable.ged_main,
        use_default_shell_env = True,
    )

    default_files = [patch_out, before_ir, after_ir, report_out]
    output_groups = dict(
        patch = depset([patch_out]),
        opt_irs = depset([before_ir, after_ir]),
        report = depset([report_out]),
    )

    return [
        DefaultInfo(files = depset(default_files)),
        OutputGroupInfo(
            **output_groups
        ),
        EcoPatchInfo(
            patch = patch_out,
            before_ir = before_ir,
            after_ir = after_ir,
            report = report_out,
        ),
    ]

xls_dslx_ir_diff_rule = rule(
    implementation = _dslx_ir_diff_impl,
    attrs = {
        "before": attr.label(
            providers = [IrFileInfo],
            doc = "Opt IR (before).",
        ),
        "after": attr.label(
            providers = [IrFileInfo],
            doc = "Opt IR (after).",
        ),
        "ged_main": attr.label(
            executable = True,
            cfg = "target",
            default = Label("//xls/contrib/eco:ged_main"),
        ),
        "mcs": attr.bool(
            default = True,
            doc = "Enable MCS when diffing; false passes --use_mcs=false to GED.",
        ),
        "mcs_cutoff": attr.int(
            default = -1,
            doc = "Stop MCS when remaining nodes <= this value; negative means run to completion.",
        ),
        "mcs_optimal": attr.bool(
            default = True,
            doc = "Require optimal MCS preprocessing; false enables heuristic early stop on long plateaus.",
        ),
        "mcs_timeout": attr.int(
            default = -1,
            doc = "Optional MCS timeout in seconds; negative means omit flag.",
        ),
    },
)

def xls_dslx_ir_diff(name, srcs, dslx_top, deps = None, mcs = None, mcs_cutoff = None, mcs_optimal = None, mcs_timeout = None, tags = None):
    """Builds opt IRs for two DSLX sources and emits a patch between them.

    Args:
      name: Base name for generated targets/outputs.
      srcs: List of two DSLX source labels [before, after].
      dslx_top: Either a single top name (applied to both) or a list of two.
      deps: Optional list of xls_dslx_library deps for imports in the sources.
      mcs: Optional boolean. If false, pass --use_mcs=false to GED; true leaves default.
      mcs_cutoff: Optional int. Stop MCS when remaining nodes <= this value; negative => run to completion.
      mcs_optimal: Optional boolean. False enables heuristic MCS early stop.
      mcs_timeout: Optional int. Timeout in seconds for MCS preprocessing.
      tags: Optional Bazel tags applied to generated targets.
    """
    if len(srcs) != 2:
        fail("xls_dslx_ir_diff.srcs must have length 2")
    if type(dslx_top) == type([]):
        if len(dslx_top) != 2:
            fail("xls_dslx_ir_diff.dslx_top list must have length 2")
        tops = dslx_top
    else:
        tops = [dslx_top, dslx_top]

    before_ir_target = name + "_before_opt_ir"
    after_ir_target = name + "_after_opt_ir"

    opt_ir_kwargs = {}
    if deps != None:
        opt_ir_kwargs["deps"] = deps

    xls_dslx_opt_ir(
        name = before_ir_target,
        srcs = [srcs[0]],
        dslx_top = tops[0],
        tags = tags,
        **opt_ir_kwargs
    )

    xls_dslx_opt_ir(
        name = after_ir_target,
        srcs = [srcs[1]],
        dslx_top = tops[1],
        tags = tags,
        **opt_ir_kwargs
    )

    xls_dslx_ir_diff_rule_kwargs = dict(
        name = name,
        before = ":" + before_ir_target,
        after = ":" + after_ir_target,
    )
    if mcs != None:
        xls_dslx_ir_diff_rule_kwargs["mcs"] = mcs
    if mcs_cutoff != None:
        xls_dslx_ir_diff_rule_kwargs["mcs_cutoff"] = mcs_cutoff
    if mcs_optimal != None:
        xls_dslx_ir_diff_rule_kwargs["mcs_optimal"] = mcs_optimal
    if mcs_timeout != None:
        xls_dslx_ir_diff_rule_kwargs["mcs_timeout"] = mcs_timeout
    if tags != None:
        xls_dslx_ir_diff_rule_kwargs["tags"] = tags

    xls_dslx_ir_diff_rule(**xls_dslx_ir_diff_rule_kwargs)

def _xls_patch_ir_impl(ctx):
    diff_target = ctx.attr.ir_diff
    if not diff_target:
        fail("xls_patch_ir: 'ir_diff' label is required.")

    diff_info = diff_target[EcoPatchInfo]
    ir_file = diff_info.before_ir
    patch_file = diff_info.patch
    after_ir = diff_info.after_ir

    output_ir = ctx.actions.declare_file(ctx.label.name + ".opt.ir")

    patch_ir_runfiles = ctx.attr.patch_ir_main[DefaultInfo].default_runfiles.files.to_list()

    inputs = [ir_file, patch_file] + ctx.files.patch_ir_main + patch_ir_runfiles
    outputs = [output_ir]
    args = [
        "--input_ir_path=" + ir_file.path,
        "--input_patch_path=" + patch_file.path,
        "--output_ir_path=" + output_ir.path,
    ]

    output_schedule = None
    output_schedule_ir = None
    schedule_report = None
    if ctx.attr.schedule:
        if CodegenInfo in ctx.attr.schedule:
            info = ctx.attr.schedule[CodegenInfo]
            schedule_file = info.schedule_file
            inputs.append(info.schedule_ir_file)
            args.append("--input_schedule_ir_path=" + info.schedule_ir_file.path)
        elif len(ctx.files.schedule) == 1:
            schedule_file = ctx.files.schedule[0]
        else:
            schedule_file = None
        if not schedule_file:
            fail("xls_patch_ir: 'schedule' must be a pipelined xls_ir_verilog target or one PackageScheduleProto textproto")
        output_schedule = ctx.actions.declare_file(ctx.label.name + ".schedule.textproto")
        output_schedule_ir = ctx.actions.declare_file(ctx.label.name + ".schedule.opt.ir")
        schedule_report = ctx.actions.declare_file(ctx.label.name + ".schedule.report")
        inputs.append(schedule_file)
        outputs += [output_schedule, output_schedule_ir, schedule_report]
        args.append("--input_schedule_path=" + schedule_file.path)
        args.append("--output_schedule_path=" + output_schedule.path)
        args.append("--output_schedule_ir_path=" + output_schedule_ir.path)
        args.append("--output_schedule_report_path=" + schedule_report.path)
        if ctx.attr.greedy_schedule:
            args.append("--greedy_schedule")
        args += _codegen_flags(ctx.attr.schedule, _SCHEDULING_FLAGS, ctx.attr.scheduling_args)
    elif ctx.attr.scheduling_args or ctx.attr.greedy_schedule:
        fail("xls_patch_ir: 'scheduling_args' and 'greedy_schedule' require 'schedule'")

    ctx.actions.run(
        inputs = inputs,
        outputs = outputs,
        executable = ctx.executable.patch_ir_main,
        arguments = args,
        mnemonic = "PatchIr",
        use_default_shell_env = True,
    )

    equivalence_report = ctx.actions.declare_file(ctx.label.name + ".equiv.report")
    common_flags = [
        "--match_exit_code=0",
        "--mismatch_exit_code=1",
        "--equivalence_report_path=" + equivalence_report.path,
    ]

    method = ctx.attr.eqv_method
    if method == "logical":
        check_inputs = [output_ir, after_ir]
        check_inputs += ctx.attr.check_ir_main[DefaultInfo].default_runfiles.files.to_list()
        check_inputs += ctx.files.check_ir_main
        logical_flags = list(common_flags)
        if ctx.attr.check_activation_count >= 0:
            logical_flags.append("--activation_count=" + str(ctx.attr.check_activation_count))
        if ctx.attr.check_top:
            logical_flags.append("--top=" + ctx.attr.check_top)
        ctx.actions.run(
            inputs = check_inputs,
            outputs = [equivalence_report],
            executable = ctx.executable.check_ir_main,
            mnemonic = "CheckIrEquivalence",
            arguments = [output_ir.path, after_ir.path] + logical_flags,
        )
    else:
        # Structural equivalence via canonical graph-hash comparison.
        check_inputs = [output_ir, after_ir]
        check_inputs += ctx.attr.check_struct_ir_main[DefaultInfo].default_runfiles.files.to_list()
        check_inputs += ctx.files.check_struct_ir_main
        ctx.actions.run(
            inputs = check_inputs,
            outputs = [equivalence_report],
            executable = ctx.executable.check_struct_ir_main,
            mnemonic = "CheckIrStructEquivalence",
            arguments = [output_ir.path, after_ir.path] + common_flags,
        )

    providers = [
        DefaultInfo(files = depset(outputs + [equivalence_report])),
        IrFileInfo(ir_file = output_ir),
    ]
    if output_schedule:
        providers.append(EcoScheduleInfo(
            schedule_ir_file = output_schedule_ir,
            schedule_file = output_schedule,
            report_file = schedule_report,
        ))
    return providers

xls_patch_ir_rule = rule(
    implementation = _xls_patch_ir_impl,
    attrs = {
        "ir_diff": attr.label(
            providers = [EcoPatchInfo],
            doc = "xls_dslx_ir_diff target providing the before IR and patch.",
        ),
        "schedule": attr.label(
            allow_files = True,
            doc = "Schedule of the before IR: a pipelined xls_ir_verilog target or a " +
                  "PackageScheduleProto textproto. When set, the patched IR is " +
                  "rescheduled keeping feasible original cycle assignments and " +
                  "<name>.schedule.textproto is emitted.",
        ),
        "scheduling_args": attr.string_dict(
            doc = "Scheduling flags for patch_ir_main (pipeline_stages, clock_period_ps, " +
                  "delay_model, ...). Inherited from an xls_ir_verilog 'schedule' " +
                  "target; entries here override.",
        ),
        "greedy_schedule": attr.bool(
            default = False,
            doc = "Pin original cycle assignments one by one (one scheduler run each) " +
                  "instead of solving once with soft constraints (baseline).",
        ),
        "patch_ir_main": attr.label(
            executable = True,
            cfg = "target",
            default = Label("//xls/contrib/eco:patch_ir_main"),
            doc = "Binary used to apply the patch.",
        ),
        "check_ir_main": attr.label(
            executable = True,
            cfg = "target",
            default = Label("//xls/dev_tools:check_ir_equivalence_main"),
            doc = "Binary used to validate equivalence of patched IR vs. revised IR.",
        ),
        "check_struct_ir_main": attr.label(
            executable = True,
            cfg = "target",
            default = Label("//xls/contrib/eco:check_ir_struct_equivalence_main"),
            doc = "Binary used by the 'structural' method: compares canonical " +
                  "graph fingerprints (StructuralHash) of patched vs. revised IR.",
        ),
        "eqv_method": attr.string(
            default = "structural",
            values = ["logical", "structural"],
            doc = "Equivalence method: 'structural' (canonical graph-hash " +
                  "comparison, default) or 'logical' (SMT/LEC via " +
                  "check_ir_equivalence_main). check_top and " +
                  "check_activation_count apply only to 'logical'.",
        ),
        "check_top": attr.string(
            default = "",
            doc = "Optional top entity passed to check_ir_equivalence_main.",
        ),
        "check_activation_count": attr.int(
            default = -1,
            doc = "Optional activation count for proc equivalence; negative omits the flag.",
        ),
    },
)

def xls_patch_ir(
        name,
        ir_diff,
        schedule = None,
        scheduling_args = None,
        greedy_schedule = None,
        top = None,
        activation_count = None,
        eqv_method = None,
        **kwargs):
    """Applies an ECO patch to the before IR and validates against the after IR.

    Args:
      name: Base name for generated target.
      ir_diff: xls_dslx_ir_diff target providing the before IR and patch.
      schedule: Optional schedule of the before IR (pipelined xls_ir_verilog target or
        PackageScheduleProto textproto); the patched IR is rescheduled to preserve it.
      scheduling_args: Optional dict of scheduling flags; overrides those of `schedule`.
      greedy_schedule: Optional bool; pin assignments one by one instead of solving
        once with soft constraints (baseline).
      top: Optional top entity passed to check_ir_equivalence_main.
      activation_count: Optional activation count for proc equivalence.
      eqv_method: Optional equivalence method: "structural" (default) or "logical".
      **kwargs: Additional parameters passed to the underlying rule.
    """
    if "check_activation_count" in kwargs and activation_count != None:
        fail("Provide only one of activation_count or check_activation_count.")
    if "check_top" in kwargs and top != None:
        fail("Provide only one of top or check_top.")

    rule_kwargs = dict(kwargs)
    rule_kwargs["ir_diff"] = ir_diff
    if schedule != None:
        rule_kwargs["schedule"] = schedule
    if scheduling_args != None:
        rule_kwargs["scheduling_args"] = scheduling_args
    if greedy_schedule != None:
        rule_kwargs["greedy_schedule"] = greedy_schedule
    if eqv_method != None:
        rule_kwargs["eqv_method"] = eqv_method
    if activation_count != None:
        rule_kwargs["check_activation_count"] = activation_count
    if top != None:
        rule_kwargs["check_top"] = top

    xls_patch_ir_rule(
        name = name,
        **rule_kwargs
    )

def _xls_eco_verilog_impl(ctx):
    src = ctx.attr.src
    info = src[EcoScheduleInfo] if EcoScheduleInfo in src else src[CodegenInfo]
    verilog_file = ctx.outputs.verilog_file
    basename = verilog_file.basename.rsplit(".", 1)[0]
    signature_file = ctx.actions.declare_file(basename + ".sig.textproto")
    block_ir_file = ctx.actions.declare_file(basename + ".block.ir")
    residual_file = ctx.actions.declare_file(basename + ".residual.textproto")

    codegen_main = ctx.attr.codegen_scheduled_main
    inputs = [info.schedule_ir_file, info.schedule_file]
    outputs = [verilog_file, signature_file, block_ir_file, residual_file]
    args = [
        info.schedule_ir_file.path,
        "--input_schedule_path=" + info.schedule_file.path,
        "--output_verilog_path=" + verilog_file.path,
        "--output_signature_path=" + signature_file.path,
        "--output_block_ir_path=" + block_ir_file.path,
        "--output_residual_data_path=" + residual_file.path,
    ]
    reference = ctx.attr.reference
    report_file = None
    if reference:
        # An xls_ir_verilog reference has no residual data; the block IR alone
        # still pins the ids.
        is_rtl = EcoRtlInfo in reference
        reference = reference[EcoRtlInfo] if is_rtl else reference[CodegenInfo]
        report_file = ctx.actions.declare_file(basename + ".rtl.report")
        inputs += [reference.block_ir_file, reference.verilog_file]
        outputs.append(report_file)
        args += [
            "--reference_block_ir_path=" + reference.block_ir_file.path,
            "--reference_verilog_path=" + reference.verilog_file.path,
            "--output_rtl_report_path=" + report_file.path,
        ]
        if is_rtl:
            inputs.append(reference.residual_file)
            args.append("--reference_residual_data_path=" + reference.residual_file.path)
        if ctx.attr.mcs_timeout >= 0:
            args.append("--align_mcs_timeout=%d" % ctx.attr.mcs_timeout)
    args += _codegen_flags(ctx.attr.codegen, _CODEGEN_FLAGS, ctx.attr.codegen_args)
    ctx.actions.run(
        inputs = inputs + ctx.files.codegen_scheduled_main +
                 codegen_main[DefaultInfo].default_runfiles.files.to_list(),
        outputs = outputs,
        executable = ctx.executable.codegen_scheduled_main,
        arguments = args,
        mnemonic = "EcoCodegen",
        use_default_shell_env = True,
    )
    if reference:
        diff_file = ctx.actions.declare_file(basename + ".rtl.diff")
        ctx.actions.run_shell(
            inputs = [reference.verilog_file, verilog_file],
            outputs = [diff_file],
            command = "diff -u '{}' '{}' > '{}'; [ $? -le 1 ]".format(
                reference.verilog_file.path,
                verilog_file.path,
                diff_file.path,
            ),
            mnemonic = "EcoRtlDiff",
            use_default_shell_env = True,
        )
        outputs.append(diff_file)
    return [
        DefaultInfo(files = depset(outputs)),
        EcoRtlInfo(
            verilog_file = verilog_file,
            block_ir_file = block_ir_file,
            residual_file = residual_file,
            report_file = report_file,
        ),
    ]

xls_eco_verilog = rule(
    doc = "Lowers a scheduled IR to RTL with codegen_scheduled_main. With 'reference', the " +
          "block is renumbered after the reference block and emitted in its order first, so " +
          "the RTL of logic the ECO did not touch is identical to the reference RTL.",
    implementation = _xls_eco_verilog_impl,
    attrs = {
        "src": attr.label(
            mandatory = True,
            providers = [[EcoScheduleInfo], [CodegenInfo]],
            doc = "xls_patch_ir target built with 'schedule', or an xls_ir_verilog target " +
                  "(its scheduled IR and schedule are lowered again, e.g. for the reference).",
        ),
        "reference": attr.label(
            providers = [[EcoRtlInfo], [CodegenInfo]],
            doc = "Design whose RTL to preserve: the before xls_eco_verilog target, or the " +
                  "before xls_ir_verilog target when re-emitting the before design itself.",
        ),
        "codegen": attr.label(
            doc = "xls_ir_verilog target of the before design; its codegen and scheduling " +
                  "flags are reused so both RTLs are generated identically.",
        ),
        "codegen_args": attr.string_dict(
            doc = "Codegen/scheduling flags; entries here override those of 'codegen'.",
        ),
        "mcs_timeout": attr.int(
            default = -1,
            doc = "Seconds before the MCS search against the reference block stops; negative disables.",
        ),
        "verilog_file": attr.output(
            mandatory = True,
            doc = "Output Verilog file; the signature and block IR take its basename.",
        ),
        "codegen_scheduled_main": attr.label(
            executable = True,
            cfg = "target",
            default = Label("//xls/contrib/eco:codegen_scheduled_main"),
        ),
    },
)

def _xls_eco_report_impl(ctx):
    reports = [
        ctx.attr.diff[EcoPatchInfo].report,
        ctx.attr.patched[EcoScheduleInfo].report_file,
        ctx.attr.rtl[EcoRtlInfo].report_file,
    ]
    out = ctx.actions.declare_file(ctx.label.name + ".eco.report")
    ctx.actions.run_shell(
        inputs = reports,
        outputs = [out],
        command = "{ echo Diff; cat '%s'; cat '%s' '%s'; } > '%s'" % (
            reports[0].path,
            reports[1].path,
            reports[2].path,
            out.path,
        ),
        mnemonic = "EcoReport",
    )
    return [DefaultInfo(files = depset([out]))]

xls_eco_report = rule(
    doc = "The blast radius of one ECO, top to bottom: IR diff, schedule, registers and " +
          "cones, RTL lines, concatenated into <name>.eco.report.",
    implementation = _xls_eco_report_impl,
    attrs = {
        "diff": attr.label(mandatory = True, providers = [EcoPatchInfo]),
        "patched": attr.label(mandatory = True, providers = [EcoScheduleInfo]),
        "rtl": attr.label(mandatory = True, providers = [EcoRtlInfo]),
    },
)

def xls_eco(
        name,
        srcs,
        dslx_top,
        deps = None,
        codegen_args = None,
        patched_codegen_args = None,
        greedy_schedule = None,
        mcs = None,
        mcs_cutoff = None,
        mcs_optimal = None,
        mcs_timeout = None,
        eqv_method = None,
        activation_count = None,
        top = None,
        tags = None):
    """End-to-end ECO: diff two DSLX sources, patch the before IR, verify.

    Generates <name>_diff and <name>_patched_opt_ir. With `codegen_args`, the
    before IR is also scheduled and lowered by xls_ir_verilog as
    <name>_before_sv (or _v), the patch preserves that schedule, and the
    patched IR is lowered under the same options as <name>_patched_sv, with
    the before RTL as reference so untouched logic keeps its RTL text
    (<name>_patched.rtl.diff, <name>_patched.rtl.report). <name>_before_rtl
    re-emits the before design with codegen_scheduled_main to capture that
    reference; <name>_before_rtl_test checks it equals xls_ir_verilog's RTL.

    Args:
      name: Base name for generated targets.
      srcs: List of two DSLX source labels [before, after].
      dslx_top: Top name for both sources, or a list of two.
      deps: Optional xls_dslx_library deps for the sources.
      codegen_args: Optional codegen/scheduling args for xls_ir_verilog
        (pipeline_stages, delay_model, reset, ...). Enables schedule preservation.
      patched_codegen_args: Optional overrides of `codegen_args` for the patched
        side only, e.g. {"pipeline_stages": "5"} for a stage-count ECO.
      greedy_schedule: Optional bool; see xls_patch_ir.
      mcs: See xls_dslx_ir_diff.
      mcs_cutoff: See xls_dslx_ir_diff.
      mcs_optimal: See xls_dslx_ir_diff.
      mcs_timeout: See xls_dslx_ir_diff.
      eqv_method: See xls_patch_ir.
      activation_count: See xls_patch_ir.
      top: See xls_patch_ir.
      tags: Optional Bazel tags applied to generated targets.
    """
    common = {"tags": tags} if tags != None else {}
    diff = name + "_diff"
    xls_dslx_ir_diff(
        name = diff,
        srcs = srcs,
        dslx_top = dslx_top,
        deps = deps,
        mcs = mcs,
        mcs_cutoff = mcs_cutoff,
        mcs_optimal = mcs_optimal,
        mcs_timeout = mcs_timeout,
        tags = tags,
    )

    schedule = None
    before_rtl = None
    ext = None
    if codegen_args != None:
        ext = "v" if codegen_args.get("use_system_verilog", "true").lower() == "false" else "sv"
        schedule = name + "_before_" + ext
        xls_ir_verilog(
            name = schedule,
            src = ":" + diff + "_before_opt_ir",
            codegen_args = codegen_args,
            verilog_file = name + "_before." + ext,
            **common
        )
        before_rtl = name + "_before_rtl"
        xls_eco_verilog(
            name = before_rtl,
            src = ":" + schedule,
            reference = ":" + schedule,
            codegen = ":" + schedule,
            verilog_file = name + "_before_rtl." + ext,
            **common
        )
        diff_test(
            name = before_rtl + "_test",
            file1 = name + "_before." + ext,
            file2 = name + "_before_rtl." + ext,
            **common
        )

    patched_args = patched_codegen_args or {}
    patched = name + "_patched_opt_ir"
    xls_patch_ir(
        name = patched,
        ir_diff = ":" + diff,
        schedule = None if schedule == None else ":" + schedule,
        scheduling_args = {k: v for k, v in patched_args.items() if k in _SCHEDULING_FLAGS} or None,
        greedy_schedule = greedy_schedule,
        eqv_method = eqv_method,
        activation_count = activation_count,
        top = top,
        **common
    )

    if schedule != None:
        xls_eco_verilog(
            name = name + "_patched_" + ext,
            src = ":" + patched,
            reference = ":" + before_rtl,
            codegen = ":" + schedule,
            codegen_args = patched_args,
            mcs_timeout = -1 if mcs_timeout == None else mcs_timeout,
            verilog_file = name + "_patched." + ext,
            **common
        )
        xls_eco_report(
            name = name + "_report",
            diff = ":" + diff,
            patched = ":" + patched,
            rtl = ":" + name + "_patched_" + ext,
            **common
        )
