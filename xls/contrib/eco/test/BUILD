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

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//xls/contrib/eco:eco_build_defs.bzl", "xls_dslx_ir_diff", "xls_eco", "xls_eco_verilog", "xls_patch_ir")

package(default_visibility = ["//xls:xls_internal"])

# TODO(xls-eco): Expand the benchmark suite - add larger industrial-scale
# IRs (5k-50k nodes) to characterize the hybrid pipeline's scalability ceiling.

# Each design: diff -> schedule-preserving patch -> RTL, all under the same
# codegen options as the before design (see xls_eco).

xls_eco(
    name = "aes_ctr",
    srcs = [
        ":aes_ctr.org.x",
        ":aes_ctr.rev.x",
    ],
    activation_count = 1,
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "4",
        "reset": "rst",
    },
    dslx_top = "aes_ctr",
    deps = [
        "//xls/modules/aes:aes_common_dslx",
        "//xls/modules/aes:aes_dslx",
    ],
)

xls_eco(
    name = "apfloat_fmac_rev_0",
    srcs = [
        ":apfloat_fmac.org.x",
        ":apfloat_fmac.rev_0.x",
    ],
    activation_count = 1,
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "3",
        "reset": "rst",
    },
    dslx_top = "fmac",
)

xls_eco(
    name = "apfloat_fmac_rev_1",
    srcs = [
        ":apfloat_fmac.org.x",
        ":apfloat_fmac.rev_1.x",
    ],
    activation_count = 1,
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "3",
        "reset": "rst",
    },
    dslx_top = "fmac",
)

xls_eco(
    name = "crc32",
    srcs = [
        ":crc32.org.x",
        ":crc32.rev.x",
    ],
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "3",
        "reset": "rst",
    },
    dslx_top = "main",
)

xls_eco(
    name = "fir_filter_fixed32",
    srcs = [
        ":fir_filter_fixed32.org.x",
        ":fir_filter_fixed32.rev.x",
    ],
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "4",
        "reset": "rst",
    },
    dslx_top = "main",
    mcs = True,
)

# MCS plateaus after ~3.7k mappings on this size imbalance; LEC on ~5k nodes is too expensive.
xls_eco(
    name = "fir_filter_fp32",
    srcs = [
        ":fir_filter_fp32.org.x",
        ":fir_filter_fp32.rev.x",
    ],
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "5",
        "reset": "rst",
    },
    dslx_top = "main",
    eqv_method = "structural",
    mcs = True,
    mcs_optimal = False,
)

xls_eco(
    name = "fir_filter_fp32_proc",
    srcs = [
        ":fir_filter_fp32_proc.org.x",
        ":fir_filter_fp32_proc.rev.x",
    ],
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "4",
        "reset": "rst",
    },
    dslx_top = "fir_filter_proc",
    eqv_method = "structural",
    mcs = True,
    mcs_optimal = False,
)

xls_eco(
    name = "ghash",
    srcs = [
        ":ghash.org.x",
        ":ghash.rev.x",
    ],
    activation_count = 1,
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "4",
        "reset": "rst",
    },
    dslx_top = "ghash",
    deps = ["//xls/modules/aes:aes_common_dslx"],
)

xls_eco(
    name = "histogram_rev_0",
    srcs = [
        ":histogram.org.x",
        ":histogram.rev_0.x",
    ],
    activation_count = 1,
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
    },
    dslx_top = "histogram",
    mcs = True,
)

# The optimal MCS recursion plateaus on this.
xls_eco(
    name = "histogram_rev_1",
    srcs = [
        ":histogram.org.x",
        ":histogram.rev_1.x",
    ],
    activation_count = 1,
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
    },
    dslx_top = "histogram",
    mcs = True,
    mcs_optimal = False,
)

xls_eco(
    name = "ram_demux",
    srcs = [
        ":ram_demux.org.x",
        ":ram_demux.rev.x",
    ],
    activation_count = 1,
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
        # State loop cannot close in one cycle at two stages.
        "worst_case_throughput": "2",
    },
    dslx_top = "ram_demux",
    deps = ["//xls/examples:ram_dslx"],
)

xls_eco(
    name = "riscv_simple",
    srcs = [
        ":riscv_simple.org.x",
        ":riscv_simple.rev.x",
    ],
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "4",
        "reset": "rst",
    },
    dslx_top = "run_instruction",
    mcs = True,
)

xls_eco(
    name = "rle_enc",
    srcs = [
        ":rle_enc.org.x",
        ":rle_enc.rev.x",
    ],
    activation_count = 1,
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "2",
        "reset": "rst",
    },
    dslx_top = "rle_enc",
)

xls_eco(
    name = "vector_core",
    srcs = [
        ":vector_core.org.x",
        ":vector_core.rev.x",
    ],
    codegen_args = {
        "generator": "pipeline",
        "delay_model": "unit",
        "pipeline_stages": "4",
        "reset": "rst",
    },
    dslx_top = "vector_core",
    mcs = True,
)

# The pieces xls_eco composes, spelled out on one function and one proc: diff,
# patch under the before schedule with the logical (Z3) equivalence check that
# xls_eco leaves at `structural`, RTL against the before RTL.
xls_dslx_ir_diff(
    name = "crc32_pieces_diff",
    srcs = [
        ":crc32.org.x",
        ":crc32.rev.x",
    ],
    dslx_top = "main",
)

xls_patch_ir(
    name = "crc32_pieces_patched_opt_ir",
    eqv_method = "logical",
    ir_diff = ":crc32_pieces_diff",
    schedule = ":crc32_before_sv",
)

xls_eco_verilog(
    name = "crc32_pieces_patched_sv",
    src = ":crc32_pieces_patched_opt_ir",
    codegen = ":crc32_before_sv",
    reference = ":crc32_before_rtl",
    verilog_file = "crc32_pieces_patched.sv",
)

xls_dslx_ir_diff(
    name = "rle_enc_pieces_diff",
    srcs = [
        ":rle_enc.org.x",
        ":rle_enc.rev.x",
    ],
    dslx_top = "rle_enc",
)

xls_patch_ir(
    name = "rle_enc_pieces_patched_opt_ir",
    activation_count = 1,
    eqv_method = "logical",
    ir_diff = ":rle_enc_pieces_diff",
    schedule = ":rle_enc_before_sv",
)

xls_eco_verilog(
    name = "rle_enc_pieces_patched_sv",
    src = ":rle_enc_pieces_patched_opt_ir",
    codegen = ":rle_enc_before_sv",
    reference = ":rle_enc_before_rtl",
    verilog_file = "rle_enc_pieces_patched.sv",
)

build_test(
    name = "benchmark_build_test",
    targets = [
        ":" + name
        for name in existing_rules()
        if name.endswith(("_patched_opt_ir", "_patched_sv", "_patched_v"))
    ],
)
