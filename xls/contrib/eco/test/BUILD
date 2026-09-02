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
load("//xls/contrib/eco:eco_build_defs.bzl", "xls_dslx_ir_diff", "xls_patch_ir")

package(default_visibility = ["//xls:xls_internal"])

# TODO(xls-eco): Expand the benchmark suite - add larger industrial-scale
# IRs (5k-50k nodes) to characterize the hybrid pipeline's scalability ceiling.

xls_dslx_ir_diff(
    name = "aes_ctr_diff",
    srcs = [
        ":aes_ctr.org.x",
        ":aes_ctr.rev.x",
    ],
    dslx_top = "aes_ctr",
    deps = [
        "//xls/modules/aes:aes_common_dslx",
        "//xls/modules/aes:aes_dslx",
    ],
)

xls_dslx_ir_diff(
    name = "apfloat_fmac_rev_0_diff",
    srcs = [
        ":apfloat_fmac.org.x",
        ":apfloat_fmac.rev_0.x",
    ],
    dslx_top = "fmac",
)

xls_dslx_ir_diff(
    name = "apfloat_fmac_rev_1_diff",
    srcs = [
        ":apfloat_fmac.org.x",
        ":apfloat_fmac.rev_1.x",
    ],
    dslx_top = "fmac",
)

xls_dslx_ir_diff(
    name = "crc32_diff",
    srcs = [
        ":crc32.org.x",
        ":crc32.rev.x",
    ],
    dslx_top = "main",
)

xls_dslx_ir_diff(
    name = "fir_filter_fixed32_diff",
    srcs = [
        ":fir_filter_fixed32.org.x",
        ":fir_filter_fixed32.rev.x",
    ],
    dslx_top = "main",
    mcs = True,
)

xls_dslx_ir_diff(
    name = "fir_filter_fp32_diff",
    srcs = [
        ":fir_filter_fp32.org.x",
        ":fir_filter_fp32.rev.x",
    ],
    dslx_top = "main",
    mcs = True,
    mcs_optimal = False,  #Plateaus after finding 3699 mappings and is probably the optimal mapping but due to large difference  between input node counts the recursion is taking a long time to find the optimal mapping, so we set it to false for now to get the results faster.
)

xls_dslx_ir_diff(
    name = "fir_filter_fp32_proc_diff",
    srcs = [
        ":fir_filter_fp32_proc.org.x",
        ":fir_filter_fp32_proc.rev.x",
    ],
    dslx_top = "fir_filter_proc",
    mcs = True,
    mcs_optimal = False,
)

xls_dslx_ir_diff(
    name = "ghash_diff",
    srcs = [
        ":ghash.org.x",
        ":ghash.rev.x",
    ],
    dslx_top = "ghash",
    deps = ["//xls/modules/aes:aes_common_dslx"],
)

xls_dslx_ir_diff(
    name = "histogram_rev_0_diff",
    srcs = [
        ":histogram.org.x",
        ":histogram.rev_0.x",
    ],
    dslx_top = "histogram",
    mcs = True,
)

xls_dslx_ir_diff(
    name = "histogram_rev_1_diff",
    srcs = [
        ":histogram.org.x",
        ":histogram.rev_1.x",
    ],
    dslx_top = "histogram",
    mcs = True,
    # The optimal MCS recursion plateaus on this
    mcs_optimal = False,
)

xls_dslx_ir_diff(
    name = "ram_demux_diff",
    srcs = [
        ":ram_demux.org.x",
        ":ram_demux.rev.x",
    ],
    dslx_top = "ram_demux",
    deps = ["//xls/examples:ram_dslx"],
)

xls_dslx_ir_diff(
    name = "riscv_simple_diff",
    srcs = [
        ":riscv_simple.org.x",
        ":riscv_simple.rev.x",
    ],
    dslx_top = "run_instruction",
    mcs = True,
)

xls_dslx_ir_diff(
    name = "rle_enc_diff",
    srcs = [
        ":rle_enc.org.x",
        ":rle_enc.rev.x",
    ],
    dslx_top = "rle_enc",
)

xls_dslx_ir_diff(
    name = "vector_core_diff",
    srcs = [
        ":vector_core.org.x",
        ":vector_core.rev.x",
    ],
    dslx_top = "vector_core",
    mcs = True,
)

xls_patch_ir(
    name = "aes_ctr_patched_opt_ir",
    activation_count = 1,
    ir_diff = ":aes_ctr_diff",
)

xls_patch_ir(
    name = "apfloat_fmac_rev_0_patched_opt_ir",
    activation_count = 1,
    ir_diff = ":apfloat_fmac_rev_0_diff",
)

xls_patch_ir(
    name = "apfloat_fmac_rev_1_patched_opt_ir",
    activation_count = 1,
    ir_diff = ":apfloat_fmac_rev_1_diff",
)

xls_patch_ir(
    name = "crc32_patched_opt_ir",
    ir_diff = ":crc32_diff",
)

xls_patch_ir(
    name = "fir_filter_fixed32_patched_opt_ir",
    ir_diff = ":fir_filter_fixed32_diff",
)

xls_patch_ir(
    name = "fir_filter_fp32_patched_opt_ir",
    # LEC on ~5k-nodes & ~9k-edges is too expensive.
    eqv_method = "structural",
    ir_diff = ":fir_filter_fp32_diff",
)

xls_patch_ir(
    name = "fir_filter_fp32_proc_patched_opt_ir",
    eqv_method = "structural",
    ir_diff = ":fir_filter_fp32_proc_diff",
)

xls_patch_ir(
    name = "ghash_patched_opt_ir",
    activation_count = 1,
    ir_diff = ":ghash_diff",
)

xls_patch_ir(
    name = "histogram_rev_0_patched_opt_ir",
    activation_count = 1,
    ir_diff = ":histogram_rev_0_diff",
)

xls_patch_ir(
    name = "histogram_rev_1_patched_opt_ir",
    activation_count = 1,
    ir_diff = ":histogram_rev_1_diff",
)

xls_patch_ir(
    name = "ram_demux_patched_opt_ir",
    activation_count = 1,
    ir_diff = ":ram_demux_diff",
)

xls_patch_ir(
    name = "riscv_simple_patched_opt_ir",
    ir_diff = ":riscv_simple_diff",
)

xls_patch_ir(
    name = "rle_enc_patched_opt_ir",
    activation_count = 1,
    ir_diff = ":rle_enc_diff",
)

xls_patch_ir(
    name = "vector_core_patched_opt_ir",
    ir_diff = ":vector_core_diff",
)

build_test(
    name = "benchmark_build_test",
    targets = [
        ":" + name
        for name in existing_rules()
        if name.endswith("_patched_opt_ir")
    ],
)
