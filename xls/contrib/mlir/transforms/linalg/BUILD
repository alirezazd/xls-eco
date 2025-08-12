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

package(default_visibility = ["//visibility:public"])

# Core types library
cc_library(
    name = "linalg_types",
    hdrs = ["types/linalg_types.h"],
    srcs = ["types/linalg_types.cc"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash:hash",
    ],
)

# Utility functions library
cc_library(
    name = "linalg_utils",
    hdrs = ["utils/helpers.h"],
    srcs = ["utils/helpers.cc"],
    deps = [
        ":linalg_types",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:LinalgDialect",
    ],
)

# Enhanced reduction utilities library
cc_library(
    name = "linalg_reduction_utils",
    hdrs = ["utils/reduction_utils.h"],
    srcs = ["utils/reduction_utils.cc"],
    deps = [
        ":linalg_types",
        "@com_google_absl//absl/types:span",
    ],
)

# Analysis libraries
cc_library(
    name = "linalg_analysis",
    hdrs = ["analysis/linalg_analysis.h"],
    srcs = ["analysis/linalg_analysis.cc"],
    deps = [
        ":linalg_types",
        ":linalg_utils",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "linalg_broadcast",
    hdrs = ["analysis/broadcast.h"],
    srcs = ["analysis/broadcast.cc"],
    deps = [
        ":linalg_types",
        ":linalg_analysis",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "linalg_schedule",
    hdrs = ["analysis/schedule.h"],
    srcs = ["analysis/schedule.cc"],
    deps = [
        ":linalg_types",
        ":linalg_broadcast",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "linalg_unroll_plan",
    hdrs = ["analysis/unroll_plan.h"],
    srcs = ["analysis/unroll_plan.cc"],
    deps = [
        ":linalg_types",
        ":linalg_broadcast",
        "@llvm-project//mlir:Support",
    ],
)

# Codegen libraries

cc_library(
    name = "linalg_lane_ops",
    hdrs = ["codegen/lane_ops.h"],
    srcs = ["codegen/lane_ops.cc"],
    deps = [
        ":linalg_types",
        ":linalg_broadcast",
        ":linalg_utils",
        ":linalg_reduction_utils",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "linalg_dslx_codegen",
    hdrs = ["codegen/dslx_codegen.h"],
    srcs = ["codegen/dslx_codegen.cc"],
    deps = [
        ":linalg_types",
        ":linalg_broadcast",
        ":linalg_schedule",
        ":linalg_unroll_plan",
        ":linalg_reduction_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "linalg_reduction",
    hdrs = ["codegen/reduction.h"],
    srcs = ["codegen/reduction.cc"],
    deps = [
        ":linalg_types",
        ":linalg_broadcast",
        ":linalg_schedule",
        ":linalg_utils",
        ":linalg_reduction_utils",
        ":linalg_lane_ops",
        "@llvm-project//mlir:Support",
    ],
)

# Main library that includes everything
cc_library(
    name = "linalg_all",
    deps = [
        ":linalg_types",
        ":linalg_utils",
        ":linalg_reduction_utils",
        ":linalg_analysis",
        ":linalg_broadcast",
        ":linalg_schedule",
        ":linalg_unroll_plan",
        ":linalg_lane_ops",
        ":linalg_dslx_codegen",
        ":linalg_reduction",
    ],
)

# Linalg to XLS transformation
cc_library(
    name = "linalg_to_xls",
    srcs = ["linalg_to_xls.cc"],
    hdrs = ["linalg_to_xls.h"],
    deps = [
        ":linalg_all",
        "//xls/contrib/mlir:mlir_xls",
        "//xls/contrib/mlir:xls_transforms_passes_inc_gen",
        "//xls/common/file:temp_file",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect", 
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

# Note: Test targets are defined in the parent package
# to avoid cross-package references
