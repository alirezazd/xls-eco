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
    hdrs = ["linalg_types.h"],
    srcs = ["linalg_types.cc"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash:hash",
    ],
)

# Utility functions library
cc_library(
    name = "linalg_utils",
    hdrs = ["helpers.h"],
    srcs = ["helpers.cc"],
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
    hdrs = ["reduction_utils.h"],
    srcs = ["reduction_utils.cc"],
    deps = [
        ":linalg_types",
        "@com_google_absl//absl/types:span",
    ],
)

# Evaluation libraries
cc_library(
    name = "linalg_schedule_gen",
    hdrs = ["schedule_gen.h"],
    srcs = ["schedule_gen.cc"],
    deps = [
        ":linalg_types",
        ":linalg_utils",
        ":linalg_eval",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "linalg_unroll_plan_gen",
    hdrs = ["unroll_plan_gen.h"],
    srcs = ["unroll_plan_gen.cc"],
    deps = [
        ":linalg_types",
        ":linalg_utils",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "linalg_eval",
    hdrs = ["linalg_eval.h"],
    srcs = ["linalg_eval.cc"],
    deps = [
        ":linalg_types",
        ":linalg_utils",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Support",
    ],
)

# Codegen libraries
cc_library(
    name = "linalg_lane_ops",
    hdrs = ["lane_ops.h"],
    srcs = ["lane_ops.cc"],
    deps = [
        ":linalg_types",
        ":linalg_eval",
        ":linalg_utils",
        ":linalg_reduction_utils",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "linalg_dslx_codegen",
    hdrs = ["dslx_codegen.h"],
    srcs = ["dslx_codegen.cc"],
    deps = [
        ":linalg_types",
        ":linalg_eval",
        ":linalg_schedule_gen",
        ":linalg_unroll_plan_gen",
        ":linalg_reduction_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "linalg_reduction",
    hdrs = ["reduction.h"],
    srcs = ["reduction.cc"],
    deps = [
        ":linalg_types",
        ":linalg_eval",
        ":linalg_schedule_gen",
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
        ":linalg_eval",
        ":linalg_schedule_gen",
        ":linalg_unroll_plan_gen",
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

# Test target for EvalOperand
cc_binary(
    name = "simple_test",
    srcs = ["simple_test.cc"],
    deps = [
        ":linalg_eval",
        ":linalg_types",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Core",
    ],
)

# Test target for operand evaluation
cc_binary(
    name = "test_eval_operand",
    srcs = ["test_eval_operand.cc"],
    deps = [
        ":linalg_eval",
        ":linalg_types",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Core",
    ],
)

# Note: Test targets are defined in the parent package
# to avoid cross-package references
