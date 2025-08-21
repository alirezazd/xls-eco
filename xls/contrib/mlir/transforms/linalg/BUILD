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

# Schedule types library
cc_library(
    name = "schedule_types",
    hdrs = ["schedule_types.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

# Configuration management library
cc_library(
    name = "linalg_config",
    hdrs = ["linalg_config.h"],
    srcs = ["linalg_config.cc"],
    deps = [":schedule_types"],
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

# Evaluation libraries
cc_library(
    name = "linalg_schedule_gen",
    hdrs = ["schedule_gen.h"],
    srcs = ["schedule_gen.cc"],
    deps = [
        ":linalg_types",
        ":schedule_types",
        ":linalg_config",
        ":linalg_utils",
        ":linalg_eval",
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
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "linalg_dslxgen",
    hdrs = ["dslxgen.h", "dslx_types.h"],
    srcs = ["dslxgen.cc"],
    deps = [
        ":linalg_types",
        ":linalg_eval",
        ":linalg_schedule_gen",
        "@com_google_absl//absl/strings",
    ],
)

# Reduction library - REMOVED (using XLS built-in reduction support)
# cc_library(
#     name = "linalg_reduction",
#     hdrs = ["reduction.h"],
#     srcs = ["reduction.cc"],
#     deps = [
#         ":linalg_types",
#         ":linalg_eval",
#         ":linalg_schedule_gen",
#         ":linalg_utils",
#         ":linalg_reduction_utils",
#         ":linalg_lane_ops",
#         "@llvm-project//mlir:Support",
#     ],
# )

# Main library that includes everything
cc_library(
    name = "linalg_all",
    deps = [
        ":linalg_types",
        ":linalg_config",
        ":linalg_utils",
        ":linalg_eval",
        ":linalg_schedule_gen",
        ":linalg_dslxgen",
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
