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

load("@rules_cc//cc:cc_test.bzl", "cc_test")

cc_test(
    name = "broadcast_analysis_test",
    srcs = ["broadcast_analysis_test.cc"],
    deps = [
        "//xls/contrib/mlir/transforms/linalg:linalg_all",
    ],
)

cc_test(
    name = "validation_test",
    srcs = ["validation_test.cc"],
    deps = [
        "//xls/contrib/mlir/transforms/linalg:linalg_all",
    ],
)

cc_test(
    name = "scalar_interpreter_test",
    srcs = ["scalar_interpreter_test.cc"],
    deps = [
        "//xls/contrib/mlir/transforms/linalg:linalg_all",
    ],
)

cc_test(
    name = "reduction_utils_test",
    srcs = ["reduction_utils_test.cc"],
    deps = [
        "//xls/contrib/mlir/transforms/linalg:linalg_all",
    ],
)

cc_test(
    name = "region_body_lowering_test",
    srcs = ["region_body_lowering_test.cc"],
    deps = [
        "//xls/contrib/mlir/transforms/linalg:linalg_all",
    ],
)

cc_test(
    name = "unroll_plan_test",
    srcs = ["unroll_plan_test.cc"],
    deps = [
        "//xls/contrib/mlir/transforms/linalg:linalg_all",
    ],
)

cc_test(
    name = "lane_ops_test",
    srcs = ["lane_ops_test.cc"],
    deps = [
        "//xls/contrib/mlir/transforms/linalg:linalg_all",
    ],
)

cc_test(
    name = "unroll_test",
    srcs = ["unroll_test.cc"],
    deps = [
        "//xls/contrib/mlir/transforms/linalg:linalg_all",
    ],
)

cc_test(
    name = "dslx_codegen_test",
    srcs = ["dslx_codegen_test.cc"],
    deps = [
        "//xls/contrib/mlir/transforms/linalg:linalg_all",
    ],
)


