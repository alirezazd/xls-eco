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

load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "ir_patch_proto",
    srcs = ["ir_patch.proto"],
    deps = [
        "//xls/ir:xls_type_proto",
        "//xls/ir:xls_value_proto",
    ],
)

cc_library(
    name = "patch_ir_lib",
    srcs = ["patch_ir.cc"],
    hdrs = ["patch_ir.h"],
    deps = [
        ":ir_patch_cc_proto",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:format_strings",
        "//xls/ir:node_util",
        "//xls/ir:op",
        "//xls/ir:source_location",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/ir:xls_type_cc_proto",
        "//xls/ir:xls_value_cc_proto",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_binary(
    name = "patch_ir_main",
    srcs = ["patch_ir_main.cc"],
    deps = [
        ":align",
        ":ir_patch_cc_proto",
        ":match",
        ":patch_ir_lib",
        ":patch_schedule",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/estimators",
        "//xls/estimators/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/scheduling:scheduling_options",
        "//xls/tools:schedule",
        "//xls/tools:scheduling_options_flags",
        "//xls/tools:scheduling_options_flags_cc_proto",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_library(
    name = "patch_schedule",
    srcs = ["patch_schedule.cc"],
    hdrs = ["patch_schedule.h"],
    deps = [
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/estimators/delay_model:delay_estimator",
        "//xls/ir",
        "//xls/ir:proc_elaboration",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/scheduling:run_pipeline_schedule",
        "//xls/scheduling:scheduling_options",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_binary(
    name = "codegen_scheduled_main",
    srcs = ["codegen_scheduled_main.cc"],
    deps = [
        ":align",
        ":block_report",
        ":match",
        "//xls/codegen:codegen_options",
        "//xls/codegen:codegen_result",
        "//xls/codegen:module_signature",
        "//xls/codegen:verilog_conversion",
        "//xls/codegen_v_1_5:passes",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "//xls/estimators",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/scheduling:pipeline_schedule",
        "//xls/scheduling:pipeline_schedule_cc_proto",
        "//xls/scheduling:scheduling_options",
        "//xls/tools:codegen",
        "//xls/tools:codegen_flags",
        "//xls/tools:codegen_flags_cc_proto",
        "//xls/tools:scheduling_options_flags",
        "//xls/tools:scheduling_options_flags_cc_proto",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_proto_library(
    name = "ir_patch_cc_proto",
    deps = [":ir_patch_proto"],
)

# Graph Edit Distance (GED) and Maximum Common Subgraph (MCS) implementation

cc_library(
    name = "graph",
    srcs = ["graph.cc"],
    hdrs = ["graph.h"],
    deps = [
        "//xls/ir:channel_cc_proto",
        "//xls/ir:op",
        "//xls/ir:xls_type_cc_proto",
        "//xls/ir:xls_value_cc_proto",
        "//xls/visualization/ir_viz:visualization_cc_proto",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "xls_ir_to_graph",
    srcs = ["xls_ir_to_graph.cc"],
    hdrs = ["xls_ir_to_graph.h"],
    deps = [
        ":graph",
        "//xls/common/file:filesystem",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:channel_ops",
        "//xls/ir:format_strings",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "//xls/ir:state_element",
        "//xls/ir:value",
        "//xls/visualization/ir_viz:node_attribute_visitor",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "lap_solver",
    srcs = ["lap_solver.cc"],
    hdrs = ["lap_solver.h"],
    deps = [
        "@abseil-cpp//absl/algorithm:container",
    ],
)

cc_library(
    name = "ged",
    srcs = [
        "ged.cc",
        "ged_cost_functions.cc",
    ],
    hdrs = [
        "ged.h",
        "ged_cost_functions.h",
    ],
    deps = [
        ":graph",
        ":lap_solver",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log",
    ],
)

cc_library(
    name = "mcs",
    srcs = ["mcs.cc"],
    hdrs = ["mcs.h"],
    deps = [
        ":graph",
        "//xls/common:stopwatch",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "match",
    srcs = ["match.cc"],
    hdrs = ["match.h"],
    deps = [
        ":ged",
        ":graph",
        ":mcs",
        "//xls/common:stopwatch",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "align",
    srcs = ["align.cc"],
    hdrs = ["align.h"],
    deps = [
        ":graph",
        ":match",
        ":xls_ir_to_graph",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@re2",
    ],
)

cc_library(
    name = "block_report",
    srcs = ["block_report.cc"],
    hdrs = ["block_report.h"],
    deps = [
        ":graph",
        ":xls_ir_to_graph",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:register",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@re2",
    ],
)

cc_test(
    name = "block_report_test",
    size = "small",
    srcs = ["block_report_test.cc"],
    deps = [
        ":block_report",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "align_test",
    size = "small",
    srcs = ["align_test.cc"],
    deps = [
        ":align",
        ":match",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:verifier",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "mcs_test",
    size = "small",
    srcs = ["mcs_test.cc"],
    deps = [
        ":mcs",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "ged_test",
    size = "small",
    srcs = ["ged_test.cc"],
    deps = [
        ":ged",
        ":graph",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "xls_ir_to_graph_test",
    size = "small",
    srcs = ["xls_ir_to_graph_test.cc"],
    deps = [
        ":graph",
        ":xls_ir_to_graph",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:op",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "struct_equivalence",
    srcs = ["struct_equivalence.cc"],
    hdrs = ["struct_equivalence.h"],
    deps = [
        ":graph",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "structural_hash_test",
    size = "small",
    srcs = ["structural_hash_test.cc"],
    deps = [
        ":graph",
        ":struct_equivalence",
        ":xls_ir_to_graph",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/ir:ir_test_base",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "viz_attribute_contract_test",
    size = "small",
    srcs = ["viz_attribute_contract_test.cc"],
    deps = [
        "//xls/common:proto_test_utils",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:channel_ops",
        "//xls/ir:function_builder",
        "//xls/ir:ir_test_base",
        "//xls/ir:value",
        "//xls/visualization/ir_viz:node_attribute_visitor",
        "//xls/visualization/ir_viz:visualization_cc_proto",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "ir_patch_gen_lib",
    srcs = ["ir_patch_gen.cc"],
    hdrs = ["ir_patch_gen.h"],
    deps = [
        ":ged",
        ":graph",
        ":ir_patch_cc_proto",
        "//xls/ir:op",
    ],
)

cc_binary(
    name = "ged_main",
    srcs = ["ged_main.cc"],
    deps = [
        ":ged",
        ":ir_patch_gen_lib",
        ":match",
        ":xls_ir_to_graph",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/status:status_macros",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:globals",
        "@abseil-cpp//absl/log:vlog_is_on",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
)

cc_binary(
    name = "check_ir_struct_equivalence_main",
    srcs = ["check_ir_struct_equivalence_main.cc"],
    deps = [
        ":graph",
        ":struct_equivalence",
        ":xls_ir_to_graph",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/status:status_macros",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
    ],
)
