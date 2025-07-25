# Copyright 2024 The XLS Authors
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
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

cc_library(
    name = "run_routines",
    srcs = ["run_routines.cc"],
    hdrs = ["run_routines.h"],
    deps = [
        ":test_xml",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/data_structures:inline_bitmap",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:create_import_data",
        "//xls/dslx:error_printer",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx:interp_value_utils",
        "//xls/dslx:mangle",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_cache",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/bytecode:bytecode_interpreter_options",
        "//xls/dslx/bytecode:proc_hierarchy_interpreter",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:bindings",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/ir_convert:conversion_info",
        "//xls/dslx/ir_convert:convert_options",
        "//xls/dslx/ir_convert:ir_converter",
        "//xls/dslx/type_system:parametric_env",
        "//xls/dslx/type_system:type",
        "//xls/dslx/type_system:type_info",
        "//xls/interpreter:random_value",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:format_preference",
        "//xls/ir:op",
        "//xls/ir:type",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/passes",
        "//xls/passes:dce_pass",
        "//xls/passes:optimization_pass",
        "//xls/passes:optimization_pass_pipeline",
        "//xls/passes:pass_base",
        "//xls/solvers:z3_ir_translator",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@re2",
    ],
)

cc_test(
    name = "run_routines_test",
    srcs = ["run_routines_test.cc"],
    deps = [
        ":ir_test_runner",
        ":run_comparator",
        ":run_routines",
        "//xls/common:xls_gunit_main",
        "//xls/common/file:filesystem",
        "//xls/common/file:temp_file",
        "//xls/common/status:matchers",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/dslx:virtualizable_file_system",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/type_system:type",
        "//xls/ir",
        "//xls/ir:bits",
        "//xls/ir:ir_parser",
        "//xls/ir:value",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
        "@re2",
    ],
)

cc_library(
    name = "test_xml",
    srcs = ["test_xml.cc"],
    hdrs = ["test_xml.h"],
    deps = [
        "//xls/common:indent",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "test_xml_test",
    srcs = ["test_xml_test.cc"],
    deps = [
        ":test_xml",
        "//xls/common:xls_gunit_main",
        "@com_google_absl//absl/time",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "run_comparator",
    srcs = ["run_comparator.cc"],
    hdrs = ["run_comparator.h"],
    deps = [
        ":run_routines",
        "//xls/common:test_macros",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:interp_value",
        "//xls/dslx:mangle",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/type_system:parametric_env",
        "//xls/interpreter:ir_interpreter",
        "//xls/ir",
        "//xls/ir:events",
        "//xls/ir:value",
        "//xls/jit:function_jit",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ir_test_runner",
    srcs = ["ir_test_runner.cc"],
    hdrs = ["ir_test_runner.h"],
    deps = [
        ":run_routines",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:warning_kind",
        "//xls/dslx/bytecode:bytecode_interpreter_options",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:proc",
        "//xls/dslx/ir_convert:conversion_info",
        "//xls/dslx/ir_convert:convert_options",
        "//xls/dslx/ir_convert:ir_converter",
        "//xls/dslx/type_system:type_info",
        "//xls/interpreter:evaluator_options",
        "//xls/interpreter:interpreter_proc_runtime",
        "//xls/interpreter:ir_interpreter",
        "//xls/interpreter:proc_runtime",
        "//xls/ir",
        "//xls/ir:channel",
        "//xls/ir:events",
        "//xls/ir:format_preference",
        "//xls/ir:proc_elaboration",
        "//xls/ir:value",
        "//xls/jit:function_jit",
        "//xls/jit:jit_proc_runtime",
        "//xls/passes:dfe_pass",
        "//xls/passes:optimization_pass",
        "//xls/passes:pass_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)
