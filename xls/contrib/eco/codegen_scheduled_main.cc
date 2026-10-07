// Copyright 2024 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_result.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/verilog_conversion.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/eco/align.h"
#include "xls/contrib/eco/block_report.h"
#include "xls/contrib/eco/match.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/tools/codegen.h"
#include "xls/tools/codegen_flags.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.h"
#include "xls/tools/scheduling_options_flags.pb.h"

ABSL_FLAG(std::string, input_schedule_path, "",
          "PackageScheduleProto textproto naming the nodes of IR_FILE, e.g. "
          "patch_ir_main --output_schedule_path.");
ABSL_FLAG(
    std::string, reference_block_ir_path, "",
    "Block IR of the before design (codegen_main --output_block_ir_path). "
    "The top block is renumbered to match it before emission, so the "
    "wires of logic the ECO did not touch keep their names; pass its "
    "residual data as --reference_residual_data_path to also replay its "
    "emission order.");
ABSL_FLAG(bool, align_mcs_optimal, false,
          "Require the optimal MCS when matching against the reference block; "
          "slow on large blocks.");
ABSL_FLAG(int, align_mcs_timeout, -1,
          "MCS timeout in seconds when matching against the reference block; "
          "negative disables it.");
ABSL_FLAG(std::string, reference_verilog_path, "",
          "Verilog of the reference design; the report then also counts the "
          "RTL lines that changed.");
ABSL_FLAG(std::string, output_rtl_report_path, "",
          "Where to write the block alignment report.");

namespace xls {
namespace {

static constexpr std::string_view kUsage = R"(
codegen_main with a given schedule instead of running the scheduler. Takes
codegen_main's codegen and scheduling flags; the scheduling flags should match
those that produced the schedule.

   codegen_scheduled_main --input_schedule_path=patched.schedule.textproto \
       --pipeline_stages=3 --delay_model=unit --reset=rst \
       --reference_block_ir_path=before.block.ir \
       --reference_residual_data_path=before.residual.textproto \
       --output_verilog_path=patched.sv patched.schedule.opt.ir
)";

// Renumbers the top block to match the reference block and emits it again.
// This runs after codegen because the reference residual data is keyed by
// ids that block finalization assigns.
absl::Status AlignAndEmit(Package* package, bool top_is_proc,
                          const CodegenFlagsProto& codegen_flags,
                          verilog::CodegenResult* result) {
  const std::string& reference_path =
      absl::GetFlag(FLAGS_reference_block_ir_path);
  XLS_ASSIGN_OR_RETURN(std::string reference_ir,
                       GetFileContents(reference_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> reference,
                       Parser::ParsePackage(reference_ir, reference_path));
  XLS_ASSIGN_OR_RETURN(Block * reference_block, reference->GetTopAsBlock());
  XLS_ASSIGN_OR_RETURN(Block * block, package->GetTopAsBlock());
  MatchOptions match_options;
  match_options.mcs_optimal = absl::GetFlag(FLAGS_align_mcs_optimal);
  match_options.mcs_timeout = absl::GetFlag(FLAGS_align_mcs_timeout);
  XLS_ASSIGN_OR_RETURN(Alignment alignment,
                       Align(block, reference_block, match_options));
  XLS_ASSIGN_OR_RETURN(BlockReport report,
                       CompareBlocks(block, reference_block));
  std::string report_text =
      absl::StrFormat("Block\n  nodes matched: %d of %d (original %d)\n",
                      alignment.matched.size(), alignment.nodes,
                      alignment.original_nodes) +
      report.ToString();

  XLS_ASSIGN_OR_RETURN(verilog::CodegenOptions codegen_options,
                       CodegenOptionsFromProto(codegen_flags));
  // codegen_main disables emit_as_pipeline for procs, and the block does not
  // record that it came from one.
  if (top_is_proc) {
    codegen_options.emit_as_pipeline(false);
  }
  result->verilog_line_map.Clear();
  result->residual_data.Clear();
  XLS_ASSIGN_OR_RETURN(result->verilog_text,
                       verilog::GenerateVerilog(block, codegen_options,
                                                &result->verilog_line_map,
                                                &result->residual_data));

  if (!absl::GetFlag(FLAGS_reference_verilog_path).empty()) {
    XLS_ASSIGN_OR_RETURN(
        std::string reference_verilog,
        GetFileContents(absl::GetFlag(FLAGS_reference_verilog_path)));
    report_text += DiffRtl(reference_verilog, result->verilog_text).ToString();
  }
  LOG(INFO) << report_text;
  if (!absl::GetFlag(FLAGS_output_rtl_report_path).empty()) {
    XLS_RETURN_IF_ERROR(SetFileContents(
        absl::GetFlag(FLAGS_output_rtl_report_path), report_text));
  }
  return absl::OkStatus();
}

absl::Status RealMain(std::string_view ir_path) {
  if (absl::GetFlag(FLAGS_input_schedule_path).empty()) {
    return absl::InvalidArgumentError("Missing --input_schedule_path flag.");
  }
  XLS_ASSIGN_OR_RETURN(std::string ir_contents, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p,
                       Parser::ParsePackage(ir_contents, ir_path));
  XLS_ASSIGN_OR_RETURN(CodegenFlagsProto codegen_flags, GetCodegenFlags());
  if (!codegen_flags.top().empty()) {
    XLS_RETURN_IF_ERROR(p->SetTopByName(codegen_flags.top()));
  }
  std::optional<FunctionBase*> top = p->GetTop();
  const bool top_is_proc = top.has_value() && (*top)->IsProc();
  XLS_ASSIGN_OR_RETURN(SchedulingOptionsFlagsProto scheduling_flags,
                       GetSchedulingOptionsFlagsProto());
  XLS_ASSIGN_OR_RETURN(bool delay_model_flag_passed,
                       IsDelayModelSpecifiedViaFlag(scheduling_flags));
  XLS_ASSIGN_OR_RETURN(PackageScheduleProto schedule_proto,
                       ParseTextProtoFile<PackageScheduleProto>(
                           absl::GetFlag(FLAGS_input_schedule_path)));
  XLS_ASSIGN_OR_RETURN(PackageSchedule schedules,
                       PackageSchedule::FromProto(p.get(), schedule_proto));
  XLS_ASSIGN_OR_RETURN(verilog::CodegenResult result,
                       Codegen(p.get(), scheduling_flags, codegen_flags,
                               delay_model_flag_passed, &schedules));
  if (!absl::GetFlag(FLAGS_reference_block_ir_path).empty()) {
    XLS_RETURN_IF_ERROR(
        AlignAndEmit(p.get(), top_is_proc, codegen_flags, &result));
  }

  if (!absl::GetFlag(FLAGS_output_block_ir_path).empty()) {
    XLS_RETURN_IF_ERROR(SetFileContents(
        absl::GetFlag(FLAGS_output_block_ir_path), p->DumpIr()));
  }
  if (!absl::GetFlag(FLAGS_output_signature_path).empty()) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(
        absl::GetFlag(FLAGS_output_signature_path), result.signature.proto()));
  }
  if (!absl::GetFlag(FLAGS_output_residual_data_path).empty()) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(
        absl::GetFlag(FLAGS_output_residual_data_path), result.residual_data));
  }
  const std::string& verilog_path = absl::GetFlag(FLAGS_output_verilog_path);
  if (!verilog_path.empty()) {
    for (int64_t i = 0; i < result.verilog_line_map.mapping_size(); ++i) {
      result.verilog_line_map.mutable_mapping(i)->set_verilog_file(
          verilog_path);
    }
  }
  if (!absl::GetFlag(FLAGS_output_verilog_line_map_path).empty()) {
    XLS_RETURN_IF_ERROR(
        SetTextProtoFile(absl::GetFlag(FLAGS_output_verilog_line_map_path),
                         result.verilog_line_map));
  }
  if (verilog_path.empty()) {
    std::cout << result.verilog_text;
    return absl::OkStatus();
  }
  return SetFileContents(verilog_path, result.verilog_text);
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(xls::kUsage, argc, argv);
  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s IR_FILE",
                                      argv[0]);
  }
  return xls::ExitStatus(xls::RealMain(positional_arguments[0]));
}
