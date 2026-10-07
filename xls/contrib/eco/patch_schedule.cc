// Copyright 2025 The XLS Authors
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

#include "xls/contrib/eco/patch_schedule.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace {

// Original assignments worth pinning, in node order so results are
// reproducible.
std::vector<NodeInCycleConstraint> Pins(
    FunctionBase* function_base, const ScheduleCycleMap& original,
    const absl::flat_hash_set<int64_t>& inserted_node_ids,
    const SchedulingOptions& options) {
  std::vector<NodeInCycleConstraint> pins;
  for (Node* node : function_base->nodes()) {
    auto it = original.find(node);
    if (it == original.end() || node->Is<Literal>() ||
        inserted_node_ids.contains(node->id()) ||
        it->second >= options.pipeline_stages().value_or(INT64_MAX)) {
      continue;
    }
    pins.push_back(NodeInCycleConstraint(node, it->second));
  }
  return pins;
}

// RunPipelineSchedule with the target clock fixed and the elaboration
// new-style procs require.
struct Scheduler {
  FunctionBase* function_base;
  const DelayEstimator* delay_estimator;
  std::optional<ProcElaboration> elab;
  SchedulingOptions options;

  static absl::StatusOr<Scheduler> Create(
      FunctionBase* function_base, const DelayEstimator& delay_estimator,
      const SchedulingOptions& options,
      std::optional<int64_t> original_clock_period_ps) {
    Scheduler scheduler{function_base, &delay_estimator, std::nullopt, options};
    if (function_base->IsProc() &&
        function_base->AsProcOrDie()->is_new_style_proc()) {
      XLS_ASSIGN_OR_RETURN(scheduler.elab, ProcElaboration::Elaborate(
                                               function_base->AsProcOrDie()));
    }
    if (!options.clock_period_ps().has_value() &&
        original_clock_period_ps.has_value()) {
      // The original clock if the patched design can meet it, else the
      // shortest it can. Without fixing it, hard pins would relax the clock
      // instead of yielding.
      SchedulingOptions probe = options;
      probe.clock_period_ps(*original_clock_period_ps)
          .minimize_clock_on_failure(true)
          .recover_after_minimizing_clock(true);
      XLS_ASSIGN_OR_RETURN(PipelineSchedule free, scheduler.Run(probe));
      XLS_RET_CHECK(free.target_clock_period_ps().has_value());
      scheduler.options.clock_period_ps(*free.target_clock_period_ps());
      VLOG(1) << "Target clock period " << *free.target_clock_period_ps()
              << " ps (original " << *original_clock_period_ps << " ps)";
    }
    return scheduler;
  }

  absl::StatusOr<PipelineSchedule> Run(const SchedulingOptions& options) const {
    std::optional<const ProcElaboration*> elab_ptr;
    if (elab.has_value()) {
      elab_ptr = &*elab;
    }
    return RunPipelineSchedule(function_base, *delay_estimator, options,
                               elab_ptr);
  }

  // Tries `pins` in order on top of `accepted`, keeping each that stays
  // feasible; `schedule` is the result under `accepted` so far.
  absl::StatusOr<PatchedSchedule> PinGreedily(
      absl::Span<const NodeInCycleConstraint> pins, SchedulingOptions accepted,
      PipelineSchedule schedule) const {
    int64_t kept = 0;
    for (const NodeInCycleConstraint& pin : pins) {
      SchedulingOptions candidate = accepted;
      candidate.add_constraint(pin);
      absl::StatusOr<PipelineSchedule> result = Run(candidate);
      if (!result.ok()) {
        VLOG(1) << "Skipping constraint for " << pin.GetNode()->GetName()
                << ": schedule infeasible";
        continue;
      }
      accepted = std::move(candidate);
      schedule = std::move(*result);
      ++kept;
    }
    XLS_RETURN_IF_ERROR(schedule.Verify());
    VLOG(1) << "Schedule constraints kept: " << kept << " of " << pins.size();
    return PatchedSchedule{std::move(schedule),
                           static_cast<int64_t>(pins.size()), kept};
  }
};

}  // namespace

absl::StatusOr<OriginalSchedule> OriginalScheduleFromProto(
    FunctionBase* function_base, const PackageScheduleProto& proto,
    FunctionBase* reference) {
  auto it = proto.schedules().find(function_base->name());
  if (it == proto.schedules().end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("No schedule for ", function_base->name()));
  }
  const PipelineScheduleProto& schedule = it->second;
  OriginalSchedule original;
  for (const StageProto& stage : schedule.stages()) {
    for (const TimedNodeProto& timed_node : stage.timed_nodes()) {
      std::optional<Node*> node =
          function_base->MaybeGetNode(timed_node.node());
      if (reference != nullptr) {
        std::optional<Node*> named = reference->MaybeGetNode(timed_node.node());
        node =
            named.has_value()
                ? function_base->GetNodeById((*named)->id()).value_or(nullptr)
                : nullptr;
      }
      if (node.value_or(nullptr) != nullptr) {
        original.cycles[*node] = stage.stage();
      }
    }
  }
  if (schedule.has_target_clock_period_ps()) {
    original.clock_period_ps = schedule.target_clock_period_ps();
  } else if (schedule.has_min_clock_period_ps()) {
    original.clock_period_ps = schedule.min_clock_period_ps();
  }
  original.stages = schedule.stages_size();
  return original;
}

std::string PatchedSchedule::Report(const OriginalSchedule& original) const {
  return absl::StrFormat(
      "  clock: %d ps (original %d ps)\n"
      "  stages: %d (original %d)\n"
      "  assignments kept: %d of %d\n",
      schedule.target_clock_period_ps().value_or(0),
      original.clock_period_ps.value_or(0), schedule.length(), original.stages,
      kept, pins);
}

absl::StatusOr<PatchedSchedule> PatchSchedule(
    FunctionBase* function_base, const OriginalSchedule& original,
    const absl::flat_hash_set<int64_t>& inserted_node_ids,
    const DelayEstimator& delay_estimator, const SchedulingOptions& options) {
  XLS_ASSIGN_OR_RETURN(Scheduler scheduler,
                       Scheduler::Create(function_base, delay_estimator,
                                         options, original.clock_period_ps));
  std::vector<NodeInCycleConstraint> pins = Pins(
      function_base, original.cycles, inserted_node_ids, scheduler.options);
  // Drift is weighted by register width, so wide values stay put first.
  SchedulingOptions soft = scheduler.options;
  for (const NodeInCycleConstraint& pin : pins) {
    double bits =
        std::max<int64_t>(pin.GetNode()->GetType()->GetFlatBitCount(), 1);
    soft.add_constraint(
        SoftNodeInCycleConstraint(pin.GetNode(), pin.GetCycle(), bits));
  }
  XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule, scheduler.Run(soft));
  int64_t kept = absl::c_count_if(pins, [&](const NodeInCycleConstraint& pin) {
    return schedule.cycle(pin.GetNode()) == pin.GetCycle();
  });
  VLOG(1) << "Schedule assignments kept: " << kept << " of " << pins.size();
  return PatchedSchedule{std::move(schedule), static_cast<int64_t>(pins.size()),
                         kept};
}

absl::StatusOr<PatchedSchedule> PatchScheduleGreedy(
    FunctionBase* function_base, const OriginalSchedule& original,
    const absl::flat_hash_set<int64_t>& inserted_node_ids,
    const DelayEstimator& delay_estimator, const SchedulingOptions& options) {
  XLS_ASSIGN_OR_RETURN(Scheduler scheduler,
                       Scheduler::Create(function_base, delay_estimator,
                                         options, original.clock_period_ps));
  XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                       scheduler.Run(scheduler.options));
  return scheduler.PinGreedily(Pins(function_base, original.cycles,
                                    inserted_node_ids, scheduler.options),
                               scheduler.options, std::move(schedule));
}

}  // namespace xls
