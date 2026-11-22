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

#ifndef XLS_CONTRIB_ECO_PATCH_SCHEDULE_H_
#define XLS_CONTRIB_ECO_PATCH_SCHEDULE_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

struct OriginalSchedule {
  // Cycle assignments of the nodes that still exist; deleted ones drop out.
  ScheduleCycleMap cycles;
  std::optional<int64_t> clock_period_ps;
  int64_t stages = 0;
};

struct PatchedSchedule {
  PipelineSchedule schedule;
  int64_t pins = 0;  // Original assignments that were asked for.
  int64_t kept = 0;  // Of those, honored.
  // Clock, stages and assignments kept, each against `original`.
  std::string Report(const OriginalSchedule& original) const;
};

// `reference` is the IR the proto names, aligned with `function_base` so
// ids agree (see Align); null looks the names up in `function_base` itself.
absl::StatusOr<OriginalSchedule> OriginalScheduleFromProto(
    FunctionBase* function_base, const PackageScheduleProto& proto,
    FunctionBase* reference);

// Both reschedule `function_base` keeping as much of `original` as possible.
// Unless `options` names a clock period, they schedule at the original one,
// relaxed to the shortest the patched design can meet if that is infeasible;
// fixing the clock first keeps preservation from being bought with a slower
// clock. Inserted nodes (by id), literals, and assignments beyond the last
// stage are never pulled.
//
// Default: one scheduler run with a soft constraint per assignment; the LP
// minimizes total drift weighted by register width, then registers within
// that optimum. Never infeasible.
absl::StatusOr<PatchedSchedule> PatchSchedule(
    FunctionBase* function_base, const OriginalSchedule& original,
    const absl::flat_hash_set<int64_t>& inserted_node_ids,
    const DelayEstimator& delay_estimator, const SchedulingOptions& options);

// Greedy baseline: pins assignments one by one in node order, keeping each
// that stays feasible. 1 + #assignments scheduler runs.
absl::StatusOr<PatchedSchedule> PatchScheduleGreedy(
    FunctionBase* function_base, const OriginalSchedule& original,
    const absl::flat_hash_set<int64_t>& inserted_node_ids,
    const DelayEstimator& delay_estimator, const SchedulingOptions& options);

}  // namespace xls

#endif  // XLS_CONTRIB_ECO_PATCH_SCHEDULE_H_
