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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_SCHEDULE_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_SCHEDULE_H_

#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"

namespace mlir::xls {

// Generates loop schedules for linalg.generic operations.
// Applies loop ordering policies, tiling, and unrolling based on configuration.
mlir::LogicalResult GenerateSchedule(const LinalgGeneric& g, const BroadcastAnalysis& A,
                                    const CodegenConfig& config, Schedule& schedule);

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_SCHEDULE_H_
