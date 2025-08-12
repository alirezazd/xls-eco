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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_UNROLL_PLAN_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_UNROLL_PLAN_H_

#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"

namespace mlir::xls {

// Picks the dimension to unroll based on schedule and configuration.
// Prioritizes explicit unroll factors, then innermost reduction loops.
int PickUnrollDim(const Schedule& sch, const CodegenConfig& cfg);

// Creates a single-dimensional unroll plan for the given schedule and configuration.
// Supports one unrolled dimension with main iterations and tail handling.
mlir::LogicalResult CreateUnrollPlan(const Schedule& sch, 
                                    const BroadcastAnalysis& A,
                                    const CodegenConfig& cfg,
                                    UnrollPlan& plan);

// Creates a multi-dimensional unroll plan supporting multiple unrolled dimensions.
// Handles lane count as product of individual unroll factors.
mlir::LogicalResult CreateMultiUnrollPlan(const Schedule& sch,
                                         const BroadcastAnalysis& A,
                                         const CodegenConfig& cfg,
                                         UnrollPlan& plan);

// Creates a lane plan for a specific dimension with given extent and unroll factor.
// Handles main iterations, tail processing, and lane count calculation.
mlir::LogicalResult CreateLanePlan(int dim, int extent, int unroll_factor,
                                  LanePlan& plan);

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_UNROLL_PLAN_H_
