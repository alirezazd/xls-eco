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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_CODEGEN_LANE_OPS_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_CODEGEN_LANE_OPS_H_

#include <array>
#include <vector>

#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"
#include "xls/contrib/mlir/transforms/linalg/utils/reduction_utils.h"

namespace mlir::xls {

// Per-lane indexing: substitute loop_var[d*] = base + u for lane u
std::array<int, 4> MakeLaneOperandIndex(int operand, 
                                       const std::array<int, 4>& current_indices,
                                       int unroll_dim, int base, int lane_u,
                                       const BroadcastAnalysis& A);

// Multi-dimensional per-lane indexing: substitute multiple loop_vars
std::array<int, 4> MakeMultiLaneOperandIndex(int operand,
                                            const std::array<int, 4>& current_indices,
                                            const MultiLanePlan& multi_plan,
                                            const std::vector<int>& bases,
                                            int lane_id,
                                            const BroadcastAnalysis& A);

// Lane combination using reduction utilities
void CombineLanes(AccState& st, const std::vector<float>& lane_values, 
                 const ReduceCtx& rctx);

// Evaluate region for a specific lane
float EvaluateRegionForLane(const LinalgGeneric& g, const BroadcastAnalysis& A,
                           const std::array<int, 4>& current_indices,
                           int unroll_dim, int base, int lane_u,
                           const AccState& st);

// Create lane bundle for unrolled dimension
mlir::LogicalResult CreateLaneBundle(const LinalgGeneric& g, 
                                    const BroadcastAnalysis& A,
                                    int unroll_dim, int lanes,
                                    LaneBundle& bundle);

// Evaluate lane bundle for all lanes
std::vector<float> EvaluateLaneBundle(const LaneBundle& bundle,
                                     const std::array<int, 4>& current_indices,
                                     int unroll_dim, int base,
                                     const AccState& st);

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_CODEGEN_LANE_OPS_H_
