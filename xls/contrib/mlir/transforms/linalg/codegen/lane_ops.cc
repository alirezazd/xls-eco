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

#include "xls/contrib/mlir/transforms/linalg/codegen/lane_ops.h"

#include <algorithm>
#include <array>
#include <vector>

#include "xls/contrib/mlir/transforms/linalg/utils/helpers.h"

namespace mlir::xls {

std::array<int, 4> MakeLaneOperandIndex(int operand, 
                                       const std::array<int, 4>& current_indices,
                                       int unroll_dim, int base, int lane_u,
                                       const BroadcastAnalysis& A) {
  std::array<int, 4> result = {0, 0, 0, 0};
  
  if (operand < 0 || operand >= A.indexers.size()) {
    return result;
  }
  
  const auto& indexer = A.indexers[operand];
  for (int axis = 0; axis < indexer.results.size() && axis < 4; ++axis) {
    const auto& result_expr = indexer.results[axis];
    if (std::holds_alternative<int>(result_expr)) {
      int dim_id = std::get<int>(result_expr);
      if (dim_id == unroll_dim) {
        result[axis] = base + lane_u;
      } else if (dim_id >= 0 && dim_id < 4) {
        result[axis] = current_indices[dim_id];
      }
    }
  }
  
  return result;
}

std::array<int, 4> MakeMultiLaneOperandIndex(int operand,
                                            const std::array<int, 4>& current_indices,
                                            const MultiLanePlan& multi_plan,
                                            const std::vector<int>& bases,
                                            int lane_id,
                                            const BroadcastAnalysis& A) {
  std::array<int, 4> result = current_indices;
  
  if (operand < 0 || operand >= A.indexers.size()) {
    return result;
  }
  
  std::vector<int> lane_offsets = multi_plan.ComputeLaneOffsets(lane_id);
  
  for (int i = 0; i < multi_plan.dims.size() && i < bases.size() && i < lane_offsets.size(); ++i) {
    int unroll_dim = multi_plan.dims[i].dim;
    int base = bases[i];
    int lane_u = lane_offsets[i];
    result[unroll_dim] = base + lane_u;
  }
  
  const auto& indexer = A.indexers[operand];
  for (int axis = 0; axis < indexer.results.size() && axis < 4; ++axis) {
    const auto& result_expr = indexer.results[axis];
    if (std::holds_alternative<int>(result_expr)) {
      int dim_id = std::get<int>(result_expr);
      
      bool is_unrolled = false;
      for (const auto& lane_plan : multi_plan.dims) {
        if (dim_id == lane_plan.dim) {
          is_unrolled = true;
          break;
        }
      }
      
      if (!is_unrolled && dim_id >= 0 && dim_id < 4) {
        result[axis] = current_indices[dim_id];
      }
    }
  }
  
  return result;
}

void CombineLanes(AccState& st, const std::vector<float>& lane_values, 
                 const ReduceCtx& rctx) {
  ReduceUpdate(st, absl::MakeSpan(lane_values), rctx);
}

float EvaluateRegionForLane(const LinalgGeneric& g, const BroadcastAnalysis& A,
                           const std::array<int, 4>& current_indices,
                           int unroll_dim, int base, int lane_u,
                           const AccState& st) {
  std::array<int, 4> lane_indices = MakeLaneOperandIndex(0, current_indices, 
                                                        unroll_dim, base, lane_u, A);
  
  float result = 0.0f;
  
  for (int o = 0; o < g.operands.size(); ++o) {
    if (!g.operands[o].is_output) {
      float val = Load(o, lane_indices);
      result += val;
    }
  }
  
  result += st.acc;
  
  return result;
}

mlir::LogicalResult CreateLaneBundle(const LinalgGeneric& g, 
                                    const BroadcastAnalysis& A,
                                    int unroll_dim, int lanes,
                                    LaneBundle& bundle) {
  bundle.lanes = lanes;
  bundle.exprs.clear();
  bundle.exprs.reserve(lanes);
  
  for (int u = 0; u < lanes; ++u) {
    ReducedExpr expr;
    expr.expr_string = "lane_" + std::to_string(u) + "_expr";
    bundle.exprs.push_back(expr);
  }
  
  return mlir::success();
}

std::vector<float> EvaluateLaneBundle(const LaneBundle& bundle,
                                     const std::array<int, 4>& current_indices,
                                     int unroll_dim, int base,
                                     const AccState& st) {
  std::vector<float> results;
  results.reserve(bundle.lanes);
  
  for (int u = 0; u < bundle.lanes; ++u) {
    float result = static_cast<float>(u) + st.acc;
    results.push_back(result);
  }
  
  return results;
}

}  // namespace mlir::xls
