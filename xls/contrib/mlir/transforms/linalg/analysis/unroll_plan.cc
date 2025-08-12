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

#include "xls/contrib/mlir/transforms/linalg/analysis/unroll_plan.h"

#include <algorithm>
#include <vector>

namespace mlir::xls {

int PickUnrollDim(const Schedule& sch, const CodegenConfig& cfg) {
  // Check for explicit unroll factors in configuration
  for (const auto& [dim, factor] : cfg.unroll) {
    if (factor > 1) {
      return dim;
    }
  }
  
  // Scan innermost reduction loops first (preferred for unrolling)
  for (auto band_it = sch.bands.rbegin(); band_it != sch.bands.rend(); ++band_it) {
    const auto& band = *band_it;
    for (auto loop_it = band.loops.rbegin(); loop_it != band.loops.rend(); ++loop_it) {
      const auto& loop = *loop_it;
      if (loop.is_reduction) {
        return loop.dim;
      }
    }
  }
  
  // Fallback: pick the innermost loop if no reduction loops found
  if (!sch.bands.empty() && !sch.bands.back().loops.empty()) {
    return sch.bands.back().loops.back().dim;
  }
  
  return -1;  // No suitable dimension found
}

mlir::LogicalResult CreateLanePlan(int dim, int extent, int unroll_factor,
                                  LanePlan& plan) {
  if (unroll_factor <= 0 || extent <= 0) {
    return mlir::failure();
  }
  
  plan.dim = dim;
  
  // Clamp unroll factor to extent if it's larger than the extent
  if (unroll_factor > extent) {
    plan.lanes = extent;
  } else {
    plan.lanes = unroll_factor;
  }
  
  // Calculate main iterations and tail
  plan.main_iters = extent / plan.lanes;
  plan.tail = extent % plan.lanes;
  plan.has_tail = (plan.tail != 0);
  
  return mlir::success();
}

mlir::LogicalResult CreateUnrollPlan(const Schedule& sch, 
                                    const BroadcastAnalysis& A,
                                    const CodegenConfig& cfg,
                                    UnrollPlan& plan) {
  plan.dims.clear();
  
  // Pick the dimension to unroll
  int unroll_dim = PickUnrollDim(sch, cfg);
  if (unroll_dim == -1) {
    // No unrolling needed
    return mlir::success();
  }
  
  // Find the unroll factor for this dimension
  int unroll_factor = 1;
  auto it = cfg.unroll.find(unroll_dim);
  if (it != cfg.unroll.end()) {
    unroll_factor = it->second;
  }
  
  if (unroll_factor <= 1) {
    // No unrolling needed
    return mlir::success();
  }
  
  // Validate dimension index
  if (unroll_dim >= A.loop_extents.size()) {
    return mlir::failure();
  }
  
  // Get the extent for this dimension (only support constant extents for now)
  const SizeExpr& size_expr = A.loop_extents[unroll_dim];
  if (!size_expr.isConstant()) {
    return mlir::failure();
  }
  
  int extent = static_cast<int>(size_expr.getConstant());
  
  // Create the lane plan for this dimension
  LanePlan lane_plan;
  if (CreateLanePlan(unroll_dim, extent, unroll_factor, lane_plan).failed()) {
    return mlir::failure();
  }
  
  plan.dims.push_back(lane_plan);
  
  return mlir::success();
}

mlir::LogicalResult CreateMultiUnrollPlan(const Schedule& sch,
                                         const BroadcastAnalysis& A,
                                         const CodegenConfig& cfg,
                                         UnrollPlan& plan) {
  plan.dims.clear();
  plan.multi_lane.dims.clear();
  
  // Find all dimensions with unroll factors > 1
  std::vector<std::pair<int, int>> unroll_dims;  // (dim, factor)
  for (const auto& [dim, factor] : cfg.unroll) {
    if (factor > 1) {
      unroll_dims.push_back({dim, factor});
    }
  }
  
  // Sort by dimension index for deterministic ordering
  std::sort(unroll_dims.begin(), unroll_dims.end());
  
  if (unroll_dims.empty()) {
    // No unrolling needed
    return mlir::success();
  }
  
  // Create lane plans for each unrolled dimension
  int total_lanes = 1;
  for (const auto& [dim, factor] : unroll_dims) {
    // Validate dimension index
    if (dim >= A.loop_extents.size()) {
      return mlir::failure();
    }
    
    // Get the extent for this dimension (only support constant extents for now)
    const SizeExpr& size_expr = A.loop_extents[dim];
    if (!size_expr.isConstant()) {
      return mlir::failure();
    }
    
    int extent = static_cast<int>(size_expr.getConstant());
    
    // Create the lane plan for this dimension
    LanePlan lane_plan;
    if (CreateLanePlan(dim, extent, factor, lane_plan).failed()) {
      return mlir::failure();
    }
    
    plan.dims.push_back(lane_plan);
    plan.multi_lane.dims.push_back(lane_plan);
    total_lanes *= factor;
  }
  
  // Set the total lane count (product of all unroll factors)
  plan.multi_lane.total_lanes = total_lanes;
  
  return mlir::success();
}

}  // namespace mlir::xls
