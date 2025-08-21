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

#include "xls/contrib/mlir/transforms/linalg/schedule_gen.h"

#include <algorithm>
#include <vector>

#include "xls/contrib/mlir/transforms/linalg/linalg_config.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_types.h"

namespace mlir::xls {

// Internal helper functions
static bool IsLastResultVar(const AffineMap& m, int dim_id) {
  if (m.results.empty()) return false;
  auto& r = m.results.back();
  return r.kind == AffineExpr::kVar && r.var == dim_id;
}

static bool OperandVariesInDim(const BroadcastAnalysis& A, int dim_id,
                               int op_idx) {
  return A.dims.operand_varies[dim_id][op_idx];
}

static int64_t GetDimensionExtent(const BroadcastAnalysis& A, int dim_id) {
  return A.loop_extents[dim_id].isConstant()
             ? A.loop_extents[dim_id].getConstant()
             : 1;
}

static int CountParallel(const LinalgGeneric& g) {
  int c = 0;
  for (auto& d : g.dims)
    if (d.kind == IterKind::kParallel) ++c;
  return c;
}

// Helper to separate parallel and reduction dimensions (avoid duplication)
static std::pair<std::vector<int>, std::vector<int>>
SeparateParallelAndReductionDims(const LinalgGeneric& g) {
  std::vector<int> parallel_dims, reduction_dims;
  const int D = g.dims.size();

  for (int i = 0; i < D; ++i) {
    if (g.dims[i].kind == IterKind::kParallel) {
      parallel_dims.push_back(i);
    } else {
      reduction_dims.push_back(i);
    }
  }

  return {std::move(parallel_dims), std::move(reduction_dims)};
}

// Loop ordering strategies
std::vector<int> BuildInitialOrder(int num_dims) {
  std::vector<int> order(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    order[i] = i;
  }
  return order;
}

std::vector<int> ApplyParallelOuterReductionInnerOrder(const LinalgGeneric& g) {
  auto [parallel_dims, reduction_dims] = SeparateParallelAndReductionDims(g);

  // Build order: parallel dimensions first, then reduction dimensions
  const int D = g.dims.size();
  std::vector<int> order(D);
  int idx = 0;
  for (int dim : parallel_dims) {
    order[idx++] = dim;
  }
  for (int dim : reduction_dims) {
    order[idx++] = dim;
  }
  return order;
}

// Dimension scoring functions
int ScoreParallelDim(const LinalgGeneric& g, const BroadcastAnalysis& A,
                     int dim_id, int preferred_vec = 8) {
  int score = 0;

  // Score outputs (weight 2) - contiguous writes
  for (int oi = 0; oi < (int)g.operands.size(); ++oi) {
    auto& op = g.operands[oi];
    if (op.is_output && IsLastResultVar(op.map, dim_id)) {
      score += 2;
    }
  }

  // Score inputs (weight 1) - contiguous reads + broadcast bonus
  for (int oi = 0; oi < (int)g.operands.size(); ++oi) {
    auto& op = g.operands[oi];
    if (op.is_output) continue;
    if (IsLastResultVar(op.map, dim_id)) {
      score += 1;
    }
    if (!OperandVariesInDim(A, dim_id, oi)) {
      score += 1;  // Broadcast bonus
    }
  }

  // Vector alignment bonus
  auto extent = GetDimensionExtent(A, dim_id);
  if (preferred_vec > 1 && extent % preferred_vec == 0) {
    score += 1;
  }

  return score;
}

int ScoreReductionDim(const LinalgGeneric& g, const BroadcastAnalysis& A,
                      int dim_id, int preferred_vec = 8) {
  int score = 0;
  // Contiguous in inputs?
  for (auto& op : g.operands) {
    if (op.is_output) continue;
    if (IsLastResultVar(op.map, dim_id))
      score += 2;  // weight contiguity higher
  }
  auto extent = GetDimensionExtent(A, dim_id);
  if (extent % preferred_vec == 0) score += 1;
  if (extent >= 32) score += 1;
  return score;
}

std::vector<int> ApplyHeuristicCacheFriendlyOrder(const LinalgGeneric& g,
                                                  const BroadcastAnalysis& A) {
  auto [parallel_dims, reduction_dims] = SeparateParallelAndReductionDims(g);
  const int D = g.dims.size();

  // Score ascending: worst first, best last (so best becomes innermost)
  std::stable_sort(
      parallel_dims.begin(), parallel_dims.end(), [&](int a, int b) {
        return ScoreParallelDim(g, A, a) < ScoreParallelDim(g, A, b);
      });

  std::stable_sort(
      reduction_dims.begin(), reduction_dims.end(), [&](int a, int b) {
        return ScoreReductionDim(g, A, a) < ScoreReductionDim(g, A, b);
      });

  // Build final order: all P (worst→best), then all R (worst→best)
  // => innermost overall = last R, i.e., best reduction
  std::vector<int> order(D);
  int idx = 0;
  for (int dim : parallel_dims) order[idx++] = dim;
  for (int dim : reduction_dims) order[idx++] = dim;
  return order;
}

std::vector<int> ApplyLoopOrderingPolicy(
    const LinalgGeneric& g, const BroadcastAnalysis& A,
    LinalgCodegenConfig::LoopOrder loop_order) {
  switch (loop_order) {
    case LinalgCodegenConfig::LoopOrder::kParallelOuter_ReductionInner:
      return ApplyParallelOuterReductionInnerOrder(g);
    case LinalgCodegenConfig::LoopOrder::kHeuristicCacheFriendly:
      return ApplyHeuristicCacheFriendlyOrder(g, A);
    default:
      return BuildInitialOrder(g.dims.size());
  }
}

// Loop band construction
LoopBand::Loop CreateSingleLoop(int dim, int64_t extent, int unroll_factor,
                                bool is_reduction) {
  return LoopBand::Loop{.dim = dim,
                        .begin = 0,
                        .end = extent,
                        .step = 1,
                        .is_tile_outer = false,
                        .is_reduction = is_reduction,
                        .unroll = std::max(1, unroll_factor)};
}

LoopBand BuildLoopBand(const LinalgGeneric& g, const BroadcastAnalysis& A,
                       const std::vector<int>& order,
                       const LinalgCodegenConfig& config) {
  LoopBand band;

  for (int dim : order) {
    int64_t extent = GetDimensionExtent(A, dim);
    int unroll_factor = config.unroll_factor(dim);
    bool is_reduction = (g.dims[dim].kind == IterKind::kReduction);

    // Create single loop (no tiling for now)
    band.loops.push_back(
        CreateSingleLoop(dim, extent, unroll_factor, is_reduction));
  }

  return band;
}

// Main schedule generation function
FailureOr<Schedule> GenerateSchedule(const LinalgEvalResults& eval_results) {
  const LinalgCodegenConfig& config = GetLinalgCodegenConfig();
  const auto& g = eval_results.linalg;
  const auto& A = eval_results.broadcast;

  // Apply loop ordering policy
  auto order = ApplyLoopOrderingPolicy(g, A, config.loop_order());

  // Build loop band
  auto band = BuildLoopBand(g, A, order, config);

  // Create and return schedule
  Schedule schedule;
  schedule.bands.push_back(std::move(band));

  // Pre-compute parallel and reduction dimensions
  schedule.ComputeParallelAndReductionDims();

  // Reverse parallel dimensions to match DSLX axis order (outer→inner = DSLX
  // first→last axis) This ensures loop variables align with DSLX array axes for
  // correct update operations
  auto& [P, R] = schedule.parallel_and_reduction_dims;
  std::reverse(P.begin(), P.end());

  // Calculate row length for reversed parallel dimensions (only if needed)
  if (CountParallel(g) >= 2 && !P.empty()) {
    schedule.row_len =
        GetDimensionExtent(A, P.back());  // Last parallel dim after reversal
  } else {
    schedule.row_len = 0;
  }

  // NEW: Build dimension mapping system
  schedule.BuildDimensionMapping();

  return schedule;
}

// NEW: Build dimension mapping system
void Schedule::BuildDimensionMapping() {
  // Clear existing mappings
  mlir_dim_to_loop_var.clear();
  loop_var_to_mlir_dim.clear();
  mlir_dim_to_dslx_dim.clear();
  dslx_dim_to_mlir_dim.clear();

  const auto& [P, R] = parallel_and_reduction_dims;

  // Build loop variable mappings for parallel dimensions
  for (size_t i = 0; i < P.size(); ++i) {
    int logical_dim = P[i];
    std::string loop_var = "p" + std::to_string(i);

    mlir_dim_to_loop_var[logical_dim] = loop_var;
    loop_var_to_mlir_dim[loop_var] = logical_dim;
  }

  // Build loop variable mappings for reduction dimensions
  for (size_t i = 0; i < R.size(); ++i) {
    int logical_dim = R[i];
    std::string loop_var = "r" + std::to_string(i);

    mlir_dim_to_loop_var[logical_dim] = loop_var;
    loop_var_to_mlir_dim[loop_var] = logical_dim;
  }

  // Build DSLX dimension mappings (accounting for XLS array representation)
  // XLS arrays are reversed: array[cols][rows] instead of array[rows][cols]
  // So we need to map MLIR dimensions to DSLX axis positions

  // Map parallel dimensions (outer to inner in MLIR = first to last in DSLX)
  for (size_t i = 0; i < P.size(); ++i) {
    int logical_dim = P[i];
    int dslx_pos = i;  // Parallel dims come first in DSLX
    mlir_dim_to_dslx_dim[logical_dim] = dslx_pos;
    dslx_dim_to_mlir_dim[dslx_pos] = logical_dim;
  }

  // Map reduction dimensions (after parallel dimensions in DSLX)
  for (size_t i = 0; i < R.size(); ++i) {
    int logical_dim = R[i];
    int dslx_pos = P.size() + i;  // Reduction dims come after parallel dims
    mlir_dim_to_dslx_dim[logical_dim] = dslx_pos;
    dslx_dim_to_mlir_dim[dslx_pos] = logical_dim;
  }
}

}  // namespace mlir::xls
