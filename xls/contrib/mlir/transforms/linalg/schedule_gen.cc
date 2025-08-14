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

#include "xls/contrib/mlir/transforms/linalg/helpers.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_config.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_types.h"

namespace mlir::xls {

std::vector<int> BuildInitialOrder(int num_dims) {
  std::vector<int> order(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    order[i] = i;
  }
  return order;
}

std::vector<int> ApplyParallelOuterReductionInnerOrder(const LinalgGeneric& g) {
  const int D = g.dims.size();
  std::vector<int> parallel_dims, reduction_dims;

  for (int i = 0; i < D; ++i) {
    if (g.dims[i].kind == IterKind::kParallel) {
      parallel_dims.push_back(i);
    } else {
      reduction_dims.push_back(i);
    }
  }

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

std::vector<int> ApplyHeuristicCacheFriendlyOrder(const LinalgGeneric& g,
                                                  const BroadcastAnalysis& A) {
  const int D = g.dims.size();
  std::vector<int> parallel_dims, reduction_dims;

  for (int i = 0; i < D; ++i) {
    if (g.dims[i].kind == IterKind::kParallel) {
      parallel_dims.push_back(i);
    } else {
      reduction_dims.push_back(i);
    }
  }

  // Sort parallel dimensions by access frequency (most accessed first)
  std::sort(parallel_dims.begin(), parallel_dims.end(), [&](int a, int b) {
    int a_count = 0, b_count = 0;
    for (const auto& indexer : A.indexers) {
      if (!indexer.results.empty()) {
        if (std::holds_alternative<int>(indexer.results.back()) &&
            std::get<int>(indexer.results.back()) == a) {
          a_count++;
        }
        if (std::holds_alternative<int>(indexer.results.back()) &&
            std::get<int>(indexer.results.back()) == b) {
          b_count++;
        }
      }
    }
    return a_count > b_count;
  });

  // Sort reduction dimensions by extent (largest first)
  std::sort(reduction_dims.begin(), reduction_dims.end(), [&](int a, int b) {
    int64_t a_extent =
        A.loop_extents[a].isConstant() ? A.loop_extents[a].getConstant() : 1;
    int64_t b_extent =
        A.loop_extents[b].isConstant() ? A.loop_extents[b].getConstant() : 1;
    return a_extent > b_extent;
  });

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

LoopBand::Loop CreateTiledLoop(int dim, int64_t extent, int tile_size,
                               int unroll_factor, bool is_reduction) {
  return LoopBand::Loop{.dim = dim,
                        .begin = 0,
                        .end = tile_size,
                        .step = 1,
                        .is_tile_outer = false,
                        .is_reduction = is_reduction,
                        .unroll = std::max(1, unroll_factor)};
}

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
    int64_t E = A.loop_extents[dim].isConstant()
                    ? A.loop_extents[dim].getConstant()
                    : 1;
    int T = config.tile_size(dim);
    int U = config.unroll_factor(dim);
    bool is_red = (g.dims[dim].kind == IterKind::kReduction);

    // Only apply tiling if explicitly enabled in config
    if (!config.enable_tiling()) {
      T = 1;
    }

    if (T > 1 && T < E) {
      // Create tiled loop structure: outer tile loop + inner element loop
      band.loops.push_back(LoopBand::Loop{.dim = dim,
                                          .begin = 0,
                                          .end = E,
                                          .step = T,
                                          .is_tile_outer = true,
                                          .is_reduction = is_red,
                                          .unroll = 1});

      band.loops.push_back(LoopBand::Loop{.dim = dim,
                                          .begin = 0,
                                          .end = T,
                                          .step = 1,
                                          .is_tile_outer = false,
                                          .is_reduction = is_red,
                                          .unroll = std::max(1, U)});
    } else {
      // Single loop without tiling
      band.loops.push_back(CreateSingleLoop(dim, E, U, is_red));
    }
  }

  return band;
}

FailureOr<Schedule> GenerateSchedule(const LinalgEvalResults& eval_results) {
  const LinalgCodegenConfig& config = GetLinalgCodegenConfig();
  const auto& g = eval_results.linalg;
  const auto& A = eval_results.broadcast;

  auto order = ApplyLoopOrderingPolicy(g, A, config.loop_order());
  auto band = BuildLoopBand(g, A, order, config);

  Schedule schedule;
  schedule.bands.push_back(std::move(band));
  return schedule;
}

}  // namespace mlir::xls
