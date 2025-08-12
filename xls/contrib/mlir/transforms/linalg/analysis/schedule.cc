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

#include "xls/contrib/mlir/transforms/linalg/analysis/schedule.h"

#include <algorithm>
#include <map>
#include <vector>

#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/broadcast.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"

namespace mlir::xls {

mlir::LogicalResult GenerateSchedule(const LinalgGeneric& g, const BroadcastAnalysis& A,
                                    const CodegenConfig& config, Schedule& schedule) {
  const int D = g.dims.size();

  // Initialize dimension order
  std::vector<int> order(D);
  for (int i = 0; i < D; ++i) {
    order[i] = i;
  }

  // Apply loop ordering policy
  switch (config.loop_order) {
    case CodegenConfig::kPreserve:
      // Keep original dimension order
      break;

    case CodegenConfig::kParallelOuter_ReductionInner: {
      // Separate parallel and reduction dimensions
      std::vector<int> parallel_dims, reduction_dims;

      for (int i = 0; i < D; ++i) {
        if (g.dims[i].kind == IterKind::kParallel) {
          parallel_dims.push_back(i);
        } else {
          reduction_dims.push_back(i);
        }
      }

      // Place parallel dimensions first, then reduction dimensions
      int idx = 0;
      for (int dim : parallel_dims) {
        order[idx++] = dim;
      }
      for (int dim : reduction_dims) {
        order[idx++] = dim;
      }
      break;
    }

    case CodegenConfig::kHeuristicCacheFriendly: {
      // Separate parallel and reduction dimensions
      std::vector<int> parallel_dims, reduction_dims;
      for (int i = 0; i < D; ++i) {
        if (g.dims[i].kind == IterKind::kParallel) {
          parallel_dims.push_back(i);
        } else {
          reduction_dims.push_back(i);
        }
      }

      // Sort parallel dimensions by access frequency (most accessed first)
      std::sort(parallel_dims.begin(), parallel_dims.end(),
                [&](int a, int b) {
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
      std::sort(reduction_dims.begin(), reduction_dims.end(),
                [&](int a, int b) {
                  int64_t a_extent = A.loop_extents[a].isConstant() ?
                      A.loop_extents[a].getConstant() : 1;
                  int64_t b_extent = A.loop_extents[b].isConstant() ?
                      A.loop_extents[b].getConstant() : 1;
                  return a_extent > b_extent;
                });

      // Place parallel dimensions first, then reduction dimensions
      int idx = 0;
      for (int dim : parallel_dims) {
        order[idx++] = dim;
      }
      for (int dim : reduction_dims) {
        order[idx++] = dim;
      }
      break;
    }
  }

  // Build loop bands with tiling and unrolling
  schedule.bands.clear();
  LoopBand band;

  for (int dim : order) {
    int64_t E = A.loop_extents[dim].isConstant() ?
        A.loop_extents[dim].getConstant() : 1;
    int T = config.tiles.size.count(dim) ? config.tiles.size.at(dim) : 1;
    bool is_red = (g.dims[dim].kind == IterKind::kReduction);

    if (T > 1 && T < E) {
      // Create tiled loop structure: outer tile loop + inner element loop
      band.loops.push_back(LoopBand::Loop{
        .dim = dim,
        .begin = 0,
        .end = E,
        .step = T,
        .is_tile_outer = true,
        .is_reduction = is_red,
        .unroll = 1
      });

      band.loops.push_back(LoopBand::Loop{
        .dim = dim,
        .begin = 0,
        .end = T,
        .step = 1,
        .is_tile_outer = false,
        .is_reduction = is_red,
        .unroll = std::max(1, config.unroll.count(dim) ? config.unroll.at(dim) : 1)
      });
    } else {
      // Single loop without tiling
      band.loops.push_back(LoopBand::Loop{
        .dim = dim,
        .begin = 0,
        .end = E,
        .step = 1,
        .is_tile_outer = false,
        .is_reduction = is_red,
        .unroll = std::max(1, config.unroll.count(dim) ? config.unroll.at(dim) : 1)
      });
    }
  }

  schedule.bands.push_back(std::move(band));
  return mlir::success();
}

}  // namespace mlir::xls
