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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_SCHEDULE_TYPES_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_SCHEDULE_TYPES_H_

#include <map>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace mlir::xls {

// Code generation configuration
class LinalgCodegenConfig {
 public:
  enum class LoopOrder {
    kPreserve,
    kParallelOuter_ReductionInner,
    kHeuristicCacheFriendly
  };

  // Builder pattern methods
  LinalgCodegenConfig& loop_order(LoopOrder value) {
    loop_order_ = value;
    return *this;
  }

  LinalgCodegenConfig& unroll_factor(int dim, int factor) {
    unroll_factors_[dim] = factor;
    return *this;
  }

  // Getters
  LoopOrder loop_order() const { return loop_order_; }

  int unroll_factor(int dim) const {
    auto it = unroll_factors_.find(dim);
    return it != unroll_factors_.end() ? it->second : 1;
  }

  // Access to internal maps for compatibility
  const std::map<int, int>& unroll() const { return unroll_factors_; }

 private:
  LoopOrder loop_order_ = LoopOrder::kPreserve;
  std::map<int, int> unroll_factors_;
};

// Loop scheduling structures
struct LoopBand {
  struct Loop {
    int dim;             // which logical dim (0..D-1)
    int64_t begin;       // 0
    int64_t end;         // loop extent or tile end
    int64_t step;        // usually 1; may equal tile size for outers
    bool is_tile_outer;  // true if this is the tile loop
    bool is_reduction;   // from dims[dim].kind
    int unroll = 1;      // static unroll factor to apply at emission
  };
  std::vector<Loop> loops;
};

struct Schedule {
  std::vector<LoopBand> bands;
  int64_t row_len =
      0;  // Row length from the last parallel dimension in final loop order

  // Pre-computed parallel and reduction dimensions
  std::pair<std::vector<int>, std::vector<int>> parallel_and_reduction_dims;  // (parallel_dims, reduction_dims)

  // Clear dimension mapping system
  absl::flat_hash_map<int, std::string> mlir_dim_to_loop_var;  // logical_dim -> "p0"/"r1"
  absl::flat_hash_map<std::string, int> loop_var_to_mlir_dim;  // "p0"/"r1" -> logical_dim
  
  // DSLX-specific mapping (accounts for XLS array representation)
  absl::flat_hash_map<int, int> mlir_dim_to_dslx_dim;  // logical_dim -> DSLX axis position
  absl::flat_hash_map<int, int> dslx_dim_to_mlir_dim;  // DSLX axis position -> logical_dim

  // Initialize parallel and reduction dimensions from bands
  void ComputeParallelAndReductionDims() {
    auto& [parallel_dims, reduction_dims] = parallel_and_reduction_dims;
    parallel_dims.clear();
    reduction_dims.clear();
    if (!bands.empty() && !bands[0].loops.empty()) {
      for (const auto& L : bands[0].loops) {
        (L.is_reduction ? reduction_dims : parallel_dims).push_back(L.dim);
      }
    }
  }
  
  // Build all dimension mappings once
  void BuildDimensionMapping();
};

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_SCHEDULE_TYPES_H_
