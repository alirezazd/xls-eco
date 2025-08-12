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

#include "xls/contrib/mlir/transforms/linalg/utils/helpers.h"

#include <algorithm>
#include <set>
#include <string>

#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"

namespace mlir::xls {

bool IsDag(const std::vector<RegionOp>& ops) {
  std::set<ValueId> defined_values;

  for (size_t i = 0; i < ops.size(); ++i) {
    defined_values.insert(ValueId(i));
  }

  for (const auto& op : ops) {
    for (const auto& input : op.inputs) {
      if (defined_values.find(input) == defined_values.end()) {
        return false;
      }
    }
    defined_values.insert(op.result);
  }

  return true;
}

bool AllYieldsDefined(const Region& region) {
  std::set<ValueId> defined_values;

  for (const auto& arg : region.args) {
    defined_values.insert(arg);
  }

  for (const auto& op : region.ops) {
    defined_values.insert(op.result);
  }

  for (const auto& yield : region.yields) {
    if (defined_values.find(yield) == defined_values.end()) {
      return false;
    }
  }

  return true;
}

std::array<int, 4> MakeOperandIndex(int operand_id, const std::array<int, 4>& loop_iter,
                                   const BroadcastAnalysis& A) {
  std::array<int, 4> idx = {0, 0, 0, 0};
  const auto& indexer = A.indexers[operand_id];
  
  for (size_t axis = 0; axis < indexer.results.size() && axis < 4; ++axis) {
    if (std::holds_alternative<int>(indexer.results[axis])) {
      int dim_id = std::get<int>(indexer.results[axis]);
      if (dim_id >= 0 && dim_id < 4) {
        idx[axis] = loop_iter[dim_id];
      }
    }
  }
  return idx;
}

float Load(int operand_id, const std::array<int, 4>& idx) {
  // Placeholder implementation for loading values from tensors.
  // In a real implementation, this would access the actual tensor data.
  // For testing purposes, returns a deterministic value based on operand and indices.
  return static_cast<float>(operand_id + idx[0] + idx[1] + idx[2] + idx[3]);
}

}  // namespace mlir::xls
