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

#include "xls/contrib/mlir/transforms/linalg/helpers.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <string>

#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_types.h"

namespace mlir::xls {

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

std::string LinalgGenericToString(const LinalgGeneric& linalg) {
  std::ostringstream oss;

  oss << "LinalgGeneric {\n";

  oss << "  dims: [";
  for (size_t i = 0; i < linalg.dims.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << linalg.dims[i].name << "("
        << (linalg.dims[i].kind == IterKind::kParallel ? "P" : "R") << ":"
        << linalg.dims[i].extent << ")";
  }
  oss << "]\n";

  oss << "  operands: [";
  for (size_t i = 0; i < linalg.operands.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << linalg.operands[i].name << ":"
        << (linalg.operands[i].is_output ? "out" : "in");
  }
  oss << "]\n";

  oss << "  region: {\n";
  oss << "  args: [";
  for (size_t i = 0; i < linalg.region.args.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << "%" << linalg.region.args[i].id;
  }
  oss << "]\n";

  oss << "  ops: [";
  for (size_t i = 0; i < linalg.region.ops.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << "%" << linalg.region.ops[i].result.id << " = ";
    switch (linalg.region.ops[i].kind) {
      case OpKind::kAddF:
        oss << "addf";
        break;
      case OpKind::kMulF:
        oss << "mulf";
        break;
      case OpKind::kSubF:
        oss << "subf";
        break;
      case OpKind::kDivF:
        oss << "divf";
        break;
      case OpKind::kMaxF:
        oss << "maxf";
        break;
      case OpKind::kMinF:
        oss << "minf";
        break;
      case OpKind::kCmpOGT:
        oss << "cmpf_ogt";
        break;
      case OpKind::kSelect:
        oss << "select";
        break;
      case OpKind::kConstF32:
        oss << "const_f32(" << linalg.region.ops[i].f32_imm << ")";
        break;
      default:
        oss << "unknown";
        break;
    }
    oss << "(";
    for (size_t j = 0; j < linalg.region.ops[i].inputs.size(); ++j) {
      if (j > 0) oss << ", ";
      oss << "%" << linalg.region.ops[i].inputs[j].id;
    }
    oss << ")";
  }
  oss << "]\n";

  oss << "  yields: [";
  for (size_t i = 0; i < linalg.region.yields.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << "%" << linalg.region.yields[i].id;
  }
  oss << "]\n";

  oss << "  }\n";
  oss << "}\n";

  return oss.str();
}

}  // namespace mlir::xls
