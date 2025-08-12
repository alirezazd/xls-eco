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

#include "xls/contrib/mlir/transforms/linalg/analysis/broadcast.h"

#include <algorithm>
#include <vector>

#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"

namespace mlir::xls {

mlir::LogicalResult AnalyzeBroadcast(const LinalgGeneric& g, BroadcastAnalysis& result) {
  const int D = g.dims.size();
  const int O = g.operands.size();

  result.loop_extents.resize(D);
  std::vector<std::vector<bool>> operand_refs_dim(O, std::vector<bool>(D, false));
  std::vector<std::vector<SizeExpr>> seen_sizes(D);
  result.dims.operand_varies.assign(O, std::vector<bool>(D, false));
  result.indexers.resize(O);

  // Analyze each operand's indexing map
  for (int o = 0; o < O; ++o) {
    const auto& opnd = g.operands[o];
    const int R = opnd.type.shape.size();
    if (R != static_cast<int>(opnd.map.results.size())) {
      return mlir::failure();
    }
    result.indexers[o].results.resize(R);

    for (int axis = 0; axis < R; ++axis) {
      const auto& e = opnd.map.results[axis];
      if (e.kind == AffineExpr::kVar) {
        int d = e.var;
        if (d < 0 || d >= D) {
          return mlir::failure();
        }

        result.dims.operand_varies[o][d] = true;
        operand_refs_dim[o][d] = true;

        SizeExpr size = (opnd.type.shape[axis] == -1)
            ? SizeExpr("dim_" + std::to_string(d) + "_" + opnd.name)
            : SizeExpr(opnd.type.shape[axis]);
        seen_sizes[d].push_back(size);
        result.indexers[o].results[axis] = d;
      } else {
        result.indexers[o].results[axis] = std::monostate{};
      }
    }
  }

  // Add broadcast sizes for dimensions not referenced by each operand
  for (int o = 0; o < O; ++o) {
    for (int d = 0; d < D; ++d) {
      if (!operand_refs_dim[o][d]) {
        seen_sizes[d].push_back(SizeExpr(1));
      }
    }
  }

  // Determine loop extents for each dimension
  for (int d = 0; d < D; ++d) {
    std::vector<SizeExpr> sizes = seen_sizes[d];
    if (sizes.empty()) sizes.push_back(SizeExpr(1));

    SizeExpr ref_size = SizeExpr(1);
    bool has_ref = false;

    // Find reference size (first non-1 size)
    for (const auto& size : sizes) {
      if (size.isConstant() && size.getConstant() == 1) continue;

      if (!has_ref) {
        ref_size = size;
        has_ref = true;
      } else if (size.isConstant() && ref_size.isConstant()) {
        if (size.getConstant() != ref_size.getConstant()) {
          return mlir::failure();
        }
      }
    }

    // Compute extent as maximum of all sizes
    SizeExpr extent = SizeExpr(1);
    for (const auto& size : sizes) {
      if (size.isConstant() && extent.isConstant()) {
        extent = SizeExpr(std::max(extent.getConstant(), size.getConstant()));
      } else if (extent.isConstant() && extent.getConstant() == 1) {
        extent = size;
      }
    }
    result.loop_extents[d] = extent;
  }

  return mlir::success();
}

mlir::LogicalResult DeriveOutputShape(const LinalgGeneric& g, const BroadcastAnalysis& A,
                                     int out_operand_index, OutputShapeInfo& info) {
  if (out_operand_index < 0 || out_operand_index >= static_cast<int>(g.operands.size())) {
    return mlir::failure();
  }

  const auto& out = g.operands[out_operand_index];
  if (!out.is_output) {
    return mlir::failure();
  }

  const int D = g.dims.size();
  const int R = out.type.shape.size();
  info.out_shape.resize(R);

  // Validate output shape derivation
  for (int axis = 0; axis < R; ++axis) {
    const auto& e = out.map.results[axis];
    if (e.kind != AffineExpr::kVar) {
      return mlir::failure();
    }

    int d = e.var;
    if (d < 0 || d >= D || g.dims[d].kind != IterKind::kParallel) {
      return mlir::failure();
    }

    info.out_shape[axis] = A.loop_extents[d];
  }

  return mlir::success();
}

}  // namespace mlir::xls
