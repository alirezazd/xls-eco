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

#include "xls/contrib/mlir/transforms/linalg/linalg_eval.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/helpers.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_types.h"

namespace mlir::xls {

mlir::LogicalResult IsDag(const std::vector<RegionOp>& ops) {
  std::set<ValueId> defined_values;

  for (size_t i = 0; i < ops.size(); ++i) {
    defined_values.insert(ValueId(i));
  }

  for (const auto& op : ops) {
    for (const auto& input : op.inputs) {
      if (defined_values.find(input) == defined_values.end()) {
        return mlir::failure();
      }
    }
    defined_values.insert(op.result);
  }

  return mlir::success();
}

mlir::LogicalResult AllYieldsDefined(const Region& region) {
  std::set<ValueId> defined_values;

  for (const auto& arg : region.args) {
    defined_values.insert(arg);
  }

  for (const auto& op : region.ops) {
    defined_values.insert(op.result);
  }

  for (const auto& yield : region.yields) {
    if (defined_values.find(yield) == defined_values.end()) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

FailureOr<AffineMap> EvalAffineMap(mlir::AffineMap mlir_map) {
  AffineMap result;

  for (auto expr : mlir_map.getResults()) {
    AffineExpr affine_expr;

    if (auto const_expr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
      if (const_expr.getValue() == 0) {
        affine_expr.kind = AffineExpr::kConst0;
        affine_expr.var = 0;
      } else {
        return failure();
      }
    } else if (auto dim_expr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      affine_expr.kind = AffineExpr::kVar;
      affine_expr.var = dim_expr.getPosition();
    } else {
      return failure();
    }

    result.results.push_back(affine_expr);
  }

  return result;
}

FailureOr<Dim> EvalDimension(mlir::utils::IteratorType iterator_type,
                             size_t dim_index) {
  Dim dim;
  dim.name = std::string("d") + std::to_string(dim_index);

  if (iterator_type == mlir::utils::IteratorType::parallel) {
    dim.kind = IterKind::kParallel;
  } else if (iterator_type == mlir::utils::IteratorType::reduction) {
    dim.kind = IterKind::kReduction;
  } else {
    return failure();
  }

  dim.extent = 1;
  return dim;
}

FailureOr<Operand> EvalOperand(mlir::Value value, const std::string& name,
                               bool is_output, mlir::AffineMap indexing_map) {
  Operand operand;
  operand.name = name;
  operand.is_output = is_output;
  
  auto map = EvalAffineMap(indexing_map);
  if (failed(map)) {
    return failure();
  }
  operand.map = std::move(*map);

  if (auto tensor_type =
          llvm::dyn_cast<mlir::RankedTensorType>(value.getType())) {
    operand.type.tag = Type::kTensorF32;
    for (auto dim : tensor_type.getShape()) {
      if (mlir::ShapedType::isDynamic(dim)) {
        return failure();
      } else {
        operand.type.shape.push_back(dim);
      }
    }
  } else if (value.getType().isF32()) {
    operand.type.tag = Type::kScalarF32;
  } else {
    return failure();
  }

  return operand;
}

FailureOr<Region> EvalRegion(mlir::Region& mlir_region) {
  if (mlir_region.empty()) {
    return failure();
  }

  Region result;
  auto& block = mlir_region.front();

  for (size_t i = 0; i < block.getNumArguments(); ++i) {
    result.args.push_back(ValueId(i));
  }

  int next_id = block.getNumArguments();
  std::vector<std::pair<mlir::Value, ValueId>> value_to_id_pairs;

  for (size_t i = 0; i < block.getNumArguments(); ++i) {
    value_to_id_pairs.push_back({block.getArgument(i), ValueId(i)});
  }

  for (auto& op : block) {
    if (mlir::isa<mlir::linalg::YieldOp>(op)) {
      for (auto operand : op.getOperands()) {
        for (const auto& pair : value_to_id_pairs) {
          if (pair.first == operand) {
            result.yields.push_back(pair.second);
            break;
          }
        }
      }
      continue;
    }

    RegionOp region_op;
    region_op.result = ValueId(next_id++);

    if (mlir::isa<mlir::arith::AddFOp>(op)) {
      region_op.kind = OpKind::kAddF;
    } else if (mlir::isa<mlir::arith::MulFOp>(op)) {
      region_op.kind = OpKind::kMulF;
    } else if (mlir::isa<mlir::arith::SubFOp>(op)) {
      region_op.kind = OpKind::kSubF;
    } else if (mlir::isa<mlir::arith::DivFOp>(op)) {
      region_op.kind = OpKind::kDivF;
    } else if (mlir::isa<mlir::arith::CmpFOp>(op)) {
      auto cmp_op = mlir::cast<mlir::arith::CmpFOp>(op);
      switch (cmp_op.getPredicate()) {
        case mlir::arith::CmpFPredicate::OGT:
          region_op.kind = OpKind::kCmpOGT;
          break;
        case mlir::arith::CmpFPredicate::OLT:
          region_op.kind = OpKind::kCmpOLT;
          break;
        case mlir::arith::CmpFPredicate::OEQ:
          region_op.kind = OpKind::kCmpOEQ;
          break;
        case mlir::arith::CmpFPredicate::OGE:
          region_op.kind = OpKind::kCmpOGE;
          break;
        case mlir::arith::CmpFPredicate::OLE:
          region_op.kind = OpKind::kCmpOLE;
          break;
        case mlir::arith::CmpFPredicate::ONE:
          region_op.kind = OpKind::kCmpONE;
          break;
        default:
          return failure();
      }
    } else if (mlir::isa<mlir::arith::SelectOp>(op)) {
      region_op.kind = OpKind::kSelect;
    } else if (mlir::isa<mlir::arith::MaximumFOp>(op)) {
      region_op.kind = OpKind::kMaxF;
    } else if (mlir::isa<mlir::arith::MinimumFOp>(op)) {
      region_op.kind = OpKind::kMinF;
    } else if (mlir::isa<mlir::arith::ConstantOp>(op)) {
      region_op.kind = OpKind::kConstF32;
      auto const_op = mlir::cast<mlir::arith::ConstantOp>(op);
      if (auto float_attr =
              llvm::dyn_cast<mlir::FloatAttr>(const_op.getValue())) {
        region_op.f32_imm = float_attr.getValueAsDouble();
      }
    } else {
      return failure();
    }

    for (auto operand : op.getOperands()) {
      for (const auto& pair : value_to_id_pairs) {
        if (pair.first == operand) {
          region_op.inputs.push_back(pair.second);
          break;
        }
      }
    }

    value_to_id_pairs.push_back({op.getResult(0), region_op.result});
    result.ops.push_back(region_op);
  }

  return result;
}

FailureOr<std::vector<Dim>> BuildDimensions(
    mlir::linalg::GenericOp& generic_op) {
  auto iterator_types = generic_op.getIteratorTypesArray();
  std::vector<Dim> dims;
  for (size_t i = 0; i < iterator_types.size(); ++i) {
    auto dim = EvalDimension(iterator_types[i], i);
    if (failed(dim)) {
      return failure();
    }
    dims.push_back(std::move(*dim));
  }
  return dims;
}

FailureOr<std::vector<Operand>> BuildOperands(
    mlir::linalg::GenericOp& generic_op) {
  auto inputs = generic_op.getInputs();
  auto outputs = generic_op.getOutputs();
  auto indexing_maps = generic_op.getIndexingMapsArray();

  std::vector<Operand> operands;

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto operand = EvalOperand(inputs[i], "input" + std::to_string(i), false,
                               indexing_maps[i]);
    if (failed(operand)) {
      return failure();
    }
    operands.push_back(std::move(*operand));
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto operand = EvalOperand(outputs[i], "output" + std::to_string(i), true,
                               indexing_maps[inputs.size() + i]);
    if (failed(operand)) {
      return failure();
    }
    operands.push_back(std::move(*operand));
  }
  return operands;
}

void UpdateDimensionExtents(LinalgGeneric& linalg,
                            const BroadcastAnalysis& broadcast) {
  for (size_t i = 0;
       i < linalg.dims.size() && i < broadcast.loop_extents.size(); ++i) {
    if (broadcast.loop_extents[i].isConstant()) {
      linalg.dims[i].extent = broadcast.loop_extents[i].getConstant();
    }
  }
}

bool HasReductionDimensions(const LinalgGeneric& linalg) {
  for (const auto& dim : linalg.dims) {
    if (dim.kind == IterKind::kReduction) {
      return true;
    }
  }
  return false;
}

bool OutputReferencesReductionDimension(const LinalgGeneric& linalg) {
  for (const auto& operand : linalg.operands) {
    if (!operand.is_output) continue;

    for (const auto& expr : operand.map.results) {
      if (expr.kind != AffineExpr::kVar) continue;

      size_t dim_idx = static_cast<size_t>(expr.var);
      if (dim_idx < linalg.dims.size() &&
          linalg.dims[dim_idx].kind == IterKind::kReduction) {
        return true;
      }
    }
  }
  return false;
}

FailureOr<LinalgEvalResults> EvalLinalgGeneric(mlir::Operation* op) {
  auto generic_op = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
  if (!generic_op) {
    return op->emitError("Failed to evaluate linalg.generic operation");
  }

  BroadcastAnalysis broadcast;
  bool has_reduction = false;
  bool is_valid_reduction = false;

  auto dims = BuildDimensions(generic_op);
  if (failed(dims)) {
    return op->emitError("Failed to evaluate dimensions");
  }

  auto operands = BuildOperands(generic_op);
  if (failed(operands)) {
    return op->emitError(
        "Failed to evaluate operands (dynamic dimensions "
        "not supported for hardware synthesis)");
  }

  auto region = EvalRegion(generic_op.getRegion());
  if (failed(region)) {
    return op->emitError("Failed to evaluate region body");
  }

  LinalgGeneric linalg;
  linalg.dims = std::move(*dims);
  linalg.operands = std::move(*operands);
  linalg.region = std::move(*region);

  auto broadcast_result = EvalBroadcast(linalg);
  if (failed(broadcast_result)) {
    return op->emitError("Failed to evaluate broadcast. Parsed structure:\n")
           << LinalgGenericToString(linalg);
  }
  broadcast = std::move(*broadcast_result);

  UpdateDimensionExtents(linalg, broadcast);

  has_reduction = HasReductionDimensions(linalg);

  if (has_reduction) {
    if (OutputReferencesReductionDimension(linalg)) {
      return op->emitError(
                 "Operation is not a valid reduction pattern (output "
                 "references reduction dimension). Structure:\n")
             << LinalgGenericToString(linalg);
    }
    is_valid_reduction = true;
  }

  return LinalgEvalResults{std::move(linalg), std::move(broadcast),
                           has_reduction, is_valid_reduction};
}



mlir::LogicalResult Validate(const LinalgGeneric& g) {
  if (g.dims.empty()) {
    return mlir::failure();
  }

  for (const auto& d : g.dims) {
    if (d.extent < 1) {
      return mlir::failure();
    }
  }

  for (const auto& opnd : g.operands) {
    if (opnd.map.results.size() != opnd.type.shape.size()) {
      return mlir::failure();
    }

    for (const auto& e : opnd.map.results) {
      if (e.kind == AffineExpr::kVar) {
        if (e.var < 0 || e.var >= static_cast<int>(g.dims.size())) {
          return mlir::failure();
        }
      }
    }
  }

  if (!IsDag(g.region.ops).succeeded() || !AllYieldsDefined(g.region).succeeded()) {
    return mlir::failure();
  }

  return mlir::success();
}

FailureOr<std::vector<std::vector<bool>>> EvalOperandDimensionRefs(
    const LinalgGeneric& g) {
  const size_t D = g.dims.size();
  const size_t O = g.operands.size();

  std::vector<std::vector<bool>> operand_refs_dim(O,
                                                  std::vector<bool>(D, false));

  for (size_t o = 0; o < O; ++o) {
    const auto& opnd = g.operands[o];
    const size_t R = opnd.type.shape.size();
    if (R != opnd.map.results.size()) {
      return mlir::failure();
    }

    for (size_t axis = 0; axis < R; ++axis) {
      const auto& e = opnd.map.results[axis];

      switch (e.kind) {
        case AffineExpr::kVar: {
          int d = e.var;
          if (d < 0 || static_cast<size_t>(d) >= D) {
            return mlir::failure();
          }
          operand_refs_dim[o][d] = true;
          break;
        }

        case AffineExpr::kConst0: {
          break;
        }

        default: {
          return mlir::failure();
        }
      }
    }
  }

  return operand_refs_dim;
}

FailureOr<std::vector<OperandIndexFn>> EvalOperandIndexers(
    const LinalgGeneric& g) {
  const size_t D = g.dims.size();
  const size_t O = g.operands.size();

  std::vector<OperandIndexFn> indexers(O);

  for (size_t o = 0; o < O; ++o) {
    const auto& opnd = g.operands[o];
    const size_t R = opnd.type.shape.size();
    if (R != opnd.map.results.size()) {
      return mlir::failure();
    }
    indexers[o].results.resize(R);

    std::vector<bool> used_dims(D, false);

    for (size_t axis = 0; axis < R; ++axis) {
      const auto& e = opnd.map.results[axis];

      switch (e.kind) {
        case AffineExpr::kVar: {
          int d = e.var;
          if (d < 0 || static_cast<size_t>(d) >= D) {
            return mlir::failure();
          }

          if (used_dims[d]) {
            return mlir::failure();
          }
          used_dims[d] = true;

          indexers[o].results[axis] = d;
          break;
        }

        case AffineExpr::kConst0: {
          if (opnd.type.shape[axis] != 1) {
            return mlir::failure();
          }
          indexers[o].results[axis] = std::monostate{};
          break;
        }

        default: {
          return mlir::failure();
        }
      }
    }
  }

  return indexers;
}

FailureOr<std::vector<SizeExpr>> EvalLoopExtents(
    const LinalgGeneric& g,
    const std::vector<std::vector<bool>>& operand_refs_dim) {
  const size_t D = g.dims.size();
  const size_t O = g.operands.size();

  std::vector<std::vector<SizeExpr>> seen_sizes(D);

  for (size_t o = 0; o < O; ++o) {
    const auto& opnd = g.operands[o];
    const size_t R = opnd.type.shape.size();

    for (size_t axis = 0; axis < R; ++axis) {
      const auto& e = opnd.map.results[axis];
      if (e.kind == AffineExpr::kVar) {
        int d = e.var;
        if (d < 0 || static_cast<size_t>(d) >= D) {
          return mlir::failure();
        }
        SizeExpr size = SizeExpr(opnd.type.shape[axis]);
        seen_sizes[d].push_back(size);
      }
    }
  }

  for (size_t o = 0; o < O; ++o) {
    for (size_t d = 0; d < D; ++d) {
      if (!operand_refs_dim[o][d]) {
        seen_sizes[d].push_back(SizeExpr(1));
      }
    }
  }

  std::vector<SizeExpr> loop_extents(D);
  for (size_t d = 0; d < D; ++d) {
    std::vector<SizeExpr> sizes = seen_sizes[d];
    if (sizes.empty()) sizes.push_back(SizeExpr(1));

    SizeExpr unique_non_one_size = SizeExpr(1);
    bool found_non_one = false;

    for (const auto& size : sizes) {
      if (size.isConstant() && size.getConstant() == 1) continue;

      if (!found_non_one) {
        unique_non_one_size = size;
        found_non_one = true;
      } else {
        if (size.isConstant() && unique_non_one_size.isConstant()) {
          if (size.getConstant() != unique_non_one_size.getConstant()) {
            return mlir::failure();
          }
        } else if (size.isConstant() && !unique_non_one_size.isConstant()) {
          if (size.getConstant() != 1) {
            return mlir::failure();
          }
        } else if (!size.isConstant() && unique_non_one_size.isConstant()) {
          if (unique_non_one_size.getConstant() != 1) {
            return mlir::failure();
          }
          unique_non_one_size = size;
        }
      }
    }

    loop_extents[d] = unique_non_one_size;
  }

  return loop_extents;
}

FailureOr<BroadcastAnalysis> EvalBroadcast(const LinalgGeneric& g) {
  const size_t D = g.dims.size();
  const size_t O = g.operands.size();

  BroadcastAnalysis result;

  result.loop_extents.resize(D);
  result.dims.operand_varies.resize(O, std::vector<bool>(D, false));
  result.indexers.resize(O);

  auto operand_refs_dim = EvalOperandDimensionRefs(g);
  if (failed(operand_refs_dim)) {
    return mlir::failure();
  }

  auto indexers = EvalOperandIndexers(g);
  if (failed(indexers)) {
    return mlir::failure();
  }

  result.indexers = std::move(*indexers);

  auto loop_extents = EvalLoopExtents(g, *operand_refs_dim);
  if (failed(loop_extents)) {
    return mlir::failure();
  }

  if (loop_extents->size() != D) {
    return mlir::failure();
  }

  result.loop_extents = std::move(*loop_extents);

  for (size_t o = 0; o < O; ++o) {
    const auto& indexer = result.indexers[o];
    for (size_t axis = 0; axis < indexer.results.size(); ++axis) {
      const auto& result_val = indexer.results[axis];
      if (std::holds_alternative<int>(result_val)) {
        int d = std::get<int>(result_val);
        if (d >= 0 && static_cast<size_t>(d) < D) {
          result.dims.operand_varies[o][d] = true;
        }
      }
    }
  }

  for (size_t o = 0; o < O; ++o) {
    for (size_t d = 0; d < D; ++d) {
      assert(result.dims.operand_varies[o][d] == (*operand_refs_dim)[o][d] &&
             "operand_varies should match operand_refs_dim");
    }
  }

  return result;
}

mlir::LogicalResult DeriveOutputShape(const LinalgGeneric& g,
                                      const BroadcastAnalysis& A,
                                      int out_operand_index,
                                      OutputShapeInfo& info) {
  if (out_operand_index < 0 ||
      out_operand_index >= static_cast<int>(g.operands.size())) {
    return mlir::failure();
  }

  const auto& out = g.operands[out_operand_index];
  if (!out.is_output) {
    return mlir::failure();
  }

  const int D = g.dims.size();
  const int R = out.type.shape.size();
  info.out_shape.resize(R);

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
