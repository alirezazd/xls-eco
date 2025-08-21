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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "xls/contrib/mlir/transforms/linalg/helpers.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_types.h"

namespace mlir::xls {

// Internal function declarations (moved from header)
LogicalResult IsDag(const std::vector<RegionOp>& ops);
LogicalResult AllYieldsDefined(const Region& region);
FailureOr<AffineMap> EvalAffineMap(mlir::AffineMap mlir_map);
FailureOr<Dim> EvalDimension(mlir::utils::IteratorType iterator_type,
                             size_t dim_index);
FailureOr<Operand> EvalOperand(mlir::Value value, const std::string& name,
                               bool is_output, mlir::AffineMap indexing_map);
FailureOr<Region> EvalRegion(mlir::Region& mlir_region);
FailureOr<std::vector<Dim>> BuildDimensions(
    mlir::linalg::GenericOp& generic_op);
FailureOr<std::vector<Operand>> BuildOperands(
    mlir::linalg::GenericOp& generic_op);
LogicalResult UpdateExtents(LinalgGeneric& linalg,
                            const BroadcastAnalysis& broadcast);
LogicalResult Validate(const LinalgGeneric& g, mlir::Operation* op);
FailureOr<std::vector<std::vector<bool>>> EvalOperandRefs(
    const LinalgGeneric& g, mlir::Operation* op);
FailureOr<std::vector<OperandIndexFn>> EvalIndexers(const LinalgGeneric& g,
                                                    mlir::Operation* op);
FailureOr<std::vector<SizeExpr>> EvalLoopExtents(
    const LinalgGeneric& g,
    const std::vector<std::vector<bool>>& operand_refs_dim,
    mlir::Operation* op);
FailureOr<BroadcastAnalysis> EvalBroadcast(const LinalgGeneric& g,
                                           mlir::Operation* op);
LogicalResult DeriveInputShape(const LinalgGeneric& g,
                               const BroadcastAnalysis& A, int operand_index,
                               InputShapeInfo& info, mlir::Operation* op);
LogicalResult DeriveOutputShape(const LinalgGeneric& g,
                                const BroadcastAnalysis& A,
                                int out_operand_index, OutputShapeInfo& info,
                                mlir::Operation* op);
FailureOr<std::string> ExtractName(mlir::Operation* op);

LogicalResult IsDag(const std::vector<RegionOp>& ops) {
  std::set<ValueId> defined_values;

  for (size_t i = 0; i < ops.size(); ++i) {
    defined_values.insert(ValueId(i));
  }

  for (const auto& op : ops) {
    for (const auto& input : op.inputs) {
      if (defined_values.find(input) == defined_values.end()) {
        return failure();
      }
    }
    defined_values.insert(op.result);
  }

  return success();
}

LogicalResult AllYieldsDefined(const Region& region) {
  std::set<ValueId> defined_values;

  for (const auto& arg : region.args) {
    defined_values.insert(arg);
  }

  for (const auto& op : region.ops) {
    defined_values.insert(op.result);
  }

  for (const auto& yield : region.yields) {
    if (defined_values.find(yield) == defined_values.end()) {
      return failure();
    }
  }

  return success();
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
    // Use simple output names: output, output_flat, etc.
    std::string name = "output";
    if (i > 0) {
      name += "_flat";
    }

    auto operand =
        EvalOperand(outputs[i], name, true, indexing_maps[inputs.size() + i]);
    if (failed(operand)) {
      return failure();
    }
    operands.push_back(std::move(*operand));
  }
  return operands;
}

LogicalResult UpdateExtents(LinalgGeneric& linalg,
                            const BroadcastAnalysis& broadcast) {
  for (size_t i = 0;
       i < linalg.dims.size() && i < broadcast.loop_extents.size(); ++i) {
    if (broadcast.loop_extents[i].isConstant()) {
      linalg.dims[i].extent = broadcast.loop_extents[i].getConstant();
    }
  }
  return success();
}

FailureOr<LinalgEvalResults> EvalLinalgGeneric(mlir::Operation* op) {
  auto generic_op = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
  if (!generic_op) {
    return op->emitError("Failed to evaluate linalg.generic operation");
  }

  BroadcastAnalysis broadcast;

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



  auto broadcast_result = EvalBroadcast(linalg, op);
  if (failed(broadcast_result)) {
    return failure();
  }
  broadcast = std::move(*broadcast_result);

  if (failed(UpdateExtents(linalg, broadcast))) {
    return failure();
  }

  // Derive shapes for all operands
  auto shapes_result = DeriveShapes(linalg, broadcast, op);
  if (failed(shapes_result)) {
    return failure();
  }

  // Validate reduction operations
  if (failed(ValidateReduction(linalg, op))) {
    return failure();
  }

  auto function_name_result = ExtractName(op);
  if (failed(function_name_result)) {
    return failure();
  }
  std::string function_name = std::move(*function_name_result);
  return LinalgEvalResults{function_name, std::move(linalg),
                           std::move(broadcast), std::move(*shapes_result)};
}



FailureOr<std::vector<std::vector<bool>>> EvalOperandRefs(
    const LinalgGeneric& g, mlir::Operation* op) {
  const size_t D = g.dims.size();
  const size_t O = g.operands.size();

  std::vector<std::vector<bool>> operand_refs_dim(O,
                                                  std::vector<bool>(D, false));

  for (size_t o = 0; o < O; ++o) {
    const auto& opnd = g.operands[o];
    const size_t R = opnd.type.shape.size();
    if (R != opnd.map.results.size()) {
      return op->emitError("Operand ")
             << o << " has mismatched affine map results ("
             << opnd.map.results.size() << ") and tensor shape (" << R
             << "). Structure:\n"
             << LinalgGenericToString(g);
    }

    for (size_t axis = 0; axis < R; ++axis) {
      const auto& e = opnd.map.results[axis];

      switch (e.kind) {
        case AffineExpr::kVar: {
          int d = e.var;
          if (d < 0 || static_cast<size_t>(d) >= D) {
            return op->emitError("Operand ")
                   << o << " affine expression " << axis
                   << " references invalid dimension " << d
                   << " (valid range: 0-" << (D - 1) << "). Structure:\n"
                   << LinalgGenericToString(g);
          }
          operand_refs_dim[o][d] = true;
          break;
        }

        case AffineExpr::kConst0: {
          break;
        }

        default: {
          return op->emitError("Operand ")
                 << o << " affine expression " << axis
                 << " has unsupported kind " << static_cast<int>(e.kind)
                 << ". Structure:\n"
                 << LinalgGenericToString(g);
        }
      }
    }
  }

  return operand_refs_dim;
}

FailureOr<std::vector<OperandIndexFn>> EvalIndexers(const LinalgGeneric& g,
                                                    mlir::Operation* op) {
  const size_t D = g.dims.size();
  const size_t O = g.operands.size();

  std::vector<OperandIndexFn> indexers(O);

  for (size_t o = 0; o < O; ++o) {
    const auto& opnd = g.operands[o];
    const size_t R = opnd.type.shape.size();
    if (R != opnd.map.results.size()) {
      return op->emitError("Operand ")
             << o << " has mismatched affine map results ("
             << opnd.map.results.size() << ") and tensor shape (" << R
             << "). Structure:\n"
             << LinalgGenericToString(g);
    }
    indexers[o].results.resize(R);

    std::vector<bool> used_dims(D, false);

    for (size_t axis = 0; axis < R; ++axis) {
      const auto& e = opnd.map.results[axis];

      switch (e.kind) {
        case AffineExpr::kVar: {
          int d = e.var;
          if (d < 0 || static_cast<size_t>(d) >= D) {
            return op->emitError("Operand ")
                   << o << " affine expression " << axis
                   << " references invalid dimension " << d
                   << " (valid range: 0-" << (D - 1) << "). Structure:\n"
                   << LinalgGenericToString(g);
          }

          if (used_dims[d]) {
            return op->emitError("Operand ")
                   << o << " affine expression " << axis
                   << " references dimension " << d
                   << " which is already used. "
                   << "Each dimension can only be referenced once per operand. "
                      "Structure:\n"
                   << LinalgGenericToString(g);
          }
          used_dims[d] = true;

          indexers[o].results[axis] = d;
          break;
        }

        case AffineExpr::kConst0: {
          if (opnd.type.shape[axis] != 1) {
            return op->emitError("Operand ")
                   << o << " affine expression " << axis
                   << " is constant 0 but tensor shape at axis " << axis
                   << " is " << opnd.type.shape[axis]
                   << " (must be 1). Structure:\n"
                   << LinalgGenericToString(g);
          }
          indexers[o].results[axis] = std::monostate{};
          break;
        }

        default: {
          return op->emitError("Operand ")
                 << o << " affine expression " << axis
                 << " has unsupported kind " << static_cast<int>(e.kind)
                 << ". Structure:\n"
                 << LinalgGenericToString(g);
        }
      }
    }
  }

  return indexers;
}

FailureOr<std::vector<SizeExpr>> EvalLoopExtents(
    const LinalgGeneric& g,
    const std::vector<std::vector<bool>>& operand_refs_dim,
    mlir::Operation* op) {
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
          return op->emitError("Operand ")
                 << o << " affine expression " << axis
                 << " references invalid dimension " << d << " (valid range: 0-"
                 << (D - 1) << "). Structure:\n"
                 << LinalgGenericToString(g);
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
            return op->emitError("Dimension ")
                   << d << " has conflicting sizes: " << size.getConstant()
                   << " and " << unique_non_one_size.getConstant()
                   << ". All operands must agree on dimension sizes. "
                      "Structure:\n"
                   << LinalgGenericToString(g);
          }
        } else if (size.isConstant() && !unique_non_one_size.isConstant()) {
          if (size.getConstant() != 1) {
            return op->emitError("Dimension ")
                   << d << " has conflicting sizes: "
                   << "symbolic " << unique_non_one_size.getSymbolic()
                   << " and constant " << size.getConstant()
                   << " (must be 1 for broadcast). Structure:\n"
                   << LinalgGenericToString(g);
          }
        } else if (!size.isConstant() && unique_non_one_size.isConstant()) {
          if (unique_non_one_size.getConstant() != 1) {
            return op->emitError("Dimension ")
                   << d << " has conflicting sizes: "
                   << "constant " << unique_non_one_size.getConstant()
                   << " and symbolic " << size.getSymbolic()
                   << " (must be 1 for broadcast). Structure:\n"
                   << LinalgGenericToString(g);
          }
          unique_non_one_size = size;
        }
      }
    }

    loop_extents[d] = unique_non_one_size;
  }

  return loop_extents;
}

FailureOr<BroadcastAnalysis> EvalBroadcast(const LinalgGeneric& g,
                                           mlir::Operation* op) {
  const size_t D = g.dims.size();
  const size_t O = g.operands.size();

  BroadcastAnalysis result;

  result.loop_extents.resize(D);
  result.dims.operand_varies.resize(O, std::vector<bool>(D, false));
  result.indexers.resize(O);

  auto operand_refs_dim = EvalOperandRefs(g, op);
  if (failed(operand_refs_dim)) {
    return failure();
  }

  auto indexers = EvalIndexers(g, op);
  if (failed(indexers)) {
    return failure();
  }

  result.indexers = std::move(*indexers);

  auto loop_extents = EvalLoopExtents(g, *operand_refs_dim, op);
  if (failed(loop_extents)) {
    return failure();
  }

  if (loop_extents->size() != D) {
    return op->emitError("Loop extents size mismatch: expected ")
           << D << " but got " << loop_extents->size() << ". Structure:\n"
           << LinalgGenericToString(g);
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

LogicalResult DeriveInputShape(const LinalgGeneric& g,
                               const BroadcastAnalysis& A, int operand_index,
                               InputShapeInfo& info, mlir::Operation* op) {
  if (operand_index < 0 ||
      operand_index >= static_cast<int>(g.operands.size())) {
    return op->emitError("Invalid operand index ")
           << operand_index << " (valid range: 0-" << (g.operands.size() - 1)
           << "). Structure:\n"
           << LinalgGenericToString(g);
  }

  const auto& operand = g.operands[operand_index];
  if (operand.is_output) {
    return op->emitError("Operand ")
           << operand_index
           << " is marked as output but should be input. Structure:\n"
           << LinalgGenericToString(g);
  }

  const int R = operand.type.shape.size();
  info.in_shape.resize(R);

  // Build input shape from MLIR tensor dimensions (keep real shape)
  for (int axis = 0; axis < R; ++axis) {
    info.in_shape[axis] = SizeExpr(operand.type.shape[axis]);
  }

  // Store same shape for DSLX (reversal will be done in codegen)
  info.dslx_shape = info.in_shape;

  return success();
}

LogicalResult DeriveOutputShape(const LinalgGeneric& g,
                                const BroadcastAnalysis& A,
                                int out_operand_index, OutputShapeInfo& info,
                                mlir::Operation* op) {
  if (out_operand_index < 0 ||
      out_operand_index >= static_cast<int>(g.operands.size())) {
    return op->emitError("Invalid output operand index ")
           << out_operand_index << " (valid range: 0-"
           << (g.operands.size() - 1) << "). Structure:\n"
           << LinalgGenericToString(g);
  }

  const auto& out = g.operands[out_operand_index];
  if (!out.is_output) {
    return op->emitError("Operand ")
           << out_operand_index
           << " is marked as input but should be output. Structure:\n"
           << LinalgGenericToString(g);
  }

  const int D = g.dims.size();
  const int R = out.type.shape.size();
  info.out_shape.resize(R);

  for (int axis = 0; axis < R; ++axis) {
    const auto& e = out.map.results[axis];
    if (e.kind != AffineExpr::kVar) {
      return op->emitError("Output operand ")
             << out_operand_index << " affine expression " << axis
             << " is not a variable (kind: " << static_cast<int>(e.kind)
             << "). Output affine maps must only use variables. Structure:\n"
             << LinalgGenericToString(g);
    }

    int d = e.var;
    if (d < 0 || d >= D) {
      return op->emitError("Output operand ")
             << out_operand_index << " affine expression " << axis
             << " references invalid dimension " << d << " (valid range: 0-"
             << (D - 1) << "). Structure:\n"
             << LinalgGenericToString(g);
    }
    if (g.dims[d].kind != IterKind::kParallel) {
      return op->emitError("Output operand ")
             << out_operand_index << " affine expression " << axis
             << " references reduction dimension " << d
             << ". Output affine maps can only reference parallel dimensions. "
                "Structure:\n"
             << LinalgGenericToString(g);
    }

    info.out_shape[axis] = A.loop_extents[d];
  }

  return success();
}

FailureOr<std::pair<std::vector<InputShapeInfo>, std::vector<OutputShapeInfo>>>
DeriveShapes(const LinalgGeneric& linalg, const BroadcastAnalysis& broadcast,
             mlir::Operation* op) {
  std::vector<InputShapeInfo> input_shapes;
  std::vector<OutputShapeInfo> output_shapes;

  // Derive shapes for all operands (inputs and outputs)
  for (size_t i = 0; i < linalg.operands.size(); ++i) {
    if (!linalg.operands[i].is_output) {
      // Handle input operand
      InputShapeInfo input_shape;
      if (failed(DeriveInputShape(linalg, broadcast, i, input_shape, op))) {
        return op->emitError("Failed to derive input shape for operand ")
               << i << ". Structure:\n"
               << LinalgGenericToString(linalg);
      }
      input_shapes.push_back(std::move(input_shape));
    } else {
      // Handle output operand
      OutputShapeInfo output_shape;
      if (failed(DeriveOutputShape(linalg, broadcast, i, output_shape, op))) {
        return op->emitError("Failed to derive output shape for operand ")
               << i << ". Structure:\n"
               << LinalgGenericToString(linalg);
      }
      output_shapes.push_back(std::move(output_shape));
    }
  }

  return std::make_pair(std::move(input_shapes), std::move(output_shapes));
}

FailureOr<std::string> ExtractName(mlir::Operation* op) {
  if (auto func_op = op->getParentOfType<mlir::func::FuncOp>()) {
    return func_op.getName().str();
  }
  return std::string("generated_linalg_function");
}

LogicalResult ValidateReduction(const LinalgGeneric& linalg,
                                mlir::Operation* op) {
  bool foundSupportedReduction = false;
  for (const auto& region_op : linalg.region.ops) {
    switch (region_op.kind) {
      case OpKind::kAddF:
      case OpKind::kSubF:
        foundSupportedReduction = true;
        break;
      case OpKind::kMulF:
      case OpKind::kDivF:
      case OpKind::kMaxF:
      case OpKind::kMinF:
        return op->emitError("LinalgReductionToXlsPattern: initialization of ")
               << (region_op.kind == OpKind::kMulF   ? "multiplication"
                   : region_op.kind == OpKind::kDivF ? "division"
                   : region_op.kind == OpKind::kMaxF ? "maximum"
                                                     : "minimum")
               << " reduction operation is not supported";
      default:
        // Other operations like constants are fine
        break;
    }
  }
  if (!foundSupportedReduction) {
    return op->emitError(
        "LinalgReductionToXlsPattern: no supported reduction operation found "
        "(only add and sub are supported)");
  }
  return success();
}

}  // namespace mlir::xls
