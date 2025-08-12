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

#include "xls/contrib/mlir/transforms/linalg/analysis/linalg_analysis.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/utils/helpers.h"

namespace mlir::xls {

// Core analysis functions for converting MLIR to internal representation

AffineMap AnalyzeAffineMap(mlir::AffineMap mlir_map) {
  AffineMap result;

  for (auto expr : mlir_map.getResults()) {
    AffineExpr affine_expr;

    if (auto const_expr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
      if (const_expr.getValue() == 0) {
        affine_expr.kind = AffineExpr::kConst0;
        affine_expr.var = 0;
      } else {
        return AffineMap{};
      }
    } else if (auto dim_expr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      affine_expr.kind = AffineExpr::kVar;
      affine_expr.var = dim_expr.getPosition();
    } else {
      return AffineMap{};
    }

    result.results.push_back(affine_expr);
  }

  return result;
}

mlir::LogicalResult AnalyzeDimension(mlir::utils::IteratorType iterator_type, 
                                    size_t dim_index, Dim& dim) {
  dim.name = std::string("d") + std::to_string(dim_index);
  
  if (iterator_type == mlir::utils::IteratorType::parallel) {
    dim.kind = IterKind::kParallel;
  } else if (iterator_type == mlir::utils::IteratorType::reduction) {
    dim.kind = IterKind::kReduction;
  } else {
    return mlir::failure();
  }
  
  dim.extent = 1;  // Default extent - should be refined based on actual tensor dimensions
  return mlir::success();
}

mlir::LogicalResult AnalyzeOperand(mlir::Value value, const std::string& name, 
                                  bool is_output, mlir::AffineMap indexing_map, 
                                  Operand& operand) {
  operand.name = name;
  operand.is_output = is_output;
  operand.map = AnalyzeAffineMap(indexing_map);
  
  if (auto tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(value.getType())) {
    operand.type.tag = Type::kTensorF32;
    for (auto dim : tensor_type.getShape()) {
      if (mlir::ShapedType::isDynamic(dim)) {
        operand.type.shape.push_back(-1);
      } else {
        operand.type.shape.push_back(dim);
      }
    }
  } else if (value.getType().isF32()) {
    operand.type.tag = Type::kScalarF32;
  } else {
    return mlir::failure();
  }
  
  return mlir::success();
}

mlir::LogicalResult AnalyzeRegion(mlir::Region& mlir_region, Region& result) {
  if (mlir_region.empty()) {
    return mlir::failure();
  }

  auto& block = mlir_region.front();

  // Analyze block arguments
  for (size_t i = 0; i < block.getNumArguments(); ++i) {
    result.args.push_back(ValueId(i));
  }

  int next_id = block.getNumArguments();
  std::vector<std::pair<mlir::Value, ValueId>> value_to_id_pairs;

  for (size_t i = 0; i < block.getNumArguments(); ++i) {
    value_to_id_pairs.push_back({block.getArgument(i), ValueId(i)});
  }

  // Analyze operations in the block
  for (auto& op : block) {
    if (mlir::isa<mlir::linalg::YieldOp>(op)) {
      // Handle yield operation
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

    // Map MLIR operations to our internal representation
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
        case mlir::arith::CmpFPredicate::OGT: region_op.kind = OpKind::kCmpOGT; break;
        case mlir::arith::CmpFPredicate::OLT: region_op.kind = OpKind::kCmpOLT; break;
        case mlir::arith::CmpFPredicate::OEQ: region_op.kind = OpKind::kCmpOEQ; break;
        case mlir::arith::CmpFPredicate::OGE: region_op.kind = OpKind::kCmpOGE; break;
        case mlir::arith::CmpFPredicate::OLE: region_op.kind = OpKind::kCmpOLE; break;
        case mlir::arith::CmpFPredicate::ONE: region_op.kind = OpKind::kCmpONE; break;
        default: return mlir::failure();
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
      if (auto float_attr = llvm::dyn_cast<mlir::FloatAttr>(const_op.getValue())) {
        region_op.f32_imm = float_attr.getValueAsDouble();
      }
    } else {
      return mlir::failure();
    }

    // Map operands to their ValueIds
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

  return mlir::success();
}

mlir::LogicalResult AnalyzeLinalgGeneric(mlir::Operation* op, LinalgGeneric& result) {
  auto generic_op = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
  if (!generic_op) {
    return mlir::failure();
  }

  result = LinalgGeneric{};

  // Analyze iterator types to determine dimensions
  auto iterator_types = generic_op.getIteratorTypesArray();
  for (size_t i = 0; i < iterator_types.size(); ++i) {
    Dim dim;
    if (failed(AnalyzeDimension(iterator_types[i], i, dim))) {
      return mlir::failure();
    }
    result.dims.push_back(dim);
  }

  // Analyze inputs and outputs
  auto inputs = generic_op.getInputs();
  auto outputs = generic_op.getOutputs();
  auto indexing_maps = generic_op.getIndexingMapsArray();

  // Process input operands
  for (size_t i = 0; i < inputs.size(); ++i) {
    Operand operand;
    if (failed(AnalyzeOperand(inputs[i], "input" + std::to_string(i), false,
                             indexing_maps[i], operand))) {
      return mlir::failure();
    }
    result.operands.push_back(operand);
  }

  // Process output operands
  for (size_t i = 0; i < outputs.size(); ++i) {
    Operand operand;
    if (failed(AnalyzeOperand(outputs[i], "output" + std::to_string(i), true,
                             indexing_maps[inputs.size() + i], operand))) {
      return mlir::failure();
    }
    result.operands.push_back(operand);
  }

  // Analyze the region body
  return AnalyzeRegion(generic_op.getRegion(), result.region);
}

mlir::LogicalResult Validate(const LinalgGeneric& g) {
  if (g.dims.empty()) {
    return mlir::failure();
  }

  // Validate dimension extents
  for (const auto& d : g.dims) {
    if (d.extent < 1) {
      return mlir::failure();
    }
  }

  // Validate operand indexing maps
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

  // Validate region structure
  if (!IsDag(g.region.ops) || !AllYieldsDefined(g.region)) {
    return mlir::failure();
  }

  return mlir::success();
}

std::string LinalgGenericToString(const LinalgGeneric& linalg) {
  std::ostringstream oss;

  oss << "LinalgGeneric {\n";

  // Print dimensions
  oss << "  dims: [";
  for (size_t i = 0; i < linalg.dims.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << linalg.dims[i].name << "("
        << (linalg.dims[i].kind == IterKind::kParallel ? "P" : "R")
        << ":" << linalg.dims[i].extent << ")";
  }
  oss << "]\n";

  // Print operands
  oss << "  operands: [";
  for (size_t i = 0; i < linalg.operands.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << linalg.operands[i].name << ":"
        << (linalg.operands[i].is_output ? "out" : "in");
  }
  oss << "]\n";

  // Print region
  oss << "  region: {\n";
  oss << "    args: [";
  for (size_t i = 0; i < linalg.region.args.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << "%" << linalg.region.args[i].id;
  }
  oss << "]\n";

  oss << "    ops: [";
  for (size_t i = 0; i < linalg.region.ops.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << "%" << linalg.region.ops[i].result.id << " = ";
    switch (linalg.region.ops[i].kind) {
      case OpKind::kAddF: oss << "addf"; break;
      case OpKind::kMulF: oss << "mulf"; break;
      case OpKind::kSubF: oss << "subf"; break;
      case OpKind::kDivF: oss << "divf"; break;
      case OpKind::kMaxF: oss << "maxf"; break;
      case OpKind::kMinF: oss << "minf"; break;
      case OpKind::kCmpOGT: oss << "cmpf_ogt"; break;
      case OpKind::kSelect: oss << "select"; break;
      case OpKind::kConstF32: oss << "const_f32(" << linalg.region.ops[i].f32_imm << ")"; break;
      default: oss << "unknown"; break;
    }
    oss << "(";
    for (size_t j = 0; j < linalg.region.ops[i].inputs.size(); ++j) {
      if (j > 0) oss << ", ";
      oss << "%" << linalg.region.ops[i].inputs[j].id;
    }
    oss << ")";
  }
  oss << "]\n";

  oss << "    yields: [";
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
