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

#include "xls/contrib/mlir/transforms/linalg_analysis.h"

#include <sstream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace xls {

mlir::LogicalResult AnalyzeLinalgGeneric(mlir::Operation* op, LinalgGeneric& result) {
  auto generic_op = mlir::dyn_cast<mlir::linalg::GenericOp>(op);
  if (!generic_op) {
    return mlir::failure();
  }
  
  // Clear previous result
  result = LinalgGeneric{};

  // Analyze dimensions
  auto iterator_types = generic_op.getIteratorTypesArray();
  for (size_t i = 0; i < iterator_types.size(); ++i) {
    Dim dim;
    dim.name = std::string("d") + std::to_string(i);  // d0, d1, d2, d3, etc.
    
    if (iterator_types[i] == mlir::utils::IteratorType::parallel) {
      dim.kind = IterKind::kParallel;
    } else if (iterator_types[i] == mlir::utils::IteratorType::reduction) {
      dim.kind = IterKind::kReduction;
    } else {
      return mlir::failure();  // Unsupported iterator type
    }
    
    // For now, assume static extents (will be refined later)
    dim.extent = 1;  // Placeholder
    
    result.dims.push_back(dim);
  }

  // Analyze operands
  auto inputs = generic_op.getInputs();
  auto outputs = generic_op.getOutputs();
  
  // Process inputs
  for (size_t i = 0; i < inputs.size(); ++i) {
    Operand operand;
    operand.name = "input" + std::to_string(i);
    operand.is_output = false;
    
    if (auto tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(inputs[i].getType())) {
      operand.type.tag = Type::kTensorF32;
      for (auto dim : tensor_type.getShape()) {
        if (mlir::ShapedType::isDynamic(dim)) {
          operand.type.shape.push_back(-1);  // Dynamic dimension
        } else {
          operand.type.shape.push_back(dim);
        }
      }
    } else if (inputs[i].getType().isF32()) {
      operand.type.tag = Type::kScalarF32;
      // Empty shape for scalars
    } else {
      return mlir::failure();  // Unsupported type
    }
    
    // Analyze affine map for this operand
    auto indexing_map = generic_op.getIndexingMapsArray()[i];
    operand.map = AnalyzeAffineMap(indexing_map);
    
    result.operands.push_back(operand);
  }
  
  // Process outputs
  for (size_t i = 0; i < outputs.size(); ++i) {
    Operand operand;
    operand.name = "output" + std::to_string(i);
    operand.is_output = true;
    
    if (auto tensor_type = llvm::dyn_cast<mlir::RankedTensorType>(outputs[i].getType())) {
      operand.type.tag = Type::kTensorF32;
      for (auto dim : tensor_type.getShape()) {
        if (mlir::ShapedType::isDynamic(dim)) {
          operand.type.shape.push_back(-1);  // Dynamic dimension
        } else {
          operand.type.shape.push_back(dim);
        }
      }
    } else if (outputs[i].getType().isF32()) {
      operand.type.tag = Type::kScalarF32;
      // Empty shape for scalars
    } else {
      return mlir::failure();  // Unsupported type
    }
    
    // Analyze affine map for this operand
    auto indexing_map = generic_op.getIndexingMapsArray()[inputs.size() + i];
    operand.map = AnalyzeAffineMap(indexing_map);
    
    result.operands.push_back(operand);
  }

  // Analyze region
  return AnalyzeRegion(generic_op.getRegion(), result.region);
}

AffineMap AnalyzeAffineMap(mlir::AffineMap mlir_map) {
  AffineMap result;
  
  for (auto expr : mlir_map.getResults()) {
    AffineExpr affine_expr;
    
    if (auto const_expr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
      if (const_expr.getValue() == 0) {
        affine_expr.kind = AffineExpr::kConst0;
        affine_expr.var = 0;
      } else {
        // For now, only support constant 0
        return AffineMap{};
      }
    } else if (auto dim_expr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      affine_expr.kind = AffineExpr::kVar;
      affine_expr.var = dim_expr.getPosition();
    } else {
      // For now, only support simple expressions
      return AffineMap{};
    }
    
    result.results.push_back(affine_expr);
  }
  
  return result;
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
  
  // Analyze operations in the block
  int next_id = block.getNumArguments();
  std::vector<std::pair<mlir::Value, ValueId>> value_to_id_pairs;
  
  // Initialize block arguments
  for (size_t i = 0; i < block.getNumArguments(); ++i) {
    value_to_id_pairs.push_back({block.getArgument(i), ValueId(i)});
  }
  
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
    
    // Analyze operation kind
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
          return mlir::failure();  // Unsupported comparison
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
      return mlir::failure();  // Unsupported operation
    }
    
    // Analyze operands
    for (auto operand : op.getOperands()) {
      for (const auto& pair : value_to_id_pairs) {
        if (pair.first == operand) {
          region_op.inputs.push_back(pair.second);
          break;
        }
      }
    }
    
    // Record the result
    value_to_id_pairs.push_back({op.getResult(0), region_op.result});
    
    result.ops.push_back(region_op);
  }
  
  return mlir::success();
}

std::string LinalgGenericToString(const LinalgGeneric& linalg) {
  std::ostringstream oss;
  
  oss << "LinalgGeneric {\n";
  
  // Dimensions
  oss << "  dims: [";
  for (size_t i = 0; i < linalg.dims.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << linalg.dims[i].name << "(" 
        << (linalg.dims[i].kind == IterKind::kParallel ? "P" : "R") 
        << ":" << linalg.dims[i].extent << ")";
  }
  oss << "]\n";
  
  // Operands
  oss << "  operands: [";
  for (size_t i = 0; i < linalg.operands.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << linalg.operands[i].name << ":" 
        << (linalg.operands[i].is_output ? "out" : "in");
  }
  oss << "]\n";
  
  // Region
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

}  // namespace xls
}  // namespace mlir
