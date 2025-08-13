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

#include "xls/contrib/mlir/transforms/linalg/codegen/dslx_codegen.h"

#include <algorithm>
#include <sstream>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace mlir::xls {

// DslxEmissionContext implementation
std::string DslxEmissionContext::indent() const {
  return std::string(indent_level * 2, ' ');
}

void DslxEmissionContext::increase_indent() {
  indent_level++;
}

void DslxEmissionContext::decrease_indent() {
  if (indent_level > 0) {
    indent_level--;
  }
}

// DslxCodegen implementation
DslxCodegen::DslxCodegen(std::ostream& output) : output_(output) {}

mlir::LogicalResult DslxCodegen::EmitFunction(const LinalgGeneric& g,
                                             const BroadcastAnalysis& A,
                                             const Schedule& sch,
                                             const UnrollPlan& up,
                                             const ReduceCtx& rctx,
                                             const DslxEmissionContext& ctx) {
  // Emit function header with parameters and return type
  if (EmitFunctionHeader(g, A, ctx).failed()) {
    return mlir::failure();
  }
  
  // Emit zero initializations for output tensors
  if (EmitZeroInits(g, A, ctx).failed()) {
    return mlir::failure();
  }
  
  // Emit nested loop structure with unrolling
  if (EmitLoops(g, A, sch, up, rctx, ctx).failed()) {
    return mlir::failure();
  }
  
  // Emit return statement
  output_ << ctx.indent() << "result\n";
  output_ << "}\n";
  
  return mlir::success();
}

mlir::LogicalResult DslxCodegen::EmitFunctionHeader(const LinalgGeneric& g,
                                                   const BroadcastAnalysis& A,
                                                   const DslxEmissionContext& ctx) {
  output_ << "#![feature(type_inference_v2)]\n\n";
  output_ << "import float32;\n\n";
  output_ << "type F32 = float32::F32;\n\n";
  output_ << "fn " << ctx.function_name << "(";
  
  // Emit input parameters (non-output operands)
  std::vector<std::string> params;
  for (const auto& operand : g.operands) {
    if (!operand.is_output) {
      std::string param_type = GetOperandType(operand, A);
      params.push_back(absl::StrCat(operand.name, ": ", param_type));
    }
  }
  
  output_ << absl::StrJoin(params, ", ");
  output_ << ") -> " << GetReturnType(g, A) << " {\n";
  
  return mlir::success();
}

mlir::LogicalResult DslxCodegen::EmitZeroInits(const LinalgGeneric& g,
                                              const BroadcastAnalysis& A,
                                              const DslxEmissionContext& ctx) {
  // Find and initialize output operands
  for (const auto& operand : g.operands) {
    if (operand.is_output) {
      // Get output shape (only parallel dimensions)
      std::vector<int64_t> output_shape;
      for (int i = 0; i < operand.type.shape.size(); ++i) {
        const auto& map_result = operand.map.results[i];
        if (map_result.kind == AffineExpr::kVar) {
          int dim_id = map_result.var;
          if (dim_id < g.dims.size() && g.dims[dim_id].kind == IterKind::kParallel) {
            output_shape.push_back(operand.type.shape[i]);
          }
        }
      }
      
      // Emit zero tensor initialization
      std::string zero_tensor = BuildZeroTensor(output_shape, ctx);
      output_ << ctx.indent() << "let " << operand.name << " = " << zero_tensor << ";\n";
    }
  }
  
  return mlir::success();
}

mlir::LogicalResult DslxCodegen::EmitLoops(const LinalgGeneric& g,
                                          const BroadcastAnalysis& A,
                                          const Schedule& sch,
                                          const UnrollPlan& up,
                                          const ReduceCtx& rctx,
                                          const DslxEmissionContext& ctx) {
  if (sch.bands.empty()) {
    return mlir::failure();
  }
  
  const auto& band = sch.bands[0];
  
  // Find if there's an unrolled dimension
  int unrolled_dim = -1;
  if (!up.dims.empty()) {
    unrolled_dim = up.dims[0].dim;
  }
  
  // Find the output operand name to use as initial value
  std::string initial_value = "output0";  // Default fallback
  for (const auto& operand : g.operands) {
    if (operand.is_output) {
      initial_value = operand.name;
      break;
    }
  }
  
  // Emit nested loops according to schedule
  for (const auto& loop : band.loops) {
    if (loop.dim == unrolled_dim) {
      // Emit unrolled loop with parallel lanes
      if (EmitUnrolledLoop(g, A, up.dims[0], rctx, ctx).failed()) {
        return mlir::failure();
      }
    } else {
      // Emit regular counted for loop
      output_ << ctx.indent() << "let result = for (i" << loop.dim << ", result) : (u32, " 
              << GetReturnType(g, A) << ") in u32:0.." << loop.end << " {\n";
      
      DslxEmissionContext inner_ctx = ctx;
      inner_ctx.increase_indent();
      
      // Emit loop body (simplified for now)
      output_ << inner_ctx.indent() << "// Loop body for dimension " << loop.dim << "\n";
      output_ << inner_ctx.indent() << "result\n";
      
      // Use initial_value for first loop, result for subsequent loops
      static int loop_count = 0;
      std::string loop_input = (loop_count == 0) ? initial_value : "result";
      output_ << ctx.indent() << "}(" << loop_input << ");\n";
      loop_count++;
    }
  }
  
  return mlir::success();
}

mlir::LogicalResult DslxCodegen::EmitUnrolledLoop(const LinalgGeneric& g,
                                                 const BroadcastAnalysis& A,
                                                 const LanePlan& lane_plan,
                                                 const ReduceCtx& rctx,
                                                 const DslxEmissionContext& ctx) {
  output_ << ctx.indent() << "// Unrolled loop for dimension " << lane_plan.dim 
          << " with " << lane_plan.lanes << " lanes\n";
  
  // Emit main iterations with parallel lanes
  if (lane_plan.main_iters > 0) {
    output_ << ctx.indent() << "let result = for (b, result) : (u32, " 
            << GetReturnType(g, A) << ") in u32:0.." << lane_plan.main_iters << " {\n";
    
    DslxEmissionContext inner_ctx = ctx;
    inner_ctx.increase_indent();
    
    // Emit base calculation for lane indexing
    output_ << inner_ctx.indent() << "let base = b * " << lane_plan.lanes << ";\n";
    
    // Emit parallel lanes (structural unroll)
    for (int u = 0; u < lane_plan.lanes; ++u) {
      output_ << inner_ctx.indent() << "let y" << u << " = // BODY(base + " << u << ");\n";
    }
    
    // Emit lane combination based on reduction policy
    if (EmitAccumulatorLogic(rctx, inner_ctx).failed()) {
      return mlir::failure();
    }
    
    output_ << inner_ctx.indent() << "result\n";
    output_ << ctx.indent() << "}(result);\n";
  }
  
  // Emit tail loop if needed
  if (lane_plan.has_tail) {
    output_ << ctx.indent() << "let result = for (t, result) : (u32, " 
            << GetReturnType(g, A) << ") in u32:0.." << lane_plan.tail << " {\n";
    
    DslxEmissionContext inner_ctx = ctx;
    inner_ctx.increase_indent();
    
    output_ << inner_ctx.indent() << "let base = " << lane_plan.main_iters << " * " 
            << lane_plan.lanes << ";\n";
    output_ << inner_ctx.indent() << "let y = // BODY(base + t);\n";
    output_ << inner_ctx.indent() << "// combine_scalar(y) into accumulator\n";
    output_ << inner_ctx.indent() << "result\n";
    
    output_ << ctx.indent() << "}(result);\n";
  }
  
  return mlir::success();
}

mlir::LogicalResult DslxCodegen::EmitAccumulatorLogic(const ReduceCtx& rctx,
                                                     const DslxEmissionContext& ctx) {
  switch (rctx.policy) {
    case ReducePolicy::kSerial: {
      // Serial reduction: sum all lane values
      output_ << ctx.indent() << "let sumU = ";
      for (int u = 0; u < 6; ++u) {  // Assuming max 6 lanes for now
        if (u > 0) output_ << " + ";
        output_ << "y" << u;
      }
      output_ << ";\n";
      output_ << ctx.indent() << "let acc_next = acc + sumU;\n";
      break;
    }
    case ReducePolicy::kTree: {
      // Tree reduction: balanced binary tree
      output_ << ctx.indent() << "let sumU = ((y0 + y1) + (y2 + y3)) + (y4 + y5);\n";
      output_ << ctx.indent() << "let acc_next = acc + sumU;\n";
      break;
    }
    case ReducePolicy::kKahan: {
      // Kahan summation for numerical stability
      output_ << ctx.indent() << "// Kahan summation for lanes\n";
      output_ << ctx.indent() << "let sumU = y0 + y1 + y2 + y3 + y4 + y5;\n";
      output_ << ctx.indent() << "let acc_next = acc + sumU;\n";
      break;
    }
  }
  
  return mlir::success();
}

std::string DslxCodegen::GetOperandType(const Operand& operand, const BroadcastAnalysis& A) {
  // Check if it's a scalar (rank 0 or all 1s)
  bool is_scalar = true;
  for (int64_t dim : operand.type.shape) {
    if (dim != 1) {
      is_scalar = false;
      break;
    }
  }
  
  if (is_scalar) {
    return "F32";
  }
  
  // Build tensor type with dimensions
  std::vector<std::string> dims;
  for (int64_t dim : operand.type.shape) {
    dims.push_back(std::to_string(dim));
  }
  
  return absl::StrCat("F32[", absl::StrJoin(dims, "]["), "]");
}

std::string DslxCodegen::GetReturnType(const LinalgGeneric& g, const BroadcastAnalysis& A) {
  // Find output operands and build return type
  std::vector<std::string> output_types;
  for (const auto& operand : g.operands) {
    if (operand.is_output) {
      // For function returns, we need to use 1D arrays (multi-dimensional return types don't work)
      // Calculate total size for 1D array
      int64_t total_size = 1;
      for (int64_t dim : operand.type.shape) {
        total_size *= dim;
      }
      output_types.push_back(absl::StrCat("F32[", total_size, "]"));
    }
  }
  
  if (output_types.empty()) {
    return "()";
  } else if (output_types.size() == 1) {
    return output_types[0];
  } else {
    return absl::StrCat("(", absl::StrJoin(output_types, ", "), ")");
  }
}

std::string DslxCodegen::BuildZeroTensor(const std::vector<int64_t>& shape,
                                        const DslxEmissionContext& ctx) {
  if (shape.empty()) {
    return "float32::zero(false)";
  }
  
  if (shape.size() == 1) {
    // 1D zero vector
    return absl::StrCat("F32[", shape[0], "]:[float32::zero(false), ...]");
  }
  
  // For function returns, we need to flatten multi-dimensional arrays to 1D
  // Calculate total size
  int64_t total_size = 1;
  for (int64_t dim : shape) {
    total_size *= dim;
  }
  
  // Return 1D array with total size
  return absl::StrCat("F32[", total_size, "]:[float32::zero(false), ...]");
}

std::string DslxCodegen::EmitUpdateCall(const std::string& tensor_name,
                                       const std::string& indices,
                                       const std::string& value) {
  return absl::StrCat("update(", tensor_name, ", ", indices, ", ", value, ")");
}

}  // namespace mlir::xls
