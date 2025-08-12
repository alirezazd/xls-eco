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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_CODEGEN_DSLX_CODEGEN_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_CODEGEN_DSLX_CODEGEN_H_

#include <string>
#include <vector>
#include <ostream>

#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/broadcast.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/schedule.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/unroll_plan.h"
#include "xls/contrib/mlir/transforms/linalg/utils/reduction_utils.h"

namespace mlir::xls {

// Context for DSLX code emission with indentation and formatting support.
struct DslxEmissionContext {
  std::string function_name;
  bool hoist_invariants = false;
  int indent_level = 0;
  
  // Helper for indentation
  std::string indent() const;
  void increase_indent();
  void decrease_indent();
};

// DSLX code generator for linalg.generic operations.
// Converts analyzed linalg operations into executable DSLX code.
class DslxCodegen {
 public:
  explicit DslxCodegen(std::ostream& output);
  
  // Main emission function for complete linalg operations.
  mlir::LogicalResult EmitFunction(const LinalgGeneric& g,
                                   const BroadcastAnalysis& A,
                                   const Schedule& sch,
                                   const UnrollPlan& up,
                                   const ReduceCtx& rctx,
                                   const DslxEmissionContext& ctx);
  
  // Function header emission (public for testing).
  mlir::LogicalResult EmitFunctionHeader(const LinalgGeneric& g,
                                        const BroadcastAnalysis& A,
                                        const DslxEmissionContext& ctx);
  
  // Zero initialization for outputs (public for testing).
  mlir::LogicalResult EmitZeroInits(const LinalgGeneric& g,
                                   const BroadcastAnalysis& A,
                                   const DslxEmissionContext& ctx);
  
  // Unrolled loop emission (public for testing).
  mlir::LogicalResult EmitUnrolledLoop(const LinalgGeneric& g,
                                      const BroadcastAnalysis& A,
                                      const LanePlan& lane_plan,
                                      const ReduceCtx& rctx,
                                      const DslxEmissionContext& ctx);
  
  // Accumulator logic emission (public for testing).
  mlir::LogicalResult EmitAccumulatorLogic(const ReduceCtx& rctx,
                                          const DslxEmissionContext& ctx);
   
 private:
  std::ostream& output_;
  
  // Loop emission for nested loop structure.
  mlir::LogicalResult EmitLoops(const LinalgGeneric& g,
                               const BroadcastAnalysis& A,
                               const Schedule& sch,
                               const UnrollPlan& up,
                               const ReduceCtx& rctx,
                               const DslxEmissionContext& ctx);
  
  // Helper functions for type and code generation.
  std::string GetOperandType(const Operand& operand, const BroadcastAnalysis& A);
  std::string GetReturnType(const LinalgGeneric& g, const BroadcastAnalysis& A);
  std::string BuildZeroTensor(const std::vector<int64_t>& shape,
                             const DslxEmissionContext& ctx);
  std::string EmitUpdateCall(const std::string& tensor_name,
                            const std::string& indices,
                            const std::string& value);
};

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_CODEGEN_DSLX_CODEGEN_H_
