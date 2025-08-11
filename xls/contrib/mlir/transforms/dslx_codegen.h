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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_DSLX_CODEGEN_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_DSLX_CODEGEN_H_

#include <string>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace xls {

/// Options for DSLX code generation strategies
struct DslxCodegenOptions {
  /// Optimization strategy: "area" or "performance"
  std::string optimization_strategy = "performance";
  
  /// Whether to generate debug comments in the DSLX code
  bool include_debug_comments = false;
  
  /// Maximum loop unroll factor for performance optimization
  int max_unroll_factor = 8;
};

/// DSLX code generator for complex linalg patterns
class DslxCodegen {
 public:
  explicit DslxCodegen(const DslxCodegenOptions& options = DslxCodegenOptions{})
      : options_(options) {}

  /// Generate DSLX code for a complex linalg.generic operation
  /// Returns the generated DSLX code as a string
  mlir::LogicalResult generateDslxCode(mlir::Operation* linalg_op,
                                      std::string& output_dslx_code);

  /// Check if the linalg operation can be handled by this codegen
  bool canHandle(mlir::Operation* linalg_op);

 private:
  /// Generate optimized DSLX code for performance
  mlir::LogicalResult generatePerformanceOptimizedCode(
      mlir::Operation* linalg_op, std::string& output_dslx_code);

  /// Generate optimized DSLX code for area
  mlir::LogicalResult generateAreaOptimizedCode(
      mlir::Operation* linalg_op, std::string& output_dslx_code);

  /// Analyze the linalg operation to determine its structure
  mlir::LogicalResult analyzeLinalgOperation(mlir::Operation* linalg_op);

  /// Generate function signature for the DSLX function
  std::string generateFunctionSignature(mlir::Operation* linalg_op);

  /// Generate the function body based on the linalg operation
  std::string generateFunctionBody(mlir::Operation* linalg_op);

  DslxCodegenOptions options_;
  
  // Analysis results
  struct LinalgAnalysis {
    std::vector<int64_t> tensor_shapes;
    std::vector<std::string> iterator_types;
    std::vector<std::string> operations;
    bool has_reduction = false;
    bool has_parallel = false;
  };
  
  LinalgAnalysis analysis_;
};

}  // namespace xls
}  // namespace mlir

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_DSLX_CODEGEN_H_
