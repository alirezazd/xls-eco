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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_DSLXGEN_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_DSLXGEN_H_

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

// IWYU pragma: begin_exports
#include "xls/contrib/mlir/transforms/linalg/schedule_types.h"
// IWYU pragma: end_exports

namespace mlir::xls {

// Forward declarations
struct LinalgEvalResults;
struct Schedule;
struct Operand;
struct InputShapeInfo;
struct OutputShapeInfo;
struct DslxStmt;
struct SizeExpr;
struct Accumulator;
struct Region;

// Class for generating DSLX code from MLIR Linalg operations. Key
// functionality:
//   (1) Converts MLIR Linalg operations to DSLX function definitions.
//   (2) Handles parallel and reduction loop generation with proper nesting.
//   (3) Manages tensor indexing and type construction for XLS array
//   representation. (4) Generates accumulator initialization and update
//   patterns for reductions.
class DslxGen {
 public:
  DslxGen();
  ~DslxGen() = default;

  FailureOr<std::string> GenerateDslxCode(const LinalgEvalResults& eval_results,
                                          const Schedule& schedule);

 private:
  // Internal stream for accumulating generated DSLX code.
  std::stringstream dslx_stream_;

  // Cached output type for reuse across loop generation.
  std::string OutT_;

  // Core emission methods
  LogicalResult InitStream();
  LogicalResult EmitHeader();
  LogicalResult EmitInits(const LinalgEvalResults& eval_results,
                          const Schedule& schedule);
  LogicalResult EmitSignature(const LinalgEvalResults& eval_results);
  LogicalResult EmitLoopBody(const LinalgEvalResults& eval_results,
                             const Schedule& schedule);

  // Sub-emit helpers for header
  LogicalResult EmitFileHeader();
  LogicalResult EmitFeatureFlag();
  LogicalResult EmitImports();
  LogicalResult EmitF32TypeAliases();
  LogicalResult EmitConstants();

  // Helper methods for signature emission
  LogicalResult EmitFunctionName(const std::string& function_name);
  LogicalResult EmitInputParameters(
      const std::vector<Operand>& operands,
      const std::vector<InputShapeInfo>& input_shapes);
  LogicalResult EmitReturnType(
      const std::vector<Operand>& operands,
      const std::vector<OutputShapeInfo>& output_shapes);

  // High-level IR construction and emission
  FailureOr<std::vector<DslxStmt>> BuildBodyStatements(
      const LinalgEvalResults& eval_results, const Schedule& schedule);

  // Utility methods
  FailureOr<std::string> BuildParameterType(const Operand& operand);
  FailureOr<std::string> BuildParameterTypeFromDerived(
      const InputShapeInfo& input_shape);
  FailureOr<std::string> BuildDslxTypeFromSizeExprs(
      const std::vector<SizeExpr>& dimensions);
  FailureOr<std::string> BuildTypedNestedZeroPattern(
      const std::vector<std::string>& dims);

  // Helper functions
  int64_t GetLoopBoundFromSchedule(const Schedule& schedule, int logical_dim);
  int CalculateLoopBitwidth(int64_t bound);

  // Recursive loop building methods
  FailureOr<DslxStmt> BuildParallel(int p_pos,
                                    const LinalgEvalResults& eval_results,
                                    const std::vector<int>& P,
                                    const std::vector<int>& R,
                                    const Schedule& schedule, int indent,
                                    const std::string& OutT);
  FailureOr<DslxStmt> BuildReduction(int r_pos,
                                     const LinalgEvalResults& eval_results,
                                     const std::vector<int>& P,
                                     const std::vector<int>& R,
                                     const Schedule& schedule, int indent);

  // Create accumulator for parallel loop
  FailureOr<Accumulator> MakeParallelAcc(int p_pos, int mP,
                                         const std::vector<int>& P,
                                         const Schedule& schedule);

  // Create accumulator for reduction loop
  FailureOr<Accumulator> MakeReductionAcc(
      int r_pos, int mR, const LinalgEvalResults& eval_results);

  LogicalResult EmitStatements(const std::vector<DslxStmt>& stmts,
                               int indent = 2);
  LogicalResult EmitStatement(const DslxStmt& stmt, int indent);

  // Helper functions for type construction using dimension mapping
  std::string MakeSuffixTypeWithMapping(int k, const Schedule& schedule);
  std::string MakeRowTWithMapping(const Schedule& schedule);

  // Helper function to build tensor indexing expression using dimension mapping
  std::string BuildTensorIndexingWithMapping(const Operand& operand,
                                             const Schedule& schedule);

  // Build expression from linalg region using sequential let statements
  FailureOr<std::pair<std::vector<DslxStmt>, std::string>>
  BuildRegionExpression(const Region& region,
                        const std::vector<Operand>& operands,
                        const std::string& accumulator_name,
                        const Schedule& schedule);

  // Implementation details are in the .cc file
};

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_DSLXGEN_H_
