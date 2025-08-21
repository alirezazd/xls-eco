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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_EVAL_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_EVAL_H_

#include <cassert>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_types.h"

namespace mlir {
class AffineMap;
class Operation;
class Region;
class Value;
}  // namespace mlir

namespace mlir::utils {
enum class IteratorType : uint32_t;
}  // namespace mlir::utils

namespace mlir::xls {

// Forward declarations
struct LinalgEvalResults;
struct InputShapeInfo;
struct OutputShapeInfo;
struct BroadcastAnalysis;
struct LinalgGeneric;

// Main function to evaluate Linalg generic operations
FailureOr<LinalgEvalResults> EvalLinalgGeneric(mlir::Operation* op);

// Helper function to derive input and output shapes
FailureOr<std::pair<std::vector<InputShapeInfo>, std::vector<OutputShapeInfo>>> 
DeriveShapes(const LinalgGeneric& linalg, const BroadcastAnalysis& broadcast, mlir::Operation* op);

// Helper function to validate reduction operations
LogicalResult ValidateReduction(const LinalgGeneric& linalg, mlir::Operation* op);

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_EVAL_H_
