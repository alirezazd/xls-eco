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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_LINALG_ANALYSIS_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_LINALG_ANALYSIS_H_

#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"

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

// Core analysis functions for converting MLIR to internal representation
AffineMap AnalyzeAffineMap(mlir::AffineMap mlir_map);
mlir::LogicalResult AnalyzeDimension(mlir::utils::IteratorType iterator_type, 
                                    size_t dim_index, Dim& dim);
mlir::LogicalResult AnalyzeOperand(mlir::Value value, const std::string& name, 
                                  bool is_output, mlir::AffineMap indexing_map, 
                                  Operand& operand);
mlir::LogicalResult AnalyzeRegion(mlir::Region& mlir_region, Region& result);

// Main analysis function for linalg.generic operations
mlir::LogicalResult AnalyzeLinalgGeneric(mlir::Operation* op, LinalgGeneric& result);

// Validation and utility functions
mlir::LogicalResult Validate(const LinalgGeneric& g);
std::string LinalgGenericToString(const LinalgGeneric& linalg);

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_LINALG_ANALYSIS_H_
