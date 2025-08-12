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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_BROADCAST_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_BROADCAST_H_

#include "mlir/Support/LogicalResult.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"

namespace mlir::xls {

// Analyzes broadcast patterns in linalg.generic operations.
// Determines loop extents, operand variation patterns, and indexing functions.
mlir::LogicalResult AnalyzeBroadcast(const LinalgGeneric& g, BroadcastAnalysis& result);

// Derives output shape information from broadcast analysis.
// Validates that output operands only reference parallel dimensions.
mlir::LogicalResult DeriveOutputShape(const LinalgGeneric& g, const BroadcastAnalysis& A,
                                     int out_operand_index, OutputShapeInfo& info);

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_BROADCAST_H_
