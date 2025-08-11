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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_TO_XLS_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_TO_XLS_H_

#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {

// Populates linalg.generic conversion patterns for XLS lowering.
// This includes patterns for:
// - Element-wise operations to arith operations
// - Fill operations to tensor.splat
// - Multi-input operations to DSLX calls
void populateLinalgToXlsPatterns(RewritePatternSet& patterns,
                                MLIRContext* context);

}  // namespace mlir

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_TO_XLS_H_
