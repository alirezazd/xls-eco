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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_CODEGEN_REDUCTION_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_CODEGEN_REDUCTION_H_

#include <array>
#include <vector>

#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"

namespace mlir::xls {

// High-level reduction orchestration
Scalar SerialReduction(const LinalgGeneric& g, const BroadcastAnalysis& A, 
                      const Schedule& sch, const CodegenConfig& cfg,
                      const std::array<int, 4>& parallel_indices);

std::vector<Scalar> MultipleOutputReduction(const LinalgGeneric& g, const BroadcastAnalysis& A,
                                           const Schedule& sch, const CodegenConfig& cfg,
                                           const std::array<int, 4>& parallel_indices);

// Loop processing functions
void ProcessSingleLaneLoop(const LinalgGeneric& g, const BroadcastAnalysis& A,
                          const LoopBand::Loop& loop, const ReduceCtx& rctx,
                          const std::array<int, 4>& parallel_indices, AccState& st);

void ProcessUnrolledLoop(const LinalgGeneric& g, const BroadcastAnalysis& A,
                        const LoopBand::Loop& loop, const ReduceCtx& rctx,
                        const std::array<int, 4>& parallel_indices, AccState& st);

// Region evaluation functions
float EvaluateRegionForLane(const LinalgGeneric& g, const BroadcastAnalysis& A,
                           const std::array<int, 4>& current_indices, const AccState& st);

std::vector<float> EvaluateRegionForMultipleOutputs(const LinalgGeneric& g, const BroadcastAnalysis& A,
                                                   const std::array<int, 4>& current_indices, 
                                                   const std::vector<Scalar>& accs);

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_CODEGEN_REDUCTION_H_
