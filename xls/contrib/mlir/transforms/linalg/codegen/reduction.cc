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

#include "xls/contrib/mlir/transforms/linalg/codegen/reduction.h"

#include <algorithm>
#include <array>
#include <vector>

#include "xls/contrib/mlir/transforms/linalg/analysis/broadcast.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/schedule.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"
#include "xls/contrib/mlir/transforms/linalg/utils/helpers.h"
#include "xls/contrib/mlir/transforms/linalg/utils/reduction_utils.h"

namespace mlir::xls {

Scalar SerialReduction(const LinalgGeneric& g, const BroadcastAnalysis& A, 
                      const Schedule& sch, const CodegenConfig& cfg,
                      const std::array<int, 4>& parallel_indices) {
  // Initialize AccState per (parallel indices)
  AccState st;  // init to {0,0}
  
  // Convert CodegenConfig to ReduceCtx
  ReduceCtx rctx;
  switch (cfg.reduce_policy) {
    case CodegenConfig::kSerial:
      rctx.policy = ReducePolicy::kSerial;
      break;
    case CodegenConfig::kKahan:
      rctx.policy = ReducePolicy::kKahan;
      break;
    case CodegenConfig::kTree:
      rctx.policy = ReducePolicy::kTree;
      break;
  }
  rctx.tree_fan_in = 2;
  rctx.deterministic = true;

  // Iterate all reduction dimensions (respecting schedule / tiles / unroll)
  for (const auto& band : sch.bands) {
    for (const auto& loop : band.loops) {
      if (!loop.is_reduction) continue;  // Skip parallel dimensions
      
      // Handle lane unrolling
      if (loop.unroll > 1) {
        ProcessUnrolledLoop(g, A, loop, rctx, parallel_indices, st);
      } else {
        ProcessSingleLaneLoop(g, A, loop, rctx, parallel_indices, st);
      }
    }
  }
  
  // Write back st.acc to output
  Scalar result;
  if (cfg.reduce_policy == CodegenConfig::kKahan) {
    result.tag = Scalar::kTupleF32;
    result.kahan = {st.acc, st.c};
  } else {
    result.tag = Scalar::kF32;
    result.f = st.acc;
  }
  
  return result;
}

void ProcessSingleLaneLoop(const LinalgGeneric& g, const BroadcastAnalysis& A,
                          const LoopBand::Loop& loop, const ReduceCtx& rctx,
                          const std::array<int, 4>& parallel_indices, AccState& st) {
  // For each iteration of this reduction loop
  for (int64_t i = loop.begin; i < loop.end; i += loop.step) {
    // Create current loop indices (parallel + current reduction)
    std::array<int, 4> current_indices = parallel_indices;
    current_indices[loop.dim] = i;
    
    // Load region args
    ValMap val_map;
    
    // For each input block-arg, compute indices and load
    for (size_t arg_idx = 0; arg_idx < g.region.args.size(); ++arg_idx) {
      if (arg_idx < g.operands.size() && !g.operands[arg_idx].is_output) {
        // Input operand
        auto indices = MakeOperandIndex(arg_idx, current_indices, A);
        float value = Load(arg_idx, indices);
        
        Scalar scalar;
        scalar.tag = Scalar::kF32;
        scalar.f = value;
        val_map[g.region.args[arg_idx]] = scalar;
      } else {
        // Output block-arg (carried partial) - bind to current accumulator
        Scalar acc_scalar;
        acc_scalar.tag = Scalar::kF32;
        acc_scalar.f = st.acc;
        val_map[g.region.args[arg_idx]] = acc_scalar;
      }
    }
    
    // Evaluate region ops in topological order
    for (const auto& op : g.region.ops) {
      Scalar result_scalar;
      
      switch (op.kind) {
        case OpKind::kAddF: {
          if (op.inputs.size() == 2) {
            float lhs = val_map[op.inputs[0]].f;
            float rhs = val_map[op.inputs[1]].f;
            result_scalar.tag = Scalar::kF32;
            result_scalar.f = lhs + rhs;
          }
          break;
        }
        case OpKind::kMulF: {
          if (op.inputs.size() == 2) {
            float lhs = val_map[op.inputs[0]].f;
            float rhs = val_map[op.inputs[1]].f;
            result_scalar.tag = Scalar::kF32;
            result_scalar.f = lhs * rhs;
          }
          break;
        }
        case OpKind::kConstF32: {
          result_scalar.tag = Scalar::kF32;
          result_scalar.f = op.f32_imm;
          break;
        }
        case OpKind::kCmpOGT: {
          if (op.inputs.size() == 2) {
            float lhs = val_map[op.inputs[0]].f;
            float rhs = val_map[op.inputs[1]].f;
            result_scalar.tag = Scalar::kF32;
            result_scalar.f = (lhs > rhs) ? 1.0f : 0.0f;  // Simplified IEEE "ordered gt"
          }
          break;
        }
        case OpKind::kSelect: {
          if (op.inputs.size() == 3) {
            float cond = val_map[op.inputs[0]].f;
            float true_val = val_map[op.inputs[1]].f;
            float false_val = val_map[op.inputs[2]].f;
            result_scalar.tag = Scalar::kF32;
            result_scalar.f = (cond != 0.0f) ? true_val : false_val;
          }
          break;
        }
        default:
          // For unsupported ops, use a default value
          result_scalar.tag = Scalar::kF32;
          result_scalar.f = 0.0f;
          break;
      }
      
      val_map[op.result] = result_scalar;
    }
    
    // On yield: compute yield (scalar) and use ReduceUpdate
    if (!g.region.yields.empty()) {
      float yield_value = val_map[g.region.yields[0]].f;
      
      // Use ReduceUpdate with single value
      ReduceUpdate(st, {yield_value}, rctx);
    }
  }
}

void ProcessUnrolledLoop(const LinalgGeneric& g, const BroadcastAnalysis& A,
                        const LoopBand::Loop& loop, const ReduceCtx& rctx,
                        const std::array<int, 4>& parallel_indices, AccState& st) {
  int unroll_factor = loop.unroll;
  
  // Calculate how many full unrolled iterations we can do
  int64_t full_iterations = (loop.end - loop.begin) / unroll_factor;
  int64_t remainder = (loop.end - loop.begin) % unroll_factor;
  
  // Process full unrolled iterations
  for (int64_t base = 0; base < full_iterations * unroll_factor; base += unroll_factor) {
    std::array<float, 16> lane_vals;  // Fixed-size array for lane values
    int n = 0;
    
    // For each unrolled lane (or just 1):
    for (int u = 0; u < unroll_factor; ++u) {
      int64_t lane_index = loop.begin + base + u;
      
      // Create lane-specific indices
      std::array<int, 4> lane_indices = parallel_indices;
      lane_indices[loop.dim] = lane_index;
      
      // Evaluate region for this lane
      lane_vals[n++] = EvaluateRegionForLane(g, A, lane_indices, st);
    }
    
    // Use ReduceUpdate with lane results
    ReduceUpdate(st, absl::MakeSpan(lane_vals.data(), n), rctx);
  }
  
  // Handle remainder (tail loop)
  if (remainder > 0) {
    for (int64_t i = loop.begin + full_iterations * unroll_factor; i < loop.end; ++i) {
      std::array<int, 4> tail_indices = parallel_indices;
      tail_indices[loop.dim] = i;
      
      float tail_value = EvaluateRegionForLane(g, A, tail_indices, st);
      
      // Use ReduceUpdate for tail value
      ReduceUpdate(st, {tail_value}, rctx);
    }
  }
}



std::vector<Scalar> MultipleOutputReduction(const LinalgGeneric& g, const BroadcastAnalysis& A,
                                           const Schedule& sch, const CodegenConfig& cfg,
                                           const std::array<int, 4>& parallel_indices) {
  std::vector<Scalar> accumulators;
  
  // Initialize one accumulator per yield
  for (size_t i = 0; i < g.region.yields.size(); ++i) {
    Scalar acc;
    if (cfg.reduce_policy == CodegenConfig::kKahan) {
      acc.tag = Scalar::kTupleF32;
      acc.kahan = {0.0f, 0.0f};
    } else {
      acc.tag = Scalar::kF32;
      acc.f = 0.0f;
    }
    accumulators.push_back(acc);
  }
  
  // Process reduction loops (same as serial reduction but with multiple outputs)
  for (const auto& band : sch.bands) {
    for (const auto& loop : band.loops) {
      if (!loop.is_reduction) continue;
      
      // TODO: Implement multiple output reduction with unrolling
      // For now, use single lane processing for all cases
      for (int64_t i = loop.begin; i < loop.end; i += loop.step) {
        std::array<int, 4> current_indices = parallel_indices;
        current_indices[loop.dim] = i;
        
        std::vector<float> yield_values = EvaluateRegionForMultipleOutputs(g, A, current_indices, accumulators);
        
        for (size_t j = 0; j < accumulators.size() && j < yield_values.size(); ++j) {
          if (cfg.reduce_policy == CodegenConfig::kKahan) {
            accumulators[j].kahan.first += yield_values[j];
          } else {
            accumulators[j].f += yield_values[j];
          }
        }
      }
    }
  }
  
  return accumulators;
}



// Region evaluation for single lane with AccState
float EvaluateRegionForLane(const LinalgGeneric& g, const BroadcastAnalysis& A,
                           const std::array<int, 4>& current_indices, const AccState& st) {
  ValMap val_map;
  
  for (size_t arg_idx = 0; arg_idx < g.region.args.size(); ++arg_idx) {
    if (arg_idx < g.operands.size() && !g.operands[arg_idx].is_output) {
      auto indices = MakeOperandIndex(arg_idx, current_indices, A);
      float value = Load(arg_idx, indices);
      
      Scalar scalar;
      scalar.tag = Scalar::kF32;
      scalar.f = value;
      val_map[g.region.args[arg_idx]] = scalar;
    } else {
      Scalar acc_scalar;
      acc_scalar.tag = Scalar::kF32;
      acc_scalar.f = st.acc;
      val_map[g.region.args[arg_idx]] = acc_scalar;
    }
  }
  
  for (const auto& op : g.region.ops) {
    Scalar result_scalar;
    
    switch (op.kind) {
      case OpKind::kAddF: {
        if (op.inputs.size() == 2) {
          float lhs = val_map[op.inputs[0]].f;
          float rhs = val_map[op.inputs[1]].f;
          result_scalar.tag = Scalar::kF32;
          result_scalar.f = lhs + rhs;
        }
        break;
      }
      case OpKind::kMulF: {
        if (op.inputs.size() == 2) {
          float lhs = val_map[op.inputs[0]].f;
          float rhs = val_map[op.inputs[1]].f;
          result_scalar.tag = Scalar::kF32;
          result_scalar.f = lhs * rhs;
        }
        break;
      }
      case OpKind::kConstF32: {
        result_scalar.tag = Scalar::kF32;
        result_scalar.f = op.f32_imm;
        break;
      }
      case OpKind::kCmpOGT: {
        if (op.inputs.size() == 2) {
          float lhs = val_map[op.inputs[0]].f;
          float rhs = val_map[op.inputs[1]].f;
          result_scalar.tag = Scalar::kF32;
          result_scalar.f = (lhs > rhs) ? 1.0f : 0.0f;
        }
        break;
      }
      case OpKind::kSelect: {
        if (op.inputs.size() == 3) {
          float cond = val_map[op.inputs[0]].f;
          float true_val = val_map[op.inputs[1]].f;
          float false_val = val_map[op.inputs[2]].f;
          result_scalar.tag = Scalar::kF32;
          result_scalar.f = (cond != 0.0f) ? true_val : false_val;
        }
        break;
      }
      default:
        result_scalar.tag = Scalar::kF32;
        result_scalar.f = 0.0f;
        break;
    }
    
    val_map[op.result] = result_scalar;
  }
  
  if (!g.region.yields.empty()) {
    return val_map[g.region.yields[0]].f;
  }
  
  return 0.0f;
}

// Region evaluation for multiple outputs
std::vector<float> EvaluateRegionForMultipleOutputs(const LinalgGeneric& g, const BroadcastAnalysis& A,
                                                   const std::array<int, 4>& current_indices, 
                                                   const std::vector<Scalar>& accs) {
  ValMap val_map;
  
  for (size_t arg_idx = 0; arg_idx < g.region.args.size(); ++arg_idx) {
    if (arg_idx < g.operands.size() && !g.operands[arg_idx].is_output) {
      auto indices = MakeOperandIndex(arg_idx, current_indices, A);
      float value = Load(arg_idx, indices);
      
      Scalar scalar;
      scalar.tag = Scalar::kF32;
      scalar.f = value;
      val_map[g.region.args[arg_idx]] = scalar;
    } else {
      size_t acc_idx = arg_idx - g.operands.size();
      if (acc_idx < accs.size()) {
        val_map[g.region.args[arg_idx]] = accs[acc_idx];
      } else {
        Scalar default_acc;
        default_acc.tag = Scalar::kF32;
        default_acc.f = 0.0f;
        val_map[g.region.args[arg_idx]] = default_acc;
      }
    }
  }
  
  for (const auto& op : g.region.ops) {
    Scalar result_scalar;
    
    switch (op.kind) {
      case OpKind::kAddF: {
        if (op.inputs.size() == 2) {
          float lhs = val_map[op.inputs[0]].f;
          float rhs = val_map[op.inputs[1]].f;
          result_scalar.tag = Scalar::kF32;
          result_scalar.f = lhs + rhs;
        }
        break;
      }
      case OpKind::kMulF: {
        if (op.inputs.size() == 2) {
          float lhs = val_map[op.inputs[0]].f;
          float rhs = val_map[op.inputs[1]].f;
          result_scalar.tag = Scalar::kF32;
          result_scalar.f = lhs * rhs;
        }
        break;
      }
      case OpKind::kConstF32: {
        result_scalar.tag = Scalar::kF32;
        result_scalar.f = op.f32_imm;
        break;
      }
      case OpKind::kCmpOGT: {
        if (op.inputs.size() == 2) {
          float lhs = val_map[op.inputs[0]].f;
          float rhs = val_map[op.inputs[1]].f;
          result_scalar.tag = Scalar::kF32;
          result_scalar.f = (lhs > rhs) ? 1.0f : 0.0f;
        }
        break;
      }
      case OpKind::kSelect: {
        if (op.inputs.size() == 3) {
          float cond = val_map[op.inputs[0]].f;
          float true_val = val_map[op.inputs[1]].f;
          float false_val = val_map[op.inputs[2]].f;
          result_scalar.tag = Scalar::kF32;
          result_scalar.f = (cond != 0.0f) ? true_val : false_val;
        }
        break;
      }
      default:
        result_scalar.tag = Scalar::kF32;
        result_scalar.f = 0.0f;
        break;
    }
    
    val_map[op.result] = result_scalar;
  }
  
  std::vector<float> yield_values;
  for (const auto& yield_id : g.region.yields) {
    yield_values.push_back(val_map[yield_id].f);
  }
  
  return yield_values;
}

}  // namespace mlir::xls
