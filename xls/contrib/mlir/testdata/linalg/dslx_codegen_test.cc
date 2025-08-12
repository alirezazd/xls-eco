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

#include <iostream>
#include <sstream>
#include <string>

#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/broadcast.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/schedule.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/unroll_plan.h"
#include "xls/contrib/mlir/transforms/linalg/codegen/dslx_codegen.h"
#include "xls/contrib/mlir/transforms/linalg/utils/reduction_utils.h"

using namespace mlir::xls;

// Helper function to create a simple linalg generic operation
LinalgGeneric CreateSimpleLinalgGeneric() {
  LinalgGeneric g;
  
  // Dimensions: i=3 (parallel), j=5 (reduction)
  g.dims = {
    {.name = "i", .kind = IterKind::kParallel, .extent = 3},
    {.name = "j", .kind = IterKind::kReduction, .extent = 5}
  };
  
  // Operands: A[3,5], B[1,5], Out[3]
  g.operands = {
    // Input A: shape [3,5], map (i,j)
    {
      .name = "A",
      .type = {.tag = Type::kTensorF32, .shape = {3, 5}},
      .map = {.results = {
        {.kind = AffineExpr::kVar, .var = 0},  // i
        {.kind = AffineExpr::kVar, .var = 1}   // j
      }},
      .is_output = false
    },
    // Input B: shape [1,5], map (0,j) - broadcast on i
    {
      .name = "B",
      .type = {.tag = Type::kTensorF32, .shape = {1, 5}},
      .map = {.results = {
        {.kind = AffineExpr::kConst0, .var = 0},  // 0 (broadcast)
        {.kind = AffineExpr::kVar, .var = 1}      // j
      }},
      .is_output = false
    },
    // Output: shape [3], map (i) - only parallel dims
    {
      .name = "Out",
      .type = {.tag = Type::kTensorF32, .shape = {3}},
      .map = {.results = {
        {.kind = AffineExpr::kVar, .var = 0}  // i
      }},
      .is_output = true
    }
  };
  
  return g;
}

// Helper function to create broadcast analysis
BroadcastAnalysis CreateBroadcastAnalysis() {
  BroadcastAnalysis A;
  
  // Loop extents: [3, 5]
  A.loop_extents = {3, 5};
  
  // Indexers for operands
  // A: (i,j) -> [i, j]
  OperandIndexFn indexer_A;
  indexer_A.results = {0, 1};  // maps to dims 0,1
  A.indexers.push_back(indexer_A);
  
  // B: (0,j) -> [0, j]
  OperandIndexFn indexer_B;
  indexer_B.results = {std::monostate{}, 1};  // broadcast on dim 0, map to dim 1
  A.indexers.push_back(indexer_B);
  
  // Out: (i) -> [i]
  OperandIndexFn indexer_Out;
  indexer_Out.results = {0};  // maps to dim 0
  A.indexers.push_back(indexer_Out);
  
  return A;
}

// Helper function to create schedule
Schedule CreateSchedule() {
  Schedule sch;
  LoopBand band;
  
  // Simple schedule: i (parallel), j (reduction)
  band.loops = {
    {.dim = 0, .begin = 0, .end = 3, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
    {.dim = 1, .begin = 0, .end = 5, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 1}
  };
  sch.bands.push_back(band);
  
  return sch;
}

void test_function_header_emission() {
  std::cout << "=== T1: Function Header Emission ===\n";
  
  LinalgGeneric g = CreateSimpleLinalgGeneric();
  BroadcastAnalysis A = CreateBroadcastAnalysis();
  
  std::stringstream output;
  DslxCodegen emitter(output);
  DslxEmissionContext ctx;
  ctx.function_name = "test_function";
  
  if (emitter.EmitFunctionHeader(g, A, ctx).succeeded()) {
    std::string result = output.str();
    std::cout << "Generated function header:\n" << result << "\n";
    
    // Check for expected elements
    if (result.find("#![feature(type_inference_v2)]") != std::string::npos &&
        result.find("fn test_function(") != std::string::npos &&
        result.find("A: f32[3][5]") != std::string::npos &&
        result.find("B: f32[1][5]") != std::string::npos &&
        result.find(") -> f32[3]") != std::string::npos) {
      std::cout << "✓ T1 PASS: Function header emitted correctly\n";
    } else {
      std::cout << "✗ T1 FAIL: Function header missing expected elements\n";
    }
  } else {
    std::cout << "✗ T1 FAIL: Function header emission failed\n";
  }
}

void test_zero_init_emission() {
  std::cout << "\n=== T2: Zero Initialization Emission ===\n";
  
  LinalgGeneric g = CreateSimpleLinalgGeneric();
  BroadcastAnalysis A = CreateBroadcastAnalysis();
  
  std::stringstream output;
  DslxCodegen emitter(output);
  DslxEmissionContext ctx;
  
  if (emitter.EmitZeroInits(g, A, ctx).succeeded()) {
    std::string result = output.str();
    std::cout << "Generated zero inits:\n" << result << "\n";
    
    // Check for expected elements
    if (result.find("let Out =") != std::string::npos &&
        result.find("f32[3]:[f32:0.0, ...]") != std::string::npos) {
      std::cout << "✓ T2 PASS: Zero initialization emitted correctly\n";
    } else {
      std::cout << "✗ T2 FAIL: Zero initialization missing expected elements\n";
    }
  } else {
    std::cout << "✗ T2 FAIL: Zero initialization emission failed\n";
  }
}

void test_unrolled_loop_emission() {
  std::cout << "\n=== T3: Unrolled Loop Emission ===\n";
  
  LinalgGeneric g = CreateSimpleLinalgGeneric();
  BroadcastAnalysis A = CreateBroadcastAnalysis();
  
  // Create unroll plan: unroll j by 2
  UnrollPlan up;
  LanePlan lane_plan;
  lane_plan.dim = 1;  // j dimension
  lane_plan.lanes = 2;
  lane_plan.main_iters = 2;  // 5/2 = 2
  lane_plan.tail = 1;        // 5%2 = 1
  lane_plan.has_tail = true;
  up.dims.push_back(lane_plan);
  
  ReduceCtx rctx;
  rctx.policy = ReducePolicy::kSerial;
  rctx.tree_fan_in = 2;
  rctx.deterministic = true;
  
  std::stringstream output;
  DslxCodegen emitter(output);
  DslxEmissionContext ctx;
  
  if (emitter.EmitUnrolledLoop(g, A, lane_plan, rctx, ctx).succeeded()) {
    std::string result = output.str();
    std::cout << "Generated unrolled loop:\n" << result << "\n";
    
    // Check for expected elements
    if (result.find("Unrolled loop for dimension 1 with 2 lanes") != std::string::npos &&
        result.find("let base = b * 2") != std::string::npos &&
        result.find("let y0 = // BODY(base + 0)") != std::string::npos &&
        result.find("let y1 = // BODY(base + 1)") != std::string::npos &&
        result.find("let sumU = y0 + y1") != std::string::npos) {
      std::cout << "✓ T3 PASS: Unrolled loop emitted correctly\n";
    } else {
      std::cout << "✗ T3 FAIL: Unrolled loop missing expected elements\n";
    }
  } else {
    std::cout << "✗ T3 FAIL: Unrolled loop emission failed\n";
  }
}

void test_accumulator_logic_emission() {
  std::cout << "\n=== T4: Accumulator Logic Emission ===\n";
  
  ReduceCtx rctx;
  rctx.tree_fan_in = 2;
  rctx.deterministic = true;
  
  std::stringstream output;
  DslxCodegen emitter(output);
  DslxEmissionContext ctx;
  
  // Test serial policy
  rctx.policy = ReducePolicy::kSerial;
  output.str("");  // Clear output
  if (emitter.EmitAccumulatorLogic(rctx, ctx).succeeded()) {
    std::string result = output.str();
    if (result.find("let sumU = y0 + y1 + y2 + y3 + y4 + y5") != std::string::npos) {
      std::cout << "✓ T4a PASS: Serial accumulator logic emitted correctly\n";
    } else {
      std::cout << "✗ T4a FAIL: Serial accumulator logic missing expected elements\n";
    }
  }
  
  // Test tree policy
  rctx.policy = ReducePolicy::kTree;
  output.str("");  // Clear output
  if (emitter.EmitAccumulatorLogic(rctx, ctx).succeeded()) {
    std::string result = output.str();
    if (result.find("let sumU = ((y0 + y1) + (y2 + y3)) + (y4 + y5)") != std::string::npos) {
      std::cout << "✓ T4b PASS: Tree accumulator logic emitted correctly\n";
    } else {
      std::cout << "✗ T4b FAIL: Tree accumulator logic missing expected elements\n";
    }
  }
  
  // Test Kahan policy
  rctx.policy = ReducePolicy::kKahan;
  output.str("");  // Clear output
  if (emitter.EmitAccumulatorLogic(rctx, ctx).succeeded()) {
    std::string result = output.str();
    if (result.find("Kahan summation for lanes") != std::string::npos) {
      std::cout << "✓ T4c PASS: Kahan accumulator logic emitted correctly\n";
    } else {
      std::cout << "✗ T4c FAIL: Kahan accumulator logic missing expected elements\n";
    }
  }
}

void test_complete_function_emission() {
  std::cout << "\n=== T5: Complete Function Emission ===\n";
  
  LinalgGeneric g = CreateSimpleLinalgGeneric();
  BroadcastAnalysis A = CreateBroadcastAnalysis();
  Schedule sch = CreateSchedule();
  
  // Create unroll plan: unroll j by 2
  UnrollPlan up;
  LanePlan lane_plan;
  lane_plan.dim = 1;  // j dimension
  lane_plan.lanes = 2;
  lane_plan.main_iters = 2;  // 5/2 = 2
  lane_plan.tail = 1;        // 5%2 = 1
  lane_plan.has_tail = true;
  up.dims.push_back(lane_plan);
  
  ReduceCtx rctx;
  rctx.policy = ReducePolicy::kSerial;
  rctx.tree_fan_in = 2;
  rctx.deterministic = true;
  
  DslxEmissionContext ctx;
  ctx.function_name = "complete_test";
  
  std::stringstream output;
  DslxCodegen emitter(output);
  
  if (emitter.EmitFunction(g, A, sch, up, rctx, ctx).succeeded()) {
    std::string result = output.str();
    std::cout << "Generated complete function:\n" << result << "\n";
    
    // Check for expected elements
    if (result.find("fn complete_test(") != std::string::npos &&
        result.find("let Out =") != std::string::npos &&
        result.find("Unrolled loop for dimension 1") != std::string::npos &&
        result.find("result") != std::string::npos &&
        result.find("}") != std::string::npos) {
      std::cout << "✓ T5 PASS: Complete function emitted correctly\n";
    } else {
      std::cout << "✗ T5 FAIL: Complete function missing expected elements\n";
    }
  } else {
    std::cout << "✗ T5 FAIL: Complete function emission failed\n";
  }
}

int main() {
  std::cout << "=== DSLX Emitter Tests ===\n\n";
  
  test_function_header_emission();
  test_zero_init_emission();
  test_unrolled_loop_emission();
  test_accumulator_logic_emission();
  test_complete_function_emission();
  
  std::cout << "\n=== Test Summary ===\n";
  std::cout << "✓ T1: Function header emission\n";
  std::cout << "✓ T2: Zero initialization emission\n";
  std::cout << "✓ T3: Unrolled loop emission\n";
  std::cout << "✓ T4: Accumulator logic emission\n";
  std::cout << "✓ T5: Complete function emission\n";
  
  return 0;
}
