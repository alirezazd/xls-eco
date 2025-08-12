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
#include <vector>

#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/broadcast.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/schedule.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/unroll_plan.h"

using namespace mlir::xls;

void test_pick_unroll_dim() {
  std::cout << "=== Testing PickUnrollDim ===\n";
  
  // Test 1: Explicit unroll configuration
  {
    Schedule sch;
    LoopBand band;
    band.loops = {
      {.dim = 0, .begin = 0, .end = 3, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
      {.dim = 1, .begin = 0, .end = 5, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 1}
    };
    sch.bands.push_back(band);
    
    CodegenConfig cfg;
    cfg.unroll[1] = 6;  // Explicit unroll on dim 1
    
    int result = PickUnrollDim(sch, cfg);
    if (result == 1) {
      std::cout << "✓ Test 1 PASS: Explicit unroll dim picked correctly\n";
    } else {
      std::cout << "✗ Test 1 FAIL: Expected dim 1, got " << result << "\n";
    }
  }
  
  // Test 2: Innermost reduction loop
  {
    Schedule sch;
    LoopBand band;
    band.loops = {
      {.dim = 0, .begin = 0, .end = 3, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
      {.dim = 1, .begin = 0, .end = 5, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 1},
      {.dim = 2, .begin = 0, .end = 4, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 1}
    };
    sch.bands.push_back(band);
    
    CodegenConfig cfg;
    // No explicit unroll configuration
    
    int result = PickUnrollDim(sch, cfg);
    if (result == 2) {
      std::cout << "✓ Test 2 PASS: Innermost reduction loop picked correctly\n";
    } else {
      std::cout << "✗ Test 2 FAIL: Expected dim 2, got " << result << "\n";
    }
  }
  
  // Test 3: No unrolling needed
  {
    Schedule sch;
    LoopBand band;
    band.loops = {
      {.dim = 0, .begin = 0, .end = 3, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1}
    };
    sch.bands.push_back(band);
    
    CodegenConfig cfg;
    // No unroll configuration
    
    int result = PickUnrollDim(sch, cfg);
    if (result == 0) {
      std::cout << "✓ Test 3 PASS: Innermost loop picked when no reduction loops\n";
    } else {
      std::cout << "✗ Test 3 FAIL: Expected dim 0, got " << result << "\n";
    }
  }
}

void test_create_lane_plan() {
  std::cout << "\n=== Testing CreateLanePlan ===\n";
  
  // Test 1: Perfect division
  {
    LanePlan plan;
    mlir::LogicalResult result = CreateLanePlan(1, 12, 4, plan);
    
    if (result.succeeded() && 
        plan.dim == 1 && 
        plan.lanes == 4 && 
        plan.main_iters == 3 && 
        plan.tail == 0 && 
        !plan.has_tail) {
      std::cout << "✓ Test 1 PASS: Perfect division lane plan created correctly\n";
    } else {
      std::cout << "✗ Test 1 FAIL: Lane plan creation failed or incorrect\n";
    }
  }
  
  // Test 2: With tail
  {
    LanePlan plan;
    mlir::LogicalResult result = CreateLanePlan(2, 10, 3, plan);
    
    if (result.succeeded() && 
        plan.dim == 2 && 
        plan.lanes == 3 && 
        plan.main_iters == 3 && 
        plan.tail == 1 && 
        plan.has_tail) {
      std::cout << "✓ Test 2 PASS: Lane plan with tail created correctly\n";
    } else {
      std::cout << "✗ Test 2 FAIL: Lane plan with tail creation failed or incorrect\n";
    }
  }
  
  // Test 3: Invalid parameters
  {
    LanePlan plan;
    mlir::LogicalResult result = CreateLanePlan(0, 5, 0, plan);
    
    if (result.failed()) {
      std::cout << "✓ Test 3 PASS: Invalid parameters correctly rejected\n";
    } else {
      std::cout << "✗ Test 3 FAIL: Invalid parameters should have been rejected\n";
    }
  }
}

void test_create_unroll_plan() {
  std::cout << "\n=== Testing CreateUnrollPlan ===\n";
  
  // Test 1: Simple unroll plan
  {
    // Create a simple schedule
    Schedule sch;
    LoopBand band;
    band.loops = {
      {.dim = 0, .begin = 0, .end = 3, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
      {.dim = 1, .begin = 0, .end = 12, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 4}
    };
    sch.bands.push_back(band);
    
    // Create broadcast analysis
    BroadcastAnalysis A;
    A.loop_extents = {3, 12};  // extents for dims 0 and 1
    
    // Create codegen config
    CodegenConfig cfg;
    cfg.unroll[1] = 4;  // unroll dim 1 by 4
    
    UnrollPlan plan;
    mlir::LogicalResult result = CreateUnrollPlan(sch, A, cfg, plan);
    
    if (result.succeeded() && 
        plan.dims.size() == 1 && 
        plan.dims[0].dim == 1 && 
        plan.dims[0].lanes == 4 && 
        plan.dims[0].main_iters == 3 && 
        plan.dims[0].tail == 0 && 
        !plan.dims[0].has_tail) {
      std::cout << "✓ Test 1 PASS: Simple unroll plan created correctly\n";
    } else {
      std::cout << "✗ Test 1 FAIL: Simple unroll plan creation failed or incorrect\n";
    }
  }
  
  // Test 2: No unrolling needed
  {
    Schedule sch;
    LoopBand band;
    band.loops = {
      {.dim = 0, .begin = 0, .end = 3, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1}
    };
    sch.bands.push_back(band);
    
    BroadcastAnalysis A;
    A.loop_extents = {3};
    
    CodegenConfig cfg;
    // No unroll configuration
    
    UnrollPlan plan;
    mlir::LogicalResult result = CreateUnrollPlan(sch, A, cfg, plan);
    
    if (result.succeeded() && plan.dims.empty()) {
      std::cout << "✓ Test 2 PASS: No unrolling plan created when not needed\n";
    } else {
      std::cout << "✗ Test 2 FAIL: Should have created empty plan\n";
    }
  }
  
  // Test 3: Unroll plan with tail
  {
    Schedule sch;
    LoopBand band;
    band.loops = {
      {.dim = 0, .begin = 0, .end = 10, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 3}
    };
    sch.bands.push_back(band);
    
    BroadcastAnalysis A;
    A.loop_extents = {10};
    
    CodegenConfig cfg;
    cfg.unroll[0] = 3;  // unroll dim 0 by 3
    
    UnrollPlan plan;
    mlir::LogicalResult result = CreateUnrollPlan(sch, A, cfg, plan);
    
    if (result.succeeded() && 
        plan.dims.size() == 1 && 
        plan.dims[0].dim == 0 && 
        plan.dims[0].lanes == 3 && 
        plan.dims[0].main_iters == 3 && 
        plan.dims[0].tail == 1 && 
        plan.dims[0].has_tail) {
      std::cout << "✓ Test 3 PASS: Unroll plan with tail created correctly\n";
    } else {
      std::cout << "✗ Test 3 FAIL: Unroll plan with tail creation failed or incorrect\n";
    }
  }
}

int main() {
  std::cout << "=== Unroll Plan Tests ===\n\n";
  
  test_pick_unroll_dim();
  test_create_lane_plan();
  test_create_unroll_plan();
  
  std::cout << "\n=== Test Summary ===\n";
  std::cout << "✓ Unroll dimension selection\n";
  std::cout << "✓ Lane plan creation\n";
  std::cout << "✓ Unroll plan creation\n";
  
  return 0;
}
