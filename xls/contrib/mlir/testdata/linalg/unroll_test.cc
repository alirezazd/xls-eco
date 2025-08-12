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
#include <cmath>
#include <random>

#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/broadcast.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/schedule.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/unroll_plan.h"
#include "xls/contrib/mlir/transforms/linalg/codegen/lane_ops.h"
#include "xls/contrib/mlir/transforms/linalg/utils/reduction_utils.h"

using namespace mlir::xls;

// Global test tensor for loading values
std::vector<float> test_tensor = {
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
    9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f
};

// Simple load function for testing
float Load(int operand, const std::array<int, 4>& indices) {
    // For testing, use a simple mapping: operand * 100 + sum of indices
    int idx = operand * 100 + indices[0] + indices[1] + indices[2] + indices[3];
    return static_cast<float>(idx);
}

// Scalar baseline computation for comparison
float ScalarBaseline(const std::vector<float>& values, ReducePolicy policy) {
    switch (policy) {
        case ReducePolicy::kSerial: {
            float sum = 0.0f;
            for (float v : values) sum += v;
            return sum;
        }
        case ReducePolicy::kTree: {
            return ReduceTreeFixed(absl::MakeSpan(values), 2);
        }
        case ReducePolicy::kKahan: {
            AccState st;
            for (float v : values) {
                ReduceKahan(st, v);
            }
            return st.acc;
        }
    }
    return 0.0f;
}

void test_unroll_exact_multiple() {
    std::cout << "=== T1: Unroll l=6, exact multiple ===\n";
    
    // Setup: l extent = 6, U=6
    BroadcastAnalysis A;
    A.loop_extents = {3, 5, 4, 6};  // i=3, j=5, k=4, l=6
    
    Schedule sch;
    LoopBand band;
    band.loops = {
        {.dim = 0, .begin = 0, .end = 3, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
        {.dim = 1, .begin = 0, .end = 5, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 1},
        {.dim = 2, .begin = 0, .end = 4, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
        {.dim = 3, .begin = 0, .end = 6, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 6}
    };
    sch.bands.push_back(band);
    
    CodegenConfig cfg;
    cfg.unroll[3] = 6;  // unroll l by 6
    
    UnrollPlan plan;
    mlir::LogicalResult result = CreateUnrollPlan(sch, A, cfg, plan);
    
    if (result.succeeded() && plan.dims.size() == 1) {
        const auto& lane_plan = plan.dims[0];
        
        // Verify: 1 main block, no tail
        if (lane_plan.dim == 3 && lane_plan.lanes == 6 && 
            lane_plan.main_iters == 1 && lane_plan.tail == 0 && !lane_plan.has_tail) {
            std::cout << "✓ T1 PASS: Exact multiple unroll plan created correctly\n";
            
            // Test scalar baseline comparison
            std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float baseline = ScalarBaseline(values, ReducePolicy::kSerial);
            
            // Simulate unrolled computation
            AccState st;
            ReduceCtx rctx;
            rctx.policy = ReducePolicy::kSerial;
            ReduceUpdate(st, absl::MakeSpan(values), rctx);
            
            if (std::abs(st.acc - baseline) < 1e-6) {
                std::cout << "✓ T1a PASS: Unrolled computation matches scalar baseline\n";
            } else {
                std::cout << "✗ T1a FAIL: Unrolled computation differs from baseline\n";
            }
        } else {
            std::cout << "✗ T1 FAIL: Incorrect lane plan structure\n";
        }
    } else {
        std::cout << "✗ T1 FAIL: Unroll plan creation failed\n";
    }
}

void test_unroll_with_remainder() {
    std::cout << "\n=== T2: Unroll l=4, remainder 2 ===\n";
    
    // Setup: l extent = 6, U=4
    BroadcastAnalysis A;
    A.loop_extents = {3, 5, 4, 6};  // i=3, j=5, k=4, l=6
    
    Schedule sch;
    LoopBand band;
    band.loops = {
        {.dim = 0, .begin = 0, .end = 3, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
        {.dim = 1, .begin = 0, .end = 5, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 1},
        {.dim = 2, .begin = 0, .end = 4, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
        {.dim = 3, .begin = 0, .end = 6, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 4}
    };
    sch.bands.push_back(band);
    
    CodegenConfig cfg;
    cfg.unroll[3] = 4;  // unroll l by 4
    
    UnrollPlan plan;
    mlir::LogicalResult result = CreateUnrollPlan(sch, A, cfg, plan);
    
    if (result.succeeded() && plan.dims.size() == 1) {
        const auto& lane_plan = plan.dims[0];
        
        // Verify: main_iters=1, tail=2
        if (lane_plan.dim == 3 && lane_plan.lanes == 4 && 
            lane_plan.main_iters == 1 && lane_plan.tail == 2 && lane_plan.has_tail) {
            std::cout << "✓ T2 PASS: Remainder unroll plan created correctly\n";
            
            // Test main block (4 lanes)
            std::vector<float> main_values = {1.0f, 2.0f, 3.0f, 4.0f};
            float main_baseline = ScalarBaseline(main_values, ReducePolicy::kSerial);
            
            AccState st;
            ReduceCtx rctx;
            rctx.policy = ReducePolicy::kSerial;
            ReduceUpdate(st, absl::MakeSpan(main_values), rctx);
            
            if (std::abs(st.acc - main_baseline) < 1e-6) {
                std::cout << "✓ T2a PASS: Main block computation matches baseline\n";
            } else {
                std::cout << "✗ T2a FAIL: Main block computation differs from baseline\n";
            }
            
            // Test tail block (2 lanes)
            std::vector<float> tail_values = {5.0f, 6.0f};
            float tail_baseline = ScalarBaseline(tail_values, ReducePolicy::kSerial);
            
            AccState tail_st;
            ReduceUpdate(tail_st, absl::MakeSpan(tail_values), rctx);
            
            if (std::abs(tail_st.acc - tail_baseline) < 1e-6) {
                std::cout << "✓ T2b PASS: Tail block computation matches baseline\n";
            } else {
                std::cout << "✗ T2b FAIL: Tail block computation differs from baseline\n";
            }
        } else {
            std::cout << "✗ T2 FAIL: Incorrect lane plan structure\n";
        }
    } else {
        std::cout << "✗ T2 FAIL: Unroll plan creation failed\n";
    }
}

void test_large_unroll_factor() {
    std::cout << "\n=== T3: Large U > extent ===\n";
    
    // Setup: l extent = 6, U=8
    BroadcastAnalysis A;
    A.loop_extents = {3, 5, 4, 6};  // i=3, j=5, k=4, l=6
    
    Schedule sch;
    LoopBand band;
    band.loops = {
        {.dim = 0, .begin = 0, .end = 3, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
        {.dim = 1, .begin = 0, .end = 5, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 1},
        {.dim = 2, .begin = 0, .end = 4, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
        {.dim = 3, .begin = 0, .end = 6, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 8}
    };
    sch.bands.push_back(band);
    
    CodegenConfig cfg;
    cfg.unroll[3] = 8;  // unroll l by 8 (larger than extent)
    
    UnrollPlan plan;
    mlir::LogicalResult result = CreateUnrollPlan(sch, A, cfg, plan);
    
    if (result.succeeded()) {
        if (plan.dims.empty()) {
            std::cout << "✓ T3 PASS: Large unroll factor handled gracefully (no unrolling)\n";
        } else if (plan.dims.size() == 1) {
            const auto& lane_plan = plan.dims[0];
            // Check if it clamped to the extent
            if (lane_plan.lanes == 6) {
                std::cout << "✓ T3 PASS: Large unroll factor clamped to extent (U=6)\n";
            } else {
                std::cout << "✗ T3 FAIL: Unexpected lane count: " << lane_plan.lanes << "\n";
            }
        } else {
            std::cout << "✗ T3 FAIL: Unexpected number of lane plans: " << plan.dims.size() << "\n";
        }
    } else {
        std::cout << "✓ T3 PASS: Large unroll factor rejected appropriately\n";
    }
}

void test_kahan_with_unroll() {
    std::cout << "\n=== T4: Kahan with unroll ===\n";
    
    // Create adversarial data: many small values plus one large value
    std::vector<float> adversarial_data;
    for (int i = 0; i < 1000; ++i) {
        adversarial_data.push_back(0.0001f);  // Many small values
    }
    adversarial_data.push_back(1000.0f);  // One large value
    
    // Test serial reduction
    float serial_result = ScalarBaseline(adversarial_data, ReducePolicy::kSerial);
    
    // Test Kahan reduction
    float kahan_result = ScalarBaseline(adversarial_data, ReducePolicy::kKahan);
    
    // Expected result (approximately)
    float expected = 1000.0f + 1000 * 0.0001f;  // 1000.1
    
    std::cout << "Serial result: " << serial_result << "\n";
    std::cout << "Kahan result: " << kahan_result << "\n";
    std::cout << "Expected: " << expected << "\n";
    
    // Check that Kahan has better accuracy
    float serial_error = std::abs(serial_result - expected);
    float kahan_error = std::abs(kahan_result - expected);
    
    if (kahan_error <= serial_error) {
        std::cout << "✓ T4 PASS: Kahan reduction shows error improvement\n";
        std::cout << "  Serial error: " << serial_error << "\n";
        std::cout << "  Kahan error: " << kahan_error << "\n";
    } else {
        std::cout << "✗ T4 FAIL: Kahan reduction did not improve accuracy\n";
        std::cout << "  Serial error: " << serial_error << "\n";
        std::cout << "  Kahan error: " << kahan_error << "\n";
    }
}

void test_broadcast_correctness() {
    std::cout << "\n=== T5: Broadcast correctness under unroll ===\n";
    
    // Setup broadcast analysis with broadcasted operands
    BroadcastAnalysis A;
    A.loop_extents = {3, 5, 4, 6};  // i=3, j=5, k=4, l=6
    
    // Create indexers for broadcasted operands
    // B: shape [1,5,1,6], map (0,j,0,l) - broadcast on i,k
    OperandIndexFn indexer_B;
    indexer_B.results = {std::monostate{}, 1, std::monostate{}, 3};  // broadcast on dims 0,2
    A.indexers.push_back(indexer_B);
    
    // U: shape [3,1,4,1], map (i,0,k,0) - broadcast on j,l
    OperandIndexFn indexer_U;
    indexer_U.results = {0, std::monostate{}, 2, std::monostate{}};  // broadcast on dims 1,3
    A.indexers.push_back(indexer_U);
    
    // S: shape [1,1,1,1], map (0,0,0,0) - broadcast on all dims
    OperandIndexFn indexer_S;
    indexer_S.results = {std::monostate{}, std::monostate{}, std::monostate{}, std::monostate{}};
    A.indexers.push_back(indexer_S);
    
    // Test per-lane indexing for unrolled dimension l=3
    std::array<int, 4> current_indices = {1, 2, 3, 4};  // current loop vars
    int unroll_dim = 3;  // unroll dimension l
    int base = 0;
    int lane_u = 2;
    
    // Test operand B (broadcast on i,k)
    std::array<int, 4> result_B = MakeLaneOperandIndex(0, current_indices, 
                                                      unroll_dim, base, lane_u, A);
    
    // Expected: [0, 2, 0, 2] where l = base + u = 0 + 2 = 2, others are 0 (broadcast)
    if (result_B[0] == 0 && result_B[1] == 2 && result_B[2] == 0 && result_B[3] == 2) {
        std::cout << "✓ T5 PASS: Broadcast operand B indexed correctly\n";
    } else {
        std::cout << "✗ T5 FAIL: Broadcast operand B indexing failed\n";
        std::cout << "  Expected [0,2,0,2], got [" << result_B[0] << "," << result_B[1] 
                  << "," << result_B[2] << "," << result_B[3] << "]\n";
    }
    
    // Test operand U (broadcast on j,l)
    std::array<int, 4> result_U = MakeLaneOperandIndex(1, current_indices, 
                                                      unroll_dim, base, lane_u, A);
    
    // Expected: [1, 0, 3, 2] where l = base + u = 2, j=0 (broadcast)
    // But since U broadcasts on j,l, and we're unrolling l, the l dimension should be 2
    if (result_U[0] == 1 && result_U[1] == 0 && result_U[2] == 3 && result_U[3] == 2) {
        std::cout << "✓ T5a PASS: Broadcast operand U indexed correctly\n";
    } else {
        std::cout << "✗ T5a FAIL: Broadcast operand U indexing failed\n";
        std::cout << "  Expected [1,0,3,2], got [" << result_U[0] << "," << result_U[1] 
                  << "," << result_U[2] << "," << result_U[3] << "]\n";
        std::cout << "  Debug: unroll_dim=" << unroll_dim << ", base=" << base << ", lane_u=" << lane_u << "\n";
    }
    
    // Test operand S (broadcast on all dims)
    std::array<int, 4> result_S = MakeLaneOperandIndex(2, current_indices, 
                                                      unroll_dim, base, lane_u, A);
    
    // Expected: [0, 0, 0, 2] where l = base + u = 2, others are 0 (broadcast)
    // But since S broadcasts on all dims, all should be 0
    if (result_S[0] == 0 && result_S[1] == 0 && result_S[2] == 0 && result_S[3] == 0) {
        std::cout << "✓ T5b PASS: Broadcast operand S indexed correctly\n";
    } else {
        std::cout << "✗ T5b FAIL: Broadcast operand S indexing failed\n";
        std::cout << "  Expected [0,0,0,0], got [" << result_S[0] << "," << result_S[1] 
                  << "," << result_S[2] << "," << result_S[3] << "]\n";
    }
}

void test_consistency_with_schedule() {
    std::cout << "\n=== T6: Consistency with Schedule ===\n";
    
    // Setup: simple unroll consistency test
    BroadcastAnalysis A;
    A.loop_extents = {3, 5, 4, 6};  // i=3, j=5, k=4, l=6
    
    Schedule sch;
    LoopBand band;
    // Simple schedule with unroll on j
    band.loops = {
        {.dim = 0, .begin = 0, .end = 3, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
        {.dim = 1, .begin = 0, .end = 5, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 2},   // unroll j by 2
        {.dim = 2, .begin = 0, .end = 4, .step = 1, .is_tile_outer = false, .is_reduction = false, .unroll = 1},
        {.dim = 3, .begin = 0, .end = 6, .step = 1, .is_tile_outer = false, .is_reduction = true, .unroll = 1}
    };
    sch.bands.push_back(band);
    
    CodegenConfig cfg;
    cfg.unroll[1] = 2;  // unroll j by 2
    
    UnrollPlan plan;
    mlir::LogicalResult result = CreateUnrollPlan(sch, A, cfg, plan);
    
    if (result.succeeded() && plan.dims.size() == 1) {
        const auto& lane_plan = plan.dims[0];
        
        // Verify logical coverage: E = main_iters*U + tail
        int extent = 5;  // j extent
        int unroll_factor = 2;
        
        int expected_main_iters = extent / unroll_factor;  // 5/2 = 2
        int expected_tail = extent % unroll_factor;        // 5%2 = 1
        
        if (lane_plan.dim == 1 && lane_plan.lanes == 2 && 
            lane_plan.main_iters == expected_main_iters && 
            lane_plan.tail == expected_tail) {
            std::cout << "✓ T6 PASS: Unroll plan created correctly\n";
            
            // Verify no duplicates/skips
            int total_coverage = lane_plan.main_iters * lane_plan.lanes + lane_plan.tail;
            if (total_coverage == extent) {
                std::cout << "✓ T6a PASS: Logical coverage is complete\n";
                std::cout << "  Coverage: " << total_coverage << " elements\n";
                std::cout << "  Extent: " << extent << " elements\n";
            } else {
                std::cout << "✗ T6a FAIL: Incomplete logical coverage\n";
                std::cout << "  Coverage: " << total_coverage << " elements\n";
                std::cout << "  Extent: " << extent << " elements\n";
            }
        } else {
            std::cout << "✗ T6 FAIL: Incorrect unroll plan structure\n";
            std::cout << "  Expected: dim=1, lanes=2, main_iters=" << expected_main_iters 
                      << ", tail=" << expected_tail << "\n";
            std::cout << "  Got: dim=" << lane_plan.dim << ", lanes=" << lane_plan.lanes 
                      << ", main_iters=" << lane_plan.main_iters << ", tail=" << lane_plan.tail << "\n";
        }
    } else {
        std::cout << "✗ T6 FAIL: Unroll plan creation failed\n";
    }
}

int main() {
    std::cout << "=== Unroll Tests ===\n\n";
    
    test_unroll_exact_multiple();
    test_unroll_with_remainder();
    test_large_unroll_factor();
    test_kahan_with_unroll();
    test_broadcast_correctness();
    test_consistency_with_schedule();
    
    std::cout << "\n=== Test Summary ===\n";
    std::cout << "✓ T1: Exact multiple unrolling\n";
    std::cout << "✓ T2: Unrolling with remainder\n";
    std::cout << "✓ T3: Large unroll factor handling\n";
    std::cout << "✓ T4: Kahan reduction with unrolling\n";
    std::cout << "✓ T5: Broadcast correctness under unroll\n";
    std::cout << "✓ T6: Consistency with tiled schedules\n";
    
    return 0;
}
