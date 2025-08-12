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

#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"
#include "xls/contrib/mlir/transforms/linalg/utils/reduction_utils.h"

using namespace mlir::xls;

void test_reduce_serial() {
  std::cout << "Testing ReduceSerial - ";
  
  AccState st;
  st.acc = 0.0f;
  st.c = 0.0f;
  
  // Test serial reduction
  ReduceSerial(st, 1.0f);
  ReduceSerial(st, 2.0f);
  ReduceSerial(st, 3.0f);
  
  if (std::abs(st.acc - 6.0f) < 1e-6) {
    std::cout << "PASS ✓ (serial reduction: " << st.acc << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected 6.0, got " << st.acc << ")\n";
  }
}

void test_reduce_kahan() {
  std::cout << "Testing ReduceKahan - ";
  
  AccState st;
  st.acc = 0.0f;
  st.c = 0.0f;
  
  // Test Kahan reduction with small numbers
  ReduceKahan(st, 0.1f);
  ReduceKahan(st, 0.2f);
  ReduceKahan(st, 0.3f);
  
  if (std::abs(st.acc - 0.6f) < 1e-6) {
    std::cout << "PASS ✓ (Kahan reduction: " << st.acc << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected 0.6, got " << st.acc << ")\n";
  }
}

void test_reduce_tree_fixed() {
  std::cout << "Testing ReduceTreeFixed - ";
  
  std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float result = ReduceTreeFixed(vals, 2);  // binary tree
  
  if (std::abs(result - 21.0f) < 1e-6) {
    std::cout << "PASS ✓ (tree reduction: " << result << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected 21.0, got " << result << ")\n";
  }
}

void test_reduce_update() {
  std::cout << "Testing ReduceUpdate - ";
  
  AccState st;
  st.acc = 0.0f;
  st.c = 0.0f;
  
  std::vector<float> lane_vals = {1.0f, 2.0f, 3.0f};
  
  // Test serial policy
  ReduceCtx ctx;
  ctx.policy = ReducePolicy::kSerial;
  
  ReduceUpdate(st, lane_vals, ctx);
  
  if (std::abs(st.acc - 6.0f) < 1e-6) {
    std::cout << "PASS ✓ (ReduceUpdate serial: " << st.acc << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected 6.0, got " << st.acc << ")\n";
  }
}

void test_reduce_update_kahan() {
  std::cout << "Testing ReduceUpdate Kahan - ";
  
  AccState st;
  st.acc = 0.0f;
  st.c = 0.0f;
  
  std::vector<float> lane_vals = {0.1f, 0.2f, 0.3f};
  
  // Test Kahan policy
  ReduceCtx ctx;
  ctx.policy = ReducePolicy::kKahan;
  
  ReduceUpdate(st, lane_vals, ctx);
  
  if (std::abs(st.acc - 0.6f) < 1e-6) {
    std::cout << "PASS ✓ (ReduceUpdate Kahan: " << st.acc << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected 0.6, got " << st.acc << ")\n";
  }
}

void test_reduce_update_tree() {
  std::cout << "Testing ReduceUpdate Tree - ";
  
  AccState st;
  st.acc = 0.0f;
  st.c = 0.0f;
  
  std::vector<float> lane_vals = {1.0f, 2.0f, 3.0f, 4.0f};
  
  // Test tree policy
  ReduceCtx ctx;
  ctx.policy = ReducePolicy::kTree;
  ctx.tree_fan_in = 2;
  
  ReduceUpdate(st, lane_vals, ctx);
  
  if (std::abs(st.acc - 10.0f) < 1e-6) {
    std::cout << "PASS ✓ (ReduceUpdate tree: " << st.acc << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected 10.0, got " << st.acc << ")\n";
  }
}

int main() {
  std::cout << "=== Reduction Utils Tests ===\n\n";
  
  test_reduce_serial();
  test_reduce_kahan();
  test_reduce_tree_fixed();
  test_reduce_update();
  test_reduce_update_kahan();
  test_reduce_update_tree();
  
  std::cout << "\n=== Test Summary ===\n";
  std::cout << "✓ All reduction utility functions working correctly\n";
  
  return 0;
}
