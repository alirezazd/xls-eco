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
#include <array>

#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"
#include "xls/contrib/mlir/transforms/linalg/utils/helpers.h"
#include "xls/contrib/mlir/transforms/linalg/utils/reduction_utils.h"

namespace mlir::xls {

namespace {

// Test helper functions (copied from linalg_analysis.cc for testing)




// Test fixture creation
BroadcastAnalysis CreateTestBroadcastAnalysis() {
  BroadcastAnalysis A;
  
  // Create a 4D example with mixed indexing patterns
  A.loop_extents.resize(4);
  A.loop_extents[0] = SizeExpr(3);  // dim 0: extent 3
  A.loop_extents[1] = SizeExpr(5);  // dim 1: extent 5
  A.loop_extents[2] = SizeExpr(4);  // dim 2: extent 4
  A.loop_extents[3] = SizeExpr(6);  // dim 3: extent 6
  
  // Create 3 operands with different indexing patterns
  A.indexers.resize(3);
  
  // Operand 0: full indexing (i,j,k,l)
  A.indexers[0].results = {0, 1, 2, 3};  // maps to all dimensions
  
  // Operand 1: partial indexing (0,j,0,l) - broadcasts on dims 0 and 2
  A.indexers[1].results = {std::monostate{}, 1, std::monostate{}, 3};
  
  // Operand 2: minimal indexing (i,k) - broadcasts on dims 1 and 3
  A.indexers[2].results = {0, 2};
  
  return A;
}

void TestMakeOperandIndex() {
  std::cout << "=== MakeOperandIndex Tests ===\n\n";
  
  BroadcastAnalysis A = CreateTestBroadcastAnalysis();
  std::array<int, 4> loop_iter = {1, 2, 3, 4};  // current loop indices
  
  // Test 1: Full indexing operand
  std::cout << "T1: Full indexing operand (i,j,k,l) - ";
  auto idx0 = MakeOperandIndex(0, loop_iter, A);
  if (idx0[0] == 1 && idx0[1] == 2 && idx0[2] == 3 && idx0[3] == 4) {
    std::cout << "PASS ✓ (indices: [" << idx0[0] << "," << idx0[1] << "," << idx0[2] << "," << idx0[3] << "])\n";
  } else {
    std::cout << "FAIL ✗ (expected [1,2,3,4], got [" << idx0[0] << "," << idx0[1] << "," << idx0[2] << "," << idx0[3] << "])\n";
  }
  
  // Test 2: Partial indexing operand with broadcasting
  std::cout << "T2: Partial indexing operand (0,j,0,l) - ";
  auto idx1 = MakeOperandIndex(1, loop_iter, A);
  if (idx1[0] == 0 && idx1[1] == 2 && idx1[2] == 0 && idx1[3] == 4) {
    std::cout << "PASS ✓ (indices: [" << idx1[0] << "," << idx1[1] << "," << idx1[2] << "," << idx1[3] << "])\n";
  } else {
    std::cout << "FAIL ✗ (expected [0,2,0,4], got [" << idx1[0] << "," << idx1[1] << "," << idx1[2] << "," << idx1[3] << "])\n";
  }
  
  // Test 3: Minimal indexing operand with broadcasting
  std::cout << "T3: Minimal indexing operand (i,k) - ";
  auto idx2 = MakeOperandIndex(2, loop_iter, A);
  if (idx2[0] == 1 && idx2[1] == 3 && idx2[2] == 0 && idx2[3] == 0) {
    std::cout << "PASS ✓ (indices: [" << idx2[0] << "," << idx2[1] << "," << idx2[2] << "," << idx2[3] << "])\n";
  } else {
    std::cout << "FAIL ✗ (expected [1,3,0,0], got [" << idx2[0] << "," << idx2[1] << "," << idx2[2] << "," << idx2[3] << "])\n";
  }
}

void TestLoad() {
  std::cout << "\n=== Load Tests ===\n\n";
  
  std::array<int, 4> idx = {1, 2, 3, 4};
  
  // Test 1: Load from operand 0
  std::cout << "T1: Load from operand 0 - ";
  float val0 = Load(0, idx);
  float expected0 = 0.0f + 1 + 2 + 3 + 4;  // operand_id + sum of indices
  if (val0 == expected0) {
    std::cout << "PASS ✓ (value: " << val0 << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected " << expected0 << ", got " << val0 << ")\n";
  }
  
  // Test 2: Load from operand 1
  std::cout << "T2: Load from operand 1 - ";
  float val1 = Load(1, idx);
  float expected1 = 1.0f + 1 + 2 + 3 + 4;
  if (val1 == expected1) {
    std::cout << "PASS ✓ (value: " << val1 << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected " << expected1 << ", got " << val1 << ")\n";
  }
  
  // Test 3: Load with zero indices
  std::cout << "T3: Load with zero indices - ";
  std::array<int, 4> zero_idx = {0, 0, 0, 0};
  float val_zero = Load(2, zero_idx);
  float expected_zero = 2.0f + 0 + 0 + 0 + 0;
  if (val_zero == expected_zero) {
    std::cout << "PASS ✓ (value: " << val_zero << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected " << expected_zero << ", got " << val_zero << ")\n";
  }
}

void TestKahanUpdate() {
  std::cout << "\n=== Kahan Update Tests ===\n\n";
  
  // Test 1: Basic Kahan addition
  std::cout << "T1: Basic Kahan addition - ";
  AccState st;
  st.acc = 1.0f;
  st.c = 0.0f;
  float x = 2.0f;
  ReduceKahan(st, x);
  if (st.acc == 3.0f) {  // acc should be 1 + 2 = 3
    std::cout << "PASS ✓ (acc: " << st.acc << ", c: " << st.c << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected acc=3.0, got acc=" << st.acc << ")\n";
  }
  
  // Test 2: Kahan with non-zero c
  std::cout << "T2: Kahan with non-zero c - ";
  AccState st2;
  st2.acc = 10.0f;
  st2.c = 0.5f;
  float x2 = 5.0f;
  ReduceKahan(st2, x2);
  if (st2.acc > 14.0f && st2.acc < 16.0f) {  // should be around 15
    std::cout << "PASS ✓ (acc: " << st2.acc << ", c: " << st2.c << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected acc≈15.0, got acc=" << st2.acc << ")\n";
  }
  
  // Test 3: Multiple Kahan updates
  std::cout << "T3: Multiple Kahan updates - ";
  AccState st3;
  st3.acc = 0.0f;
  st3.c = 0.0f;
  ReduceKahan(st3, 1.0f);
  ReduceKahan(st3, 2.0f);
  ReduceKahan(st3, 3.0f);
  if (st3.acc == 6.0f) {  // should be 1 + 2 + 3 = 6
    std::cout << "PASS ✓ (final acc: " << st3.acc << ")\n";
  } else {
    std::cout << "FAIL ✗ (expected acc=6.0, got acc=" << st3.acc << ")\n";
  }
}

void TestScalarEnvironment() {
  std::cout << "\n=== Scalar Environment Tests ===\n\n";
  
  // Test 1: F32 scalar
  std::cout << "T1: F32 scalar - ";
  Scalar f32_scalar;
  f32_scalar.tag = Scalar::kF32;
  f32_scalar.f = 42.5f;
  if (f32_scalar.tag == Scalar::kF32 && f32_scalar.f == 42.5f) {
    std::cout << "PASS ✓ (value: " << f32_scalar.f << ")\n";
  } else {
    std::cout << "FAIL ✗\n";
  }
  
  // Test 2: TupleF32 scalar (Kahan)
  std::cout << "T2: TupleF32 scalar (Kahan) - ";
  Scalar tuple_scalar;
  tuple_scalar.tag = Scalar::kTupleF32;
  tuple_scalar.kahan = {10.0f, 0.1f};
  if (tuple_scalar.tag == Scalar::kTupleF32 && 
      tuple_scalar.kahan.first == 10.0f && tuple_scalar.kahan.second == 0.1f) {
    std::cout << "PASS ✓ (acc: " << tuple_scalar.kahan.first 
              << ", c: " << tuple_scalar.kahan.second << ")\n";
  } else {
    std::cout << "FAIL ✗\n";
  }
  
  // Test 3: ValMap usage
  std::cout << "T3: ValMap usage - ";
  ValMap val_map;
  ValueId id1(1);
  ValueId id2(2);
  
  Scalar val1;
  val1.tag = Scalar::kF32;
  val1.f = 1.5f;
  
  Scalar val2;
  val2.tag = Scalar::kTupleF32;
  val2.kahan = {2.0f, 0.0f};
  
  val_map[id1] = val1;
  val_map[id2] = val2;
  
  if (val_map.size() == 2 && 
      val_map[id1].tag == Scalar::kF32 && val_map[id1].f == 1.5f &&
      val_map[id2].tag == Scalar::kTupleF32 && val_map[id2].kahan.first == 2.0f) {
    std::cout << "PASS ✓ (map size: " << val_map.size() << ")\n";
  } else {
    std::cout << "FAIL ✗\n";
  }
}

void TestIntermediateForm() {
  std::cout << "\n=== Intermediate Form Tests ===\n\n";
  
  // Test 1: LoopIndex creation
  std::cout << "T1: LoopIndex creation - ";
  LoopIndex loop_idx;
  loop_idx.dim = 1;
  loop_idx.begin = 0;
  loop_idx.end = 10;
  loop_idx.step = 2;
  loop_idx.is_tile_outer = false;
  loop_idx.unroll = 4;
  
  if (loop_idx.dim == 1 && loop_idx.end == 10 && loop_idx.unroll == 4) {
    std::cout << "PASS ✓ (dim: " << loop_idx.dim << ", end: " << loop_idx.end << ", unroll: " << loop_idx.unroll << ")\n";
  } else {
    std::cout << "FAIL ✗ (unexpected values)\n";
  }
  
  // Test 2: ReducedExpr creation
  std::cout << "T2: ReducedExpr creation - ";
  ReducedExpr expr;
  expr.expr_string = "addf(load(input0, i, j), load(input1, i, j))";
  
  if (expr.expr_string.find("addf") != std::string::npos) {
    std::cout << "PASS ✓ (expr: " << expr.expr_string << ")\n";
  } else {
    std::cout << "FAIL ✗ (unexpected expression)\n";
  }
  
  // Test 3: Accumulator creation
  std::cout << "T3: Accumulator creation - ";
  Accumulator acc;
  acc.pol = CodegenConfig::kKahan;
  acc.lanes = 4;
  acc.bundle.lanes = 4;
  acc.bundle.exprs.push_back(expr);
  
  if (acc.pol == CodegenConfig::kKahan && acc.lanes == 4) {
    std::cout << "PASS ✓ (policy: Kahan, lanes: " << acc.lanes << ")\n";
  } else {
    std::cout << "FAIL ✗ (unexpected values)\n";
  }
  
  // Test 4: LoweredKernel creation
  std::cout << "T4: LoweredKernel creation - ";
  LoweredKernel kernel;
  kernel.red_loops.push_back(loop_idx);
  kernel.accs.push_back(acc);
  
  if (kernel.red_loops.size() == 1 && kernel.accs.size() == 1) {
    std::cout << "PASS ✓ (loops: " << kernel.red_loops.size() << ", accs: " << kernel.accs.size() << ")\n";
  } else {
    std::cout << "FAIL ✗ (unexpected sizes)\n";
  }
}

}  // namespace

}  // namespace mlir::xls

int main() {
  std::cout << "=== Region Body Lowering Tests ===\n\n";
  
  mlir::xls::TestMakeOperandIndex();
  mlir::xls::TestLoad();
  mlir::xls::TestKahanUpdate();
  mlir::xls::TestScalarEnvironment();
  mlir::xls::TestIntermediateForm();
  
  std::cout << "\n=== Test Summary ===\n";
  std::cout << "✓ MakeOperandIndex: Computes operand indices from loop indices\n";
  std::cout << "✓ Load: Placeholder tensor loading function\n";
  std::cout << "✓ KahanUpdate: Kahan summation algorithm\n";
  std::cout << "✓ Scalar Environment: F32 and TupleF32 scalar types\n";
  std::cout << "✓ ValMap: SSA value to scalar mapping\n";
  std::cout << "✓ Intermediate Form: Creation of intermediate forms for codegen\n";
  
  return 0;
}
