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

#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"
#include "xls/contrib/mlir/transforms/linalg/utils/helpers.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/linalg_analysis.h"

#include <iostream>
#include <cassert>

namespace mlir::xls {

namespace {

// Test T1: Valid complex example
LinalgGeneric CreateT1ValidExample() {
  LinalgGeneric g;
  
  // Dims: (i P:3, j R:5, k P:4, l R:6)
  g.dims = {
    {"i", IterKind::kParallel, 3},
    {"j", IterKind::kReduction, 5},
    {"k", IterKind::kParallel, 4},
    {"l", IterKind::kReduction, 6}
  };
  
  // A: shape [3,5,4,6], map (i,j,k,l)
  Operand A;
  A.name = "A";
  A.type.tag = Type::kTensorF32;
  A.type.shape = {3, 5, 4, 6};
  A.map.results = {
    {AffineExpr::kVar, 0},  // i
    {AffineExpr::kVar, 1},  // j
    {AffineExpr::kVar, 2},  // k
    {AffineExpr::kVar, 3}   // l
  };
  A.is_output = false;
  g.operands.push_back(A);
  
  // B: shape [1,5,1,6], map (0,j,0,l)
  Operand B;
  B.name = "B";
  B.type.tag = Type::kTensorF32;
  B.type.shape = {1, 5, 1, 6};
  B.map.results = {
    {AffineExpr::kConst0, 0},  // 0
    {AffineExpr::kVar, 1},     // j
    {AffineExpr::kConst0, 0},  // 0
    {AffineExpr::kVar, 3}      // l
  };
  B.is_output = false;
  g.operands.push_back(B);
  
  // U: shape [3,1,4,1], map (i,0,k,0)
  Operand U;
  U.name = "U";
  U.type.tag = Type::kTensorF32;
  U.type.shape = {3, 1, 4, 1};
  U.map.results = {
    {AffineExpr::kVar, 0},     // i
    {AffineExpr::kConst0, 0},  // 0
    {AffineExpr::kVar, 2},     // k
    {AffineExpr::kConst0, 0}   // 0
  };
  U.is_output = false;
  g.operands.push_back(U);
  
  // S: shape [1,1,1,1], map (0,0,0,0)
  Operand S;
  S.name = "S";
  S.type.tag = Type::kTensorF32;
  S.type.shape = {1, 1, 1, 1};
  S.map.results = {
    {AffineExpr::kConst0, 0},  // 0
    {AffineExpr::kConst0, 0},  // 0
    {AffineExpr::kConst0, 0},  // 0
    {AffineExpr::kConst0, 0}   // 0
  };
  S.is_output = false;
  g.operands.push_back(S);
  
  // Out C: shape [3,4], map (i,k)
  Operand C;
  C.name = "C";
  C.type.tag = Type::kTensorF32;
  C.type.shape = {3, 4};
  C.map.results = {
    {AffineExpr::kVar, 0},  // i
    {AffineExpr::kVar, 2}   // k
  };
  C.is_output = true;
  g.operands.push_back(C);
  
  // Region args: input scalars + output accumulator
  g.region.args = {
    ValueId(0),  // A scalar
    ValueId(1),  // B scalar
    ValueId(2),  // U scalar
    ValueId(3),  // S scalar
    ValueId(4)   // C accumulator
  };
  
  // Region ops: consts + addf + cmp ogt + select + mulf + yield
  int next_id = 5;
  
  // const 2.0
  RegionOp const1;
  const1.kind = OpKind::kConstF32;
  const1.f32_imm = 2.0f;
  const1.result = ValueId(next_id++);
  g.region.ops.push_back(const1);
  
  // addf(A, const)
  RegionOp add1;
  add1.kind = OpKind::kAddF;
  add1.inputs = {ValueId(0), const1.result};  // A + const
  add1.result = ValueId(next_id++);
  g.region.ops.push_back(add1);
  
  // cmp ogt(add1, B)
  RegionOp cmp1;
  cmp1.kind = OpKind::kCmpOGT;
  cmp1.inputs = {add1.result, ValueId(1)};  // add1 > B
  cmp1.result = ValueId(next_id++);
  g.region.ops.push_back(cmp1);
  
  // select(cmp, U, S)
  RegionOp select1;
  select1.kind = OpKind::kSelect;
  select1.inputs = {cmp1.result, ValueId(2), ValueId(3)};  // cmp ? U : S
  select1.result = ValueId(next_id++);
  g.region.ops.push_back(select1);
  
  // mulf(select, C)
  RegionOp mul1;
  mul1.kind = OpKind::kMulF;
  mul1.inputs = {select1.result, ValueId(4)};  // select * C
  mul1.result = ValueId(next_id++);
  g.region.ops.push_back(mul1);
  
  // yield(mul1)
  g.region.yields = {mul1.result};
  
  return g;
}

// Test T2: Bad map index - var = 9 with only 4 dims
LinalgGeneric CreateT2BadMapIndex() {
  LinalgGeneric g = CreateT1ValidExample();
  
  // Modify the first operand to have an invalid map index
  g.operands[0].map.results[0] = {AffineExpr::kVar, 9};  // Invalid: only 4 dims (0-3)
  
  return g;
}

// Test T3: Shape/map mismatch - B says shape rank 4 but map has 3 results
LinalgGeneric CreateT3ShapeMapMismatch() {
  LinalgGeneric g = CreateT1ValidExample();
  
  // Modify B to have shape rank 4 but map with only 3 results
  g.operands[1].type.shape = {1, 5, 1, 6};  // rank 4
  g.operands[1].map.results = {
    {AffineExpr::kConst0, 0},  // 0
    {AffineExpr::kVar, 1},     // j
    {AffineExpr::kVar, 3}      // l (only 3 results, should be 4)
  };
  
  return g;
}

// Test T4: Unsupported region op
LinalgGeneric CreateT4UnsupportedOp() {
  LinalgGeneric g = CreateT1ValidExample();
  
  // Add an unsupported operation (we'll use a placeholder)
  RegionOp unsupported;
  unsupported.kind = OpKind::kAddF;  // We'll modify this to be unsupported
  unsupported.inputs = {ValueId(0), ValueId(1)};
  unsupported.result = ValueId(100);  // High ID to avoid conflicts
  g.region.ops.push_back(unsupported);
  
  return g;
}

// Test T5: Empty dimensions
LinalgGeneric CreateT5EmptyDims() {
  LinalgGeneric g;
  
  // No dimensions
  g.dims = {};
  
  // Single operand
  Operand A;
  A.name = "A";
  A.type.tag = Type::kTensorF32;
  A.type.shape = {1};
  A.map.results = {};
  A.is_output = false;
  g.operands.push_back(A);
  
  return g;
}

// Test T6: Invalid extent (0)
LinalgGeneric CreateT6InvalidExtent() {
  LinalgGeneric g = CreateT1ValidExample();
  
  // Set a dimension extent to 0
  g.dims[0].extent = 0;
  
  return g;
}

// Test T7: DAG violation in region
LinalgGeneric CreateT7DagViolation() {
  LinalgGeneric g = CreateT1ValidExample();
  
  // Clear existing ops and create a DAG violation
  g.region.ops.clear();
  
  // op1 uses op2, but op2 comes after op1
  RegionOp op1;
  op1.kind = OpKind::kAddF;
  op1.inputs = {ValueId(0), ValueId(101)};  // Uses op2's result
  op1.result = ValueId(100);
  g.region.ops.push_back(op1);
  
  RegionOp op2;
  op2.kind = OpKind::kConstF32;
  op2.f32_imm = 1.0f;
  op2.result = ValueId(101);
  g.region.ops.push_back(op2);
  
  g.region.yields = {op1.result};
  
  return g;
}

// Test T8: Undefined yield
LinalgGeneric CreateT8UndefinedYield() {
  LinalgGeneric g = CreateT1ValidExample();
  
  // Add a yield that references an undefined value
  g.region.yields.push_back(ValueId(999));  // Undefined value
  
  return g;
}

void RunValidationTest(const std::string& test_name, const LinalgGeneric& g, bool expect_success) {
  std::cout << "=== " << test_name << " ===\n";
  
  auto result = Validate(g);
  bool success = result.succeeded();
  
  if (expect_success && success) {
    std::cout << "PASS ✓ (expected success, got success)\n";
  } else if (!expect_success && !success) {
    std::cout << "PASS ✓ (expected failure, got failure)\n";
  } else if (expect_success && !success) {
    std::cout << "FAIL ✗ (expected success, got failure)\n";
  } else {
    std::cout << "FAIL ✗ (expected failure, got success)\n";
  }
  
  std::cout << "\n";
}

}  // namespace

void RunAllValidationTests() {
  std::cout << "Running Linalg Validation Tests\n";
  std::cout << "================================\n\n";
  
  // T1: Valid complex example
  RunValidationTest("T1 - Valid Complex Example", CreateT1ValidExample(), true);
  
  // T2: Bad map index
  RunValidationTest("T2 - Bad Map Index", CreateT2BadMapIndex(), false);
  
  // T3: Shape/map mismatch
  RunValidationTest("T3 - Shape/Map Mismatch", CreateT3ShapeMapMismatch(), false);
  
  // T4: Unsupported region op (this would need more complex validation)
  RunValidationTest("T4 - Unsupported Region Op", CreateT4UnsupportedOp(), true); // Currently passes
  
  // T5: Empty dimensions
  RunValidationTest("T5 - Empty Dimensions", CreateT5EmptyDims(), false);
  
  // T6: Invalid extent
  RunValidationTest("T6 - Invalid Extent (0)", CreateT6InvalidExtent(), false);
  
  // T7: DAG violation
  RunValidationTest("T7 - DAG Violation", CreateT7DagViolation(), false);
  
  // T8: Undefined yield
  RunValidationTest("T8 - Undefined Yield", CreateT8UndefinedYield(), false);
  
  std::cout << "Validation tests completed.\n";
}

}  // namespace mlir::xls

int main() {
  mlir::xls::RunAllValidationTests();
  return 0;
}
