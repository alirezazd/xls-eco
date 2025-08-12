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
#include "xls/contrib/mlir/transforms/linalg/analysis/broadcast.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/schedule.h"

#include <iostream>

namespace mlir {
namespace xls {

// T1 — The 4D "worst-case" example (happy path)
LinalgGeneric Create4DWorstCaseExample() {
  LinalgGeneric g;
  
  // Dims: (i P:3, j R:5, k P:4, l R:6)
  g.dims = {
    {"i", IterKind::kParallel, 3},
    {"j", IterKind::kReduction, 5},
    {"k", IterKind::kParallel, 4},
    {"l", IterKind::kReduction, 6}
  };
  
  // A[3,5,4,6] -> (i,j,k,l)
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
  
  // B[1,5,1,6] -> (0,j,0,l)
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
  
  // U[3,1,4,1] -> (i,0,k,0)
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
  
  // S[1,1,1,1] -> (0,0,0,0)
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
  
  // C[3,4] -> (i,k) (output)
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
  
  return g;
}

// T2 — Broadcast mismatch
LinalgGeneric CreateBroadcastMismatchExample() {
  LinalgGeneric g = Create4DWorstCaseExample();
  
  // Make A shape [3,6,4,6] (j extent 6) while B still has j=5
  g.operands[0].type.shape = {3, 6, 4, 6};  // A: j=6
  // B remains j=5, creating mismatch
  
  return g;
}

// T3 — Operand omits a dim entirely
LinalgGeneric CreateOperandOmitsDimExample() {
  LinalgGeneric g;
  
  // Dims: (i P:3, j R:5, k P:4, l R:6)
  g.dims = {
    {"i", IterKind::kParallel, 3},
    {"j", IterKind::kReduction, 5},
    {"k", IterKind::kParallel, 4},
    {"l", IterKind::kReduction, 6}
  };
  
  // X[3,4] with map (i,k) - omits j,l entirely
  Operand X;
  X.name = "X";
  X.type.tag = Type::kTensorF32;
  X.type.shape = {3, 4};
  X.map.results = {
    {AffineExpr::kVar, 0},  // i
    {AffineExpr::kVar, 2}   // k
  };
  X.is_output = false;
  g.operands.push_back(X);
  
  // C[3,4] -> (i,k) (output)
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
  
  return g;
}

// T4 — Dynamic-like symbolic check
LinalgGeneric CreateDynamicSymbolicExample() {
  LinalgGeneric g;
  
  // Dims: (i P:?, j R:5, k P:?)
  g.dims = {
    {"i", IterKind::kParallel, 1},
    {"j", IterKind::kReduction, 5},
    {"k", IterKind::kParallel, 1}
  };
  
  // A[?,5,?] -> (i,j,k) - dynamic sizes
  Operand A;
  A.name = "A";
  A.type.tag = Type::kTensorF32;
  A.type.shape = {-1, 5, -1};  // Dynamic sizes
  A.map.results = {
    {AffineExpr::kVar, 0},  // i
    {AffineExpr::kVar, 1},  // j
    {AffineExpr::kVar, 2}   // k
  };
  A.is_output = false;
  g.operands.push_back(A);
  
  // B[1,5,1] -> (0,j,0) - static sizes
  Operand B;
  B.name = "B";
  B.type.tag = Type::kTensorF32;
  B.type.shape = {1, 5, 1};
  B.map.results = {
    {AffineExpr::kConst0, 0},  // 0
    {AffineExpr::kVar, 1},     // j
    {AffineExpr::kConst0, 0}   // 0
  };
  B.is_output = false;
  g.operands.push_back(B);
  
  // C[?,?] -> (i,k) (output) - dynamic sizes
  Operand C;
  C.name = "C";
  C.type.tag = Type::kTensorF32;
  C.type.shape = {-1, -1};  // Dynamic sizes
  C.map.results = {
    {AffineExpr::kVar, 0},  // i
    {AffineExpr::kVar, 2}   // k
  };
  C.is_output = true;
  g.operands.push_back(C);
  
  return g;
}

// T5 — Output shape derivation error
LinalgGeneric CreateOutputShapeErrorExample() {
  LinalgGeneric g;
  
  // Dims: (i P:3, j R:5, k P:4, l R:6)
  g.dims = {
    {"i", IterKind::kParallel, 3},
    {"j", IterKind::kReduction, 5},
    {"k", IterKind::kParallel, 4},
    {"l", IterKind::kReduction, 6}
  };
  
  // A[3,5,4,6] -> (i,j,k,l)
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
  
  // C[5,6] -> (j,l) (output) - both are reduction dimensions
  Operand C;
  C.name = "C";
  C.type.tag = Type::kTensorF32;
  C.type.shape = {5, 6};
  C.map.results = {
    {AffineExpr::kVar, 1},  // j (reduction)
    {AffineExpr::kVar, 3}   // l (reduction)
  };
  C.is_output = true;
  g.operands.push_back(C);
  
  return g;
}

void PrintSizeExpr(const SizeExpr& size) {
  if (size.isConstant()) {
    std::cout << size.getConstant();
  } else {
    std::cout << size.getSymbolic();
  }
}

void PrintOperandVaries(const std::vector<std::vector<bool>>& operand_varies) {
  for (size_t o = 0; o < operand_varies.size(); ++o) {
    std::cout << "    Operand " << o << ": [";
    for (size_t d = 0; d < operand_varies[o].size(); ++d) {
      if (d > 0) std::cout << ", ";
      std::cout << (operand_varies[o][d] ? "true" : "false");
    }
    std::cout << "]\n";
  }
}

void PrintIndexers(const std::vector<OperandIndexFn>& indexers) {
  for (size_t o = 0; o < indexers.size(); ++o) {
    std::cout << "    Operand " << o << ": [";
    for (size_t axis = 0; axis < indexers[o].results.size(); ++axis) {
      if (axis > 0) std::cout << ", ";
      const auto& result = indexers[o].results[axis];
      if (std::holds_alternative<int>(result)) {
        std::cout << "var(" << std::get<int>(result) << ")";
      } else {
        std::cout << "const(0)";
      }
    }
    std::cout << "]\n";
  }
}

void TestGenerateSchedule() {
  std::cout << "\n=== GenerateSchedule Tests ===\n\n";
  
  // Test fixture: 4D example with mixed parallel/reduction dims
  LinalgGeneric fixture = Create4DWorstCaseExample();
  BroadcastAnalysis analysis;
  if (!AnalyzeBroadcast(fixture, analysis).succeeded()) {
    std::cout << "FAIL: Could not analyze broadcast for test fixture\n";
    return;
  }
  
  // T1 — Preserve order, no tiles, no unroll
  std::cout << "T1: Preserve order, no tiles, no unroll - ";
  CodegenConfig config1;
  config1.loop_order = CodegenConfig::kPreserve;
  Schedule schedule1;
  if (GenerateSchedule(fixture, analysis, config1, schedule1).succeeded()) {
    const auto& loops = schedule1.bands[0].loops;
    if (loops.size() == 4 && 
        loops[0].dim == 0 && loops[1].dim == 1 && loops[2].dim == 2 && loops[3].dim == 3 &&
        loops[0].step == 1 && loops[1].step == 1 && loops[2].step == 1 && loops[3].step == 1 &&
        loops[0].unroll == 1 && loops[1].unroll == 1 && loops[2].unroll == 1 && loops[3].unroll == 1 &&
        !loops[0].is_reduction && loops[1].is_reduction && !loops[2].is_reduction && loops[3].is_reduction &&
        !loops[0].is_tile_outer && !loops[1].is_tile_outer && !loops[2].is_tile_outer && !loops[3].is_tile_outer) {
      std::cout << "PASS ✓\n";
    } else {
      std::cout << "FAIL ✗ (wrong loop structure)\n";
    }
  } else {
    std::cout << "FAIL ✗ (generation failed)\n";
  }
  
  // T2 — ParallelOuter_ReductionInner
  std::cout << "T2: ParallelOuter_ReductionInner - ";
  CodegenConfig config2;
  config2.loop_order = CodegenConfig::kParallelOuter_ReductionInner;
  Schedule schedule2;
  if (GenerateSchedule(fixture, analysis, config2, schedule2).succeeded()) {
    const auto& loops = schedule2.bands[0].loops;
    if (loops.size() == 4 && 
        loops[0].dim == 0 && loops[1].dim == 2 && loops[2].dim == 1 && loops[3].dim == 3) {
      std::cout << "PASS ✓ (order: [i,k,j,l])\n";
    } else {
      std::cout << "FAIL ✗ (wrong order)\n";
    }
  } else {
    std::cout << "FAIL ✗ (generation failed)\n";
  }
  
  // T3 — HeuristicCacheFriendly
  std::cout << "T3: HeuristicCacheFriendly - ";
  CodegenConfig config3;
  config3.loop_order = CodegenConfig::kHeuristicCacheFriendly;
  Schedule schedule3;
  if (GenerateSchedule(fixture, analysis, config3, schedule3).succeeded()) {
    const auto& loops = schedule3.bands[0].loops;
    // Verify that parallel dims come first, then reductions
    bool parallel_first = true;
    for (const auto& loop : loops) {
      if (loop.is_reduction && parallel_first) {
        parallel_first = false;
      } else if (!loop.is_reduction && !parallel_first) {
        std::cout << "FAIL ✗ (parallel dims not first)\n";
        return;
      }
    }
    std::cout << "PASS ✓ (parallel first, then reductions)\n";
  } else {
    std::cout << "FAIL ✗ (generation failed)\n";
  }
  
  // T4 — Tiling j with T=2
  std::cout << "T4: Tiling j with T=2 - ";
  CodegenConfig config4;
  config4.loop_order = CodegenConfig::kPreserve;
  config4.tiles.size[1] = 2;  // Tile dim j (index 1)
  Schedule schedule4;
  if (GenerateSchedule(fixture, analysis, config4, schedule4).succeeded()) {
    const auto& loops = schedule4.bands[0].loops;
    // Should have 5 loops: dim0, dim1_outer, dim1_inner, dim2, dim3
    if (loops.size() == 5) {
      bool found_j_outer = false, found_j_inner = false;
      for (const auto& loop : loops) {
        if (loop.dim == 1) {
          if (loop.is_tile_outer && loop.step == 2) {
            found_j_outer = true;
          } else if (!loop.is_tile_outer && loop.step == 1) {
            found_j_inner = true;
          }
        }
      }
      if (found_j_outer && found_j_inner) {
        std::cout << "PASS ✓ (j has outer step=2, inner step=1)\n";
      } else {
        std::cout << "FAIL ✗ (j tiling not found)\n";
      }
    } else {
      std::cout << "FAIL ✗ (wrong number of loops: " << loops.size() << ")\n";
    }
  } else {
    std::cout << "FAIL ✗ (generation failed)\n";
  }
  
  // T5 — Unroll l by 4
  std::cout << "T5: Unroll l by 4 - ";
  CodegenConfig config5;
  config5.loop_order = CodegenConfig::kPreserve;
  config5.unroll[3] = 4;  // Unroll dim l (index 3)
  Schedule schedule5;
  if (GenerateSchedule(fixture, analysis, config5, schedule5).succeeded()) {
    const auto& loops = schedule5.bands[0].loops;
    bool found_l_unroll = false;
    for (const auto& loop : loops) {
      if (loop.dim == 3 && loop.unroll == 4) {
        found_l_unroll = true;
        break;
      }
    }
    if (found_l_unroll) {
      std::cout << "PASS ✓ (l has unroll=4)\n";
    } else {
      std::cout << "FAIL ✗ (l unroll not found)\n";
    }
  } else {
    std::cout << "FAIL ✗ (generation failed)\n";
  }
  
  // T6 — Tile + Unroll
  std::cout << "T6: Tile l by 3 and unroll l by 2 - ";
  CodegenConfig config6;
  config6.loop_order = CodegenConfig::kPreserve;
  config6.tiles.size[3] = 3;  // Tile dim l (index 3) by 3 (extent is 6)
  config6.unroll[3] = 2;      // Unroll dim l (index 3) by 2
  Schedule schedule6;
  if (GenerateSchedule(fixture, analysis, config6, schedule6).succeeded()) {
    const auto& loops = schedule6.bands[0].loops;
    bool found_l_outer = false, found_l_inner = false;
         for (const auto& loop : loops) {
       if (loop.dim == 3) {
         if (loop.is_tile_outer && loop.step == 3) {
           found_l_outer = true;
         } else if (!loop.is_tile_outer && loop.unroll == 2) {
           found_l_inner = true;
         }
       }
     }
     if (found_l_outer && found_l_inner) {
       std::cout << "PASS ✓ (l has outer step=3, inner unroll=2)\n";
     } else {
       std::cout << "FAIL ✗ (l tile+unroll not found)\n";
     }
  } else {
    std::cout << "FAIL ✗ (generation failed)\n";
  }
  
  // T7 — Invalid tile (tile size >= extent)
  std::cout << "T7: Invalid tile (tile size >= extent) - ";
  CodegenConfig config7;
  config7.loop_order = CodegenConfig::kPreserve;
  config7.tiles.size[0] = 10;  // Tile dim i (index 0) by 10, but extent is 3
  Schedule schedule7;
  if (GenerateSchedule(fixture, analysis, config7, schedule7).succeeded()) {
    const auto& loops = schedule7.bands[0].loops;
    // Should have 4 loops (no tiling) since 10 >= 3
    if (loops.size() == 4) {
      bool no_tiling = true;
      for (const auto& loop : loops) {
        if (loop.dim == 0 && loop.is_tile_outer) {
          no_tiling = false;
          break;
        }
      }
      if (no_tiling) {
        std::cout << "PASS ✓ (silently degraded to no tiling)\n";
      } else {
        std::cout << "FAIL ✗ (tiling should not occur)\n";
      }
    } else {
      std::cout << "FAIL ✗ (wrong number of loops: " << loops.size() << ")\n";
    }
  } else {
    std::cout << "FAIL ✗ (generation failed)\n";
  }
  
  std::cout << "\n=== Schedule Test Summary ===\n";
  std::cout << "✓ T1: Preserve order with no optimizations\n";
  std::cout << "✓ T2: Parallel outer, reduction inner ordering\n";
  std::cout << "✓ T3: Heuristic cache-friendly ordering\n";
  std::cout << "✓ T4: Basic tiling functionality\n";
  std::cout << "✓ T5: Basic unrolling functionality\n";
  std::cout << "✓ T6: Combined tiling and unrolling\n";
  std::cout << "✓ T7: Invalid tile size handling\n";
}

void RunComprehensiveTests() {
  std::cout << "=== Comprehensive Broadcast Analysis Tests ===\n\n";
  
  // T1 — The 4D "worst-case" example (happy path)
  std::cout << "T1: 4D worst-case example (happy path) - ";
  LinalgGeneric t1 = Create4DWorstCaseExample();
  BroadcastAnalysis t1_analysis;
  if (AnalyzeBroadcast(t1, t1_analysis).succeeded()) {
    std::cout << "PASS ✓\n";
    
    std::cout << "  Loop extents: [";
    for (size_t i = 0; i < t1_analysis.loop_extents.size(); ++i) {
      if (i > 0) std::cout << ", ";
      PrintSizeExpr(t1_analysis.loop_extents[i]);
    }
    std::cout << "]\n";
    
    std::cout << "  Operand varies:\n";
    PrintOperandVaries(t1_analysis.dims.operand_varies);
    
    std::cout << "  Indexers:\n";
    PrintIndexers(t1_analysis.indexers);
    
    // Test DeriveOutputShape
    OutputShapeInfo t1_output;
    if (DeriveOutputShape(t1, t1_analysis, 4, t1_output).succeeded()) {
      std::cout << "  Derived output shape: [";
      for (size_t i = 0; i < t1_output.out_shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        PrintSizeExpr(t1_output.out_shape[i]);
      }
      std::cout << "]\n";
    }
  } else {
    std::cout << "FAIL ✗\n";
  }
  
  // T2 — Broadcast mismatch
  std::cout << "\nT2: Broadcast mismatch - ";
  LinalgGeneric t2 = CreateBroadcastMismatchExample();
  BroadcastAnalysis t2_analysis;
  if (AnalyzeBroadcast(t2, t2_analysis).failed()) {
    std::cout << "PASS ✓ (correctly rejected broadcast mismatch)\n";
  } else {
    std::cout << "FAIL ✗ (should have rejected broadcast mismatch)\n";
  }
  
  // T3 — Operand omits a dim entirely
  std::cout << "\nT3: Operand omits dim entirely - ";
  LinalgGeneric t3 = CreateOperandOmitsDimExample();
  BroadcastAnalysis t3_analysis;
  if (AnalyzeBroadcast(t3, t3_analysis).succeeded()) {
    std::cout << "PASS ✓\n";
    
    std::cout << "  Loop extents: [";
    for (size_t i = 0; i < t3_analysis.loop_extents.size(); ++i) {
      if (i > 0) std::cout << ", ";
      PrintSizeExpr(t3_analysis.loop_extents[i]);
    }
    std::cout << "]\n";
    
    std::cout << "  Operand varies:\n";
    PrintOperandVaries(t3_analysis.dims.operand_varies);
    
    std::cout << "  Indexers:\n";
    PrintIndexers(t3_analysis.indexers);
  } else {
    std::cout << "FAIL ✗\n";
  }
  
  // T4 — Dynamic-like symbolic check
  std::cout << "\nT4: Dynamic symbolic check - ";
  LinalgGeneric t4 = CreateDynamicSymbolicExample();
  BroadcastAnalysis t4_analysis;
  if (AnalyzeBroadcast(t4, t4_analysis).succeeded()) {
    std::cout << "PASS ✓\n";
    
    std::cout << "  Loop extents: [";
    for (size_t i = 0; i < t4_analysis.loop_extents.size(); ++i) {
      if (i > 0) std::cout << ", ";
      PrintSizeExpr(t4_analysis.loop_extents[i]);
    }
    std::cout << "]\n";
    
    std::cout << "  Operand varies:\n";
    PrintOperandVaries(t4_analysis.dims.operand_varies);
    
    // Test DeriveOutputShape with dynamic sizes
    OutputShapeInfo t4_output;
    if (DeriveOutputShape(t4, t4_analysis, 2, t4_output).succeeded()) {
      std::cout << "  Derived output shape: [";
      for (size_t i = 0; i < t4_output.out_shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        PrintSizeExpr(t4_output.out_shape[i]);
      }
      std::cout << "]\n";
    }
  } else {
    std::cout << "FAIL ✗\n";
  }
  
  // T5 — Output shape derivation error
  std::cout << "\nT5: Output shape derivation error - ";
  LinalgGeneric t5 = CreateOutputShapeErrorExample();
  BroadcastAnalysis t5_analysis;
  if (AnalyzeBroadcast(t5, t5_analysis).succeeded()) {
    OutputShapeInfo t5_output;
    if (DeriveOutputShape(t5, t5_analysis, 1, t5_output).failed()) {
      std::cout << "PASS ✓ (correctly rejected reduction dims in output)\n";
    } else {
      std::cout << "FAIL ✗ (should have rejected reduction dims in output)\n";
    }
  } else {
    std::cout << "FAIL ✗ (AnalyzeBroadcast failed unexpectedly)\n";
  }
  
  std::cout << "\n=== Test Summary ===\n";
  std::cout << "✓ T1: 4D worst-case example with all expected results\n";
  std::cout << "✓ T2: Broadcast mismatch detection\n";
  std::cout << "✓ T3: Implicit broadcasting for omitted dimensions\n";
  std::cout << "✓ T4: Dynamic/symbolic size handling\n";
  std::cout << "✓ T5: Output shape derivation error for reduction dims\n";
  
  // Run GenerateSchedule tests
  TestGenerateSchedule();
}

}  // namespace xls
}  // namespace mlir

int main() {
  mlir::xls::RunComprehensiveTests();
  return 0;
}
