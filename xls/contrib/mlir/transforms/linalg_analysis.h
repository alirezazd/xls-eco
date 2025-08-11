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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_H_

#include <map>
#include <string>
#include <vector>

#include "mlir/Support/LogicalResult.h"

// Forward declarations
namespace mlir {
class Operation;
class AffineMap;
class Region;
}

namespace mlir {
namespace xls {

enum class IterKind { kParallel, kReduction };

struct Dim {
  std::string name;   // e.g., "i","j","k","l" (optional)
  IterKind kind;      // parallel or reduction
  int64_t extent;     // >=1 (assume static)
};

struct AffineExpr {
  // Enough to handle maps like (i,j,k,l)->(i,0,k,0) or permutations.
  enum Kind { kVar, kConst0 } kind;
  int var; // index into dims for kVar; ignored for kConst0
};

struct AffineMap {
  // D dims -> R result indices (one per operand axis)
  std::vector<AffineExpr> results; // size == operand.rank
};

struct Type {
  enum { kTensorF32, kScalarF32 } tag;  // f32 tensors and scalars
  std::vector<int64_t> shape;     // tensor extents per operand axis (empty for scalars)
};

struct Operand {
  std::string name;               // A,B,U,S,Out0,...
  Type type;
  AffineMap map;                  // how loop dims index this operand
  bool is_output;                 // true for outs
};

enum class OpKind { 
  kAddF, kMulF, kSubF, kDivF, 
  kCmpOGT, kCmpOLT, kCmpOEQ, kCmpOGE, kCmpOLE, kCmpONE,
  kSelect, kYield, kConstF32, kMaxF, kMinF
};

struct ValueId { 
  int id; 
  ValueId(int i) : id(i) {}
  ValueId() : id(-1) {}  // Default constructor
  bool operator==(const ValueId& other) const { return id == other.id; }
  bool operator<(const ValueId& other) const { return id < other.id; }
};

struct RegionOp {
  OpKind kind;
  std::vector<ValueId> inputs;    // operands by id
  ValueId result;                 // produced id
  float f32_imm = 0.0f;           // for constants
};

struct Region {
  // Region args: ordered like MLIR block args (input scalars + %out accumulators).
  std::vector<ValueId> args;
  std::vector<RegionOp> ops;      // A straight-line DAG in topo order
  std::vector<ValueId> yields;    // values yielded to outs
};

struct LinalgGeneric {
  std::vector<Dim> dims;          // iterator order e.g., [i,j,k,l]
  std::vector<Operand> operands;  // ins + outs
  Region region;
};

struct CodegenConfig {
  enum LoopOrder { kPreserve, kParallelOuter_ReductionInner } loop_order;
  enum ReducePolicy { kSerial, kTree, kKahan } reduce_policy;
  std::map<std::string,int> unroll;  // unroll factors per reduction dim name/index (default 1)
};

// Analysis functions
mlir::LogicalResult AnalyzeLinalgGeneric(mlir::Operation* op, LinalgGeneric& result);
AffineMap AnalyzeAffineMap(mlir::AffineMap mlir_map);
mlir::LogicalResult AnalyzeRegion(mlir::Region& mlir_region, Region& result);
std::string LinalgGenericToString(const LinalgGeneric& linalg);

}  // namespace xls
}  // namespace mlir

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_ANALYSIS_H_
