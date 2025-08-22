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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_TYPES_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_TYPES_H_

#include <array>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"

namespace mlir::xls {

// Core data structures for linalg.generic evaluation
enum class IterKind { kParallel, kReduction };

struct Dim {
  std::string name;
  IterKind kind;
  int64_t extent;
};

struct AffineExpr {
  enum Kind { kVar, kConst0 } kind;
  int var;
};

struct AffineMap {
  std::vector<AffineExpr> results;
};

struct Type {
  enum { kTensorF32, kScalarF32 } tag;
  std::vector<int64_t> shape;
};

struct Operand {
  std::string name;
  Type type;
  AffineMap map;
  bool is_output;
};

enum class OpKind {
  kAddF,
  kMulF,
  kSubF,
  kDivF,
  kCmpOGT,
  kCmpOLT,
  kCmpOEQ,
  kCmpOGE,
  kCmpOLE,
  kCmpONE,
  kSelect,
  kYield,
  kConstF32,
  kMaxF,
  kMinF
};

struct ValueId {
  int id;
  ValueId(int i = -1);
  bool operator==(const ValueId& other) const;
  bool operator<(const ValueId& other) const;
};

// Hash function for ValueId
template <typename H>
H AbslHashValue(H h, const ValueId& value_id) {
  return H::combine(std::move(h), value_id.id);
}

struct RegionOp {
  OpKind kind;
  std::vector<ValueId> inputs;
  ValueId result;
  float f32_imm = 0.0f;
};

struct Region {
  std::vector<ValueId> args;
  std::vector<RegionOp> ops;
  std::vector<ValueId> yields;
};

struct LinalgGeneric {
  std::vector<Dim> dims;
  std::vector<Operand> operands;
  Region region;
};

// Broadcast evaluation structures
struct SizeExpr {
  enum Kind { kConstant, kSymbolic } kind;
  int64_t constant_value;
  std::string symbolic_name;

  SizeExpr();
  SizeExpr(int64_t value);
  SizeExpr(std::string name);

  bool isConstant() const;
  bool isSymbolic() const;
  int64_t getConstant() const;
  const std::string& getSymbolic() const;
};

struct DimAnalysis {
  SizeExpr extent;
  std::vector<std::vector<bool>> operand_varies;
};

struct OperandIndexFn {
  std::vector<std::variant<int, std::monostate>> results;
};

struct BroadcastAnalysis {
  DimAnalysis dims;
  std::vector<SizeExpr> loop_extents;
  std::vector<OperandIndexFn> indexers;
};

struct InputShapeInfo {
  std::vector<SizeExpr> in_shape;
  std::vector<SizeExpr> dslx_shape;  // Reversed for DSLX syntax
};

struct OutputShapeInfo {
  std::vector<SizeExpr> out_shape;
};

// Comprehensive evaluation result that includes reduction detection and
// broadcast evaluation
struct LinalgEvalResults {
  std::string function_name;  // High-level identifier
  LinalgGeneric linalg;       // Core operation structure
  BroadcastAnalysis broadcast; // Broadcast analysis results
  std::pair<std::vector<InputShapeInfo>, std::vector<OutputShapeInfo>>
      shapes;  // Derived shapes for all operands (inputs and outputs)
  OpKind reduction_op;  // Actual reduction operation detected

  // Default constructor
  LinalgEvalResults()
      : function_name("generated_linalg_function"), reduction_op(OpKind::kYield) {}

  // Constructor with all parameters
  LinalgEvalResults(const std::string& name, const LinalgGeneric& l, const BroadcastAnalysis& b,
                    std::pair<std::vector<InputShapeInfo>, std::vector<OutputShapeInfo>> shapes,
                    OpKind reduction_op)
      : function_name(name),
        linalg(l),
        broadcast(b),
        shapes(std::move(shapes)),
        reduction_op(reduction_op) {}
};

// Forward declarations for schedule types
struct Schedule;
class LinalgCodegenConfig;



}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_TYPES_H_
