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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_TYPES_LINALG_TYPES_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_TYPES_LINALG_TYPES_H_

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
  kAddF, kMulF, kSubF, kDivF,
  kCmpOGT, kCmpOLT, kCmpOEQ, kCmpOGE, kCmpOLE, kCmpONE,
  kSelect, kYield, kConstF32, kMaxF, kMinF
};

struct ValueId {
  int id;
  ValueId(int i = -1) : id(i) {}
  bool operator==(const ValueId& other) const { return id == other.id; }
  bool operator<(const ValueId& other) const { return id < other.id; }
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

  SizeExpr() : kind(kConstant), constant_value(1), symbolic_name("") {}
  SizeExpr(int64_t value) : kind(kConstant), constant_value(value), symbolic_name("") {}
  SizeExpr(std::string name) : kind(kSymbolic), constant_value(0), symbolic_name(std::move(name)) {}

  bool isConstant() const { return kind == kConstant; }
  bool isSymbolic() const { return kind == kSymbolic; }
  int64_t getConstant() const { return constant_value; }
  const std::string& getSymbolic() const { return symbolic_name; }
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

// Comprehensive evaluation result that includes reduction detection and broadcast evaluation
struct LinalgEvalResults {
  LinalgGeneric linalg;
  BroadcastAnalysis broadcast;
  bool has_reduction;
  bool is_valid_reduction;
  
  // Default constructor
  LinalgEvalResults() : has_reduction(false), is_valid_reduction(false) {}
  
  // Constructor with all parameters
  LinalgEvalResults(const LinalgGeneric& l, const BroadcastAnalysis& b, bool has_red, bool is_valid)
      : linalg(l), broadcast(b), has_reduction(has_red), is_valid_reduction(is_valid) {}
};

struct OutputShapeInfo {
  std::vector<SizeExpr> out_shape;
};

// Code generation configuration
struct TileSpec {
  std::map<int /*dim id*/, int /*tile size*/> size;
};

class LinalgCodegenConfig {
 public:
  enum class LoopOrder { kPreserve, kParallelOuter_ReductionInner, kHeuristicCacheFriendly };
  enum class ReducePolicy { kSerial, kTree, kKahan };
  
  // Builder pattern methods
  LinalgCodegenConfig& loop_order(LoopOrder value) {
    loop_order_ = value;
    return *this;
  }
  
  LinalgCodegenConfig& reduce_policy(ReducePolicy value) {
    reduce_policy_ = value;
    return *this;
  }
  
  LinalgCodegenConfig& unroll_factor(int dim, int factor) {
    unroll_factors_[dim] = factor;
    return *this;
  }
  
  LinalgCodegenConfig& tile_size(int dim, int size) {
    tile_sizes_[dim] = size;
    return *this;
  }
  
  LinalgCodegenConfig& hoist_invariants(bool value) {
    hoist_invariants_ = value;
    return *this;
  }
  
  LinalgCodegenConfig& enable_tiling(bool value) {
    enable_tiling_ = value;
    return *this;
  }
  
  // Getters
  LoopOrder loop_order() const { return loop_order_; }
  ReducePolicy reduce_policy() const { return reduce_policy_; }
  bool enable_tiling() const { return enable_tiling_; }
  
  int unroll_factor(int dim) const {
    auto it = unroll_factors_.find(dim);
    return it != unroll_factors_.end() ? it->second : 1;
  }
  
  int tile_size(int dim) const {
    auto it = tile_sizes_.find(dim);
    return it != tile_sizes_.end() ? it->second : 1;
  }
  
  bool hoist_invariants() const { return hoist_invariants_; }
  
  // Access to internal maps for compatibility
  const std::map<int, int>& unroll() const { return unroll_factors_; }
  const std::map<int, int>& tiles() const { return tile_sizes_; }
  
  // Legacy compatibility - create TileSpec from internal state
  TileSpec get_tiles() const {
    TileSpec tiles;
    tiles.size = tile_sizes_;
    return tiles;
  }
  
 private:
  LoopOrder loop_order_ = LoopOrder::kPreserve;
  ReducePolicy reduce_policy_ = ReducePolicy::kSerial;
  std::map<int, int> unroll_factors_;
  std::map<int, int> tile_sizes_;
  bool hoist_invariants_ = true;
  bool enable_tiling_ = false;  // Default to no tiling for safety
};

// Legacy compatibility alias
using CodegenConfig = LinalgCodegenConfig;

// Loop scheduling structures
struct LoopBand {
  struct Loop {
    int dim;            // which logical dim (0..D-1)
    int64_t begin;      // 0
    int64_t end;        // loop extent or tile end
    int64_t step;       // usually 1; may equal tile size for outers
    bool is_tile_outer; // true if this is the tile loop
    bool is_reduction;  // from dims[dim].kind
    int unroll = 1;     // static unroll factor to apply at emission
  };
  std::vector<Loop> loops;
};

struct Schedule {
  std::vector<LoopBand> bands;
};

// Region-body lowering structures
struct Scalar {
  enum { kF32, kTupleF32 /*for Kahan: (acc,c)*/ } tag;
  float f;                // if kF32
  std::pair<float, float> kahan; // if kTupleF32
};

using ValMap = absl::flat_hash_map<ValueId, Scalar>; // SSA id -> scalar

// Intermediate form for region-body lowering
struct LoopIndex {
  int dim;  // dimension index
  int64_t begin;
  int64_t end;
  int64_t step;
  bool is_tile_outer;
  int unroll;
};

struct ReducedExpr {
  std::string expr_string;  // placeholder for actual AST or code emitter callback
};

struct LaneBundle {
  int lanes;                   // U
  std::vector<ReducedExpr> exprs;  // size U, same DAG param'd by lane index
};

struct Accumulator {
  LinalgCodegenConfig::ReducePolicy pol;
  int lanes;                   // U
  LaneBundle bundle;           // per-iteration lanes
};

struct LoweredKernel {
  std::vector<LoopIndex> red_loops;  // innermost structure, possibly tiled
  std::vector<Accumulator> accs;     // usually size 1
};

// Reduction structures
using ReducePolicy = CodegenConfig::ReducePolicy;

struct ReduceCtx {
  ReducePolicy policy;
  int tree_fan_in = 2;          // binary tree by default
  bool deterministic = true;    // keep fixed grouping
};

struct AccState {
  float acc = 0.0f;
  float c = 0.0f;               // compensation term for Kahan
};

// Unroll planning structures
struct LanePlan {
  int dim;            // which loop dim is unrolled
  int lanes;          // U
  bool has_tail;      // (extent % U) != 0
  int main_iters;     // extent / U
  int tail;           // extent % U
};

struct MultiLanePlan {
  std::vector<LanePlan> dims;  // e.g., {j:6, l:4} -> 24 total lanes
  int total_lanes;             // product of all lane counts (Uj * Ul * ...)
  
  // Compute lane_id from individual dimension offsets
  int ComputeLaneId(const std::vector<int>& lane_offsets) const;
  
  // Compute individual dimension offsets from lane_id
  std::vector<int> ComputeLaneOffsets(int lane_id) const;
};

struct UnrollPlan {
  std::vector<LanePlan> dims;  // e.g., { LanePlan{dim=j, lanes=6, ...}, LanePlan{dim=l, lanes=4, ...} }
  MultiLanePlan multi_lane;    // For handling multiple unrolled dimensions together
};

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_TYPES_LINALG_TYPES_H_
