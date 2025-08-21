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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_DSLX_TYPES_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_DSLX_TYPES_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::xls {

// Minimal loop/body IR for DSLX emission with fine-grained control.
struct DslxStmt {
  enum Kind { kLet, kFor, kUpdate };

  // let <name>: <type> = <rhs>;
  struct LetStmt {
    std::string name;  // e.g. "row_final"
    std::string type;  // e.g. "F32[32]"
    std::string rhs;   // e.g. "acc_row" or zero-literal
  };

  // for (dim_name, acc): (u32, <acc_type>) in u32:<begin>..u32:<end> {
  //   ...body_stmts...
  //   <body_expr>
  // }(<seed>);
  // Or: let name = for (dim_name, acc): (u32, <acc_type>) in
  // u32:<begin>..u32:<end> { ... }(<seed>);
  struct ForStmt {
    std::optional<std::string> name;  // Optional name for "let name = for ..."
                                      // (std::nullopt if no let)
    std::string dim_name;  // "d0", "d1", "d2", ... (dimension-based naming)
    std::string acc;       // "acc0", "row", "s0", ...
    std::string acc_type;  // "F32[32][8]" / "F32[32]" / "F32"
    int64_t begin = 0;
    int64_t end = 0;
    int64_t step = 1;                  // Always 1 for now
    std::vector<DslxStmt> body_stmts;  // Nested lets/loops
    std::string body_expr;             // Final expr returned by loop body
    std::string seed;                  // Seed after the close
  };

  // update(array, index, new_value)
  struct UpdateStmt {
    std::string array;      // "acc0", "row", "s0", ...
    std::string index;      // "p0", "r0", "(p0, r0)" for multi-dimensional
    std::string new_value;  // "s0_final", "row_final", ...
  };

  // Storage
  Kind kind;
  LetStmt let;
  ForStmt for_loop;
  UpdateStmt update;

  // Constructors
  static DslxStmt MakeLet(std::string name, std::string type, std::string rhs) {
    DslxStmt stmt;
    stmt.kind = kLet;
    stmt.let = {std::move(name), std::move(type), std::move(rhs)};
    return stmt;
  }

  static DslxStmt MakeFor(std::string dim_name, std::string acc,
                          std::string acc_type, int64_t begin, int64_t end,
                          std::vector<DslxStmt> body_stmts,
                          std::string body_expr, std::string seed,
                          std::optional<std::string> name = std::nullopt) {
    DslxStmt stmt;
    stmt.kind = kFor;
    stmt.for_loop = {std::move(name),
                     std::move(dim_name),
                     std::move(acc),
                     std::move(acc_type),
                     begin,
                     end,
                     1,
                     std::move(body_stmts),
                     std::move(body_expr),
                     std::move(seed)};
    return stmt;
  }

  static DslxStmt MakeUpdate(std::string array, std::string index,
                             std::string new_value) {
    DslxStmt stmt;
    stmt.kind = kUpdate;
    stmt.update = {std::move(array), std::move(index), std::move(new_value)};
    return stmt;
  }
};

// Accumulator structure for managing loop accumulators
struct Accumulator {
  std::string name;  // "acc0", "row", "s0"
  std::string type;  // "F32[32][8]", "F32[32]", "F32"
  std::string init;  // "F32[32][8]:[F32[32]:[F32_ZERO, ...], ...]", "F32_ZERO"

  // Default constructor (required for std::map)
  Accumulator() : name(""), type(""), init("") {}

  // Constructor to define accumulator in place
  Accumulator(std::string n, std::string t)
      : name(std::move(n)), type(std::move(t)), init("") {}

  // Constructor with init value
  Accumulator(std::string n, std::string t, std::string i)
      : name(std::move(n)), type(std::move(t)), init(std::move(i)) {}
};

// Accumulator management system
class AccumulatorManager {
 public:
  // Add an accumulator for a specific loop level
  void AddAccumulator(int loop_level, const Accumulator& acc) {
    accumulators_[loop_level] = acc;
  }

  // Get accumulator for a loop level
  const Accumulator* GetAccumulator(int loop_level) const {
    auto it = accumulators_.find(loop_level);
    return it != accumulators_.end() ? &it->second : nullptr;
  }

  // Get all accumulators
  const std::map<int, Accumulator>& GetAllAccumulators() const {
    return accumulators_;
  }

  // Clear all accumulators
  void Clear() { accumulators_.clear(); }

  // Check if accumulator exists for loop level
  bool HasAccumulator(int loop_level) const {
    return accumulators_.find(loop_level) != accumulators_.end();
  }

 private:
  std::map<int, Accumulator> accumulators_;  // loop_level -> accumulator
};

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_DSLX_TYPES_H_
