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

#ifndef XLS_CONTRIB_ECO_BLOCK_REPORT_H_
#define XLS_CONTRIB_ECO_BLOCK_REPORT_H_

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/block.h"

namespace xls {

// Compares a patched block against the original on what a netlist ECO
// depends on. A
// register is kept when name and type agree, moved when only its pipeline
// stage prefix (p<k>_) changed, retyped when its type changed; a cone is
// identical when the logic feeding it hashes the same with register reads,
// ports and literals as leaves.
struct BlockReport {
  int64_t original_registers = 0;
  int64_t registers = 0;
  int64_t registers_kept = 0;
  int64_t registers_kept_bits = 0;
  int64_t registers_moved = 0;
  int64_t registers_moved_bits = 0;
  int64_t registers_retyped = 0;
  int64_t registers_removed = 0;
  int64_t registers_removed_bits = 0;
  int64_t registers_added = 0;
  int64_t registers_added_bits = 0;
  // The value feeding a kept register hashes the same.
  int64_t register_data_cones_identical = 0;
  int64_t register_data_cones_identical_bits = 0;
  // The whole write, load enable and reset included, hashes the same. In a
  // proc these come from shared control, so one changed control net makes
  // every register differ.
  int64_t register_cones_identical = 0;

  int64_t original_ports = 0;
  int64_t ports = 0;
  int64_t ports_kept = 0;
  int64_t ports_retyped = 0;
  int64_t ports_removed = 0;
  int64_t ports_added = 0;
  int64_t output_ports_kept = 0;
  int64_t output_cones_identical = 0;

  std::string ToString() const;
};

absl::StatusOr<BlockReport> CompareBlocks(Block* block, Block* original);

struct RtlDiff {
  int64_t original_lines = 0;
  int64_t lines = 0;
  int64_t changed = 0;                     // Lines a unified diff would show.
  int64_t changed_disregarding_order = 0;  // Lines not present on both sides.
  std::string ToString() const;
};

// Line-level distance between two RTL texts. Codegen may move a whole cone
// when the ECO makes an early node read a late one, so the order-blind count
// is what a netlist ECO sees.
RtlDiff DiffRtl(std::string_view original, std::string_view text);

}  // namespace xls

#endif  // XLS_CONTRIB_ECO_BLOCK_REPORT_H_
