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

#include "xls/contrib/eco/block_report.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "re2/re2.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/eco/graph.h"
#include "xls/contrib/eco/xls_ir_to_graph.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"

namespace xls {
namespace {

// Merkle hash of every node's input cone, by node name, with register reads,
// ports and literals as leaves (their labels carry the name or value).
absl::StatusOr<absl::flat_hash_map<std::string, size_t>> ConeHashes(
    Block* block) {
  XLS_ASSIGN_OR_RETURN(XLSGraph graph, XlsIrToGraph(block));
  std::vector<size_t> hashes = graph.StructuralHash();
  absl::flat_hash_map<std::string, size_t> by_name;
  for (int64_t i = 0; i < graph.nodes.size(); ++i) {
    by_name[graph.nodes[i].name] = hashes[i];
  }
  return by_name;
}

// Hashes of a register's writes (or of just the values they write), so two
// blocks' registers compare by the logic that feeds them.
std::vector<size_t> WriteHashes(
    Block* block, Register* reg,
    const absl::flat_hash_map<std::string, size_t>& cones, bool data_only) {
  std::vector<size_t> hashes;
  absl::StatusOr<absl::Span<RegisterWrite* const>> writes =
      block->GetRegisterWrites(reg);
  if (writes.ok()) {
    for (RegisterWrite* write : *writes) {
      Node* node = data_only ? write->data() : write;
      hashes.push_back(cones.at(node->GetName()));
    }
  }
  std::sort(hashes.begin(), hashes.end());
  return hashes;
}

// Pipeline registers are named p<k>_<node>; the same node in another stage is
// the same register moved.
std::string WithoutStage(std::string_view name) {
  std::string out(name);
  RE2::Replace(&out, "^p[0-9]+_", "");
  return out;
}

absl::Status CompareRegisters(Block* block, Block* original,
                              BlockReport& report) {
  XLS_ASSIGN_OR_RETURN(auto cones, ConeHashes(block));
  XLS_ASSIGN_OR_RETURN(auto original_cones, ConeHashes(original));
  absl::flat_hash_map<std::string, std::vector<Register*>> by_base;
  for (Register* reg : block->GetRegisters()) {
    by_base[WithoutStage(reg->name())].push_back(reg);
  }
  absl::flat_hash_set<Register*> paired;
  report.original_registers = original->GetRegisters().size();
  report.registers = block->GetRegisters().size();
  for (Register* original_reg : original->GetRegisters()) {
    const int64_t bits = original_reg->type()->GetFlatBitCount();
    const std::string type = original_reg->type()->ToString();
    absl::StatusOr<Register*> same_name =
        block->GetRegister(original_reg->name());
    if (same_name.ok()) {
      paired.insert(*same_name);
      if ((*same_name)->type()->ToString() != type) {
        ++report.registers_retyped;
        continue;
      }
      ++report.registers_kept;
      report.registers_kept_bits += bits;
      if (WriteHashes(block, *same_name, cones, /*data_only=*/true) ==
          WriteHashes(original, original_reg, original_cones, true)) {
        ++report.register_data_cones_identical;
        report.register_data_cones_identical_bits += bits;
      }
      report.register_cones_identical +=
          WriteHashes(block, *same_name, cones, /*data_only=*/false) ==
          WriteHashes(original, original_reg, original_cones, false);
      continue;
    }
    Register* moved = nullptr;
    for (Register* candidate : by_base[WithoutStage(original_reg->name())]) {
      if (!paired.contains(candidate) &&
          candidate->type()->ToString() == type) {
        moved = candidate;
        break;
      }
    }
    if (moved != nullptr) {
      paired.insert(moved);
      ++report.registers_moved;
      report.registers_moved_bits += bits;
    } else {
      ++report.registers_removed;
      report.registers_removed_bits += bits;
    }
  }
  for (Register* reg : block->GetRegisters()) {
    if (!paired.contains(reg)) {
      ++report.registers_added;
      report.registers_added_bits += reg->type()->GetFlatBitCount();
    }
  }

  // Ports: name and direction identify one, the type says whether it is kept.
  absl::flat_hash_map<std::string, std::pair<std::string, Node*>> ports;
  auto collect = [](Block* b) {
    absl::flat_hash_map<std::string, std::pair<std::string, Node*>> out;
    for (InputPort* port : b->GetInputPorts()) {
      out[absl::StrCat("in ", port->name())] = {port->GetType()->ToString(),
                                                port};
    }
    for (OutputPort* port : b->GetOutputPorts()) {
      out[absl::StrCat("out ", port->name())] = {
          port->operand(0)->GetType()->ToString(), port};
    }
    return out;
  };
  ports = collect(block);
  auto original_ports = collect(original);
  report.original_ports = original_ports.size();
  report.ports = ports.size();
  for (const auto& [key, original_port] : original_ports) {
    auto it = ports.find(key);
    if (it == ports.end()) {
      ++report.ports_removed;
    } else if (it->second.first != original_port.first) {
      ++report.ports_retyped;
    } else {
      ++report.ports_kept;
      if (original_port.second->Is<OutputPort>()) {
        ++report.output_ports_kept;
        report.output_cones_identical +=
            cones.at(it->second.second->GetName()) ==
            original_cones.at(original_port.second->GetName());
      }
    }
  }
  report.ports_added =
      report.ports - (report.ports_kept + report.ports_retyped);
  return absl::OkStatus();
}

}  // namespace

std::string BlockReport::ToString() const {
  return absl::StrFormat(
      "Registers: %d (original %d)\n"
      "  kept: %d (%d bits)\n"
      "  moved: %d (%d bits)\n"
      "  retyped: %d\n"
      "  removed: %d (%d bits)\n"
      "  added: %d (%d bits)\n"
      "  data cones identical: %d of %d kept (%d of %d bits)\n"
      "  cones identical with enables: %d of %d kept\n"
      "Ports: %d (original %d)\n"
      "  kept: %d\n"
      "  retyped: %d\n"
      "  removed: %d\n"
      "  added: %d\n"
      "  output cones identical: %d of %d kept\n",
      registers, original_registers, registers_kept, registers_kept_bits,
      registers_moved, registers_moved_bits, registers_retyped,
      registers_removed, registers_removed_bits, registers_added,
      registers_added_bits, register_data_cones_identical, registers_kept,
      register_data_cones_identical_bits, registers_kept_bits,
      register_cones_identical, registers_kept, ports, original_ports,
      ports_kept, ports_retyped, ports_removed, ports_added,
      output_cones_identical, output_ports_kept);
}

absl::StatusOr<BlockReport> CompareBlocks(Block* block, Block* original) {
  BlockReport report;
  XLS_RETURN_IF_ERROR(CompareRegisters(block, original, report));
  return report;
}

std::string RtlDiff::ToString() const {
  return absl::StrFormat(
      "RTL lines: %d (original %d)\n"
      "  changed: %d\n"
      "  changed disregarding order: %d\n",
      lines, original_lines, changed, changed_disregarding_order);
}

RtlDiff DiffRtl(std::string_view original, std::string_view text) {
  std::vector<std::string_view> a = absl::StrSplit(original, '\n');
  std::vector<std::string_view> b = absl::StrSplit(text, '\n');
  RtlDiff diff{.original_lines = static_cast<int64_t>(a.size()),
               .lines = static_cast<int64_t>(b.size())};

  // Longest common subsequence, keeping two rows of the table.
  std::vector<int64_t> previous(b.size() + 1, 0);
  std::vector<int64_t> current(b.size() + 1, 0);
  for (std::string_view line : a) {
    for (int64_t j = 0; j < b.size(); ++j) {
      current[j + 1] = line == b[j] ? previous[j] + 1
                                    : std::max(previous[j + 1], current[j]);
    }
    std::swap(previous, current);
  }
  const int64_t common = previous[b.size()];
  diff.changed = (a.size() - common) + (b.size() - common);

  absl::btree_map<std::string_view, int64_t> counts;
  for (std::string_view line : a) {
    ++counts[line];
  }
  int64_t shared = 0;
  for (std::string_view line : b) {
    auto it = counts.find(line);
    if (it != counts.end() && it->second > 0) {
      --it->second;
      ++shared;
    }
  }
  diff.changed_disregarding_order = (a.size() - shared) + (b.size() - shared);
  return diff;
}

}  // namespace xls
