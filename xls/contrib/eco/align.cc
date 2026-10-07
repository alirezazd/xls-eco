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

#include "xls/contrib/eco/align.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "re2/re2.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/eco/graph.h"
#include "xls/contrib/eco/match.h"
#include "xls/contrib/eco/xls_ir_to_graph.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

// A converted block lists its nodes in creation order and a parsed one in dump
// order; going through the text puts both in dump order, which is what the
// identity check below compares.
absl::StatusOr<XLSGraph> GraphInDumpOrder(FunctionBase* function_base) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Package> package,
      Parser::ParsePackage(function_base->package()->DumpIr()));
  XLS_ASSIGN_OR_RETURN(FunctionBase * copy,
                       package->GetFunctionBaseByName(function_base->name()));
  return XlsIrToGraph(copy);
}

// When both graphs are identical, e.g. the original block re-emitted from its
// own IR, maps nodes by position and skips the search. Returns nullopt if any
// label or edge differs.
std::optional<std::vector<std::pair<std::string, std::string>>> IdentityMap(
    const XLSGraph& original, const XLSGraph& graph) {
  if (original.nodes.size() != graph.nodes.size() ||
      original.edges.size() != graph.edges.size()) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < original.edges.size(); ++i) {
    const XLSEdge& a = original.edges[i];
    const XLSEdge& b = graph.edges[i];
    if (a.endpoints != b.endpoints || a.label != b.label ||
        a.index != b.index) {
      return std::nullopt;
    }
  }
  std::vector<std::pair<std::string, std::string>> node_map;
  node_map.reserve(original.nodes.size());
  for (int64_t i = 0; i < original.nodes.size(); ++i) {
    if (original.nodes[i].label != graph.nodes[i].label &&
        original.nodes[i].cost_attributes.op != Op::kAssert) {
      return std::nullopt;
    }
    node_map.emplace_back(original.nodes[i].name, graph.nodes[i].name);
  }
  return node_map;
}

// Replaces `op.id` tokens, the pre-finalization names codegen bakes into its
// invariant assertions, so two messages that differ only there compare equal.
std::string WithoutNodeIds(std::string_view message) {
  std::string out(message);
  RE2::GlobalReplace(&out, "[a-z_]+\\.[0-9]+", "#");
  return out;
}

// Codegen embeds the selector's node name in its one-hot invariant assertion,
// so the message changes whenever ids do. Pairs the assertions the match left
// out through their condition, and gives every paired assertion the original
// message.
absl::Status AdoptAssertMessages(
    FunctionBase* function_base, FunctionBase* original,
    std::vector<std::pair<Node*, Node*>>& partners) {
  absl::flat_hash_map<Node*, Node*> partner_of(partners.begin(),
                                               partners.end());
  absl::flat_hash_set<Node*> matched_original;
  for (const auto& [node, original_node] : partners) {
    matched_original.insert(original_node);
  }
  // Unmatched original assertions keyed by condition and id-free message. A
  // state element's assertion and its non-synthesizable copy share both keys.
  absl::flat_hash_map<std::pair<Node*, std::string>, std::vector<Assert*>>
      original_asserts;
  for (Node* node : original->nodes()) {
    if (node->Is<Assert>() && !matched_original.contains(node)) {
      Assert* assert = node->As<Assert>();
      original_asserts[{assert->condition(), WithoutNodeIds(assert->message())}]
          .push_back(assert);
    }
  }
  for (Node* node : function_base->nodes()) {
    if (!node->Is<Assert>() || partner_of.contains(node)) {
      continue;
    }
    Assert* assert = node->As<Assert>();
    auto condition = partner_of.find(assert->condition());
    if (condition == partner_of.end()) {
      continue;
    }
    auto it = original_asserts.find(
        {condition->second, WithoutNodeIds(assert->message())});
    if (it != original_asserts.end() && !it->second.empty()) {
      partners.emplace_back(assert, it->second.back());
      it->second.pop_back();
    }
  }

  for (auto& [node, original_node] : partners) {
    if (!node->Is<Assert>() || !original_node->Is<Assert>()) {
      continue;
    }
    Assert* assert = node->As<Assert>();
    const std::string& message = original_node->As<Assert>()->message();
    if (assert->message() == message ||
        WithoutNodeIds(assert->message()) != WithoutNodeIds(message)) {
      continue;
    }
    std::string name = assert->HasAssignedName() ? assert->GetName() : "";
    XLS_ASSIGN_OR_RETURN(Assert * replacement,
                         assert->ReplaceUsesWithNew<Assert>(
                             assert->token(), assert->condition(), message,
                             assert->label(), assert->original_label()));
    XLS_RETURN_IF_ERROR(function_base->RemoveNode(assert));
    if (!name.empty()) {
      replacement->SetName(name);
    }
    node = replacement;
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Alignment> Align(FunctionBase* function_base,
                                FunctionBase* original,
                                const MatchOptions& options) {
  XLS_ASSIGN_OR_RETURN(XLSGraph original_graph, GraphInDumpOrder(original));
  XLS_ASSIGN_OR_RETURN(XLSGraph graph, GraphInDumpOrder(function_base));
  std::vector<std::pair<std::string, std::string>> node_map;
  if (auto identity = IdentityMap(original_graph, graph)) {
    node_map = std::move(*identity);
  } else {
    node_map = MatchGraphs(original_graph, graph, options).node_map;
  }

  // A proc graph also has channel and state-element nodes; those are not IR
  // nodes and carry no id.
  std::vector<std::pair<Node*, Node*>> partners;
  partners.reserve(node_map.size());
  for (const auto& [original_name, name] : node_map) {
    std::optional<Node*> original_node = original->MaybeGetNode(original_name);
    std::optional<Node*> node = function_base->MaybeGetNode(name);
    if (original_node.has_value() && node.has_value()) {
      partners.emplace_back(*node, *original_node);
    }
  }
  XLS_RETURN_IF_ERROR(AdoptAssertMessages(function_base, original, partners));

  // User lists are sorted by id, so two nodes must never share an id during
  // renumbering. Move every node above both packages' ids first.
  Package* package = function_base->package();
  int64_t next_id =
      std::max(package->next_node_id(), original->package()->next_node_id());
  for (FunctionBase* fb : package->GetFunctionBases()) {
    for (Node* node : fb->nodes()) {
      node->SetId(next_id++);
    }
  }
  Alignment alignment{.original_nodes = original->node_count(),
                      .nodes = function_base->node_count()};
  for (const auto& [node, original_node] : partners) {
    node->SetId(original_node->id());
    if (!node->HasAssignedName() && original_node->HasAssignedName()) {
      node->SetName(original_node->GetName());
    }
    alignment.matched.insert(node);
  }
  package->set_next_node_id(next_id);
  return alignment;
}

}  // namespace xls
