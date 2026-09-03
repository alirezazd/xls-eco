// Copyright 2026 The XLS Authors
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

#include "xls/contrib/eco/xls_ir_to_graph.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/eco/graph.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/ir/value.h"
#include "xls/visualization/ir_viz/node_attribute_visitor.h"

namespace xls {
namespace {

// Element identity shared by a state read and its state-element node.
absl::Status AddStateElementAttributes(Proc* proc, StateElement* element,
                                       NodeCostAttributes* attrs) {
  attrs->state_element = element->name();
  attrs->state_non_synthesizable = element->non_synthesizable();
  XLS_ASSIGN_OR_RETURN(attrs->state_initial_value,
                       element->initial_value().AsProto());
  // Placement only; not part of the label.
  XLS_ASSIGN_OR_RETURN(attrs->state_index,
                       proc->GetStateElementIndex(element));
  return absl::OkStatus();
}

absl::StatusOr<NodeCostAttributes> GetNodeCostAttributes(Node* node) {
  NodeCostAttributes attrs;
  attrs.op = node->op();
  attrs.data_type = node->GetType()->ToProto();
  attrs.operand_data_types.reserve(node->operand_count());
  for (Node* operand : node->operands()) {
    attrs.operand_data_types.push_back(operand->GetType()->ToProto());
  }
  if (node->Is<Literal>()) {
    XLS_ASSIGN_OR_RETURN(attrs.literal_value,
                         node->As<Literal>()->value().AsProto());
  }
  if (node->Is<ArrayIndex>()) {
    attrs.array_assumed_in_bounds = node->As<ArrayIndex>()->assumed_in_bounds();
  } else if (node->Is<ArrayUpdate>()) {
    attrs.array_assumed_in_bounds =
        node->As<ArrayUpdate>()->assumed_in_bounds();
  }
  if (node->Is<Param>()) {
    XLS_ASSIGN_OR_RETURN(
        attrs.param_index,
        node->function_base()->GetParamIndex(node->As<Param>()));
  }
  // A Next binds its state element as metadata, not an operand, so the graph
  // has no edge for it; carry the element name in the label instead.
  if (node->Is<Next>()) {
    attrs.state_element = node->As<Next>()->state_element()->name();
  } else if (node->Is<StateRead>()) {
    XLS_RETURN_IF_ERROR(AddStateElementAttributes(
        node->function_base()->AsProcOrDie(),
        node->As<StateRead>()->state_element(), &attrs));
  }
  if (node->Is<Trace>()) {
    attrs.trace_xls_format =
        StepsToXlsFormatString(node->As<Trace>()->format());
  }

  AttributeVisitor visitor;
  XLS_RETURN_IF_ERROR(node->VisitSingleNode(&visitor));
  attrs.node_attributes = visitor.attributes();
  return attrs;
}

EdgeCostAttributes GetEdgeCostAttributes(Node* operand, Node* user,
                                         int64_t index) {
  EdgeCostAttributes attrs;
  attrs.source_data_type = operand->GetType()->ToProto();
  attrs.source_op = operand->op();
  attrs.sink_data_type = user->GetType()->ToProto();
  attrs.sink_op = user->op();
  if (!OpIsCommutative(user->op())) {
    attrs.index = index;
  }
  return attrs;
}

// Label for a synthetic channel node (no xls::Op; identity is the channel).
NodeCostAttributes GetNodeCostAttributes(const ChannelInterface* interface) {
  NodeCostAttributes attrs;
  ChannelInfo& info = attrs.channel.emplace();
  info.name = std::string(interface->name());
  info.data_type = interface->type()->ToProto();
  info.kind = ChannelKindToString(interface->kind());
  info.direction = ChannelDirectionToString(interface->direction());
  info.flow_control = FlowControlToString(interface->flow_control());
  if (interface->strictness().has_value()) {
    info.strictness = ChannelStrictnessToString(*interface->strictness());
  }
  info.flop_kind = FlopKindToString(interface->flop_kind());
  return attrs;
}

// Label for a synthetic state-element node (no xls::Op; identity is the
// element: name, type, init, non_synthesizable).
absl::StatusOr<NodeCostAttributes> GetNodeCostAttributes(
    Proc* proc, StateElement* element) {
  NodeCostAttributes attrs;
  attrs.data_type = element->type()->ToProto();
  XLS_RETURN_IF_ERROR(AddStateElementAttributes(proc, element, &attrs));
  return attrs;
}

// Canonical edge order so patches depend on the graph, not construction order.
void SortEdgesAndRefresh(XLSGraph& graph) {
  absl::c_sort(graph.edges, [](const XLSEdge& a, const XLSEdge& b) {
    if (a.endpoints.first != b.endpoints.first) {
      return a.endpoints.first < b.endpoints.first;
    }
    if (a.endpoints.second != b.endpoints.second) {
      return a.endpoints.second < b.endpoints.second;
    }
    return a.index < b.index;
  });
  graph.RefreshAdjacency();
  graph.RefreshEdgeCounts();
}

// Models channels as first-class nodes with synthetic binding edges to their
// Send/Receive nodes, so channel add/remove/retype are detected and patched
// through the ordinary node/edge machinery.
absl::Status AddChannelNodesAndBindings(Proc* proc, XLSGraph& graph) {
  absl::flat_hash_map<std::string, int> channel_name_to_index;
  for (const ChannelInterface* interface : proc->interface()) {
    const std::string name(interface->name());
    XLSNode channel_node(absl::StrCat("chan.", name),
                         GetNodeCostAttributes(interface));
    channel_node.all_attributes = {
        {"name", name},
        {"op", "channel"},
    };
    channel_name_to_index[name] = graph.add_node(channel_node);
  }

  for (Node* node : proc->nodes()) {
    std::optional<std::string> channel_name;
    if (node->Is<Send>()) {
      channel_name = std::string(node->As<Send>()->channel_name());
    } else if (node->Is<Receive>()) {
      channel_name = std::string(node->As<Receive>()->channel_name());
    }
    if (!channel_name.has_value()) {
      continue;
    }
    auto it = channel_name_to_index.find(*channel_name);
    if (it == channel_name_to_index.end()) {
      continue;
    }
    const int channel_index = it->second;
    XLS_RET_CHECK(graph.node_name_to_index.contains(node->GetName()))
        << "Missing graph node for channel op " << node->GetName();
    const int node_index = graph.node_name_to_index.at(node->GetName());
    EdgeCostAttributes edge_attrs;
    edge_attrs.channel_binding = true;
    // Send writes into the channel; the channel feeds Receive.
    if (node->Is<Send>()) {
      graph.add_edge(XLSEdge(node_index, channel_index, edge_attrs, 0));
    } else {
      graph.add_edge(XLSEdge(channel_index, node_index, edge_attrs, 0));
    }
  }
  return absl::OkStatus();
}

// Models proc state elements as first-class nodes with synthetic binding
// edges to their reads and next_values, so element add/remove/move carries
// element identity through ordinary node edits (mirrors channel nodes).
absl::Status AddStateElementNodesAndBindings(Proc* proc, XLSGraph& graph) {
  absl::flat_hash_map<std::string, int> element_name_to_index;
  for (StateElement* element : proc->StateElements()) {
    XLS_ASSIGN_OR_RETURN(NodeCostAttributes attrs,
                         GetNodeCostAttributes(proc, element));
    XLSNode element_node(absl::StrCat("state.", element->name()), attrs);
    element_node.all_attributes = {
        {"name", element->name()},
        {"op", "state_element"},
    };
    element_name_to_index[element->name()] = graph.add_node(element_node);
  }

  for (Node* node : proc->nodes()) {
    std::optional<std::string> element_name;
    if (node->Is<StateRead>()) {
      element_name = node->As<StateRead>()->state_element()->name();
    } else if (node->Is<Next>()) {
      element_name = node->As<Next>()->state_element()->name();
    }
    if (!element_name.has_value()) {
      continue;
    }
    const int element_index = element_name_to_index.at(*element_name);
    XLS_RET_CHECK(graph.node_name_to_index.contains(node->GetName()))
        << "Missing graph node for state op " << node->GetName();
    const int node_index = graph.node_name_to_index.at(node->GetName());
    EdgeCostAttributes edge_attrs;
    edge_attrs.channel_binding = true;  // Generic binding edge; skipped by
                                        // PatchIr like channel bindings.
    // The element feeds its read; a next_value writes into the element.
    if (node->Is<StateRead>()) {
      graph.add_edge(XLSEdge(element_index, node_index, edge_attrs, 0));
    } else {
      graph.add_edge(XLSEdge(node_index, element_index, edge_attrs, 0));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<XLSGraph> XlsIrToGraph(FunctionBase* function_base) {
  XLS_RET_CHECK(function_base != nullptr);

  XLSGraph graph;
  for (Node* node : function_base->nodes()) {
    XLS_ASSIGN_OR_RETURN(NodeCostAttributes cost_attributes,
                         GetNodeCostAttributes(node));
    XLSNode graph_node(node->GetName(), cost_attributes);
    graph_node.all_attributes = {
        {"id", absl::StrCat(node->id())},
        {"name", node->GetName()},
        {"op", OpToString(node->op())},
        {"ir", node->ToStringWithOperandTypes()},
        {"cost_attributes", graph_node.cost_attributes.DebugString()},
    };
    graph.add_node(graph_node);
  }

  for (Node* node : function_base->nodes()) {
    XLS_RET_CHECK(graph.node_name_to_index.contains(node->GetName()))
        << "Missing graph node for IR node " << node->GetName();
    const int sink = graph.node_name_to_index.at(node->GetName());
    for (int64_t index = 0; index < node->operand_count(); ++index) {
      Node* operand = node->operand(index);
      XLS_RET_CHECK(graph.node_name_to_index.contains(operand->GetName()))
          << "Missing graph node for operand " << operand->GetName();
      const int source = graph.node_name_to_index.at(operand->GetName());
      graph.add_edge(XLSEdge(source, sink,
                             GetEdgeCostAttributes(operand, node, index),
                             static_cast<int>(index)));
    }
  }

  if (function_base->IsFunction()) {
    Function* function = function_base->AsFunctionOrDie();
    if (function->return_value() != nullptr) {
      graph.return_node_name = function->return_value()->GetName();
    }
  }
  if (function_base->IsProc()) {
    Proc* proc = function_base->AsProcOrDie();
    if (proc->is_new_style_proc()) {
      XLS_RETURN_IF_ERROR(AddChannelNodesAndBindings(proc, graph));
    }
    XLS_RETURN_IF_ERROR(AddStateElementNodesAndBindings(proc, graph));
  }

  SortEdgesAndRefresh(graph);
  graph.populate_node_signatures();
  graph.RefreshReturnAndIndex();
  return graph;
}

absl::StatusOr<XLSGraph> ParseIrFileToGraph(std::string_view ir_path) {
  if (ir_path == "-") {
    ir_path = "/dev/stdin";
  }

  XLS_ASSIGN_OR_RETURN(std::string ir, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir, ir_path));
  std::optional<FunctionBase*> top = package->GetTop();
  XLS_RET_CHECK(top.has_value()) << "IR package has no top entity";
  return XlsIrToGraph(*top);
}

}  // namespace xls
