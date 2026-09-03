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

#include "xls/contrib/eco/ir_patch_gen.h"

#include <cstdint>

#include "xls/contrib/eco/ged.h"
#include "xls/contrib/eco/graph.h"
#include "xls/contrib/eco/ir_patch.pb.h"
#include "xls/ir/op.h"

namespace {

void PopulateUniqueArgs(const NodeCostAttributes& attrs,
                        xls_eco::NodeProto* node_proto) {
  if (node_proto == nullptr || !attrs.op.has_value()) {
    return;
  }

  const xls::viz::NodeAttributes& node_attributes = attrs.node_attributes;
  switch (*attrs.op) {
    case xls::Op::kLiteral:
      if (attrs.literal_value.has_value()) {
        node_proto->add_unique_args()->mutable_value()->CopyFrom(
            *attrs.literal_value);
      }
      break;
    case xls::Op::kBitSlice:
      if (node_attributes.has_start()) {
        node_proto->add_unique_args()->set_start(
            static_cast<uint64_t>(node_attributes.start()));
      }
      break;
    case xls::Op::kTupleIndex:
      if (node_attributes.has_index()) {
        node_proto->add_unique_args()->set_index(
            static_cast<uint64_t>(node_attributes.index()));
      }
      break;
    case xls::Op::kArrayIndex:
    case xls::Op::kArrayUpdate:
      if (attrs.array_assumed_in_bounds.has_value()) {
        node_proto->add_unique_args()->set_assumed_in_bounds(
            *attrs.array_assumed_in_bounds);
      }
      break;
    case xls::Op::kOneHot:
      if (node_attributes.has_lsb_prio()) {
        node_proto->add_unique_args()->set_lsb_prio(node_attributes.lsb_prio());
      }
      break;
    case xls::Op::kSignExt:
      if (node_attributes.has_new_bit_count()) {
        node_proto->add_unique_args()->set_new_bit_count(
            static_cast<uint64_t>(node_attributes.new_bit_count()));
      }
      break;
    case xls::Op::kSel:
    case xls::Op::kPrioritySel:
      if (node_attributes.has_has_default()) {
        node_proto->add_unique_args()->set_has_default_value(
            node_attributes.has_default());
      }
      break;
    case xls::Op::kReceive:
      if (node_attributes.has_channel()) {
        node_proto->add_unique_args()->set_channel(node_attributes.channel());
      }
      if (node_attributes.has_blocking()) {
        node_proto->add_unique_args()->set_blocking(node_attributes.blocking());
      }
      break;
    case xls::Op::kSend:
      if (node_attributes.has_channel()) {
        node_proto->add_unique_args()->set_channel(node_attributes.channel());
      }
      break;
    case xls::Op::kAssert:
      if (node_attributes.has_message_()) {
        node_proto->add_unique_args()->set_message(node_attributes.message_());
      }
      if (node_attributes.has_label()) {
        node_proto->add_unique_args()->set_label(node_attributes.label());
      }
      break;
    case xls::Op::kTrace:
      if (attrs.trace_xls_format.has_value()) {
        node_proto->add_unique_args()->set_format(*attrs.trace_xls_format);
      }
      if (node_attributes.has_verbosity()) {
        node_proto->add_unique_args()->set_verbosity(
            node_attributes.verbosity());
      }
      break;
    case xls::Op::kNext:
      // A Next has no state-read operand; PatchIr binds the element by name.
      if (attrs.state_element.has_value()) {
        node_proto->add_unique_args()->set_state_element(*attrs.state_element);
      }
      break;
    case xls::Op::kStateRead:
      // Element identity/placement ride on the state-element node; the read
      // only names its element (like next_value).
      if (attrs.state_element.has_value()) {
        node_proto->add_unique_args()->set_state_element(*attrs.state_element);
      }
      break;
    default:
      break;
  }
}

void ExportChannelProto(const ChannelInfo& info,
                        xls_eco::ChannelProto* channel_proto) {
  channel_proto->set_name(info.name);
  if (info.data_type.has_value()) {
    channel_proto->mutable_data_type()->CopyFrom(*info.data_type);
  }
  channel_proto->set_kind(info.kind);
  channel_proto->set_direction(info.direction);
  channel_proto->set_flow_control(info.flow_control);
  channel_proto->set_strictness(info.strictness);
  channel_proto->set_flop_kind(info.flop_kind);
}

void ExportEdgeProto(const XLSGraph& graph, const XLSEdge& edge,
                     xls_eco::EdgeProto* edge_proto) {
  edge_proto->set_from_node(graph.nodes[edge.endpoints.first].name);
  edge_proto->set_to_node(graph.nodes[edge.endpoints.second].name);
  edge_proto->set_index(edge.index);
  edge_proto->set_channel_binding(edge.cost_attributes.channel_binding);
}

void ExportNodeProto(const XLSNode& node, xls_eco::NodeProto* node_proto) {
  const NodeCostAttributes& attrs = node.cost_attributes;
  node_proto->set_name(node.name);
  if (attrs.data_type.has_value()) {
    node_proto->mutable_data_type()->CopyFrom(*attrs.data_type);
  }
  if (attrs.op.has_value()) {
    node_proto->set_op(xls::OpToString(*attrs.op));
  }
  for (const xls::TypeProto& operand_data_type : attrs.operand_data_types) {
    node_proto->add_operand_data_types()->CopyFrom(operand_data_type);
  }
  // Channel nodes carry no xls::Op; tag them with a synthetic "channel" op.
  if (attrs.channel.has_value()) {
    node_proto->set_op("channel");
    ExportChannelProto(*attrs.channel, node_proto->mutable_channel());
  }
  // State-element nodes carry no xls::Op; tag them "state_element"
  // with fixed unique_args [index, state_element, init, non_synthesizable]
  // (index is placement in the revised layout, not identity).
  if (!attrs.op.has_value() && attrs.state_element.has_value()) {
    node_proto->set_op("state_element");
    node_proto->add_unique_args()->set_index(
        static_cast<uint64_t>(attrs.state_index.value_or(0)));
    node_proto->add_unique_args()->set_state_element(*attrs.state_element);
    if (attrs.state_initial_value.has_value()) {
      node_proto->add_unique_args()->mutable_init()->CopyFrom(
          *attrs.state_initial_value);
    }
    node_proto->add_unique_args()->set_non_synthesizable(
        attrs.state_non_synthesizable.value_or(false));
  }

  PopulateUniqueArgs(attrs, node_proto);
}

}  // namespace

xls_eco::IrPatchProto GenerateIrPatchProto(const XLSGraph& original_graph,
                                           const XLSGraph& modified_graph,
                                           const ged::GEDResult& ged_result) {
  xls_eco::IrPatchProto patch_proto;
  uint32_t id = 0;
  auto add_edit_path = [&](xls_eco::Operation operation) {
    auto* edit_path = patch_proto.add_edit_paths();
    edit_path->set_id(id++);
    edit_path->set_operation(operation);
    return edit_path;
  };

  for (const auto& [original_node_id, modified_node_id] :
       ged_result.node_substitutions) {
    auto* node_edit_path =
        add_edit_path(xls_eco::Operation::UPDATE)->mutable_node_edit_path();
    ExportNodeProto(original_graph.nodes[original_node_id],
                    node_edit_path->mutable_node());
    ExportNodeProto(modified_graph.nodes[modified_node_id],
                    node_edit_path->mutable_updated_node());
  }
  for (int original_node_id : ged_result.node_deletions) {
    auto* edit_path = add_edit_path(xls_eco::Operation::DELETE);
    ExportNodeProto(original_graph.nodes[original_node_id],
                    edit_path->mutable_node_edit_path()->mutable_node());
  }
  for (int modified_node_id : ged_result.node_insertions) {
    auto* edit_path = add_edit_path(xls_eco::Operation::INSERT);
    ExportNodeProto(modified_graph.nodes[modified_node_id],
                    edit_path->mutable_node_edit_path()->mutable_node());
  }
  for (const auto& [original_edge_id, modified_edge_id] :
       ged_result.edge_substitutions) {
    auto* edge_edit_path =
        add_edit_path(xls_eco::Operation::UPDATE)->mutable_edge_edit_path();
    ExportEdgeProto(original_graph, original_graph.edges[original_edge_id],
                    edge_edit_path->mutable_edge());
    ExportEdgeProto(modified_graph, modified_graph.edges[modified_edge_id],
                    edge_edit_path->mutable_updated_edge());
  }
  for (int original_edge_id : ged_result.edge_deletions) {
    auto* edit_path = add_edit_path(xls_eco::Operation::DELETE);
    ExportEdgeProto(original_graph, original_graph.edges[original_edge_id],
                    edit_path->mutable_edge_edit_path()->mutable_edge());
  }
  for (int modified_edge_id : ged_result.edge_insertions) {
    auto* edit_path = add_edit_path(xls_eco::Operation::INSERT);
    ExportEdgeProto(modified_graph, modified_graph.edges[modified_edge_id],
                    edit_path->mutable_edge_edit_path()->mutable_edge());
  }

  // Always record the return node when present. RestoreReturnNode() needs it
  // whenever the return node was isolated via UPDATE or DELETE, even if the
  // return-node name did not change between the two versions.
  if (modified_graph.return_node_name.has_value()) {
    auto it = modified_graph.node_name_to_index.find(
        *modified_graph.return_node_name);
    if (it != modified_graph.node_name_to_index.end()) {
      xls_eco::NodeProto* ret_proto = patch_proto.mutable_return_node();
      ret_proto->set_name(*modified_graph.return_node_name);
      ExportNodeProto(modified_graph.nodes[it->second], ret_proto);
    }
  }

  return patch_proto;
}
