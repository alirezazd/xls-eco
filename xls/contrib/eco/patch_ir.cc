// Copyright 2020 The XLS Authors
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

#include "xls/contrib/eco/patch_ir.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/eco/ir_patch.pb.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/function_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/xls_type.pb.h"
#include "xls/ir/xls_value.pb.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/tools/codegen_flags.h"
#include "xls/tools/scheduling_options_flags.h"

namespace xls {

namespace {

const absl::flat_hash_map<std::string, Op>& PatchToIrOpMap() {
  static const auto* kMap = new absl::flat_hash_map<std::string, Op>{
      {"literal", Op::kLiteral},
      {"param", Op::kParam},
      {"sub", Op::kSub},
      {"add", Op::kAdd},
      {"and_reduce", Op::kAndReduce},
      {"one_hot", Op::kOneHot},
      {"shrl", Op::kShrl},
      {"shll", Op::kShll},
      {"neg", Op::kNeg},
      {"eq", Op::kEq},
      {"ne", Op::kNe},
      {"ugt", Op::kUGt},
      {"ult", Op::kULt},
      {"ule", Op::kULe},
      {"sle", Op::kSLe},
      {"umul", Op::kUMul},
      {"smul", Op::kSMul},
      {"udiv", Op::kUDiv},
      {"sdiv", Op::kSDiv},
      {"concat", Op::kConcat},
      {"bit_slice", Op::kBitSlice},
      {"one_hot_sel", Op::kOneHotSel},
      {"or", Op::kOr},
      {"and", Op::kAnd},
      {"xor", Op::kXor},
      {"nand", Op::kNand},
      {"nor", Op::kNor},
      {"not", Op::kNot},
      {"sel", Op::kSel},
      {"priority_sel", Op::kPrioritySel},
      {"or_reduce", Op::kOrReduce},
      {"tuple", Op::kTuple},
      {"tuple_index", Op::kTupleIndex},
      {"array", Op::kArray},
      {"array_index", Op::kArrayIndex},
      {"array_update", Op::kArrayUpdate},
      {"sign_ext", Op::kSignExt},
      {"state_read", Op::kStateRead},
      {"after_all", Op::kAfterAll},
      {"receive", Op::kReceive},
      {"send", Op::kSend},
      {"next_value", Op::kNext},
      {"assert", Op::kAssert},
      {"trace", Op::kTrace},
      {"gate", Op::kGate},
  };
  return *kMap;
}

const absl::flat_hash_map<std::pair<bool, xls_eco::Operation>,
                          PatchIr::EditPathPriority>&
EditPathPriorityMap() {
  static const auto* kMap =
      new absl::flat_hash_map<std::pair<bool, xls_eco::Operation>,
                              PatchIr::EditPathPriority>{
          {{true, xls_eco::UPDATE}, PatchIr::EditPathPriority::kNodeUpdate},
          {{true, xls_eco::INSERT}, PatchIr::EditPathPriority::kNodeInsert},
          {{true, xls_eco::DELETE}, PatchIr::EditPathPriority::kNodeDelete},
          {{false, xls_eco::UPDATE}, PatchIr::EditPathPriority::kEdgeUpdate},
          {{false, xls_eco::INSERT}, PatchIr::EditPathPriority::kEdgeInsert},
          {{false, xls_eco::DELETE}, PatchIr::EditPathPriority::kEdgeDelete},
      };
  return *kMap;
}

}  // namespace

PatchIr::PatchIr(FunctionBase* function_base, xls_eco::IrPatchProto& patch)
    : patch_(patch), function_base_(function_base), schedule_(std::nullopt) {
  std::copy(patch_.edit_paths().begin(), patch_.edit_paths().end(),
            std::back_inserter(sorted_edit_paths_));
  absl::c_sort(sorted_edit_paths_, [this](const xls_eco::EditPathProto& lhs,
                                          const xls_eco::EditPathProto& rhs) {
    return this->CompareEditPaths(lhs, rhs);
  });
  package_ = function_base_->package();
}
absl::StatusOr<std::vector<Node*>> PatchIr::MakeDummyNodes(
    absl::Span<Type*> types) {
  std::vector<Node*> dummy_nodes;
  for (Type* type : types) {
    XLS_ASSIGN_OR_RETURN(Node * dummy_node, MakeDummyNode(type));
    dummy_nodes.push_back(dummy_node);
  }
  return dummy_nodes;
}
absl::StatusOr<Node*> PatchIr::MakeDummyNode(Type* type) {
  XLS_ASSIGN_OR_RETURN(Node * dummy_node,
                       function_base_->MakeNodeWithName<Literal>(
                           SourceInfo(), ZeroOfType(type), "Dummy"));
  return dummy_node;
}
void PatchIr::UpdateNodeMaps(Node* n, absl::Span<Node*> dummy_operands,
                             std::string_view node_name) {
  auto& dummy_nodes = dummy_nodes_map_[n];
  dummy_nodes.insert(dummy_nodes.begin(), dummy_operands.begin(),
                     dummy_operands.end());
  patch_to_ir_node_map_[node_name] = n->GetName();
}
absl::Status PatchIr::CleanupDummyNodes(Node* node) {
  auto& dummy_nodes = dummy_nodes_map_[node];
  for (auto it = dummy_nodes.begin(); it != dummy_nodes.end();) {
    Node* dummy_node = *it;
    XLS_RETURN_IF_ERROR(function_base_->RemoveNode(dummy_node));
    it = dummy_nodes.erase(it);
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> PatchIr::GetProtoBitCount(const TypeProto& type) {
  XLS_ASSIGN_OR_RETURN(Type * t, package_->GetTypeFromProto(type));
  XLS_ASSIGN_OR_RETURN(BitsType * b, t->AsBits());
  return b->bit_count();
}

absl::StatusOr<Node*> PatchIr::ResolveNodeByPatchName(
    std::string_view patch_name) {
  auto it = patch_to_ir_node_map_.find(patch_name);
  if (it != patch_to_ir_node_map_.end()) {
    return function_base_->GetNode(it->second);
  }
  XLS_ASSIGN_OR_RETURN(Node * n, function_base_->GetNode(patch_name));
  return n;
}

absl::Status PatchIr::ApplyPatch() {
  for (const xls_eco::EditPathProto& edit_path : sorted_edit_paths_) {
    XLS_RETURN_IF_ERROR(ApplyPath(edit_path));
  }
  XLS_RETURN_IF_ERROR(ApplyStateElementLayout());
  if (function_base_->IsFunction() && dummy_return_node_ != nullptr) {
    XLS_RETURN_IF_ERROR(RestoreReturnNode());
  }
  XLS_RETURN_IF_ERROR(ValidatePatch());
  return absl::OkStatus();
}

absl::Status PatchIr::ApplyPath(const xls_eco::EditPathProto& edit_path) {
  switch (edit_path.operation()) {
    case xls_eco::Operation::DELETE:
      XLS_RETURN_IF_ERROR(edit_path.has_node_edit_path()
                              ? ApplyDeletePath(edit_path.node_edit_path())
                              : ApplyDeletePath(edit_path.edge_edit_path()));
      break;
    case xls_eco::Operation::INSERT:
      XLS_RETURN_IF_ERROR(edit_path.has_node_edit_path()
                              ? ApplyInsertPath(edit_path.node_edit_path())
                              : ApplyInsertPath(edit_path.edge_edit_path()));
      break;
    case xls_eco::Operation::UPDATE:
      XLS_RETURN_IF_ERROR(edit_path.has_node_edit_path()
                              ? ApplyUpdatePath(edit_path.node_edit_path())
                              : ApplyUpdatePath(edit_path.edge_edit_path()));
      break;
    default:
      return absl::InvalidArgumentError("Invalid operation");
  }
  return absl::OkStatus();
}
absl::Status PatchIr::ApplyDeletePath(
    const xls_eco::NodeEditPathProto& node_delete) {
  if (node_delete.node().has_channel()) {
    return RemoveChannelFromProto(node_delete.node().channel());
  }
  if (node_delete.node().op() == "state_element") {
    // Reads and next_values of the element are deleted first (delete tier
    // ranks them earlier), so the element dies with no live references.
    XLS_ASSIGN_OR_RETURN(Proc * proc, package_->GetTopAsProc());
    XLS_ASSIGN_OR_RETURN(
        StateElement * se,
        proc->GetStateElementByName(
            node_delete.node().unique_args(1).state_element()));
    XLS_ASSIGN_OR_RETURN(int64_t index, proc->GetStateElementIndex(se));
    return proc->RemoveStateElement(index);
  }
  XLS_ASSIGN_OR_RETURN(Node * n,
                       function_base_->GetNode(node_delete.node().name()));
  if (function_base_->IsFunction() && function_base_->HasImplicitUse(n)) {
    XLS_RETURN_IF_ERROR(IsolateReturnNode());
  }
  // Drop any index remaps recorded against the node being deleted.
  for (auto it = edge_index_remap_.begin(); it != edge_index_remap_.end();) {
    if (it->first.first == n) {
      edge_index_remap_.erase(it++);
    } else {
      ++it;
    }
  }
  // Defensive cleanup for debug-node users. The direct IR graph represents
  // assert/trace nodes, so ordinary patches should delete their edges before
  // deleting this node.
  for (Node* user : n->users()) {
    if (user->op() != Op::kAssert && user->op() != Op::kTrace) {
      return absl::InternalError(absl::StrFormat(
          "Unexpected non-debug remaining user '%s' (op=%s) "
          "of node '%s' to be deleted",
          user->GetName(), OpToString(user->op()), n->GetName()));
    }
  }
  while (!n->users().empty()) {
    Node* user = *n->users().begin();
    for (int64_t i = 0; i < user->operand_count(); ++i) {
      if (user->operand(i) == n) {
        XLS_ASSIGN_OR_RETURN(Node * dummy, MakeDummyNode(n->GetType()));
        dummy_nodes_map_[user].push_back(dummy);
        XLS_RETURN_IF_ERROR(user->ReplaceOperandNumber(i, dummy, false));
      }
    }
  }
  XLS_RETURN_IF_ERROR(function_base_->RemoveNode(n));
  XLS_RETURN_IF_ERROR(CleanupDummyNodes(n));
  return absl::OkStatus();
}
absl::Status PatchIr::ApplyDeletePath(
    const xls_eco::EdgeEditPathProto& edge_delete) {
  if (edge_delete.edge().channel_binding()) {
    // Send/Receive bind to channels by name; binding edges are diagnostic only.
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(Node * from_node,
                       function_base_->GetNode(edge_delete.edge().from_node()));
  XLS_ASSIGN_OR_RETURN(Node * to_node,
                       function_base_->GetNode(edge_delete.edge().to_node()));
  XLS_ASSIGN_OR_RETURN(Node * dummy_node, MakeDummyNode(from_node->GetType()));
  dummy_nodes_map_[to_node].push_back(dummy_node);
  XLS_RETURN_IF_ERROR(to_node->ReplaceOperandNumber(edge_delete.edge().index(),
                                                    dummy_node, false));
  return absl::OkStatus();
}
absl::Status PatchIr::ApplyInsertPath(
    const xls_eco::NodeEditPathProto& node_insert) {
  const xls_eco::NodeProto& patch_node = node_insert.node();
  if (patch_node.has_channel()) {
    return CreateChannelFromProto(patch_node.channel());
  }
  if (patch_node.op() == "state_element") {
    // Appended read-less; reads bind later (insert tier ranks them after) and
    // ApplyStateElementLayout settles the final position.
    XLS_ASSIGN_OR_RETURN(Proc * proc, package_->GetTopAsProc());
    const std::string& element_name =
        patch_node.unique_args(1).state_element();
    XLS_ASSIGN_OR_RETURN(Value init,
                         Value::FromProto(patch_node.unique_args(2).init()));
    XLS_ASSIGN_OR_RETURN(
        StateElement * se,
        proc->AppendUnreadStateElement(
            element_name, init,
            patch_node.unique_args(3).non_synthesizable()));
    // Deleted names are released by Proc::RemoveStateElement, so the name
    // must come back verbatim; a uniquified one would diverge from after IR.
    XLS_RET_CHECK_EQ(se->name(), element_name);
    return absl::OkStatus();
  }
  const auto& op_map = PatchToIrOpMap();
  if (op_map.find(patch_node.op()) == op_map.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported operation: ", patch_node.op()));
  }
  const Op op = op_map.at(patch_node.op());
  std::vector<Type*> operand_types = {};
  if (patch_node.operand_data_types_size() > 0) {
    std::transform(patch_node.operand_data_types().begin(),
                   patch_node.operand_data_types().end(),
                   std::back_inserter(operand_types),
                   [&](const TypeProto& type) {
                     return package_->GetTypeFromProto(type).value();
                   });
  }
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> dummy_operands,
                       MakeDummyNodes(absl::MakeSpan(operand_types)));
  // Captured before any case pops a predicate/default off the vector, so
  // every dummy is tracked for cleanup.
  const absl::Span<Node*> all_dummy_operands = absl::MakeSpan(dummy_operands);
  Node* n = nullptr;
  switch (op) {
    case (Op::kLiteral): {
      XLS_ASSIGN_OR_RETURN(Value v,
                           Value::FromProto(patch_node.unique_args(0).value()));
      XLS_ASSIGN_OR_RETURN(n,
                           function_base_->MakeNode<Literal>(SourceInfo(), v));
      break;
    }
    case (Op::kSignExt): {
      XLS_ASSIGN_OR_RETURN(n,
                           function_base_->MakeNode<ExtendOp>(
                               SourceInfo(), dummy_operands[0],
                               patch_node.unique_args(0).new_bit_count(), op));
      break;
    }
    case (Op::kBitSlice): {
      XLS_ASSIGN_OR_RETURN(int64_t width,
                           GetProtoBitCount(patch_node.data_type()));
      XLS_ASSIGN_OR_RETURN(n, function_base_->MakeNode<BitSlice>(
                                  SourceInfo(), dummy_operands[0],
                                  patch_node.unique_args(0).start(), width));
      break;
    }
    case (Op::kTuple): {
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<Tuple>(SourceInfo(),
                                             absl::MakeSpan(dummy_operands)));
      break;
    }
    case (Op::kTupleIndex): {
      XLS_ASSIGN_OR_RETURN(n, function_base_->MakeNode<TupleIndex>(
                                  SourceInfo(), dummy_operands[0],
                                  patch_node.unique_args(0).index()));
      break;
    }
    case (Op::kArrayIndex): {
      XLS_ASSIGN_OR_RETURN(
          n, patch_node.unique_args_size() > 0 &&
                     patch_node.unique_args(0).has_assumed_in_bounds()
                 ? function_base_->MakeNode<ArrayIndex>(
                       SourceInfo(), dummy_operands[0],
                       absl::MakeConstSpan(dummy_operands).subspan(1),
                       patch_node.unique_args(0).assumed_in_bounds())
                 : function_base_->MakeNode<ArrayIndex>(
                       SourceInfo(), dummy_operands[0],
                       absl::MakeConstSpan(dummy_operands).subspan(1)));
      break;
    }

    case (Op::kArrayUpdate): {
      bool assumed_in_bounds = false;
      if (patch_node.unique_args_size() > 0 &&
          patch_node.unique_args(0).has_assumed_in_bounds()) {
        assumed_in_bounds = patch_node.unique_args(0).assumed_in_bounds();
      }
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<ArrayUpdate>(
                 SourceInfo(), dummy_operands[0], dummy_operands[1],
                 absl::MakeConstSpan(dummy_operands).subspan(2),
                 assumed_in_bounds));
      break;
    }

    case (Op::kAfterAll): {
      XLS_ASSIGN_OR_RETURN(n,
                           function_base_->MakeNode<AfterAll>(
                               SourceInfo(), absl::MakeSpan(dummy_operands)));
      break;
    }

    case (Op::kArray): {
      // The node's data type is the array type; elements are the operands.
      XLS_ASSIGN_OR_RETURN(Type * array_type,
                           package_->GetTypeFromProto(patch_node.data_type()));
      XLS_ASSIGN_OR_RETURN(n, function_base_->MakeNode<Array>(
                                  SourceInfo(), absl::MakeSpan(dummy_operands),
                                  array_type->AsArrayOrDie()->element_type()));
      break;
    }

    case (Op::kSMul):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kUMul): {
      XLS_ASSIGN_OR_RETURN(int64_t width,
                           GetProtoBitCount(patch_node.data_type()));
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<ArithOp>(SourceInfo(), dummy_operands[0],
                                               dummy_operands[1], width, op));
      break;
    }
    case (Op::kSLe):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kSGe):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kSLt):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kSGt):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kNe):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kUGt):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kULt):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kEq): {
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<CompareOp>(
                 SourceInfo(), dummy_operands[0], dummy_operands[1], op));
      break;
    }
    case (Op::kAdd):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kSub):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kUDiv):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kSDiv):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kULe):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kShrl):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kShll): {
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<BinOp>(SourceInfo(), dummy_operands[0],
                                             dummy_operands[1], op));
      break;
    }
    case (Op::kConcat): {
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<Concat>(SourceInfo(),
                                              absl::MakeSpan(dummy_operands)));
      break;
    }
    case (Op::kOneHot): {
      LsbOrMsb priority = patch_node.unique_args(0).lsb_prio() == true
                              ? LsbOrMsb::kLsb
                              : LsbOrMsb::kMsb;
      XLS_ASSIGN_OR_RETURN(n, function_base_->MakeNode<OneHot>(
                                  SourceInfo(), dummy_operands[0], priority));
      break;
    }
    case (Op::kOrReduce):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kAndReduce): {
      XLS_ASSIGN_OR_RETURN(n, function_base_->MakeNode<BitwiseReductionOp>(
                                  SourceInfo(), dummy_operands[0], op));
      break;
    }
    case (Op::kNeg):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kNot): {
      XLS_ASSIGN_OR_RETURN(n, function_base_->MakeNode<UnOp>(
                                  SourceInfo(), dummy_operands[0], op));
      break;
    }
    case (Op::kOr):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kNor):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kNand):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kXor):
      ABSL_FALLTHROUGH_INTENDED;
    case (Op::kAnd): {
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<NaryOp>(
                 SourceInfo(), absl::MakeSpan(dummy_operands), op));
      break;
    }
    case (Op::kStateRead): {
      // The read only binds an existing element (created or preserved by its
      // state-element node); like next_value, unique_args(0) names it.
      std::optional<Node*> predicate;
      if (patch_node.operand_data_types_size() > 0) {
        predicate = dummy_operands.back();
        dummy_operands.pop_back();
      }
      XLS_ASSIGN_OR_RETURN(Proc * proc, package_->GetTopAsProc());
      XLS_ASSIGN_OR_RETURN(
          StateElement * se,
          proc->GetStateElementByName(
              patch_node.unique_args(0).state_element()));
      XLS_ASSIGN_OR_RETURN(StateRead * read, proc->AddStateRead(se, predicate));
      n = read;
      break;
    }
    case (Op::kPrioritySel): {
      Node* default_value = dummy_operands.back();
      dummy_operands.pop_back();
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<PrioritySelect>(
                 SourceInfo(), dummy_operands[0],
                 absl::MakeSpan(dummy_operands).subspan(1), default_value));
      break;
    }
    case (Op::kSel): {
      std::optional<Node*> default_value;
      if (patch_node.unique_args(0).has_default_value()) {
        default_value = dummy_operands.back();
        dummy_operands.pop_back();
      }
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<Select>(
                 SourceInfo(), dummy_operands[0],
                 absl::MakeSpan(dummy_operands).subspan(1), default_value));
      break;
    }
    case (Op::kOneHotSel): {
      std::vector<Node*> cases;
      for (auto it = dummy_operands.begin() + 1; it != dummy_operands.end();
           ++it) {
        cases.push_back(*it);
      }
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<OneHotSelect>(
                 SourceInfo(), dummy_operands[0], absl::MakeSpan(cases)));
      break;
    }
    case (Op::kParam): {
      XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                           GetProtoBitCount(patch_node.data_type()));
      XLS_ASSIGN_OR_RETURN(
          n,
          function_base_->MakeNodeWithName<Param>(
              SourceInfo(), function_base_->package()->GetBitsType(bit_count),
              patch_node.name()));
      break;
    }
    // The named channel is (re)created by its channel node, sequenced first.
    case (Op::kSend): {
      std::optional<Node*> predicate;
      if (patch_node.operand_data_types_size() > 2) {
        predicate = dummy_operands.back();
        dummy_operands.pop_back();
      }
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<Send>(
                 SourceInfo(), dummy_operands[0], dummy_operands[1], predicate,
                 patch_node.unique_args(0).channel()));
      break;
    }
    case (Op::kReceive): {
      std::optional<Node*> predicate;
      if (patch_node.operand_data_types_size() > 1) {
        predicate = dummy_operands.back();
        dummy_operands.pop_back();
      }
      // Get the payload type from the Receive node's tuple return type
      // Receive returns a tuple (token, payload), so we need element 1
      XLS_ASSIGN_OR_RETURN(Type * node_type,
                           package_->GetTypeFromProto(patch_node.data_type()));
      TupleType* tuple_type = node_type->AsTupleOrDie();
      Type* payload_type = tuple_type->element_type(1);
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<Receive>(
                 SourceInfo(), dummy_operands[0], predicate,
                 patch_node.unique_args(0).channel(),
                 patch_node.unique_args(1).blocking(), payload_type));
      break;
    }
    case (Op::kNext): {
      XLS_ASSIGN_OR_RETURN(Proc * proc, package_->GetTopAsProc());
      // Next binds its element as metadata, not an operand; the operands are
      // the value and an optional predicate.
      XLS_RET_CHECK_GT(patch_node.unique_args_size(), 0)
          << "next_value insert is missing its state_element arg: "
          << patch_node.name();
      absl::StatusOr<StateElement*> state_element_or =
          proc->GetStateElementByName(
              patch_node.unique_args(0).state_element());
      if (!state_element_or.ok()) {
        std::string live;
        for (StateElement* se : proc->StateElements()) {
          absl::StrAppend(&live, se->name(), " ");
        }
        return absl::NotFoundError(absl::StrCat(
            "next_value insert ", patch_node.name(), ": no state element ",
            patch_node.unique_args(0).state_element(), "; live: ", live));
      }
      StateElement* state_element = *state_element_or;
      std::optional<Node*> predicate;
      if (patch_node.operand_data_types_size() > 1) {
        predicate = dummy_operands.back();
      }
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<Next>(SourceInfo(), state_element,
                                            dummy_operands[0], predicate,
                                            /*label=*/std::nullopt));
      break;
    }
    case (Op::kAssert): {
      const std::string message =
          patch_node.unique_args_size() > 0 &&
                  patch_node.unique_args(0).has_message()
              ? patch_node.unique_args(0).message()
              : "";
      std::optional<std::string> label;
      if (patch_node.unique_args_size() > 1 &&
          patch_node.unique_args(1).has_label()) {
        label = patch_node.unique_args(1).label();
      }
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<Assert>(SourceInfo(), dummy_operands[0],
                                              dummy_operands[1], message, label,
                                              std::nullopt));
      break;
    }
    case (Op::kTrace): {
      const std::string format_string =
          patch_node.unique_args_size() > 0 &&
                  patch_node.unique_args(0).has_format()
              ? patch_node.unique_args(0).format()
              : "";
      XLS_ASSIGN_OR_RETURN(std::vector<FormatStep> format,
                           ParseFormatString(format_string));
      int64_t verbosity = 0;
      if (patch_node.unique_args_size() > 1 &&
          patch_node.unique_args(1).has_verbosity()) {
        verbosity = patch_node.unique_args(1).verbosity();
      }
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<Trace>(
                 SourceInfo(), dummy_operands[0], dummy_operands[1],
                 absl::MakeConstSpan(dummy_operands).subspan(2),
                 absl::MakeConstSpan(format), verbosity));
      break;
    }
    case (Op::kGate): {
      XLS_ASSIGN_OR_RETURN(
          n, function_base_->MakeNode<Gate>(SourceInfo(), dummy_operands[0],
                                            dummy_operands[1]));
      break;
    }
    default:
      return absl::InvalidArgumentError("Invalid operation");
  }
  UpdateNodeMaps(n, all_dummy_operands, patch_node.name());
  inserted_node_names_.insert(n->GetName());
  return absl::OkStatus();
}
absl::Status PatchIr::ApplyInsertPath(
    const xls_eco::EdgeEditPathProto& edge_insert) {
  const xls_eco::EdgeProto& patch_edge = edge_insert.edge();
  if (patch_edge.channel_binding()) {
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(Node * from_node,
                       ResolveNodeByPatchName(patch_edge.from_node()));
  XLS_ASSIGN_OR_RETURN(Node * to_node,
                       ResolveNodeByPatchName(patch_edge.to_node()));
  const bool is_commutative = OpIsCommutative(to_node->op());
  int64_t position = static_cast<int64_t>(patch_edge.index());
  if (!is_commutative) {
    auto map_it = edge_index_remap_.find(
        {to_node, static_cast<int64_t>(patch_edge.index())});
    if (map_it != edge_index_remap_.end()) {
      position = map_it->second;
    }
  }

  if (is_commutative) {
    // Order is not part of a commutative node's identity, so the patch index
    // is advisory: fill any dummy slot.
    bool found_dummy = false;
    for (int64_t i = 0; i < to_node->operand_count(); ++i) {
      Node* existing_operand = to_node->operand(i);
      auto it_dummy =
          std::find(dummy_nodes_map_[to_node].begin(),
                    dummy_nodes_map_[to_node].end(), existing_operand);
      if (it_dummy != dummy_nodes_map_[to_node].end()) {
        position = i;
        found_dummy = true;
        break;
      }
    }
    if (!found_dummy) {
      return absl::InvalidArgumentError(absl::StrCat(
          "No dummy operand slot on commutative node ", to_node->GetName(),
          " for insertion at index ", patch_edge.index(),
          "; operand_count=", to_node->operand_count()));
    }
  } else {
    while (position < to_node->operand_count()) {
      Node* existing_operand = to_node->operands()[position];
      auto it_dummy =
          std::find(dummy_nodes_map_[to_node].begin(),
                    dummy_nodes_map_[to_node].end(), existing_operand);
      if (it_dummy == dummy_nodes_map_[to_node].end()) {
        position++;
      } else {
        break;
      }
    }
    if (position >= to_node->operand_count()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Insert edge target index out of range for node ", to_node->GetName(),
          ": requested ", patch_edge.index(), " (remapped to ", position,
          "), operand_count=", to_node->operand_count()));
    }
  }
  Node* node_to_remove = nullptr;
  if (position < to_node->operand_count()) {
    node_to_remove = to_node->operands()[position];
  }

  XLS_RETURN_IF_ERROR(
      to_node->ReplaceOperandNumber(position, from_node, false));

  if (node_to_remove != nullptr) {
    XLS_RETURN_IF_ERROR(function_base_->RemoveNode(node_to_remove));
    auto it = std::remove(dummy_nodes_map_[to_node].begin(),
                          dummy_nodes_map_[to_node].end(), node_to_remove);
    dummy_nodes_map_[to_node].erase(it, dummy_nodes_map_[to_node].end());
  }
  return absl::OkStatus();
}

absl::Status PatchIr::ApplyUpdatePath(
    const xls_eco::NodeEditPathProto& node_update) {
  if (node_update.node().has_channel()) {
    // Unchanged channel (a real change surfaces as channel delete+insert).
    return absl::OkStatus();
  }
  if (node_update.node().op() == "state_element") {
    // Surviving element; ApplyStateElementLayout settles its position.
    return absl::OkStatus();
  }
  patch_to_ir_node_map_[node_update.updated_node().name()] =
      node_update.node().name();
  XLS_ASSIGN_OR_RETURN(Node * n,
                       function_base_->GetNode(node_update.node().name()));
  if (function_base_->IsFunction() && function_base_->HasImplicitUse(n)) {
    XLS_RETURN_IF_ERROR(IsolateReturnNode());
  }
  return absl::OkStatus();
}
absl::Status PatchIr::ApplyUpdatePath(
    const xls_eco::EdgeEditPathProto& edge_update) {
  if (edge_update.edge().channel_binding()) {
    return absl::OkStatus();
  }
  if (edge_update.edge().index() != edge_update.updated_edge().index()) {
    XLS_ASSIGN_OR_RETURN(
        Node * n, ResolveNodeByPatchName(edge_update.updated_edge().to_node()));
    if (OpIsCommutative(n->op())) {
      // Ignore index reorders on commutative ops.
      return absl::OkStatus();
    }
    edge_index_remap_[{n, static_cast<int64_t>(edge_update.edge().index())}] =
        static_cast<int64_t>(edge_update.updated_edge().index());
  }
  return absl::OkStatus();
}
absl::Status PatchIr::IsolateReturnNode() {
  for (Node* n : function_base_->nodes()) {
    if (function_base_->HasImplicitUse(n)) {
      XLS_ASSIGN_OR_RETURN(dummy_return_node_,
                           function_base_->MakeNode<Literal>(
                               SourceInfo(), Value(ZeroOfType(n->GetType()))));
      XLS_ASSIGN_OR_RETURN(bool changed,
                           n->ReplaceImplicitUsesWith(dummy_return_node_));
      if (!changed) {
        return absl::InternalError("Failed to replace implicit uses");
      }
      return absl::OkStatus();
    }
  }
  return absl::InternalError("No return node found");
}

absl::Status PatchIr::RestoreReturnNode() {
  XLS_ASSIGN_OR_RETURN(Node * return_node,
                       ResolveNodeByPatchName(patch_.return_node().name()));
  XLS_ASSIGN_OR_RETURN(
      bool _, dummy_return_node_->ReplaceImplicitUsesWith(return_node));
  XLS_RETURN_IF_ERROR(function_base_->RemoveNode(dummy_return_node_));
  return absl::OkStatus();
}
absl::Status PatchIr::ValidatePatch() {
  for (const auto& [node, dummies] : dummy_nodes_map_) {
    if (dummies.empty()) {
      continue;
    }
    std::string names;
    for (Node* dummy : dummies) {
      absl::StrAppend(&names, dummy->GetName(), " ");
    }
    LOG(WARNING) << "Dummy operands left on " << node->GetName() << ": "
                 << names;
  }
  return absl::OkStatus();
}

absl::Status PatchIr::CreateChannelFromProto(
    const xls_eco::ChannelProto& channel) {
  XLS_ASSIGN_OR_RETURN(Type * type,
                       package_->GetTypeFromProto(channel.data_type()));
  XLS_ASSIGN_OR_RETURN(ChannelKind kind, StringToChannelKind(channel.kind()));
  XLS_ASSIGN_OR_RETURN(Proc * proc, package_->GetTopAsProc());
  const ChannelDirection direction = channel.direction() == "send"
                                         ? ChannelDirection::kSend
                                         : ChannelDirection::kReceive;
  XLS_ASSIGN_OR_RETURN(FlowControl flow_control,
                       StringToFlowControl(channel.flow_control()));
  std::optional<ChannelStrictness> strictness;
  if (!channel.strictness().empty()) {
    XLS_ASSIGN_OR_RETURN(strictness,
                         ChannelStrictnessFromString(channel.strictness()));
  }
  XLS_ASSIGN_OR_RETURN(
      ChannelInterface * interface,
      proc->AddInterfaceChannel(channel.name(), direction, type, kind,
                                flow_control, strictness));
  if (!channel.flop_kind().empty()) {
    XLS_ASSIGN_OR_RETURN(FlopKind flop_kind,
                         StringToFlopKind(channel.flop_kind()));
    interface->SetFlopKind(flop_kind);
  }
  return absl::OkStatus();
}

absl::Status PatchIr::RemoveChannelFromProto(
    const xls_eco::ChannelProto& channel) {
  XLS_ASSIGN_OR_RETURN(Proc * proc, package_->GetTopAsProc());
  const ChannelDirection direction = channel.direction() == "send"
                                         ? ChannelDirection::kSend
                                         : ChannelDirection::kReceive;
  XLS_ASSIGN_OR_RETURN(ChannelInterface * interface,
                       proc->GetChannelInterface(channel.name(), direction));
  return proc->RemoveChannelInterface(interface);
}

absl::Status PatchIr::ApplyStateElementLayout() {
  // Element INSERT and UPDATE paths carry indices in the revised layout, but
  // coverage is partial: the patch only describes changed nodes (MCS cuts
  // unchanged ones before GED). Place covered elements at their revised
  // indices; uncovered ones keep their relative order in the remaining slots.
  // Order is not element identity, so this is fidelity, not correctness.
  absl::flat_hash_map<std::string, int64_t> target_index;
  for (const xls_eco::EditPathProto& edit_path : patch_.edit_paths()) {
    if (!edit_path.has_node_edit_path()) {
      continue;
    }
    const xls_eco::NodeProto* node = nullptr;
    if (edit_path.operation() == xls_eco::INSERT &&
        edit_path.node_edit_path().node().op() == "state_element") {
      node = &edit_path.node_edit_path().node();
    } else if (edit_path.operation() == xls_eco::UPDATE &&
               edit_path.node_edit_path().updated_node().op() ==
                   "state_element") {
      node = &edit_path.node_edit_path().updated_node();
    }
    if (node != nullptr) {
      target_index[node->unique_args(1).state_element()] =
          static_cast<int64_t>(node->unique_args(0).index());
    }
  }
  if (target_index.empty()) {
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(Proc * proc, package_->GetTopAsProc());
  const int64_t element_count = proc->GetStateElementCount();
  std::vector<std::string> desired(element_count);
  std::vector<std::string> uncovered;
  for (StateElement* element : proc->StateElements()) {
    auto it = target_index.find(element->name());
    if (it == target_index.end()) {
      uncovered.push_back(element->name());
      continue;
    }
    XLS_RET_CHECK_LT(it->second, element_count);
    XLS_RET_CHECK(desired[it->second].empty())
        << "duplicate target state index " << it->second;
    desired[it->second] = element->name();
  }
  auto uncovered_it = uncovered.begin();
  for (std::string& slot : desired) {
    if (slot.empty()) {
      XLS_RET_CHECK(uncovered_it != uncovered.end())
          << "patch names a state element the live proc does not have";
      slot = *uncovered_it++;
    }
  }
  XLS_RET_CHECK(uncovered_it == uncovered.end());
  for (int64_t to = 0; to < element_count; ++to) {
    XLS_ASSIGN_OR_RETURN(StateElement * se,
                         proc->GetStateElementByName(desired[to]));
    XLS_ASSIGN_OR_RETURN(int64_t from, proc->GetStateElementIndex(se));
    if (from != to) {
      XLS_RETURN_IF_ERROR(proc->MoveStateElement(from, to));
    }
  }
  return absl::OkStatus();
}

absl::Status PatchIr::PatchSchedule(const PipelineSchedule& schedule) {
  XLS_ASSIGN_OR_RETURN(
      SchedulingOptionsFlagsProto scheduling_options_flags_proto,
      GetSchedulingOptionsFlagsProto());
  XLS_ASSIGN_OR_RETURN(
      SchedulingOptions scheduling_options,
      SetUpSchedulingOptions(scheduling_options_flags_proto, package_));
  SchedulingOptions tmp_scheduling_options = scheduling_options;
  XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                       SetUpDelayEstimator(scheduling_options_flags_proto));
  decltype(package_->GetNodeCount()) constraint_count = 0;
  for (const auto& [node, cycle] : schedule.GetCycleMap()) {
    if (inserted_node_names_.contains(node->GetName())) {
      VLOG(1) << "Skipping constraint for newly inserted node "
              << node->GetName();
      continue;
    }
    if (std::find(function_base_->nodes().begin(),
                  function_base_->nodes().end(),
                  node) == function_base_->nodes().end()) {
      VLOG(1) << "Skipping constraint for node not in IR: "
              << node->GetName();
      continue;
    }
    if (node->op() == Op::kLiteral) {
      VLOG(1) << "Skipping constraint for literal " << node->GetName();
      continue;
    }
    tmp_scheduling_options.add_constraint(NodeInCycleConstraint(node, cycle));
    // check if schedule is feasible; if not, then we need remove the
    // constraint
    if (!RunPipelineSchedule(function_base_, *delay_estimator,
                             tmp_scheduling_options)
             .ok()) {
      tmp_scheduling_options.clear_constraints();
      VLOG(1) << "Skipping constraint for " << node->GetName()
              << ": schedule infeasible";
    } else {
      scheduling_options = tmp_scheduling_options;
      constraint_count++;
    }
    tmp_scheduling_options = scheduling_options;
    XLS_ASSIGN_OR_RETURN(schedule_,
                         RunPipelineSchedule(function_base_, *delay_estimator,
                                             scheduling_options));
  }
  XLS_RETURN_IF_ERROR(schedule_->Verify());
  VLOG(1) << "Schedule constraints kept: " << constraint_count << " of "
          << package_->GetNodeCount() << " nodes";
  return absl::OkStatus();
}

absl::StatusOr<PipelineSchedule> PatchIr::GetPatchedSchedule() {
  if (schedule_.has_value()) {
    return schedule_.value();
  }
  return absl::InternalError("No schedule found");
}

absl::Status PatchIr::ExportIr(const std::string& export_path) const {
  std::string ir_data = package_->DumpIr();
  std::ofstream out_file(export_path);
  if (out_file.is_open()) {
    out_file << ir_data;
    out_file.close();
    return absl::OkStatus();
  }
  return absl::InternalError("Failed to open file: " + export_path);
}
absl::Status PatchIr::ExportScheduleProto() {
  XLS_ASSIGN_OR_RETURN(
      SchedulingOptionsFlagsProto scheduling_options_flags_proto,
      GetSchedulingOptionsFlagsProto());
  XLS_ASSIGN_OR_RETURN(DelayEstimator * delay_estimator,
                       SetUpDelayEstimator(scheduling_options_flags_proto));
  XLS_ASSIGN_OR_RETURN(PipelineScheduleProto schedule_proto,
                       schedule_->ToProto(*delay_estimator));
  XLS_RETURN_IF_ERROR(SetTextProtoFile(
      absl::GetFlag(FLAGS_output_schedule_path), schedule_proto));
  return absl::OkStatus();
}
PatchIr::EditPathPriority PatchIr::PriorityOf(
    const xls_eco::EditPathProto& edit_path) const {
  return EditPathPriorityMap().at(
      {edit_path.has_node_edit_path(), edit_path.operation()});
}

bool PatchIr::CompareEditPaths(const xls_eco::EditPathProto& lhs,
                               const xls_eco::EditPathProto& rhs) {
  EditPathPriority lhs_priority = PriorityOf(lhs);
  EditPathPriority rhs_priority = PriorityOf(rhs);
  if (lhs_priority != rhs_priority) {
    return lhs_priority < rhs_priority;
  }
  if (lhs_priority == EditPathPriority::kNodeDelete) {
    // Ordinary nodes first, then state elements, then channels: removing an
    // element dereferences it from every read/next_value still being removed,
    // so those node deletes must run before the element dies.
    auto delete_rank = [](const xls_eco::EditPathProto& path) {
      const xls_eco::NodeProto& node = path.node_edit_path().node();
      if (node.has_channel()) {
        return 2;
      }
      return node.op() == "state_element" ? 1 : 0;
    };
    const int lhs_rank = delete_rank(lhs);
    const int rhs_rank = delete_rank(rhs);
    if (lhs_rank != rhs_rank) {
      return lhs_rank < rhs_rank;
    }
  }
  if (lhs_priority == EditPathPriority::kNodeInsert) {
    // Channels and state elements first: Send/Receive bind to channels and
    // state reads/next_values to elements, both by name.
    auto insert_rank = [](const xls_eco::EditPathProto& p) {
      const xls_eco::NodeProto& node = p.node_edit_path().node();
      if (node.has_channel()) {
        return 0;
      }
      return node.op() == "state_element" ? 1 : 2;
    };
    const auto lhs_rank = insert_rank(lhs);
    const auto rhs_rank = insert_rank(rhs);
    if (lhs_rank != rhs_rank) {
      return lhs_rank < rhs_rank;
    }
  }
  return lhs.id() < rhs.id();
}
}  // namespace xls
