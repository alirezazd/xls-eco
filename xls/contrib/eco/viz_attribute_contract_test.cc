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

// Pins the viz::AttributeVisitor output that ECO depends on.
//
// NodeCostAttributes::Hash() (graph.cc) folds the visitor's proto output
// wholesale into every node label, and ir_patch_gen.cc copies several of its
// fields into patch protos. Any upstream change to what the visitor emits
// therefore silently shifts all ECO node labels (invalidating stored hash
// comparisons) or changes patch contents. These tests fail loudly instead:
// when one breaks, decide deliberately whether the new/changed field belongs
// in the ECO label, then update the golden.
//
// Block-only ops (ports, registers, instantiations) are not exercised because
// ECO graphs are built from functions and procs; the field-set pin below still
// catches proto growth for those.

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/visualization/ir_viz/node_attribute_visitor.h"
#include "xls/visualization/ir_viz/visualization.pb.h"

namespace xls {
namespace {

using ::xls::proto_testing::EqualsProto;

class VizAttributeContractTest : public IrTestBase {};

viz::NodeAttributes AttributesFor(BValue v) {
  AttributeVisitor visitor;
  XLS_EXPECT_OK(v.node()->VisitSingleNode(&visitor));
  return visitor.attributes();
}

TEST_F(VizAttributeContractTest, FunctionOpAttributes) {
  std::unique_ptr<VerifiedPackage> p = CreatePackage();
  Function* callee;
  {
    FunctionBuilder cb("callee_fn", p.get());
    cb.Not(cb.Param("v", p->GetBitsType(8)));
    XLS_ASSERT_OK_AND_ASSIGN(callee, cb.Build());
  }
  Function* loop_body;
  {
    FunctionBuilder lb("loop_body_fn", p.get());
    BValue i = lb.Param("i", p->GetBitsType(32));
    BValue acc = lb.Param("acc", p->GetBitsType(32));
    lb.Add(i, acc);
    XLS_ASSERT_OK_AND_ASSIGN(loop_body, lb.Build());
  }

  FunctionBuilder b("attrs_fn", p.get());
  BValue x8 = b.Param("x8", p->GetBitsType(8));
  BValue x32 = b.Param("x32", p->GetBitsType(32));
  BValue arr = b.Param("arr", p->GetArrayType(4, p->GetBitsType(8)));
  BValue tok = b.AfterAll({});
  BValue pred = b.Eq(x8, x8);

  EXPECT_THAT(AttributesFor(b.Literal(Value(UBits(0xAB, 8)))),
              EqualsProto(R"pb(value: "0xab")pb"));
  EXPECT_THAT(AttributesFor(b.BitSlice(x32, /*start=*/2, /*width=*/4)),
              EqualsProto(R"pb(start: 2 width: 4)pb"));
  EXPECT_THAT(AttributesFor(b.DynamicBitSlice(x32, x8, /*width=*/4)),
              EqualsProto(R"pb(width: 4)pb"));
  EXPECT_THAT(AttributesFor(b.ArraySlice(arr, x8, /*width=*/2)),
              EqualsProto(R"pb(width: 2)pb"));
  EXPECT_THAT(AttributesFor(b.Decode(x8, /*width=*/16)),
              EqualsProto(R"pb(width: 16)pb"));
  EXPECT_THAT(AttributesFor(b.OneHot(x8, LsbOrMsb::kLsb)),
              EqualsProto(R"pb(lsb_prio: true)pb"));
  EXPECT_THAT(AttributesFor(b.OneHot(x8, LsbOrMsb::kMsb)),
              EqualsProto(R"pb(lsb_prio: false)pb"));
  EXPECT_THAT(AttributesFor(b.Select(pred, {x8, x8})),
              EqualsProto(R"pb(has_default: false)pb"));
  const std::vector<BValue> one_case = {x8};
  EXPECT_THAT(AttributesFor(b.Select(pred, one_case, /*default_value=*/x8)),
              EqualsProto(R"pb(has_default: true)pb"));
  EXPECT_THAT(AttributesFor(b.SignExtend(x8, 16)),
              EqualsProto(R"pb(new_bit_count: 16)pb"));
  EXPECT_THAT(AttributesFor(b.ZeroExtend(x8, 16)),
              EqualsProto(R"pb(new_bit_count: 16)pb"));
  EXPECT_THAT(AttributesFor(b.TupleIndex(b.Tuple({x8, x32}), 1)),
              EqualsProto(R"pb(index: 1)pb"));
  EXPECT_THAT(AttributesFor(b.MinDelay(tok, /*delay=*/3)),
              EqualsProto(R"pb(delay: 3)pb"));
  EXPECT_THAT(
      AttributesFor(b.Assert(tok, pred, /*message=*/"assert message",
                             /*label=*/"assert_label")),
      EqualsProto(R"pb(message_: "assert message" label: "assert_label")pb"));
  EXPECT_THAT(AttributesFor(b.Assert(tok, pred, /*message=*/"assert message")),
              EqualsProto(R"pb(message_: "assert message")pb"));
  EXPECT_THAT(AttributesFor(b.Cover(pred, /*label=*/"cover_label")),
              EqualsProto(R"pb(label: "cover_label")pb"));
  // The visitor serializes trace formats in *Verilog* syntax ({} becomes %d).
  // ECO must not treat this as the XLS format string; see trace_xls_format in
  // xls_ir_to_graph.cc and the kTrace case in ir_patch_gen.cc.
  EXPECT_THAT(AttributesFor(b.Trace(tok, pred, /*args=*/{x8}, "x is {}")),
              EqualsProto(R"pb(format: "x is %d" verbosity: 0)pb"));
  EXPECT_THAT(AttributesFor(b.Invoke({x8}, callee)),
              EqualsProto(R"pb(to_apply: "callee_fn")pb"));
  EXPECT_THAT(AttributesFor(b.Map(arr, callee)),
              EqualsProto(R"pb(to_apply: "callee_fn")pb"));
  EXPECT_THAT(AttributesFor(
                  b.CountedFor(x32, /*trip_count=*/4, /*stride=*/2, loop_body)),
              EqualsProto(R"pb(trip_count: 4 stride: 2)pb"));
  EXPECT_THAT(AttributesFor(b.DynamicCountedFor(x32, /*trip_count=*/x8,
                                                /*stride=*/x8, loop_body)),
              EqualsProto(R"pb(body: "loop_body_fn")pb"));
  // Ops without a dedicated handler must contribute nothing to the label.
  EXPECT_THAT(AttributesFor(b.Add(x32, x32)),
              EqualsProto(viz::NodeAttributes()));
  XLS_ASSERT_OK(b.Build().status());
}

TEST_F(VizAttributeContractTest, ProcOpAttributes) {
  std::unique_ptr<VerifiedPackage> p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * in_ch,
      p->CreateStreamingChannel("in_ch", ChannelOps::kReceiveOnly,
                                p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * out_ch,
      p->CreateStreamingChannel("out_ch", ChannelOps::kSendOnly,
                                p->GetBitsType(32)));

  ProcBuilder pb("attrs_proc", p.get());
  BValue tok = pb.AfterAll({});
  BValue recv = pb.Receive(in_ch, tok);
  BValue recv_nb = pb.ReceiveNonBlocking(in_ch, tok);
  BValue send = pb.Send(out_ch, tok, pb.TupleIndex(recv, 1));

  EXPECT_THAT(AttributesFor(recv), EqualsProto(R"pb(channel: "in_ch"
                                                    blocking: true)pb"));
  EXPECT_THAT(AttributesFor(recv_nb), EqualsProto(R"pb(channel: "in_ch"
                                                       blocking: false)pb"));
  EXPECT_THAT(AttributesFor(send), EqualsProto(R"pb(channel: "out_ch")pb"));
  XLS_ASSERT_OK(pb.Build({}).status());
}

// Catches upstream growth of the proto itself, including fields for ops the
// tests above do not exercise. On failure, decide whether ECO's labels should
// depend on the new field, then update this list.
TEST_F(VizAttributeContractTest, NodeAttributesFieldSetIsPinned) {
  std::vector<std::string> fields;
  const auto* descriptor = viz::NodeAttributes::descriptor();
  fields.reserve(descriptor->field_count());
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const auto* field = descriptor->field(i);
    fields.push_back(absl::StrCat(field->number(), ":", field->name()));
  }
  EXPECT_THAT(
      fields,
      ::testing::UnorderedElementsAre(
          "1:value", "2:start", "3:width", "4:index", "5:delay_ps",
          "6:known_bits", "7:on_critical_path", "8:cycle",
          "9:state_param_index", "10:initial_value", "11:area_um",
          "12:lsb_prio", "13:new_bit_count", "14:channel", "15:blocking",
          "16:to_apply", "17:trip_count", "18:stride", "19:body",
          "20:has_default", "21:delay", "22:message_", "23:label", "24:format",
          "25:verbosity", "26:name", "27:register_", "28:has_load_enable",
          "29:has_reset", "30:instantiation", "31:port_name", "32:ranges",
          "33:critical_path_delay_ps", "34:leading_signs",
          "35:dead_after_synthesis"));
}

}  // namespace
}  // namespace xls
