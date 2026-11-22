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

#include <memory>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/eco/match.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace xls {
namespace {

constexpr std::string_view kReference = R"(package p

top block b(clk: clock, x: bits[8], out: bits[8]) {
  reg r(bits[8])
  x: bits[8] = input_port(name=x, id=1)
  not.2: bits[8] = not(x, id=2)
  r_write: () = register_write(not.2, register=r, id=3)
  r_read: bits[8] = register_read(register=r, id=4)
  neg.5: bits[8] = neg(r_read, id=5)
  out: () = output_port(neg.5, name=out, id=6)
}
)";

// The same block as codegen renumbers it.
constexpr std::string_view kRenumbered = R"(package p

top block b(clk: clock, x: bits[8], out: bits[8]) {
  reg r(bits[8])
  x: bits[8] = input_port(name=x, id=101)
  not.102: bits[8] = not(x, id=102)
  r_write: () = register_write(not.102, register=r, id=103)
  r_read: bits[8] = register_read(register=r, id=104)
  neg.105: bits[8] = neg(r_read, id=105)
  out: () = output_port(neg.105, name=out, id=106)
}
)";

// Renumbered, with an ECO in front of the register.
constexpr std::string_view kPatched = R"(package p

top block b(clk: clock, x: bits[8], out: bits[8]) {
  reg r(bits[8])
  x: bits[8] = input_port(name=x, id=101)
  not.102: bits[8] = not(x, id=102)
  not.107: bits[8] = not(not.102, id=107)
  r_write: () = register_write(not.107, register=r, id=103)
  r_read: bits[8] = register_read(register=r, id=104)
  neg.105: bits[8] = neg(r_read, id=105)
  out: () = output_port(neg.105, name=out, id=106)
}
)";

struct Parsed {
  std::unique_ptr<Package> package;
  Block* block;
};

absl::StatusOr<Parsed> Parse(std::string_view ir) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir));
  XLS_ASSIGN_OR_RETURN(Block * block, package->GetTopAsBlock());
  return Parsed{std::move(package), block};
}

TEST(BlockAlignTest, RenumberedBlockGetsReferenceIds) {
  XLS_ASSERT_OK_AND_ASSIGN(Parsed reference, Parse(kReference));
  XLS_ASSERT_OK_AND_ASSIGN(Parsed renumbered, Parse(kRenumbered));
  XLS_ASSERT_OK_AND_ASSIGN(
      Alignment alignment,
      Align(renumbered.block, reference.block, MatchOptions()));
  EXPECT_EQ(alignment.matched.size(), 6);
  EXPECT_EQ(renumbered.package->DumpIr(), reference.package->DumpIr());
  XLS_ASSERT_OK(VerifyPackage(renumbered.package.get()));
}

TEST(BlockAlignTest, InsertedNodeKeepsFreshIdOthersKeepReferenceIds) {
  XLS_ASSERT_OK_AND_ASSIGN(Parsed reference, Parse(kReference));
  XLS_ASSERT_OK_AND_ASSIGN(Parsed patched, Parse(kPatched));
  XLS_ASSERT_OK_AND_ASSIGN(
      Alignment alignment,
      Align(patched.block, reference.block, MatchOptions()));
  EXPECT_EQ(alignment.matched.size(), 6);
  EXPECT_EQ(alignment.nodes, 7);
  for (std::string_view name : {"x", "not.2", "r_write", "r_read", "neg.5"}) {
    XLS_ASSERT_OK_AND_ASSIGN(Node * node, patched.block->GetNode(name));
    XLS_ASSERT_OK_AND_ASSIGN(Node * ref, reference.block->GetNode(name));
    EXPECT_EQ(node->id(), ref->id()) << name;
  }
  XLS_ASSERT_OK_AND_ASSIGN(Node * r_write, patched.block->GetNode("r_write"));
  EXPECT_GT(r_write->operand(0)->id(), 106);
  XLS_ASSERT_OK(VerifyPackage(patched.package.get()));
}

constexpr std::string_view kAssertReference = R"(package p

top block b(clk: clock, x: bits[1], out: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  tok: token = after_all(id=2)
  assert.3: token = assert(tok, x, message="Selector concat.7 was expected to be one-hot, and is not.", label="one_hot_A", id=3)
  out: () = output_port(x, name=out, id=4)
}
)";

constexpr std::string_view kAssertRenumbered = R"(package p

top block b(clk: clock, x: bits[1], out: bits[1]) {
  x: bits[1] = input_port(name=x, id=101)
  tok: token = after_all(id=102)
  assert.103: token = assert(tok, x, message="Selector concat.9 was expected to be one-hot, and is not.", label="one_hot_A", id=103)
  out: () = output_port(x, name=out, id=104)
}
)";

TEST(BlockAlignTest, AssertionMessageNamingANodeIsAdopted) {
  XLS_ASSERT_OK_AND_ASSIGN(Parsed reference, Parse(kAssertReference));
  XLS_ASSERT_OK_AND_ASSIGN(Parsed renumbered, Parse(kAssertRenumbered));
  XLS_ASSERT_OK_AND_ASSIGN(
      Alignment alignment,
      Align(renumbered.block, reference.block, MatchOptions()));
  EXPECT_EQ(alignment.matched.size(), 4);
  EXPECT_EQ(renumbered.package->DumpIr(), reference.package->DumpIr());
}

constexpr std::string_view kTwinAssertReference = R"(package p

top block b(clk: clock, x: bits[1], out: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  tok: token = after_all(id=2)
  assert.3: token = assert(tok, x, message="Selector concat.7 is not one-hot", label="A", id=3)
  assert.4: token = assert(tok, x, message="Selector concat.7 is not one-hot", label="B", id=4)
  out: () = output_port(x, name=out, id=5)
}
)";

constexpr std::string_view kTwinAssertRenumbered = R"(package p

top block b(clk: clock, x: bits[1], out: bits[1]) {
  x: bits[1] = input_port(name=x, id=101)
  tok: token = after_all(id=102)
  assert.103: token = assert(tok, x, message="Selector concat.9 is not one-hot", label="A", id=103)
  assert.104: token = assert(tok, x, message="Selector concat.9 is not one-hot", label="B", id=104)
  out: () = output_port(x, name=out, id=105)
}
)";

TEST(BlockAlignTest, AssertionsSharingAConditionPairOneToOne) {
  XLS_ASSERT_OK_AND_ASSIGN(Parsed reference, Parse(kTwinAssertReference));
  XLS_ASSERT_OK_AND_ASSIGN(Parsed renumbered, Parse(kTwinAssertRenumbered));
  XLS_ASSERT_OK_AND_ASSIGN(
      Alignment alignment,
      Align(renumbered.block, reference.block, MatchOptions()));
  EXPECT_EQ(alignment.matched.size(), 5);
  XLS_ASSERT_OK(VerifyPackage(renumbered.package.get()));
  EXPECT_EQ(renumbered.package->DumpIr(), reference.package->DumpIr());
}

// A function as the scheduling passes leave it: unchanged nodes keep their
// names, nodes the passes add have fresh ids.
constexpr std::string_view kFunctionReference = R"(package p

top fn f(x: bits[8]) -> bits[8] {
  a: bits[8] = not(x, id=2)
  add.3: bits[8] = add(a, x, id=3)
  ret neg.4: bits[8] = neg(add.3, id=4)
}
)";

constexpr std::string_view kFunctionPatched = R"(package p

top fn f(x: bits[8]) -> bits[8] {
  a: bits[8] = not(x, id=2)
  add.13: bits[8] = add(a, x, id=13)
  not.15: bits[8] = not(add.13, id=15)
  ret neg.14: bits[8] = neg(not.15, id=14)
}
)";

TEST(AlignTest, NameAnchorsPinRenumberedNodesAndLeaveNewOnesFree) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> reference,
                           Parser::ParsePackage(kFunctionReference));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> patched,
                           Parser::ParsePackage(kFunctionPatched));
  MatchOptions options;
  options.anchor_names = true;
  XLS_ASSERT_OK_AND_ASSIGN(
      Alignment alignment,
      Align(*patched->GetTop(), *reference->GetTop(), options));
  EXPECT_EQ(alignment.matched.size(), 4);
  EXPECT_EQ(alignment.nodes, 5);
  FunctionBase* f = *patched->GetTop();
  for (std::string_view name : {"x", "a", "add.3", "neg.4"}) {
    XLS_ASSERT_OK_AND_ASSIGN(Node * node, f->GetNode(name));
    XLS_ASSERT_OK_AND_ASSIGN(Node * ref, (*reference->GetTop())->GetNode(name));
    EXPECT_EQ(node->id(), ref->id()) << name;
    EXPECT_TRUE(alignment.matched.contains(node)) << name;
  }
  XLS_ASSERT_OK_AND_ASSIGN(Node * neg, f->GetNode("neg.4"));
  EXPECT_FALSE(alignment.matched.contains(neg->operand(0)));
  EXPECT_GT(neg->operand(0)->id(), 15);
  XLS_ASSERT_OK(VerifyPackage(patched.get()));
}

}  // namespace
}  // namespace xls
