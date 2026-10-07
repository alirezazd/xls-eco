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

#include <memory>
#include <string_view>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/block.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

constexpr std::string_view kOriginal = R"(package p

top block b(clk: clock, x: bits[8], out: bits[8]) {
  reg r(bits[8])
  reg p0_x(bits[8])
  x: bits[8] = input_port(name=x, id=1)
  not.2: bits[8] = not(x, id=2)
  r_write: () = register_write(not.2, register=r, id=3)
  r_read: bits[8] = register_read(register=r, id=4)
  p0_x_write: () = register_write(x, register=p0_x, id=7)
  p0_x_read: bits[8] = register_read(register=p0_x, id=8)
  neg.5: bits[8] = neg(r_read, id=5)
  out: () = output_port(neg.5, name=out, id=6)
}
)";

// The ECO negates before the register and moves p0_x to the next stage.
constexpr std::string_view kPatched = R"(package p

top block b(clk: clock, x: bits[8], out: bits[8]) {
  reg r(bits[8])
  reg p1_x(bits[8])
  x: bits[8] = input_port(name=x, id=1)
  neg.2: bits[8] = neg(x, id=2)
  r_write: () = register_write(neg.2, register=r, id=3)
  r_read: bits[8] = register_read(register=r, id=4)
  p1_x_write: () = register_write(x, register=p1_x, id=7)
  p1_x_read: bits[8] = register_read(register=p1_x, id=8)
  neg.5: bits[8] = neg(r_read, id=5)
  out: () = output_port(neg.5, name=out, id=6)
}
)";

TEST(BlockReportTest, RegistersPortsAndCones) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> original,
                           Parser::ParsePackage(kOriginal));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> patched,
                           Parser::ParsePackage(kPatched));
  XLS_ASSERT_OK_AND_ASSIGN(
      BlockReport report,
      CompareBlocks(*patched->GetTopAsBlock(), *original->GetTopAsBlock()));
  EXPECT_EQ(report.registers_kept, 1);
  EXPECT_EQ(report.registers_kept_bits, 8);
  EXPECT_EQ(report.registers_moved, 1);
  EXPECT_EQ(report.registers_removed, 0);
  EXPECT_EQ(report.registers_added, 0);
  EXPECT_EQ(report.register_data_cones_identical, 0);  // r now holds -x.
  EXPECT_EQ(report.ports_kept, 2);
  EXPECT_EQ(report.output_ports_kept, 1);
  EXPECT_EQ(report.output_cones_identical, 1);  // out reads r; r is a leaf.
}

TEST(RtlDiffTest, CountsChangedLinesWithAndWithoutOrder) {
  RtlDiff diff = DiffRtl("a\nb\nc\n", "a\nc\nb\nd\n");
  EXPECT_EQ(diff.original_lines, 4);
  EXPECT_EQ(diff.lines, 5);
  EXPECT_EQ(diff.changed, 3);
  EXPECT_EQ(diff.changed_disregarding_order, 1);
}

}  // namespace
}  // namespace xls
