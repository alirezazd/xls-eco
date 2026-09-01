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

// Tests XLSGraph::StructuralHash and MismatchRoots.
//
// StructuralHash is a bottom-up Merkle fold over the dataflow DAG: a node's
// hash depends on its own label and its operands' hashes, never on node names
// or IR ids. It returns the per-node hashes in node-index order with the
// whole-graph hash appended last -- hence the GraphHash()/NodeHashes() helpers
// below.
//
// The tests come in two kinds: differences that are NOT structure (names, ids,
// commutative operand order) must be absorbed and hash equal; real structural
// differences must hash unequal AND be localized by MismatchRoots to the
// topologically minimal divergent nodes, not their downstream users.
//
// Hashes are only comparable within a single binary (std::hash is
// implementation-defined), so these tests always compare two graphs built
// here -- never against a hard-coded constant.

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/eco/graph.h"
#include "xls/contrib/eco/struct_equivalence.h"
#include "xls/contrib/eco/xls_ir_to_graph.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

class StructuralHashTest : public IrTestBase {};

absl::StatusOr<XLSGraph> GraphFromIr(std::string_view ir) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedPackage> package,
                       IrTestBase::ParsePackage(ir));
  std::optional<FunctionBase*> top = package->GetTop();
  XLS_RET_CHECK(top.has_value());
  return XlsIrToGraph(*top);
}

std::size_t GraphHash(const std::vector<std::size_t>& v) { return v.back(); }

absl::Span<const std::size_t> NodeHashes(const std::vector<std::size_t>& v) {
  return absl::MakeConstSpan(v).first(v.size() - 1);
}

// Node names and IR ids are not structure: identical dataflow hashes equal.
TEST_F(StructuralHashTest, EquivalentModuloNamesAndIds) {
  XLS_ASSERT_OK_AND_ASSIGN(XLSGraph g1, GraphFromIr(R"(
package a

top fn main(x: bits[32], y: bits[32]) -> bits[32] {
  add.3: bits[32] = add(x, y, id=3)
  ret result: bits[32] = not(add.3, id=4)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(XLSGraph g2, GraphFromIr(R"(
package b

top fn main(x: bits[32], y: bits[32]) -> bits[32] {
  mysum: bits[32] = add(x, y, id=7)
  ret out: bits[32] = not(mysum, id=8)
}
)"));
  EXPECT_EQ(GraphHash(g1.StructuralHash()), GraphHash(g2.StructuralHash()));
}

// Operand order of a commutative op is not structure: add(x, y) == add(y, x).
TEST_F(StructuralHashTest, CommutativeOperandSwapIsAbsorbed) {
  XLS_ASSERT_OK_AND_ASSIGN(XLSGraph g1, GraphFromIr(R"(
package a

top fn main(x: bits[32], y: bits[32]) -> bits[32] {
  ret sum: bits[32] = add(x, y, id=3)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(XLSGraph g2, GraphFromIr(R"(
package b

top fn main(x: bits[32], y: bits[32]) -> bits[32] {
  ret sum: bits[32] = add(y, x, id=3)
}
)"));
  EXPECT_EQ(GraphHash(g1.StructuralHash()), GraphHash(g2.StructuralHash()));
}

// Deliberate mismatch: the two IRs differ in one literal. The check must
// fail, and the mismatch roots must name exactly the literal on each side,
// not its downstream users.
TEST_F(StructuralHashTest, LiteralChangeFailsAndIsLocalized) {
  XLS_ASSERT_OK_AND_ASSIGN(XLSGraph g1, GraphFromIr(R"(
package a

top fn main(x: bits[32]) -> bits[32] {
  one: bits[32] = literal(value=1, id=2)
  bumped: bits[32] = add(x, one, id=3)
  ret result: bits[32] = not(bumped, id=4)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(XLSGraph g2, GraphFromIr(R"(
package b

top fn main(x: bits[32]) -> bits[32] {
  two: bits[32] = literal(value=2, id=2)
  bumped: bits[32] = add(x, two, id=3)
  ret result: bits[32] = not(bumped, id=4)
}
)"));
  const std::vector<std::size_t> v1 = g1.StructuralHash();
  const std::vector<std::size_t> v2 = g2.StructuralHash();
  EXPECT_NE(GraphHash(v1), GraphHash(v2));
  EXPECT_EQ(MismatchRoots(g1, NodeHashes(v1), NodeHashes(v2)),
            std::vector<std::string>{"one"});
  EXPECT_EQ(MismatchRoots(g2, NodeHashes(v2), NodeHashes(v1)),
            std::vector<std::string>{"two"});
}

// Deliberate mismatch: swapping the operands of a NON-commutative op is a
// real structural change. The check must fail, rooted at the swapped op.
TEST_F(StructuralHashTest, NonCommutativeOperandSwapFailsAndIsLocalized) {
  XLS_ASSERT_OK_AND_ASSIGN(XLSGraph g1, GraphFromIr(R"(
package a

top fn main(x: bits[32], y: bits[32]) -> bits[32] {
  ret diff: bits[32] = sub(x, y, id=3)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(XLSGraph g2, GraphFromIr(R"(
package b

top fn main(x: bits[32], y: bits[32]) -> bits[32] {
  ret diff: bits[32] = sub(y, x, id=3)
}
)"));
  const std::vector<std::size_t> v1 = g1.StructuralHash();
  const std::vector<std::size_t> v2 = g2.StructuralHash();
  EXPECT_NE(GraphHash(v1), GraphHash(v2));
  EXPECT_EQ(MismatchRoots(g1, NodeHashes(v1), NodeHashes(v2)),
            std::vector<std::string>{"diff"});
  EXPECT_EQ(MismatchRoots(g2, NodeHashes(v2), NodeHashes(v1)),
            std::vector<std::string>{"diff"});
}

}  // namespace
}  // namespace xls
