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

#include "xls/contrib/eco/ged.h"

#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "xls/contrib/eco/ged_cost_functions.h"
#include "xls/contrib/eco/graph.h"

namespace ged {
namespace {

NodeCostAttributes AttributesForLabel(const std::string& label) {
  NodeCostAttributes attrs;
  if (label == "A") {
    attrs.op = xls::Op::kAdd;
  } else if (label == "B") {
    attrs.op = xls::Op::kAnd;
  } else if (label == "C") {
    attrs.op = xls::Op::kOr;
  } else {
    attrs.op = xls::Op::kParam;
  }
  return attrs;
}

// Builds a graph from node labels and (from, to, index) edges. The edge's
// operand index is folded into its cost attributes so distinct indices give
// distinct edge labels (mirroring real IR, where labels distinguish operands).
XLSGraph MakeGraph(const std::vector<std::string>& labels,
                   const std::vector<std::tuple<int, int, int>>& edges) {
  XLSGraph graph;
  for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
    graph.add_node(
        XLSNode("n" + std::to_string(i), AttributesForLabel(labels[i])));
  }
  for (const auto& [from, to, index] : edges) {
    EdgeCostAttributes attrs;
    attrs.index = index;
    graph.add_edge(XLSEdge(from, to, attrs, index));
  }
  graph.populate_node_signatures();
  return graph;
}

// Asserts that `result` is a complete, valid edit path: every node and edge of
// both graphs is covered exactly once, every node substitution maps equal
// labels, and every edge substitution is label- and endpoint-consistent. This
// is the property that guarantees the patch replays graph1 into graph2.
void ExpectValidEditPath(const GEDResult& result, const XLSGraph& g1,
                         const XLSGraph& g2) {
  std::vector<int> g1_node_cov(g1.nodes.size(), 0);
  std::vector<int> g2_node_cov(g2.nodes.size(), 0);
  std::vector<int> node_map(g1.nodes.size(), -1);
  for (const auto& [i, j] : result.node_substitutions) {
    g1_node_cov[i]++;
    g2_node_cov[j]++;
    node_map[i] = j;
    EXPECT_EQ(g1.nodes[i].label, g2.nodes[j].label)
        << "node substitution " << i << "->" << j << " is not label-equal";
  }
  for (int i : result.node_deletions) {
    g1_node_cov[i]++;
  }
  for (int j : result.node_insertions) {
    g2_node_cov[j]++;
  }
  for (int i = 0; i < static_cast<int>(g1.nodes.size()); ++i) {
    EXPECT_EQ(g1_node_cov[i], 1) << "graph1 node " << i << " not covered once";
  }
  for (int j = 0; j < static_cast<int>(g2.nodes.size()); ++j) {
    EXPECT_EQ(g2_node_cov[j], 1) << "graph2 node " << j << " not covered once";
  }

  std::vector<int> g1_edge_cov(g1.edges.size(), 0);
  std::vector<int> g2_edge_cov(g2.edges.size(), 0);
  for (const auto& [a, b] : result.edge_substitutions) {
    g1_edge_cov[a]++;
    g2_edge_cov[b]++;
    EXPECT_EQ(g1.edges[a].label, g2.edges[b].label)
        << "edge substitution " << a << "->" << b << " is not label-equal";
    EXPECT_EQ(node_map[g1.edges[a].endpoints.first],
              g2.edges[b].endpoints.first);
    EXPECT_EQ(node_map[g1.edges[a].endpoints.second],
              g2.edges[b].endpoints.second);
  }
  for (int a : result.edge_deletions) {
    g1_edge_cov[a]++;
  }
  for (int b : result.edge_insertions) {
    g2_edge_cov[b]++;
  }
  for (int a = 0; a < static_cast<int>(g1.edges.size()); ++a) {
    EXPECT_EQ(g1_edge_cov[a], 1) << "graph1 edge " << a << " not covered once";
  }
  for (int b = 0; b < static_cast<int>(g2.edges.size()); ++b) {
    EXPECT_EQ(g2_edge_cov[b], 1) << "graph2 edge " << b << " not covered once";
  }
}

TEST(GedTest, IdenticalGraphsCostZeroNoInsertsOrDeletes) {
  XLSGraph g1 = MakeGraph({"A", "B", "C"}, {{0, 1, 0}, {1, 2, 1}});
  XLSGraph g2 = MakeGraph({"A", "B", "C"}, {{0, 1, 0}, {1, 2, 1}});

  GEDResult result = SolveGED(g1, g2, CreateUserCosts(), Method::kBranchFast);

  ExpectValidEditPath(result, g1, g2);
  EXPECT_EQ(result.total_cost, 0);
  EXPECT_TRUE(result.node_deletions.empty());
  EXPECT_TRUE(result.node_insertions.empty());
  EXPECT_TRUE(result.edge_deletions.empty());
  EXPECT_TRUE(result.edge_insertions.empty());
  EXPECT_EQ(result.node_substitutions.size(), 3u);
  EXPECT_EQ(result.edge_substitutions.size(), 2u);
}

TEST(GedTest, ImbalancedGraphInducesValidPathWithInsertions) {
  XLSGraph g1 = MakeGraph({"A", "B"}, {{0, 1, 0}});
  XLSGraph g2 = MakeGraph({"A", "B", "C"}, {{0, 1, 0}, {1, 2, 1}});

  GEDResult result = SolveGED(g1, g2, CreateUserCosts(), Method::kBranchFast);

  ExpectValidEditPath(result, g1, g2);
  // The extra "C" node and its incoming edge are pure insertions.
  EXPECT_EQ(result.node_insertions.size(), 1u);
  EXPECT_TRUE(result.node_deletions.empty());
  EXPECT_EQ(result.edge_insertions.size(), 1u);
  EXPECT_TRUE(result.edge_deletions.empty());
  EXPECT_EQ(result.total_cost, 2);  // 1 node insert + 1 edge insert.
}

TEST(GedTest, SymmetricSurplusBecomesDeletion) {
  // Two "A" nodes vs one: exactly one A must be deleted (the surplus per label
  // class is invariant across any assignment).
  XLSGraph g1 = MakeGraph({"A", "A", "B"}, {});
  XLSGraph g2 = MakeGraph({"A", "B"}, {});

  GEDResult result = SolveGED(g1, g2, CreateUserCosts(), Method::kBranchFast);

  ExpectValidEditPath(result, g1, g2);
  EXPECT_EQ(result.node_deletions.size(), 1u);
  EXPECT_TRUE(result.node_insertions.empty());
  EXPECT_EQ(result.node_substitutions.size(), 2u);
}

TEST(GedTest, InduceEditPathFromHandBuiltMapWithDeletion) {
  XLSGraph g1 = MakeGraph({"A", "B", "C"}, {{0, 1, 0}, {1, 2, 1}});
  XLSGraph g2 = MakeGraph({"A", "B"}, {{0, 1, 0}});

  // Map A->A, B->B, delete C; nothing inserted.
  std::vector<int> match_g1 = {0, 1, -1};
  std::vector<int> match_g2 = {0, 1};

  GEDResult result =
      internal::InduceEditPath(match_g1, match_g2, g1, g2, CreateUserCosts());

  ExpectValidEditPath(result, g1, g2);
  ASSERT_EQ(result.node_deletions.size(), 1u);
  EXPECT_EQ(result.node_deletions[0], 2);
  EXPECT_TRUE(result.node_insertions.empty());
  // The B->C edge (incident to deleted C) is deleted; the A->B edge
  // substitutes.
  EXPECT_EQ(result.edge_substitutions.size(), 1u);
  EXPECT_EQ(result.edge_deletions.size(), 1u);
}

TEST(GedTest, BranchFastCostMatrixIncludesIncidentEdgeTerm) {
  // g1 has a lone "A"; g2 has "A" with one outgoing edge to "B".
  XLSGraph g1 = MakeGraph({"A"}, {});
  XLSGraph g2 = MakeGraph({"A", "B"}, {{0, 1, 0}});

  RawCostMatrix matrix = internal::BuildLsapeCostMatrix(
      g1, g2, CreateUserCosts(), Method::kBranchFast);

  // Substituting g1.A -> g2.A: node cost 0 (equal label), plus one out-edge of
  // g2.A that g1.A lacks => +1 (out-edge multiset mismatch). Scaled costs.
  EXPECT_EQ(matrix.Get(0, 0), 1);
  // Deleting g1.A (index 0; deletion column is n+i == 2+0): 2*del(1) +
  // degree(0) = 2.
  EXPECT_EQ(matrix.Get(0, 2), 2);

  // The kNode variant drops the edge term: substitution becomes 0.
  RawCostMatrix node_only =
      internal::BuildLsapeCostMatrix(g1, g2, CreateUserCosts(), Method::kNode);
  EXPECT_EQ(node_only.Get(0, 0), 0);
}

}  // namespace
}  // namespace ged
