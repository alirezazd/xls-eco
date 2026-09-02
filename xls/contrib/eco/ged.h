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

#ifndef XLS_ECO_GED_H_
#define XLS_ECO_GED_H_

#include <climits>
#include <limits>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "xls/contrib/eco/graph.h"

namespace ged {

// Dense cost matrix stored in row-major order.
struct RawCostMatrix {
  std::vector<int> data;
  int n_rows = 0;
  int n_cols = 0;

  static constexpr int INF = std::numeric_limits<int>::max();

  RawCostMatrix() = default;
  RawCostMatrix(int rows, int cols)
      : data(static_cast<size_t>(rows) * cols, INF),  // Initialize with INF
        n_rows(rows),
        n_cols(cols) {}

  int Get(int i, int j) const { return data[i * n_cols + j]; }
  void Set(int i, int j, int c) { data[i * n_cols + j] = c; }
};

// Result of a rectangular LSAP over a node or edge cost matrix, classified into
// substitutions (i<m, j<n), deletions (i<m), and insertions (j<n).
struct AssignmentResult {
  int cost = 0;
  std::vector<std::pair<int, int>> subs;
  std::vector<int> dels;
  std::vector<int> ins;
};

// Solves the rectangular linear sum assignment problem on the (m+n)x(m+n)
// cost matrix M (substitution / deletion / insertion / dummy blocks) using the
// shortest-augmenting-path solver in lap_solver, and classifies the assignment
// into subs/dels/ins. `dense_buffer` is a reusable scratch buffer.
AssignmentResult SolveLSAP(const RawCostMatrix& M, int m, int n,
                           std::vector<std::vector<double>>& dense_buffer);

struct NodeCostFunctions {
  absl::AnyInvocable<int(const XLSNode&, const XLSNode&) const> subst;
  absl::AnyInvocable<int(const XLSNode&) const> ins;
  absl::AnyInvocable<int(const XLSNode&) const> del;
};

struct EdgeCostFunctions {
  absl::AnyInvocable<int(const XLSEdge&, const XLSEdge&) const> subst;
  absl::AnyInvocable<int(const XLSEdge&) const> ins;
  absl::AnyInvocable<int(const XLSEdge&) const> del;
};

struct GEDOptions {
  NodeCostFunctions nodeCosts;
  EdgeCostFunctions edgeCosts;
};

struct GEDResult {
  std::vector<std::pair<int, int>> node_substitutions;
  std::vector<int> node_insertions;
  std::vector<int> node_deletions;

  std::vector<std::pair<int, int>> edge_substitutions;
  std::vector<int> edge_insertions;
  std::vector<int> edge_deletions;

  int node_cost = INT_MAX;
  int edge_cost = INT_MAX;
  int total_cost = INT_MAX;
};

// Selects how local edge structure is folded into each node-substitution entry
// of the assignment cost matrix (Blumenthal et al. 2020; see ged.cc). All
// variants yield a complete, valid edit path; they trade tightness (less patch
// drift) for speed.
enum class Method {
  kNode,        // NODE (Section 5.2.1): node label only, no edge term.
  kBranch,      // BRANCH (Section 5.2.3): optimal incident-edge LSAPE C_{i,k}.
  kBranchFast,  // BRANCH-FAST (Section 5.2.4): sorted multiset match (default).
};

// Computes the graph edit distance between graph1 and graph2 via the LSAPE-GED
// paradigm (Algorithm 1 of Blumenthal et al., The VLDB Journal 2020; see
// ged.cc): a single bipartite node assignment whose induced edit path is
// returned. Runs in polynomial time and returns an UPPER BOUND on the true GED
// whose induced edit path provably transforms graph1 into graph2 -- correctness
// is independent of optimality (suboptimality only enlarges the patch) and is
// certified downstream by the equivalence check.
GEDResult SolveGED(const XLSGraph& graph1, const XLSGraph& graph2,
                   const GEDOptions& options,
                   Method method = Method::kBranchFast);

// Utility functions exposed for testing.
namespace internal {

// Builds the (n+m)x(n+m) LSAPE node-assignment cost matrix (Blumenthal et al.
// 2020, Section 5.2.3 BRANCH / 5.2.4 BRANCH-FAST). Costs are scaled by 2 so the
// BRANCH 0.5 half-edge term stays integral. The deletion/insertion blocks
// absorb node-count imbalance.
RawCostMatrix BuildLsapeCostMatrix(const XLSGraph& graph1,
                                   const XLSGraph& graph2,
                                   const GEDOptions& options, Method method);

// Induces a complete, valid edit path from a node map, per Definition 3
// (induced edit path) of Blumenthal et al. 2020: matched nodes become
// substitutions, unmatched graph1/graph2 nodes become deletions/insertions,
// then every edge is reconciled into an edge substitution / deletion /
// insertion. `match_g1[i]` is the graph2 node matched to graph1 node i, or -1
// if i is deleted; `match_g2[j]` is the graph1 node matched to graph2 node j,
// or -1 if j is inserted. The result covers every node and edge of both graphs
// exactly once and therefore replays to graph2 exactly.
GEDResult InduceEditPath(const std::vector<int>& match_g1,
                         const std::vector<int>& match_g2,
                         const XLSGraph& graph1, const XLSGraph& graph2,
                         const GEDOptions& options);

}  // namespace internal

}  // namespace ged

#endif  // XLS_ECO_GED_H_
