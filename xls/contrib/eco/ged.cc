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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "xls/contrib/eco/graph.h"
#include "xls/contrib/eco/lap_solver.h"

// Graph Edit Distance via the BRANCH-FAST heuristic of:
//   David B. Blumenthal, Nicolas Boria, Johann Gamper, Sebastien Bougleux, and
//   Luc Brun. "Comparing heuristics for graph edit distance computation."
//   The VLDB Journal 29 (2020), 419-458.
//   https://doi.org/10.1007/s00778-019-00544-1
//
// All section/definition/algorithm references below are to that paper:
//   * Algorithm 1 ("The paradigm LSAPE-GED"): build a node edit cost matrix C,
//     solve the LSAPE for a node map pi, and return the upper bound
//     UB := c(P_pi) (Algorithm 1, line 3) -- the cost of pi's induced edit
//     path.
//   * Definition 2 (node map) and Definition 3 (induced edit path): a node map
//     induces a complete edit path, and "induced edit paths are indeed edit
//     paths", so the constructed result provably transforms G into H. Any
//     feasible node map is therefore a valid upper bound; correctness here is
//     by construction and is certified downstream by the equivalence check,
//     never by GED optimality.
//   * Section 5.2.3 (algorithm BRANCH): the LSAPE substitution entry is
//         c_{i,k} := cV(u_i, v_k) + 0.5 * C_{i,k}(pi^{i,k}),
//     the deletion entry cV(u_i, eps) + 0.5 * sum_{j} cE((u_i, u_ij), eps), and
//     the insertion entry symmetrically, where C_{i,k} is the optimal LSAPE
//     over the edges incident with u_i and v_k (Section 5.2.2, algorithm BP).
//     The 0.5 halves each edge because it is shared by its two endpoints.
//   * Section 5.2.4 (algorithm BRANCH-FAST): replaces the optimal incident-edge
//     LSAPE with a sorted incident-edge-label multiset match. Per the paper,
//     "For constant edge edit costs, BRANCH and BRANCH-FAST are equivalent";
//     ged_cost_functions uses constant (0 / high) edge costs, so this single
//     implementation realizes both.
//   * Complexity (Section 5.2): solving C optimally is
//     O(min{|VG|,|VH|}^2 * max{|VG|,|VH|}); greedily O(|VG||VH|); the induced
//     cost is O(max{|EG|,|EH|}).
//
// XLS adaptation: the paper targets undirected graphs. XLS ECO graphs are
// directed and edge-labeled, so the incident-edge term C_{i,k} is split into
// separate in-edge and out-edge multiset matches, and InduceEditPath reconciles
// each directed edge independently. The LSAPE instance C is solved by
// SolveLSAP, a wrapper over the shortest-augmenting-path solver in lap_solver
// (Crouse 2016).

namespace ged {
namespace {

// Sorted label multisets of a node's incident edges, split by direction so the
// branch cost respects edge orientation in the directed IR graph.
struct IncidentLabels {
  std::vector<std::size_t> in;
  std::vector<std::size_t> out;
};

std::vector<IncidentLabels> ComputeIncidentLabels(const XLSGraph& graph) {
  std::vector<IncidentLabels> incident(graph.nodes.size());
  for (const XLSEdge& edge : graph.edges) {
    incident[edge.endpoints.first].out.push_back(edge.label);
    incident[edge.endpoints.second].in.push_back(edge.label);
  }
  for (IncidentLabels& il : incident) {
    absl::c_sort(il.in);
    absl::c_sort(il.out);
  }
  return incident;
}

// Incident-edge term C_{i,k} for one sorted label-multiset pair, computed the
// BRANCH-FAST way (Blumenthal et al. 2020, Section 5.2.4): edges with equal
// labels substitute at cost 0, the rest are inserted/deleted at cost 1 each.
// Under the constant (0 / high) edge-substitution cost of ged_cost_functions
// this sorted-multiset match equals BP's optimal incident-edge LSAPE C_{i,k}
// (Section 5.2.2), so kBranch and kBranchFast coincide here ("For constant edge
// edit costs, BRANCH and BRANCH-FAST are equivalent", Section 5.2.4).
int MultisetMismatch(const std::vector<std::size_t>& a,
                     const std::vector<std::size_t>& b) {
  int matched = 0;
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] == b[j]) {
      ++matched;
      ++i;
      ++j;
    } else if (a[i] < b[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return static_cast<int>(a.size() + b.size()) - 2 * matched;
}

}  // namespace

AssignmentResult SolveLSAP(const RawCostMatrix& M, int m, int n,
                           std::vector<std::vector<double>>& dense) {
  AssignmentResult result;
  VLOG(2) << "SolveLSAP start: m=" << m << " n=" << n << " rows=" << M.n_rows
          << " cols=" << M.n_cols;

  const int size = std::max(M.n_rows, M.n_cols);
  if (dense.size() != static_cast<size_t>(size)) {
    dense.assign(size, std::vector<double>(size, (double)RawCostMatrix::INF));
  } else {
    for (auto& row : dense) {
      std::fill(row.begin(), row.end(), (double)RawCostMatrix::INF);
    }
  }
  for (int i = 0; i < M.n_rows; ++i) {
    for (int j = 0; j < M.n_cols; ++j) {
      dense[i][j] = static_cast<double>(M.Get(i, j));
    }
  }

  auto [row_ind, col_ind] = LinearSumAssignment(dense);

  int64_t total_cost = 0;
  for (size_t p = 0; p < row_ind.size(); ++p) {
    int i = row_ind[p];
    int j = col_ind[p];
    double c = dense[i][j];
    if (c >= (double)RawCostMatrix::INF) {
      result.cost = RawCostMatrix::INF;
      VLOG(1) << "SolveLSAP infeasible assignment detected";
      return result;
    }
    total_cost += (int64_t)c;

    if (i < m && j < n) {
      result.subs.emplace_back(i, j);
    } else if (i < m && j >= n) {
      result.dels.push_back(i);
    } else if (i >= m && j < n) {
      result.ins.push_back(j);
    }
  }

  result.cost = (total_cost > RawCostMatrix::INF)
                    ? RawCostMatrix::INF
                    : static_cast<int>(total_cost);
  VLOG(2) << "SolveLSAP done: cost=" << result.cost
          << " subs=" << result.subs.size() << " dels=" << result.dels.size()
          << " ins=" << result.ins.size();
  return result;
}

namespace internal {

RawCostMatrix BuildLsapeCostMatrix(const XLSGraph& graph1,
                                   const XLSGraph& graph2,
                                   const GEDOptions& options, Method method) {
  const int m = static_cast<int>(graph1.nodes.size());
  const int n = static_cast<int>(graph2.nodes.size());
  RawCostMatrix matrix(m + n, m + n);

  const bool with_edges = method != Method::kNode;
  std::vector<IncidentLabels> inc1 = ComputeIncidentLabels(graph1);
  std::vector<IncidentLabels> inc2 = ComputeIncidentLabels(graph2);

  // BRANCH cost matrix C (Blumenthal et al. 2020, Section 5.2.3):
  //   substitution c_{i,k}    = cV(u_i, v_k) + 0.5 * C_{i,k}
  //   deletion  c_{i,|VH|+1}  = cV(u_i, eps) + 0.5 * (incident edges of u_i)
  //   insertion c_{|VG|+1,k}  = cV(eps, v_k) + 0.5 * (incident edges of v_k)
  // Costs are scaled by 2 here so the 0.5 half-edge term stays integral:
  // 2*cV + C_{i,k} == 2*(cV + 0.5*C_{i,k}); the scaling does not change the
  // resulting node map.

  // Substitution block (m x n).
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int cost = 2 * options.nodeCosts.subst(graph1.nodes[i], graph2.nodes[j]);
      if (with_edges) {
        cost += MultisetMismatch(inc1[i].in, inc2[j].in) +
                MultisetMismatch(inc1[i].out, inc2[j].out);
      }
      matrix.Set(i, j, cost);
    }
  }
  // Deletion block (m x m): diagonal = delete node + delete its incident edges.
  for (int i = 0; i < m; ++i) {
    const int degree = static_cast<int>(inc1[i].in.size() + inc1[i].out.size());
    matrix.Set(
        i, n + i,
        2 * options.nodeCosts.del(graph1.nodes[i]) + (with_edges ? degree : 0));
  }
  // Insertion block (n x n): diagonal = insert node + insert its incident
  // edges.
  for (int j = 0; j < n; ++j) {
    const int degree = static_cast<int>(inc2[j].in.size() + inc2[j].out.size());
    matrix.Set(
        m + j, j,
        2 * options.nodeCosts.ins(graph2.nodes[j]) + (with_edges ? degree : 0));
  }
  // Dummy block (n x m): all zeros.
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      matrix.Set(m + i, n + j, 0);
    }
  }
  return matrix;
}

GEDResult InduceEditPath(const std::vector<int>& match_g1,
                         const std::vector<int>& match_g2,
                         const XLSGraph& graph1, const XLSGraph& graph2,
                         const GEDOptions& options) {
  GEDResult result;
  const int num_g1_nodes = static_cast<int>(match_g1.size());
  const int num_g2_nodes = static_cast<int>(match_g2.size());
  const int num_g1_edges = static_cast<int>(graph1.edges.size());
  const int num_g2_edges = static_cast<int>(graph2.edges.size());

  // Node operations straight from the node map.
  for (int i = 0; i < num_g1_nodes; ++i) {
    if (match_g1[i] >= 0) {
      result.node_substitutions.emplace_back(i, match_g1[i]);
    } else {
      result.node_deletions.push_back(i);
    }
  }
  for (int j = 0; j < num_g2_nodes; ++j) {
    if (match_g2[j] < 0) {
      result.node_insertions.push_back(j);
    }
  }

  // Edge reconciliation. Bucket graph2 edge indices by endpoint pair; a graph1
  // edge whose endpoints both map and whose label matches an unclaimed graph2
  // edge between the mapped endpoints becomes an edge substitution (cost 0),
  // otherwise an edge deletion. Unclaimed graph2 edges become insertions. This
  // covers every edge of both graphs exactly once.
  absl::flat_hash_map<std::pair<int, int>, std::vector<int>, PairHash>
      g2_buckets;
  for (int k = 0; k < num_g2_edges; ++k) {
    g2_buckets[graph2.edges[k].endpoints].push_back(k);
  }
  std::vector<bool> claimed(num_g2_edges, false);

  for (int k1 = 0; k1 < num_g1_edges; ++k1) {
    const XLSEdge& e1 = graph1.edges[k1];
    const int u2 = match_g1[e1.endpoints.first];
    const int v2 = match_g1[e1.endpoints.second];
    int matched_k2 = -1;
    if (u2 >= 0 && v2 >= 0) {
      auto it = g2_buckets.find(std::pair<int, int>(u2, v2));
      if (it != g2_buckets.end()) {
        for (int k2 : it->second) {
          if (!claimed[k2] && graph2.edges[k2].label == e1.label) {
            matched_k2 = k2;
            break;
          }
        }
      }
    }
    if (matched_k2 >= 0) {
      claimed[matched_k2] = true;
      result.edge_substitutions.emplace_back(k1, matched_k2);
    } else {
      result.edge_deletions.push_back(k1);
    }
  }
  for (int k2 = 0; k2 < num_g2_edges; ++k2) {
    if (!claimed[k2]) {
      result.edge_insertions.push_back(k2);
    }
  }

  // Accumulate final costs.
  int node_cost = 0;
  for (const auto& [i, j] : result.node_substitutions) {
    node_cost += options.nodeCosts.subst(graph1.nodes[i], graph2.nodes[j]);
  }
  for (int i : result.node_deletions) {
    node_cost += options.nodeCosts.del(graph1.nodes[i]);
  }
  for (int j : result.node_insertions) {
    node_cost += options.nodeCosts.ins(graph2.nodes[j]);
  }
  int edge_cost = 0;
  for (const auto& [a, b] : result.edge_substitutions) {
    edge_cost += options.edgeCosts.subst(graph1.edges[a], graph2.edges[b]);
  }
  for (int a : result.edge_deletions) {
    edge_cost += options.edgeCosts.del(graph1.edges[a]);
  }
  for (int b : result.edge_insertions) {
    edge_cost += options.edgeCosts.ins(graph2.edges[b]);
  }
  result.node_cost = node_cost;
  result.edge_cost = edge_cost;
  result.total_cost = node_cost + edge_cost;
  return result;
}

}  // namespace internal

GEDResult SolveGED(const XLSGraph& graph1, const XLSGraph& graph2,
                   const GEDOptions& options, Method method) {
  // Algorithm 1 ("The paradigm LSAPE-GED", Blumenthal et al. 2020): build the
  // node edit cost matrix C, solve the LSAPE for a node map pi, and return the
  // upper bound c(P_pi) given by pi's induced edit path.
  const int m = static_cast<int>(graph1.nodes.size());
  const int n = static_cast<int>(graph2.nodes.size());
  VLOG(1) << "SolveGED start: G1 nodes=" << m
          << " edges=" << graph1.edges.size() << " | G2 nodes=" << n
          << " edges=" << graph2.edges.size();

  RawCostMatrix cost_matrix =
      internal::BuildLsapeCostMatrix(graph1, graph2, options, method);
  std::vector<std::vector<double>> dense_buffer;
  AssignmentResult assignment = SolveLSAP(cost_matrix, m, n, dense_buffer);

  // Turn the assignment into a node map: match_g1[i] = matched graph2 node (or
  // -1 if deleted); match_g2[j] = matched graph1 node (or -1 if inserted).
  std::vector<int> match_g1(m, -1);
  std::vector<int> match_g2(n, -1);
  for (const auto& [i, j] : assignment.subs) {
    match_g1[i] = j;
    match_g2[j] = i;
  }

  GEDResult result =
      internal::InduceEditPath(match_g1, match_g2, graph1, graph2, options);
  VLOG(1) << "SolveGED done: total_cost=" << result.total_cost
          << " node_subs=" << result.node_substitutions.size()
          << " node_dels=" << result.node_deletions.size()
          << " node_ins=" << result.node_insertions.size()
          << " edge_subs=" << result.edge_substitutions.size()
          << " edge_dels=" << result.edge_deletions.size()
          << " edge_ins=" << result.edge_insertions.size();
  return result;
}

}  // namespace ged
