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

#include "xls/contrib/eco/match.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "xls/common/stopwatch.h"
#include "xls/contrib/eco/ged.h"
#include "xls/contrib/eco/ged_cost_functions.h"
#include "xls/contrib/eco/graph.h"
#include "xls/contrib/eco/mcs.h"

namespace xls {
namespace {

void AnnotateMcsMappings(
    const std::vector<std::pair<std::string, std::string>>& boundary_pairs,
    XLSGraph& graph1, XLSGraph& graph2) {
  for (const auto& [g1_name, g2_name] : boundary_pairs) {
    auto g1_it = graph1.node_name_to_index.find(g1_name);
    auto g2_it = graph2.node_name_to_index.find(g2_name);
    if (g1_it == graph1.node_name_to_index.end() ||
        g2_it == graph2.node_name_to_index.end()) {
      VLOG(1) << "Skip MCS map annotation for pair (" << g1_name << ", "
              << g2_name << ") because one node is missing post-cut";
      continue;
    }
    graph1.nodes[g1_it->second].mcs_map_index = g2_it->second;
    graph2.nodes[g2_it->second].mcs_map_index = g1_it->second;
    VLOG(2) << "Annotated MCS pair: " << g1_name << " <-> " << g2_name;
  }
}

mcs::MCSResult NameAnchors(const XLSGraph& graph1, const XLSGraph& graph2) {
  mcs::MCSResult anchors;
  for (int64_t u = 0; u < graph1.nodes.size(); ++u) {
    auto it = graph2.node_name_to_index.find(graph1.nodes[u].name);
    if (it != graph2.node_name_to_index.end() &&
        graph2.nodes[it->second].label == graph1.nodes[u].label) {
      anchors.mapping.emplace_back(u, it->second);
    }
  }
  anchors.size = anchors.mapping.size();
  return anchors;
}

// Pins the boundary of the common subgraph, cuts its interior, and records
// the mapping.
void Preprocess(XLSGraph& graph1, XLSGraph& graph2, const MatchOptions& options,
                MatchResult& result) {
  Stopwatch mcs_stopwatch;
  mcs::MCSResult mcs;
  if (options.anchor_names) {
    LOG(INFO) << "Anchoring nodes by name";
    mcs = NameAnchors(graph1, graph2);
  } else {
    LOG(INFO) << "MCS preprocessing enabled";
    mcs = mcs::SolveMCS(graph1, graph2, options.mcs_cutoff, options.mcs_optimal,
                        options.mcs_timeout);
  }
  result.stats.mcs_runtime_sec =
      absl::ToDoubleSeconds(mcs_stopwatch.GetElapsedTime());
  result.stats.mcs_matched_nodes = mcs.mapping.size();
  result.stats.mcs_matched_edges = mcs.edge_size;
  for (const auto& [u, v] : mcs.mapping) {
    result.node_map.emplace_back(graph1.nodes[u].name, graph2.nodes[v].name);
  }

  auto boundary_nodes = mcs::GetBoundaryNodes(mcs, graph1, graph2);
  std::vector<int> boundary_g1_indices;
  std::vector<int> boundary_g2_indices;
  std::vector<std::pair<std::string, std::string>> boundary_names;
  boundary_g1_indices.reserve(boundary_nodes.size());
  boundary_g2_indices.reserve(boundary_nodes.size());
  boundary_names.reserve(boundary_nodes.size());
  for (const auto& [u, v] : boundary_nodes) {
    std::string g1_name = (u >= 0 && u < (int)graph1.nodes.size())
                              ? graph1.nodes[u].name
                              : "<invalid>";
    std::string g2_name = (v >= 0 && v < (int)graph2.nodes.size())
                              ? graph2.nodes[v].name
                              : "<invalid>";
    VLOG(2) << "Pin boundary pair: G1 idx=" << u << " name=" << g1_name
            << " G2 idx=" << v << " name=" << g2_name;
    boundary_g1_indices.push_back(u);
    boundary_g2_indices.push_back(v);
    boundary_names.emplace_back(g1_name, g2_name);
  }
  graph1.PinNodes(boundary_g1_indices);
  graph2.PinNodes(boundary_g2_indices);
  VLOG(1) << "Pinned boundary nodes: " << boundary_nodes.size();

  result.stats.mcs_boundary_nodes = boundary_nodes.size();
  int boundary_edges = 0;
  for (const auto& e : graph1.edges) {
    for (const auto& idx : boundary_g1_indices) {
      if (e.endpoints.first == idx || e.endpoints.second == idx) {
        boundary_edges++;
        break;
      }
    }
  }
  result.stats.mcs_boundary_edges = boundary_edges;

  std::vector<int> to_cut_g1, to_cut_g2;
  for (const auto& [u, v] : mcs.mapping) {
    if (!graph1.nodes[u].pinned) to_cut_g1.push_back(u);
    if (!graph2.nodes[v].pinned) to_cut_g2.push_back(v);
  }
  VLOG(2) << "Cut candidates: G1=" << to_cut_g1.size()
          << " G2=" << to_cut_g2.size();

  graph1.Cut(to_cut_g1);
  graph2.Cut(to_cut_g2);
  graph1.RemoveDanglingEdges();
  graph2.RemoveDanglingEdges();
  AnnotateMcsMappings(boundary_names, graph1, graph2);
  VLOG(1) << "Post-MCS graph sizes: G1 nodes=" << graph1.nodes.size()
          << " edges=" << graph1.edges.size()
          << " | G2 nodes=" << graph2.nodes.size()
          << " edges=" << graph2.edges.size();
}

}  // namespace

MatchResult MatchGraphs(XLSGraph& graph1, XLSGraph& graph2,
                        const MatchOptions& options) {
  MatchResult result;
  if (options.use_mcs || options.anchor_names) {
    Preprocess(graph1, graph2, options, result);
  }

  LOG(INFO) << "Running GED solver";
  Stopwatch ged_stopwatch;
  result.ged = ged::SolveGED(graph1, graph2, CreateUserCosts());
  result.stats.ged_runtime_sec =
      absl::ToDoubleSeconds(ged_stopwatch.GetElapsedTime());

  // Pinned boundary nodes are substituted with their MCS partner and are
  // already in the map.
  absl::flat_hash_set<std::string> mapped;
  for (const auto& [g1_name, g2_name] : result.node_map) {
    mapped.insert(g1_name);
  }
  for (const auto& [i1, i2] : result.ged.node_substitutions) {
    const std::string& g1_name = graph1.nodes[i1].name;
    if (mapped.insert(g1_name).second) {
      result.node_map.emplace_back(g1_name, graph2.nodes[i2].name);
    }
  }
  return result;
}

}  // namespace xls
