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

#ifndef XLS_CONTRIB_ECO_MATCH_H_
#define XLS_CONTRIB_ECO_MATCH_H_

#include <string>
#include <utility>
#include <vector>

#include "xls/contrib/eco/ged.h"
#include "xls/contrib/eco/graph.h"

namespace xls {

struct MatchOptions {
  bool use_mcs = true;
  // Pair nodes that share name and label instead of searching for the MCS:
  // exact for a patched IR against its original, where names survive.
  bool anchor_names = false;
  // See mcs::SolveMCS.
  int mcs_cutoff = -1;
  bool mcs_optimal = true;
  int mcs_timeout = -1;
};

struct MatchStats {
  double mcs_runtime_sec = 0.0;
  int mcs_matched_nodes = 0;
  int mcs_matched_edges = 0;
  int mcs_boundary_nodes = 0;
  int mcs_boundary_edges = 0;
  double ged_runtime_sec = 0.0;
};

struct MatchResult {
  // Every (before, after) node pair: the MCS mapping plus GED substitutions.
  std::vector<std::pair<std::string, std::string>> node_map;
  // Edit path over the residual graphs; indices refer to the cut graphs.
  ged::GEDResult ged;
  MatchStats stats;
};

// Matches the nodes of two graphs: MCS preprocessing pins the boundary of the
// maximum common subgraph and cuts its interior, then GED computes an edit path
// over what remains. On return the graphs are the residual graphs.
MatchResult MatchGraphs(XLSGraph& graph1, XLSGraph& graph2,
                        const MatchOptions& options);

}  // namespace xls

#endif  // XLS_CONTRIB_ECO_MATCH_H_
