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

#include "xls/contrib/eco/struct_equivalence.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xls/contrib/eco/graph.h"

namespace xls {

std::vector<std::string> MismatchRoots(const XLSGraph& graph,
                                       absl::Span<const std::size_t> own,
                                       absl::Span<const std::size_t> other) {
  absl::flat_hash_map<std::size_t, int> budget;
  for (std::size_t h : other) {
    ++budget[h];
  }
  const int n = static_cast<int>(own.size());
  std::vector<bool> divergent(n, false);
  for (int i = 0; i < n; ++i) {
    auto it = budget.find(own[i]);
    if (it != budget.end() && it->second > 0) {
      --it->second;
    } else {
      divergent[i] = true;
    }
  }
  std::vector<std::string> roots;
  for (int i = 0; i < n; ++i) {
    if (!divergent[i]) {
      continue;
    }
    bool minimal = true;
    for (int pred : graph.get_incoming_neighbors(i)) {
      if (divergent[pred]) {
        minimal = false;
        break;
      }
    }
    if (minimal) {
      roots.push_back(graph.nodes[i].name);
    }
  }
  return roots;
}

}  // namespace xls
