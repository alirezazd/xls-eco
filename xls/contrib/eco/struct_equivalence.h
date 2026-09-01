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

#ifndef XLS_ECO_STRUCT_EQUIVALENCE_H_
#define XLS_ECO_STRUCT_EQUIVALENCE_H_

#include <cstddef>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "xls/contrib/eco/graph.h"

namespace xls {

// Names of the topologically minimal divergent nodes of `graph`: nodes whose
// subtree hash (`own`, the per-node prefix of XLSGraph::StructuralHash) has
// no counterpart in the other graph's hashes (`other`) while all of their
// operands' hashes do. Merkle divergence propagates downstream, so these are
// the roots of a structural mismatch.
std::vector<std::string> MismatchRoots(const XLSGraph& graph,
                                       absl::Span<const std::size_t> own,
                                       absl::Span<const std::size_t> other);

}  // namespace xls

#endif  // XLS_ECO_STRUCT_EQUIVALENCE_H_
