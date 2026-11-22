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

#ifndef XLS_CONTRIB_ECO_ALIGN_H_
#define XLS_CONTRIB_ECO_ALIGN_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/contrib/eco/match.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

struct Alignment {
  int64_t original_nodes = 0;
  int64_t nodes = 0;
  absl::flat_hash_set<Node*> matched;  // Nodes that took an original id.
};

// Renumbers `function_base` so that every node the match pairs with a node of
// `original` carries that node's id, and its name if it had none; unmatched
// nodes get fresh ids above both packages. Codegen then emits the same
// identifiers for logic the ECO did not touch, and a schedule of the original
// maps onto the function base by id.
absl::StatusOr<Alignment> Align(FunctionBase* function_base,
                                FunctionBase* original,
                                const MatchOptions& options);

}  // namespace xls

#endif  // XLS_CONTRIB_ECO_ALIGN_H_
