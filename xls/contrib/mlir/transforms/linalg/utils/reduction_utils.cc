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

#include "xls/contrib/mlir/transforms/linalg/utils/reduction_utils.h"

#include <vector>

namespace mlir::xls {

float ReduceTreeFixed(absl::Span<const float> vals, int fan_in) {
  std::vector<float> cur(vals.begin(), vals.end());
  while (cur.size() > 1) {
    std::vector<float> nxt;
    nxt.reserve((cur.size() + fan_in - 1) / fan_in);
    for (size_t i = 0; i < cur.size(); i += fan_in) {
      float s = 0.0f;
      for (int k = 0; k < fan_in && (i + k) < cur.size(); ++k) s += cur[i + k];
      nxt.push_back(s);
    }
    cur.swap(nxt);
  }
  return cur[0];
}

}  // namespace mlir::xls
