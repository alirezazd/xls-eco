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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_UTILS_REDUCTION_UTILS_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_UTILS_REDUCTION_UTILS_H_

#include "absl/types/span.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"

namespace mlir::xls {

// Low-level reduction algorithms
inline void ReduceSerial(AccState& st, float x) {
  st.acc += x;
}

inline void ReduceKahan(AccState& st, float x) {
  float y = x - st.c;
  float t = st.acc + y;
  st.c = (t - st.acc) - y;
  st.acc = t;
}

float ReduceTreeFixed(absl::Span<const float> vals, int fan_in = 2);

inline void ReduceUpdate(AccState& st,
                         absl::Span<const float> lane_vals,
                         const ReduceCtx& ctx) {
  switch (ctx.policy) {
    case ReducePolicy::kSerial: {
      float s = 0.0f;
      for (float v : lane_vals) s += v;
      st.acc += s;
      break;
    }
    case ReducePolicy::kKahan: {
      for (float v : lane_vals) ReduceKahan(st, v);
      break;
    }
    case ReducePolicy::kTree: {
      float s = ReduceTreeFixed(lane_vals, ctx.tree_fan_in);
      st.acc += s;
      break;
    }
  }
}

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_UTILS_REDUCTION_UTILS_H_
