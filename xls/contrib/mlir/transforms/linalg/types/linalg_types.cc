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

#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"

namespace mlir::xls {

int MultiLanePlan::ComputeLaneId(const std::vector<int>& lane_offsets) const {
  int lane_id = 0;
  int stride = 1;
  
  for (int i = dims.size() - 1; i >= 0 && i < lane_offsets.size(); --i) {
    lane_id += lane_offsets[i] * stride;
    stride *= dims[i].lanes;
  }
  
  return lane_id;
}

std::vector<int> MultiLanePlan::ComputeLaneOffsets(int lane_id) const {
  std::vector<int> offsets;
  offsets.reserve(dims.size());
  
  int remaining = lane_id;
  
  for (int i = dims.size() - 1; i >= 0; --i) {
    int lanes = dims[i].lanes;
    offsets.insert(offsets.begin(), remaining % lanes);
    remaining /= lanes;
  }
  
  return offsets;
}

}  // namespace mlir::xls
