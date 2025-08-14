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

#ifndef XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_CONFIG_H_
#define XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_CONFIG_H_

#include "xls/contrib/mlir/transforms/linalg/linalg_types.h"

namespace mlir::xls {

// Manager for LinalgCodegenConfig singleton instance.
class LinalgCodegenConfigManager {
 public:
  // Returns the singleton manager instance.
  static LinalgCodegenConfigManager& GetLinalgCodegenConfigManagerSingleton() {
    static LinalgCodegenConfigManager instance;
    return instance;
  }

  // Get the current configuration.
  LinalgCodegenConfig& GetConfig() { return config_; }
  const LinalgCodegenConfig& GetConfig() const { return config_; }
  
  // Set the configuration.
  void SetConfig(const LinalgCodegenConfig& config) { config_ = config; }

 private:
  LinalgCodegenConfig config_;
  LinalgCodegenConfigManager() = default;
};

// Convenience functions for accessing the global configuration.
inline LinalgCodegenConfig& GetLinalgCodegenConfig() {
  return LinalgCodegenConfigManager::GetLinalgCodegenConfigManagerSingleton().GetConfig();
}

inline void SetLinalgCodegenConfig(const LinalgCodegenConfig& config) {
  LinalgCodegenConfigManager::GetLinalgCodegenConfigManagerSingleton().SetConfig(config);
}

// Convenience functions for setting individual config options
inline void SetLinalgCodegenConfig(
    LinalgCodegenConfig::LoopOrder loop_order,
    LinalgCodegenConfig::ReducePolicy reduce_policy,
    bool hoist_invariants = false,
    bool enable_tiling = false) {
  LinalgCodegenConfig config;
  config.loop_order(loop_order);
  config.reduce_policy(reduce_policy);
  config.hoist_invariants(hoist_invariants);
  config.enable_tiling(enable_tiling);
  SetLinalgCodegenConfig(config);
}

// Overload for setting just loop order and reduce policy
inline void SetLinalgCodegenConfig(
    LinalgCodegenConfig::LoopOrder loop_order,
    LinalgCodegenConfig::ReducePolicy reduce_policy) {
  SetLinalgCodegenConfig(loop_order, reduce_policy, false, false);
}

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TRANSFORMS_LINALG_LINALG_CONFIG_H_
