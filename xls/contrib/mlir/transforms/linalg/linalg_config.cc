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

#include "xls/contrib/mlir/transforms/linalg/linalg_config.h"

#include "xls/contrib/mlir/transforms/linalg/schedule_types.h"

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
LinalgCodegenConfig& GetLinalgCodegenConfig() {
  return LinalgCodegenConfigManager::GetLinalgCodegenConfigManagerSingleton().GetConfig();
}

void SetLinalgCodegenConfig(const LinalgCodegenConfig& config) {
  LinalgCodegenConfigManager::GetLinalgCodegenConfigManagerSingleton().SetConfig(config);
}

// Convenience function for setting loop order
void SetLinalgCodegenConfig(LinalgCodegenConfig::LoopOrder loop_order) {
  LinalgCodegenConfig config;
  config.loop_order(loop_order);
  SetLinalgCodegenConfig(config);
}

}  // namespace mlir::xls
