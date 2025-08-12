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

#ifndef XLS_CONTRIB_MLIR_TESTDATA_LINALG_TEST_UTILS_H_
#define XLS_CONTRIB_MLIR_TESTDATA_LINALG_TEST_UTILS_H_

#include <iostream>
#include <string>

#include "xls/contrib/mlir/transforms/linalg/analysis/broadcast.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"

namespace mlir::xls {
namespace test {

// Test result reporting
struct TestResult {
  bool passed;
  std::string message;
  
  TestResult(bool p, std::string m) : passed(p), message(std::move(m)) {}
  
  static TestResult Pass(std::string message = "") {
    return TestResult(true, std::move(message));
  }
  
  static TestResult Fail(std::string message) {
    return TestResult(false, std::move(message));
  }
};

// Test fixture for linalg analysis
class LinalgTestFixture {
 public:
  // Create a 4D worst-case example with mixed indexing patterns
  static LinalgGeneric Create4DWorstCaseExample();
  
  // Create broadcast mismatch example
  static LinalgGeneric CreateBroadcastMismatchExample();
  
  // Create operand omits dim example
  static LinalgGeneric CreateOperandOmitsDimExample();
  
  // Create dynamic symbolic example
  static LinalgGeneric CreateDynamicSymbolicExample();
  
  // Create output shape error example
  static LinalgGeneric CreateOutputShapeErrorExample();
  
  // Create test broadcast analysis
  static BroadcastAnalysis CreateTestBroadcastAnalysis();
  
  // Print test results
  static void PrintTestResult(const std::string& test_name, const TestResult& result);
  
  // Print size expression
  static void PrintSizeExpr(const SizeExpr& size);
  
  // Print operand varies matrix
  static void PrintOperandVaries(const std::vector<std::vector<bool>>& operand_varies);
  
  // Print indexers
  static void PrintIndexers(const std::vector<OperandIndexFn>& indexers);
};

// Test runner utilities
class TestRunner {
 public:
  static void RunTest(const std::string& test_name, 
                     std::function<TestResult()> test_func);
  
  static void PrintTestSummary();
  
 private:
  static int total_tests_;
  static int passed_tests_;
  static int failed_tests_;
};

}  // namespace test
}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TESTDATA_LINALG_TEST_UTILS_H_
