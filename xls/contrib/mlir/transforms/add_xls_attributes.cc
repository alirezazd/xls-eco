// Copyright 2024 The XLS Authors
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

#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::xls {

#define GEN_PASS_DEF_ADDXLSATTRIBUTESPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

class AddXlsAttributesPass
    : public impl::AddXlsAttributesPassBase<AddXlsAttributesPass> {
 public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool modified = false;

    module.walk([&](mlir::func::FuncOp funcOp) {
      if (!funcOp->hasAttr("xls")) {
        funcOp->setAttr("xls", mlir::BoolAttr::get(&getContext(), true));
        modified = true;
      }
    });

    if (modified) {
      markAllAnalysesPreserved();
    }
  }
};

}  // namespace

}  // namespace mlir::xls
