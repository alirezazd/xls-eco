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

#include "xls/contrib/mlir/transforms/linalg/linalg_to_xls.h"

#include <cstdint>
#include <sstream>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/contrib/mlir/transforms/linalg/dslx_codegen.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_config.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_eval.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_types.h"
#include "xls/contrib/mlir/transforms/linalg/reduction.h"
#include "xls/contrib/mlir/transforms/linalg/schedule_gen.h"
#include "xls/contrib/mlir/transforms/linalg/unroll_plan_gen.h"

#define GEN_PASS_DEF_LINALGTOXLSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace mlir::xls {

struct LinalgElementwiseToArithPattern
    : public OpRewritePattern<linalg::GenericOp> {
  LinalgElementwiseToArithPattern(MLIRContext* context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/20) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter& rewriter) const override {
    auto iteratorTypes = op.getIteratorTypesArray();
    bool allParallel =
        llvm::all_of(iteratorTypes, [](utils::IteratorType iter) {
          return iter == utils::IteratorType::parallel;
        });

    if (!allParallel) {
      op.emitError(
          "LinalgElementwiseToArithPattern: operation must have all parallel "
          "iterator types");
      return failure();
    }

    if (op.getOutputs().size() != 1) {
      op.emitError(
          "LinalgElementwiseToArithPattern: operation must have exactly one "
          "output");
      return failure();
    }

    auto inputs = op.getInputs();
    if (inputs.size() < 2) {
      op.emitError(
          "LinalgElementwiseToArithPattern: operation must have at least 2 "
          "inputs");
      return failure();
    }

    bool has_tensor = false;
    for (auto input : inputs) {
      if (llvm::isa<RankedTensorType>(input.getType())) {
        has_tensor = true;
        break;
      }
    }

    if (!has_tensor) {
      op.emitError(
          "LinalgElementwiseToArithPattern: operation must have at least one "
          "tensor input");
      return failure();
    }

    mlir::Region& bodyRegion = op.getRegion();
    mlir::Block& bodyBlock = bodyRegion.front();

    Operation* arithOp = nullptr;
    int arithOpCount = 0;

    for (Operation& bodyOp : bodyBlock) {
      if (isa<arith::AddFOp>(bodyOp) || isa<arith::MulFOp>(bodyOp) ||
          isa<arith::SubFOp>(bodyOp)) {
        arithOpCount++;
        if (arithOpCount == 1) {
          arithOp = &bodyOp;
        } else {
          arithOp = nullptr;
          break;
        }
      }
    }

    if (arithOpCount != 1) {
      op.emitError(
          "LinalgElementwiseToArithPattern: operation must have exactly one "
          "arithmetic operation (add, mul, or sub)");
      return failure();
    }

    if (!arithOp || !bodyBlock.getTerminator() ||
        !isa<linalg::YieldOp>(bodyBlock.getTerminator())) {
      op.emitError(
          "LinalgElementwiseToArithPattern: operation must have a valid "
          "terminator (linalg.yield)");
      return failure();
    }

    Value lhs, rhs;
    if (isa<arith::AddFOp>(arithOp)) {
      auto addOp = cast<arith::AddFOp>(arithOp);
      lhs = addOp.getLhs();
      rhs = addOp.getRhs();
    } else if (isa<arith::MulFOp>(arithOp)) {
      auto mulOp = cast<arith::MulFOp>(arithOp);
      lhs = mulOp.getLhs();
      rhs = mulOp.getRhs();
    } else if (isa<arith::SubFOp>(arithOp)) {
      auto subOp = cast<arith::SubFOp>(arithOp);
      lhs = subOp.getLhs();
      rhs = subOp.getRhs();
    } else {
      op.emitError(
          "LinalgElementwiseToArithPattern: unsupported arithmetic operation "
          "type");
      return failure();
    }

    auto yieldOp = cast<linalg::YieldOp>(bodyBlock.getTerminator());
    Value yieldedValue = yieldOp.getValues()[0];

    if (yieldedValue != arithOp->getResult(0)) {
      op.emitError(
          "LinalgElementwiseToArithPattern: yielded value must be the result "
          "of the arithmetic operation");
      return failure();
    }

    auto resultType =
        llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType) {
      op.emitError(
          "LinalgElementwiseToArithPattern: operation result must be a ranked "
          "tensor type");
      return failure();
    }

    auto loc = op.getLoc();

    if (isa<arith::AddFOp>(arithOp)) {
      auto addOp = rewriter.create<arith::AddFOp>(loc, lhs, rhs);
      rewriter.replaceOp(op, addOp);
    } else if (isa<arith::MulFOp>(arithOp)) {
      auto mulOp = rewriter.create<arith::MulFOp>(loc, lhs, rhs);
      rewriter.replaceOp(op, mulOp);
    } else if (isa<arith::SubFOp>(arithOp)) {
      auto subOp = rewriter.create<arith::SubFOp>(loc, lhs, rhs);
      rewriter.replaceOp(op, subOp);
    }

    return success();
  }
};

struct LinalgFillToTensorSplatPattern
    : public OpRewritePattern<linalg::GenericOp> {
  LinalgFillToTensorSplatPattern(MLIRContext* context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/25) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.getInputs().empty() || op.getOutputs().size() != 1) {
      op.emitError(
          "LinalgFillToTensorSplatPattern: operation must have no inputs and "
          "exactly one output");
      return failure();
    }

    auto iteratorTypes = op.getIteratorTypesArray();
    bool allParallel =
        llvm::all_of(iteratorTypes, [](utils::IteratorType iter) {
          return iter == utils::IteratorType::parallel;
        });

    if (!allParallel) {
      op.emitError(
          "LinalgFillToTensorSplatPattern: operation must have all parallel "
          "iterator types");
      return failure();
    }

    mlir::Region& bodyRegion = op.getRegion();
    if (bodyRegion.empty()) {
      op.emitError(
          "LinalgFillToTensorSplatPattern: operation must have a non-empty "
          "region");
      return failure();
    }

    mlir::Block& bodyBlock = bodyRegion.front();

    if (!bodyBlock.getTerminator() ||
        !isa<linalg::YieldOp>(bodyBlock.getTerminator())) {
      op.emitError(
          "LinalgFillToTensorSplatPattern: operation must have a valid "
          "terminator (linalg.yield)");
      return failure();
    }

    auto yieldOp = cast<linalg::YieldOp>(bodyBlock.getTerminator());
    if (yieldOp.getValues().size() != 1) {
      op.emitError(
          "LinalgFillToTensorSplatPattern: yield operation must have exactly "
          "one value");
      return failure();
    }

    Value yieldedValue = yieldOp.getValues()[0];
    auto constOp = yieldedValue.getDefiningOp<arith::ConstantOp>();
    if (!constOp) {
      op.emitError(
          "LinalgFillToTensorSplatPattern: yielded value must be a constant "
          "operation");
      return failure();
    }

    auto resultType =
        llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType) {
      op.emitError(
          "LinalgFillToTensorSplatPattern: operation result must be a ranked "
          "tensor type");
      return failure();
    }

    auto constValueAttr = constOp.getValue();
    auto loc = op.getLoc();

    auto scalarConstant =
        rewriter.create<arith::ConstantOp>(loc, constValueAttr);
    auto splatTensor =
        rewriter.create<tensor::SplatOp>(loc, scalarConstant, resultType);

    rewriter.replaceOp(op, splatTensor);
    return success();
  }
};

struct LinalgReductionToXlsPattern
    : public OpRewritePattern<linalg::GenericOp> {
  LinalgReductionToXlsPattern(MLIRContext* context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/30) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter& rewriter) const override {
    auto eval_res = EvalLinalgGeneric(op);
    if (failed(eval_res)) {
      return failure();
    }

    // Set global configuration with convenient interface
    SetLinalgCodegenConfig(LinalgCodegenConfig::LoopOrder::kPreserve,
                           LinalgCodegenConfig::ReducePolicy::kSerial,
                           false,  // hoist_invariants
                           false   // enable_tiling
    );

    auto sch = GenerateSchedule(*eval_res);
    if (failed(sch)) {
      op.emitError(
          "LinalgReductionToXlsPattern: failed to generate schedule for "
          "reduction operation");
      return failure();
    }

    UnrollPlan up;
    if (failed(GenerateUnrollPlan(eval_res->linalg, eval_res->broadcast, up))) {
      return failure();
    }

    std::stringstream dslx_output;
    DslxCodegen codegen(dslx_output);

    if (failed(codegen.EmitFunction(eval_res->linalg, eval_res->broadcast, *sch,
                                    up))) {
      return failure();
    }

    return failure();
  }
};

struct LinalgToXlsPass : public ::impl::LinalgToXlsPassBase<LinalgToXlsPass> {
  using LinalgToXlsPassBase::LinalgToXlsPassBase;

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);

    populateLinalgToXlsPatterns(patterns, ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect,
                           xls::XlsDialect>();
    target.addIllegalOp<linalg::GenericOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return;
    }
  }
};

}  // namespace mlir::xls

namespace mlir::xls {

std::unique_ptr<Pass> createLinalgToXlsPass() {
  return std::make_unique<LinalgToXlsPass>();
}

}  // namespace mlir::xls

namespace mlir {

void populateLinalgToXlsPatterns(RewritePatternSet& patterns,
                                 MLIRContext* context) {
  patterns.add<mlir::xls::LinalgElementwiseToArithPattern>(context);
  patterns.add<mlir::xls::LinalgFillToTensorSplatPattern>(context);
  patterns.add<mlir::xls::LinalgReductionToXlsPattern>(context);
}

}  // namespace mlir
