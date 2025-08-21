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
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/contrib/mlir/transforms/linalg/dslxgen.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_config.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_eval.h"
#include "xls/contrib/mlir/transforms/linalg/linalg_types.h"
// #include "xls/contrib/mlir/transforms/linalg/reduction.h"  // REMOVED - using XLS built-in capabilities
#include "xls/contrib/mlir/transforms/linalg/schedule_gen.h"
// #include "xls/contrib/mlir/transforms/linalg/unroll_plan_gen.h"  // REMOVED - using XLS built-in unrolling

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
    // Step 0: Check that this is actually a reduction operation
    auto iteratorTypes = op.getIteratorTypesArray();
    bool hasReduction = llvm::any_of(iteratorTypes, [](utils::IteratorType iter) {
      return iter == utils::IteratorType::reduction;
    });
    
    if (!hasReduction) {
      return failure(); // Not a reduction, let other patterns handle it
    }

    // Step 1: Check that all output operands are zero-initialized splat tensors
    auto outputs = op.getOutputs();
    for (auto output : outputs) {
      auto splatOp = output.getDefiningOp<tensor::SplatOp>();
      if (!splatOp) {
        op.emitError(
            "LinalgReductionToXlsPattern: output operand must be a tensor.splat operation");
        return failure();
      }
      
      auto constOp = splatOp.getInput().getDefiningOp<arith::ConstantOp>();
      if (!constOp) {
        op.emitError(
            "LinalgReductionToXlsPattern: splat input must be an arith.constant operation");
        return failure();
      }
      
      // Check that the constant is numeric (int or float)
      auto constValue = constOp.getValue();
      if (!llvm::isa<mlir::FloatAttr>(constValue) && !llvm::isa<mlir::IntegerAttr>(constValue)) {
        op.emitError(
            "LinalgReductionToXlsPattern: constant must be a numeric value (int or float)");
        return failure();
      }
      
      // Temporarily restrict to zero initialization only
      bool isZero = false;
      if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(constValue)) {
        isZero = floatAttr.getValueAsDouble() == 0.0;
      } else if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constValue)) {
        isZero = intAttr.getValue().isZero();
      }
      
      if (!isZero) {
        op.emitError(
            "LinalgReductionToXlsPattern: currently only zero initialization is supported");
        return failure();
      }
    }

    // Step 1: Evaluate the linalg.generic operation
    auto eval_res = EvalLinalgGeneric(op);
    if (failed(eval_res)) {
      return failure();
    }



    // Step 2: Set global configuration with convenient interface
    SetLinalgCodegenConfig(LinalgCodegenConfig::LoopOrder::kHeuristicCacheFriendly);

    // Step 3: Generate schedule
    auto schedule = GenerateSchedule(*eval_res);
    if (failed(schedule)) {
      op.emitError(
          "LinalgReductionToXlsPattern: failed to generate schedule for "
          "reduction operation");
      return failure();
    }

    // Step 4: Generate DSLX code (let XLS handle all unrolling automatically)
    DslxGen codegen;
    auto dslx_code_result = codegen.GenerateDslxCode(*eval_res, *schedule);
    if (failed(dslx_code_result)) {
      op.emitError("LinalgReductionToXlsPattern: failed to generate DSLX code");
      return failure();
    }
    std::string dslx_code = *dslx_code_result;

    // Step 6: Write DSLX code to file (optional, for debugging)
    // TODO: Make this configurable or remove in production
    std::ofstream dslx_file("/home/the1/Gits/xls-eco/xls/contrib/mlir/testdata/tmp/generated_dslx.x");
    if (dslx_file.is_open()) {
      dslx_file << dslx_code;
      dslx_file.close();
      llvm::errs() << "DSLX code written to /home/the1/Gits/xls-eco/xls/contrib/mlir/testdata/tmp/generated_dslx.x\n";
    } else {
      llvm::errs() << "Failed to open /home/the1/Gits/xls-eco/xls/contrib/mlir/testdata/tmp/generated_dslx.x for writing\n";
    }

    // Step 7: Replace the linalg.generic operation with a placeholder
    // For now, we replace with a zero tensor since we only generate function
    // headers
    auto resultType = op.getResult(0).getType();
    auto zeroConst = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getFloatAttr(rewriter.getF32Type(), 0.0));
    auto zeroTensor =
        rewriter.create<tensor::SplatOp>(op.getLoc(), zeroConst, resultType);

    rewriter.replaceOp(op, zeroTensor);
    return success();
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
