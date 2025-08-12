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
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/contrib/mlir/transforms/linalg/types/linalg_types.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/linalg_analysis.h"

#define GEN_PASS_DEF_LINALGTOXLSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace mlir::xls {

struct LinalgElementwiseToArithPattern : public OpRewritePattern<linalg::GenericOp> {
  LinalgElementwiseToArithPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/20) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto iteratorTypes = op.getIteratorTypesArray();
    bool allParallel = llvm::all_of(iteratorTypes, [](utils::IteratorType iter) {
      return iter == utils::IteratorType::parallel;
    });

    if (!allParallel) {
      return failure();
    }

    if (op.getOutputs().size() != 1) {
      return failure();
    }

    auto inputs = op.getInputs();
    if (inputs.size() < 2) {
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
      return failure();
    }

    mlir::Region &bodyRegion = op.getRegion();
    mlir::Block &bodyBlock = bodyRegion.front();

    Operation* arithOp = nullptr;
    int arithOpCount = 0;

    for (Operation &bodyOp : bodyBlock) {
      if (isa<arith::AddFOp>(bodyOp) || isa<arith::MulFOp>(bodyOp) || isa<arith::SubFOp>(bodyOp)) {
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
      return failure();
    }

    if (!arithOp || !bodyBlock.getTerminator() || !isa<linalg::YieldOp>(bodyBlock.getTerminator())) {
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
      return failure();
    }

    auto loc = op.getLoc();

    Value tensorOperand, scalarOperand;
    bool need_scalar_to_tensor = false;

    auto findOperand = [&](Value value) -> Value {
      for (size_t i = 0; i < inputs.size(); i++) {
        if (value == bodyBlock.getArgument(i)) {
          return inputs[i];
        }
      }
      return nullptr;
    };

    Value input1 = findOperand(lhs);
    Value input2 = findOperand(rhs);

    if (!input1 || !input2) {
      return failure();
    }

    bool input1_is_tensor = llvm::isa<RankedTensorType>(input1.getType());
    bool input2_is_tensor = llvm::isa<RankedTensorType>(input2.getType());

    if (input1_is_tensor && !input2_is_tensor) {
      tensorOperand = input1;
      scalarOperand = input2;
      need_scalar_to_tensor = true;
    } else if (!input1_is_tensor && input2_is_tensor) {
      tensorOperand = input2;
      scalarOperand = input1;
      need_scalar_to_tensor = true;
    } else {
      tensorOperand = input1;
      scalarOperand = input2;
    }

    if (need_scalar_to_tensor) {
      auto tensorType = llvm::dyn_cast<RankedTensorType>(tensorOperand.getType());
      if (!tensorType) {
        return failure();
      }
      scalarOperand = rewriter.create<tensor::SplatOp>(loc, scalarOperand, tensorType);
    }

    Value result;
    if (isa<arith::AddFOp>(arithOp)) {
      result = rewriter.create<arith::AddFOp>(loc, tensorOperand, scalarOperand);
    } else if (isa<arith::MulFOp>(arithOp)) {
      result = rewriter.create<arith::MulFOp>(loc, tensorOperand, scalarOperand);
    } else if (isa<arith::SubFOp>(arithOp)) {
      result = rewriter.create<arith::SubFOp>(loc, tensorOperand, scalarOperand);
    } else {
      return failure();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LinalgFillToTensorSplatPattern : public OpRewritePattern<linalg::GenericOp> {
  LinalgFillToTensorSplatPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/25) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getInputs().empty() || op.getOutputs().size() != 1) {
      return failure();
    }

    auto iteratorTypes = op.getIteratorTypesArray();
    bool allParallel = llvm::all_of(iteratorTypes, [](utils::IteratorType iter) {
      return iter == utils::IteratorType::parallel;
    });

    if (!allParallel) {
      return failure();
    }

    mlir::Region &bodyRegion = op.getRegion();
    if (bodyRegion.empty()) {
      return failure();
    }

    mlir::Block &bodyBlock = bodyRegion.front();

    if (!bodyBlock.getTerminator() || !isa<linalg::YieldOp>(bodyBlock.getTerminator())) {
      return failure();
    }

    auto yieldOp = cast<linalg::YieldOp>(bodyBlock.getTerminator());
    if (yieldOp.getValues().size() != 1) {
      return failure();
    }

    Value yieldedValue = yieldOp.getValues()[0];
    auto constOp = yieldedValue.getDefiningOp<arith::ConstantOp>();
    if (!constOp) {
      return failure();
    }

    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType) {
      return failure();
    }

    auto constValueAttr = constOp.getValue();
    auto loc = op.getLoc();

    auto scalarConstant = rewriter.create<arith::ConstantOp>(loc, constValueAttr);
    auto splatTensor = rewriter.create<tensor::SplatOp>(loc, scalarConstant, resultType);

    rewriter.replaceOp(op, splatTensor);
    return success();
  }
};



struct LinalgToXlsPass : public ::impl::LinalgToXlsPassBase<LinalgToXlsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    populateLinalgToXlsPatterns(patterns, ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect, xls::XlsDialect>();
    target.addIllegalOp<linalg::GenericOp>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
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
}

}  // namespace mlir
