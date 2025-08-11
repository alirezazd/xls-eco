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

#include "xls/contrib/mlir/transforms/linalg_to_xls.h"

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

// Get auto-generated pass definitions
#define GEN_PASS_DEF_LINALGTOXLSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace mlir::xls {



// Converts element-wise linalg.generic operations to arith operations
struct LinalgElementwiseToArithPattern : public OpRewritePattern<linalg::GenericOp> {
  LinalgElementwiseToArithPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle element-wise operations (all parallel iterators)
    auto iteratorTypes = op.getIteratorTypesArray();
    bool allParallel = llvm::all_of(iteratorTypes, [](utils::IteratorType iter) {
      return iter == utils::IteratorType::parallel;
    });
    
    if (!allParallel) {
      return failure();
    }

    // Check if it's a simple element-wise operation with one input and one output
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) {
      return failure();
    }

    // Analyze the body to see if it's a simple arithmetic operation
    Region &bodyRegion = op.getRegion();
    Block &bodyBlock = bodyRegion.front();
    
    // Look for: input + constant -> yield pattern
    Operation* arithOp = nullptr;
    Operation* yieldOp = nullptr;
    Value inputArg = nullptr;
    Value constantValue = nullptr;
    
    // Get the input argument (first argument)
    if (bodyBlock.getNumArguments() >= 2) {
      inputArg = bodyBlock.getArgument(0);
    }
    
    for (Operation &bodyOp : bodyBlock) {
      if (isa<arith::AddFOp>(bodyOp) || isa<arith::MulFOp>(bodyOp) || isa<arith::SubFOp>(bodyOp)) {
        arithOp = &bodyOp;
        
        // Check if one operand is the input and the other is a constant
        Value lhs, rhs;
        if (isa<arith::AddFOp>(bodyOp)) {
          auto addOp = cast<arith::AddFOp>(bodyOp);
          lhs = addOp.getLhs();
          rhs = addOp.getRhs();
        } else if (isa<arith::MulFOp>(bodyOp)) {
          auto mulOp = cast<arith::MulFOp>(bodyOp);
          lhs = mulOp.getLhs();
          rhs = mulOp.getRhs();
        } else if (isa<arith::SubFOp>(bodyOp)) {
          auto subOp = cast<arith::SubFOp>(bodyOp);
          lhs = subOp.getLhs();
          rhs = subOp.getRhs();
        }
        
        // Find which is the input and which is the constant
        if (lhs == inputArg) {
          if (isa<arith::ConstantOp>(rhs.getDefiningOp())) {
            constantValue = rhs;
          } else {
            // Try to find the constant in the parent function
            auto parentFunc = op->getParentOfType<func::FuncOp>();
            if (parentFunc) {
              for (auto &parentOp : parentFunc.getBody().front()) {
                if (isa<arith::ConstantOp>(parentOp) && parentOp.getResult(0) == rhs) {
                  constantValue = rhs;
                  break;
                }
              }
            }
          }
        } else if (rhs == inputArg) {
          if (isa<arith::ConstantOp>(lhs.getDefiningOp())) {
            constantValue = lhs;
          } else {
            // Try to find the constant in the parent function
            auto parentFunc = op->getParentOfType<func::FuncOp>();
            if (parentFunc) {
              for (auto &parentOp : parentFunc.getBody().front()) {
                if (isa<arith::ConstantOp>(parentOp) && parentOp.getResult(0) == lhs) {
                  constantValue = lhs;
                  break;
                }
              }
            }
          }
        }
        
        if (constantValue) {
          break;
        }
      }
    }
    
    // Check if there's a terminator (yield operation)
    if (!yieldOp && bodyBlock.getTerminator()) {
      if (isa<linalg::YieldOp>(bodyBlock.getTerminator())) {
        yieldOp = bodyBlock.getTerminator();
      }
    }
    
    if (!arithOp || !yieldOp || !constantValue) {
      return failure();
    }
    
    // Extract the result type (should be the output tensor type)
    auto resultType = llvm::cast<RankedTensorType>(op.getResult(0).getType());
    
    // Get the constant value
    auto constOp = cast<arith::ConstantOp>(constantValue.getDefiningOp());
    auto constValueAttr = constOp.getValue();
    auto loc = op.getLoc();
    
    // Create scalar constant
    auto scalarConstant = rewriter.create<arith::ConstantOp>(loc, constValueAttr);
    
    // Create splat tensor
    auto splatTensor = rewriter.create<tensor::SplatOp>(
        loc, scalarConstant, resultType);
    
    // Get input tensor
    auto inputTensor = op.getInputs()[0];
    
    // Create the appropriate arith operation based on the body operation
    Value result;
    if (isa<arith::AddFOp>(arithOp)) {
      result = rewriter.create<arith::AddFOp>(loc, inputTensor, splatTensor);
    } else if (isa<arith::MulFOp>(arithOp)) {
      result = rewriter.create<arith::MulFOp>(loc, inputTensor, splatTensor);
    } else if (isa<arith::SubFOp>(arithOp)) {
      result = rewriter.create<arith::SubFOp>(loc, inputTensor, splatTensor);
    } else {
      return failure();
    }
    
    // Replace the linalg.generic with the arith operation
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Converts fill linalg.generic operations to tensor.splat
struct LinalgFillToTensorSplatPattern : public OpRewritePattern<linalg::GenericOp> {
  LinalgFillToTensorSplatPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/3) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle fill operations (no inputs, only outputs)
    if (!op.getInputs().empty() || op.getOutputs().size() != 1) {
      return failure();
    }

    // Check if it's a simple fill operation with all parallel iterators
    auto iteratorTypes = op.getIteratorTypesArray();
    bool allParallel = llvm::all_of(iteratorTypes, [](utils::IteratorType iter) {
      return iter == utils::IteratorType::parallel;
    });
    
    if (!allParallel) {
      return failure();
    }

    // Analyze the body to see if it's a simple constant fill
    Region &bodyRegion = op.getRegion();
    if (bodyRegion.empty()) {
      return failure();
    }
    
    Block &bodyBlock = bodyRegion.front();
    
    // Check if there's a terminator (yield operation)
    if (!bodyBlock.getTerminator() || !isa<linalg::YieldOp>(bodyBlock.getTerminator())) {
      return failure();
    }
    
    // Get the yielded value
    auto yieldOp = cast<linalg::YieldOp>(bodyBlock.getTerminator());
    if (yieldOp.getValues().size() != 1) {
      return failure();
    }
    
    Value yieldedValue = yieldOp.getValues()[0];
    
    // Check if the yielded value is a constant
    auto constOp = yieldedValue.getDefiningOp<arith::ConstantOp>();
    if (!constOp) {
      return failure();
    }
    
    // Extract the result type (should be the output tensor type)
    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType) {
      return failure();
    }
    
    // Get the constant value
    auto constValueAttr = constOp.getValue();
    auto loc = op.getLoc();
    
    // Create scalar constant
    auto scalarConstant = rewriter.create<arith::ConstantOp>(loc, constValueAttr);
    
    // Create tensor.splat operation
    auto splatTensor = rewriter.create<tensor::SplatOp>(
        loc, scalarConstant, resultType);
    
    // Replace the linalg.generic with the tensor.splat operation
    rewriter.replaceOp(op, splatTensor);
    return success();
  }
};

// Unrolls linalg.generic operations into individual tensor operations
struct LinalgGenericUnrollPattern : public OpRewritePattern<linalg::GenericOp> {
  LinalgGenericUnrollPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle element-wise operations (all parallel iterators)
    auto iteratorTypes = op.getIteratorTypesArray();
    bool allParallel = llvm::all_of(iteratorTypes, [](utils::IteratorType it) {
      return it == utils::IteratorType::parallel;
    });
    if (!allParallel) {
      return failure();
    }

    // Get the body block
    Block &bodyBlock = op.getRegion().front();
    
    // Check if the body has at least one operation (plus the terminator)
    if (bodyBlock.getOperations().size() < 2) {
      return failure();
    }

    // Extract all operations from the body and create tensor versions
    return unrollGenericOp(op, bodyBlock, rewriter);
  }

private:
  LogicalResult unrollGenericOp(linalg::GenericOp op, Block &bodyBlock,
                                PatternRewriter &rewriter) const {
    // Handle multiple operations by creating a sequence of tensor operations
    SmallVector<Operation*> bodyOps;
    
    // Collect all operations except the terminator
    for (Operation &bodyOp : bodyBlock.getOperations()) {
      if (!isa<linalg::YieldOp>(bodyOp)) {
        bodyOps.push_back(&bodyOp);
      }
    }
    
    if (bodyOps.empty()) {
      return failure();
    }
    
    // If there's only one operation, handle it directly
    if (bodyOps.size() == 1) {
      return handleSingleOperation(op, bodyOps[0], rewriter);
    }
    
    // Handle multiple operations by creating a sequence
    return handleMultipleOperations(op, bodyOps, rewriter);
  }

  LogicalResult handleSingleOperation(linalg::GenericOp op, Operation *bodyOp,
                                     PatternRewriter &rewriter) const {
    if (auto addOp = dyn_cast<arith::AddFOp>(bodyOp)) {
      return createTensorBinaryOp<arith::AddFOp>(op, addOp, rewriter);
    } else if (auto mulOp = dyn_cast<arith::MulFOp>(bodyOp)) {
      return createTensorBinaryOp<arith::MulFOp>(op, mulOp, rewriter);
    } else if (auto subOp = dyn_cast<arith::SubFOp>(bodyOp)) {
      return createTensorBinaryOp<arith::SubFOp>(op, subOp, rewriter);
    } else if (auto cmpOp = dyn_cast<arith::CmpFOp>(bodyOp)) {
      return createTensorCmpOp(op, cmpOp, rewriter);
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(bodyOp)) {
      return createTensorSelectOp(op, selectOp, rewriter);
    }

    return failure();
  }

  LogicalResult handleMultipleOperations(linalg::GenericOp op, 
                                        SmallVector<Operation*> &bodyOps,
                                        PatternRewriter &rewriter) const {
    // Handle arbitrary sequences of operations by creating a sequence of tensor operations
    return createArbitrarySequence(op, bodyOps, rewriter);
  }

  template<typename OpType>
  LogicalResult createTensorBinaryOp(linalg::GenericOp op, OpType binOp,
                                    PatternRewriter &rewriter) const {
    // Verify we have exactly 2 inputs for binary operation
    if (op.getInputs().size() != 2) {
      return failure();
    }

    // Verify the operands are block arguments
    Value lhs = binOp.getLhs();
    Value rhs = binOp.getRhs();
    
    if (!isa<BlockArgument>(lhs) || !isa<BlockArgument>(rhs)) {
      return failure();
    }
    
    // Verify the operands are the first two block arguments
    auto lhsArg = cast<BlockArgument>(lhs);
    auto rhsArg = cast<BlockArgument>(rhs);
    
    if (lhsArg.getArgNumber() != 0 || rhsArg.getArgNumber() != 1) {
      return failure();
    }

    // Create the tensor binary operation
    auto result = rewriter.create<OpType>(
        op.getLoc(), op.getInputs()[0], op.getInputs()[1]);

    // Replace the linalg.generic with the tensor operation
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult createTensorCmpOp(linalg::GenericOp op, arith::CmpFOp cmpOp,
                                  PatternRewriter &rewriter) const {
    // Verify we have exactly 2 inputs for comparison
    if (op.getInputs().size() != 2) {
      return failure();
    }

    // Verify the operands are block arguments
    Value lhs = cmpOp.getLhs();
    Value rhs = cmpOp.getRhs();
    
    if (!isa<BlockArgument>(lhs) || !isa<BlockArgument>(rhs)) {
      return failure();
    }
    
    // Verify the operands are the first two block arguments
    auto lhsArg = cast<BlockArgument>(lhs);
    auto rhsArg = cast<BlockArgument>(rhs);
    
    if (lhsArg.getArgNumber() != 0 || rhsArg.getArgNumber() != 1) {
      return failure();
    }

    // Create the tensor comparison operation
    auto result = rewriter.create<arith::CmpFOp>(
        op.getLoc(), cmpOp.getPredicate(), op.getInputs()[0], op.getInputs()[1]);

    // Replace the linalg.generic with the tensor operation
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult createTensorSelectOp(linalg::GenericOp op, arith::SelectOp selectOp,
                                    PatternRewriter &rewriter) const {
    // Verify we have exactly 3 inputs for select (condition, trueValue, falseValue)
    if (op.getInputs().size() != 3) {
      return failure();
    }

    // Verify the operands are block arguments
    Value condition = selectOp.getCondition();
    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();
    
    if (!isa<BlockArgument>(condition) || !isa<BlockArgument>(trueValue) || !isa<BlockArgument>(falseValue)) {
      return failure();
    }
    
    // Verify the operands are the first three block arguments
    auto conditionArg = cast<BlockArgument>(condition);
    auto trueValueArg = cast<BlockArgument>(trueValue);
    auto falseValueArg = cast<BlockArgument>(falseValue);
    
    if (conditionArg.getArgNumber() != 0 || trueValueArg.getArgNumber() != 1 || falseValueArg.getArgNumber() != 2) {
      return failure();
    }

    // Create the tensor select operation
    auto result = rewriter.create<arith::SelectOp>(
        op.getLoc(), op.getInputs()[0], op.getInputs()[1], op.getInputs()[2]);

    // Replace the linalg.generic with the tensor operation
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult createArbitrarySequence(linalg::GenericOp op, 
                                       SmallVector<Operation*> &bodyOps,
                                       PatternRewriter &rewriter) const {
    // Create a mapping from block arguments to tensor inputs and outputs
    SmallVector<Value> inputs = op.getInputs();
    SmallVector<Value> outputs = op.getOutputs();
    DenseMap<Value, Value> argToTensorMap;
    
    Block &bodyBlock = op.getRegion().front();
    
    // Map input arguments to tensor inputs
    for (size_t i = 0; i < inputs.size(); ++i) {
      argToTensorMap[bodyBlock.getArgument(i)] = inputs[i];
    }
    
    // Map output arguments to tensor outputs (for reduction patterns)
    for (size_t i = 0; i < outputs.size(); ++i) {
      argToTensorMap[bodyBlock.getArgument(inputs.size() + i)] = outputs[i];
    }
    
    // Create a mapping from operation results to tensor results
    DenseMap<Value, Value> resultMap;
    
    // Process each operation in sequence
    for (Operation *bodyOp : bodyOps) {
      Value result = createTensorOperation(bodyOp, argToTensorMap, resultMap, rewriter);
      if (!result) {
        return failure();
      }
      resultMap[bodyOp->getResult(0)] = result;
    }
    
    // The last operation's result is our final result
    Value finalResult = resultMap[bodyOps.back()->getResult(0)];
    
    // Replace the linalg.generic with the final result
    rewriter.replaceOp(op, finalResult);
    return success();
  }
  
  Value createTensorOperation(Operation *bodyOp, 
                             DenseMap<Value, Value> &argToTensorMap,
                             DenseMap<Value, Value> &resultMap,
                             PatternRewriter &rewriter) const {
    // Handle different types of operations
    if (auto addOp = dyn_cast<arith::AddFOp>(bodyOp)) {
      Value lhs = getTensorOperand(addOp.getLhs(), argToTensorMap, resultMap);
      Value rhs = getTensorOperand(addOp.getRhs(), argToTensorMap, resultMap);
      if (!lhs || !rhs) return nullptr;
      return rewriter.create<arith::AddFOp>(addOp.getLoc(), lhs, rhs);
    } else if (auto mulOp = dyn_cast<arith::MulFOp>(bodyOp)) {
      Value lhs = getTensorOperand(mulOp.getLhs(), argToTensorMap, resultMap);
      Value rhs = getTensorOperand(mulOp.getRhs(), argToTensorMap, resultMap);
      if (!lhs || !rhs) return nullptr;
      return rewriter.create<arith::MulFOp>(mulOp.getLoc(), lhs, rhs);
    } else if (auto subOp = dyn_cast<arith::SubFOp>(bodyOp)) {
      Value lhs = getTensorOperand(subOp.getLhs(), argToTensorMap, resultMap);
      Value rhs = getTensorOperand(subOp.getRhs(), argToTensorMap, resultMap);
      if (!lhs || !rhs) return nullptr;
      return rewriter.create<arith::SubFOp>(subOp.getLoc(), lhs, rhs);
    } else if (auto cmpOp = dyn_cast<arith::CmpFOp>(bodyOp)) {
      Value lhs = getTensorOperand(cmpOp.getLhs(), argToTensorMap, resultMap);
      Value rhs = getTensorOperand(cmpOp.getRhs(), argToTensorMap, resultMap);
      if (!lhs || !rhs) return nullptr;
      return rewriter.create<arith::CmpFOp>(cmpOp.getLoc(), cmpOp.getPredicate(), lhs, rhs);
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(bodyOp)) {
      Value condition = getTensorOperand(selectOp.getCondition(), argToTensorMap, resultMap);
      Value trueValue = getTensorOperand(selectOp.getTrueValue(), argToTensorMap, resultMap);
      Value falseValue = getTensorOperand(selectOp.getFalseValue(), argToTensorMap, resultMap);
      if (!condition || !trueValue || !falseValue) return nullptr;
      return rewriter.create<arith::SelectOp>(selectOp.getLoc(), condition, trueValue, falseValue);
    }
    
    return nullptr;
  }
  
  Value getTensorOperand(Value operand, 
                        DenseMap<Value, Value> &argToTensorMap,
                        DenseMap<Value, Value> &resultMap) const {
    // If it's a block argument, map it to the corresponding tensor input/output
    if (isa<BlockArgument>(operand)) {
      auto it = argToTensorMap.find(operand);
      if (it != argToTensorMap.end()) {
        return it->second;
      }
    }
    
    // If it's a result from a previous operation, map it to the tensor result
    auto it = resultMap.find(operand);
    if (it != resultMap.end()) {
      return it->second;
    }
    
    return nullptr;
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
  patterns.add<mlir::xls::LinalgGenericUnrollPattern>(context);
}

}  // namespace mlir
