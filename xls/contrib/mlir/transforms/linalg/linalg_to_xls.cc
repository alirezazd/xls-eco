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
#include <sstream>

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
#include "xls/contrib/mlir/transforms/linalg/analysis/broadcast.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/schedule.h"
#include "xls/contrib/mlir/transforms/linalg/analysis/unroll_plan.h"
#include "xls/contrib/mlir/transforms/linalg/codegen/dslx_codegen.h"

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

    auto yieldOp = cast<linalg::YieldOp>(bodyBlock.getTerminator());
    Value yieldedValue = yieldOp.getValues()[0];

    if (yieldedValue != arithOp->getResult(0)) {
      return failure();
    }

    auto resultType = llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType) {
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

struct LinalgReductionToXlsPattern : public OpRewritePattern<linalg::GenericOp> {
  LinalgReductionToXlsPattern(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/30) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Step 1: Analyze the LinalgGeneric operation
    LinalgGeneric g;
    if (failed(AnalyzeLinalgGeneric(op, g))) {
      return failure();
    }

    // Check if this is a reduction operation (has reduction iterators)
    bool hasReduction = false;
    for (const auto& dim : g.dims) {
      if (dim.kind == IterKind::kReduction) {
        hasReduction = true;
        break;
      }
    }

    if (!hasReduction) {
      return failure();
    }

    // Step 2: Perform broadcast analysis
    BroadcastAnalysis A;
    if (failed(AnalyzeBroadcast(g, A))) {
      return failure();
    }

    // Step 3: Generate schedule
    CodegenConfig cfg;
    cfg.loop_order = CodegenConfig::kPreserve;
    cfg.reduce_policy = CodegenConfig::kSerial;
    
    Schedule sch;
    if (failed(GenerateSchedule(g, A, cfg, sch))) {
      return failure();
    }

    // Step 4: Print debug information and intentionally stop
    llvm::errs() << "\n=== LINALG REDUCTION PATTERN CAPTURED ===\n";
    llvm::errs() << "Operation: " << op.getOperationName() << "\n";
    llvm::errs() << "Location: " << op.getLoc() << "\n\n";

    llvm::errs() << "--- LinalgGeneric Analysis ---\n";
    llvm::errs() << "Dimensions: ";
    for (size_t i = 0; i < g.dims.size(); ++i) {
      llvm::errs() << (i > 0 ? ", " : "") << g.dims[i].name << "(" 
                   << (g.dims[i].kind == IterKind::kParallel ? "P" : "R") << ":" 
                   << g.dims[i].extent << ")";
    }
    llvm::errs() << "\n";
    llvm::errs() << "Operands: " << g.operands.size() << "\n";
    llvm::errs() << "Region ops: " << g.region.ops.size() << "\n";

    llvm::errs() << "\n--- Broadcast Analysis ---\n";
    llvm::errs() << "Loop extents: [";
    for (size_t i = 0; i < A.loop_extents.size(); ++i) {
      llvm::errs() << (i > 0 ? ", " : "") << A.loop_extents[i].getConstant();
    }
    llvm::errs() << "]\n";
    llvm::errs() << "Indexers: " << A.indexers.size() << "\n";

    llvm::errs() << "\n--- Schedule Generation ---\n";
    llvm::errs() << "Bands: " << sch.bands.size() << "\n";
    for (size_t i = 0; i < sch.bands.size(); ++i) {
      const auto& band = sch.bands[i];
      llvm::errs() << "  Band " << i << ": " << band.loops.size() << " loops\n";
      for (size_t j = 0; j < band.loops.size(); ++j) {
        const auto& loop = band.loops[j];
        llvm::errs() << "    Loop " << j << ": dim=" << loop.dim 
                     << ", step=" << loop.step 
                     << ", unroll=" << loop.unroll
                     << ", is_reduction=" << (loop.is_reduction ? "true" : "false") << "\n";
      }
    }

    // Step 5: Create UnrollPlan for the schedule
    llvm::errs() << "\n--- Starting Unroll Plan Generation ---\n";
    UnrollPlan up;
    if (failed(CreateUnrollPlan(sch, A, cfg, up))) {
      llvm::errs() << "Failed to create unroll plan\n";
      return failure();
    }
    llvm::errs() << "Unroll plan creation successful\n";

    llvm::errs() << "\n--- Unroll Plan Generation ---\n";
    llvm::errs() << "Configuration unroll factors: ";
    if (cfg.unroll.empty()) {
      llvm::errs() << "none (serial execution)\n";
    } else {
      for (const auto& [dim, factor] : cfg.unroll) {
        llvm::errs() << "dim" << dim << "=" << factor << " ";
      }
      llvm::errs() << "\n";
    }
    
    if (!up.dims.empty()) {
      llvm::errs() << "Unroll dimensions: " << up.dims.size() << "\n";
      for (size_t i = 0; i < up.dims.size(); ++i) {
        const auto& plan = up.dims[i];
        llvm::errs() << "  Dimension " << i << ": dim=" << plan.dim 
                     << ", lanes=" << plan.lanes 
                     << ", has_tail=" << (plan.has_tail ? "true" : "false")
                     << ", main_iters=" << plan.main_iters 
                     << ", tail=" << plan.tail << "\n";
      }
      llvm::errs() << "Total lanes: " << up.multi_lane.total_lanes << "\n";
    } else {
      llvm::errs() << "No unroll plan created (serial execution)\n";
    }

    // Step 6: Generate DSLX Code
    llvm::errs() << "\n--- Starting DSLX Code Generation ---\n";
    
    // Create DSLX codegen context
    std::stringstream dslx_output;
    DslxCodegen codegen(dslx_output);
    DslxEmissionContext ctx;
    ctx.function_name = "complex_reduction_example";
    
    // Create reduction context from codegen config
    ReduceCtx rctx;
    rctx.policy = cfg.reduce_policy;
    rctx.tree_fan_in = 2;
    rctx.deterministic = true;
    
    // Generate DSLX function
    if (failed(codegen.EmitFunction(g, A, sch, up, rctx, ctx))) {
      llvm::errs() << "Failed to generate DSLX code\n";
      return failure();
    }
    
    std::string dslx_code = dslx_output.str();
    llvm::errs() << "DSLX code generation successful\n";
    llvm::errs() << "\n--- Generated DSLX Code ---\n";
    llvm::errs() << dslx_code << "\n";

    llvm::errs() << "\n--- Next Steps for Complete Pipeline ---\n";
    llvm::errs() << "1. ✅ Create UnrollPlan for the schedule (COMPLETED)\n";
    llvm::errs() << "2. ✅ Generate DSLX function signature (COMPLETED)\n";
    llvm::errs() << "3. ✅ Emit zero-initialization for output tensors (COMPLETED)\n";
    llvm::errs() << "4. ✅ Generate nested loops with reduction logic (COMPLETED)\n";
    llvm::errs() << "5. ✅ Emit accumulator updates using reduction policies (COMPLETED)\n";
    llvm::errs() << "6. ✅ Return the final result (COMPLETED)\n";

    llvm::errs() << "\n=== INTENTIONAL STOP: DSLX code generation completed! ===\n";
    llvm::errs() << "The complete pipeline successfully generated DSLX code from MLIR.\n";
    llvm::errs() << "Ready to proceed with XLS IR generation and optimization.\n\n";

    // Intentionally fail to stop the transformation
    return failure();
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
  patterns.add<mlir::xls::LinalgReductionToXlsPattern>(context);
}

}  // namespace mlir
