#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_LOWERMULTIREDUCTION
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// This pass exists because LowerVectorMultiReductionPass can be run on
// func::FuncOp only and we translate triton::FuncOp directly into llvm::FuncOp.
// So we run the same set of patterns on triton::FuncOp.
struct LowerMultiReduction
    : public mlir::triton::impl::LowerMultiReductionBase<LowerMultiReduction> {
  using LowerMultiReductionBase::LowerMultiReductionBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet loweringPatterns(context);
    vector::VectorMultiReductionLowering options;
    vector::populateVectorMultiReductionLoweringPatterns(loweringPatterns,
                                                         options);

    if (failed(applyPatternsAndFoldGreedily(op, std::move(loweringPatterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<triton::FuncOp>> createLowerMultiReductionPass() {
  return std::make_unique<LowerMultiReduction>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
