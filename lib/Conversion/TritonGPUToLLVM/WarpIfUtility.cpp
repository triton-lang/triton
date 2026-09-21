#include "triton/Conversion/TritonGPUToLLVM/WarpIfUtility.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

LogicalResult mlir::triton::lowerWarpIfOps(ModuleOp module,
                                           const LLVMTypeConverter &converter,
                                           const TargetInfoBase &target) {
  // Convert the legal boundary ops after their bodies have been converted.
  // Walk postorder: yields first, then the parent (which may be replaced).
  module.walk([&](Operation *op) {
    if (isa<WarpIfOp, WarpIfYieldOp>(op))
      convertOpTypes(op, converter);
  });
  SmallVector<WarpIfOp> ops;
  module.walk([&](WarpIfOp op) { ops.push_back(op); });
  for (WarpIfOp op : ops) {
    if (!llvm::hasSingleElement(op.getBody()))
      return op.emitError(
          "warp_if requires a single-block body after conversion");
    auto yield = dyn_cast<WarpIfYieldOp>(op.getBody().front().getTerminator());
    if (!yield)
      return op.emitError("warp_if body is missing its yield");
    IRRewriter rewriter(op.getContext());
    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    // The boundary operands now have LLVM types; do not use the tensor-typed
    // ODS accessor during this temporary conversion state.
    auto elements = unpackLLElements(loc, op->getOperand(0), rewriter);
    if (elements.empty())
      return op.emitError("warp_if condition has no register elements");
    Value laneAny = elements.front();
    for (Value element : llvm::drop_begin(elements))
      laneAny = LLVM::OrOp::create(rewriter, loc, laneAny, element);
    Type ballotTy =
        rewriter.getIntegerType(TritonGPUDialect::getThreadsPerWarp(module));
    Value ballot = target.ballot(rewriter, loc, ballotTy, laneAny);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value warpAny =
        b.icmp_ne(ballot, b.int_val(ballotTy.getIntOrFloatBitWidth(), 0));

    Block *entry = op->getBlock();
    Block *merge = rewriter.splitBlock(entry, Block::iterator(op));
    auto results = merge->addArguments(
        op.getResultTypes(), SmallVector<Location>(op.getNumResults(), loc));
    Block *body = &op.getBody().front();
    rewriter.inlineRegionBefore(op.getBody(), merge);
    rewriter.setInsertionPoint(yield);
    LLVM::BrOp::create(rewriter, loc, yield.getValues(), merge);
    rewriter.eraseOp(yield);
    rewriter.setInsertionPointToEnd(entry);
    LLVM::CondBrOp::create(rewriter, loc, warpAny, body, ValueRange{}, merge,
                           op.getInputs());
    rewriter.replaceOp(op, SmallVector<Value>(results.begin(), results.end()));
  }
  return success();
}
