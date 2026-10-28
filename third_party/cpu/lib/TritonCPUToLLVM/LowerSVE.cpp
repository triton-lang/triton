#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::cpu {
#define GEN_PASS_DEF_LOWERSVE
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::cpu

using namespace mlir;

namespace {

/// Lower a two-column transpose with scalable interleave operations. This
/// keeps the generated IR proportional to the number of operations rather
/// than to the fixed vector length.
struct LowerTranspose : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = op.getSourceVectorType();
    auto dstTy = op.getResultVectorType();
    if (srcTy.isScalable() || srcTy.getRank() != 2 ||
        op.getPermutation() != ArrayRef<int64_t>({1, 0}))
      return failure();

    ArrayRef<int64_t> shape = srcTy.getShape();
    bool rowsToColumns = shape[1] == 2;
    bool columnsToRows = shape[0] == 2;
    if (!rowsToColumns && !columnsToRows)
      return failure();

    Type elemTy = srcTy.getElementType();
    if (!elemTy.isIntOrFloat())
      return failure();
    unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
    if (bitWidth == 0 || bitWidth > 128 || 128 % bitWidth != 0)
      return failure();

    Location loc = op.getLoc();
    int64_t rows = rowsToColumns ? shape[0] : shape[1];
    int64_t numElems = srcTy.getNumElements();
    int64_t minLanes = 128 / bitWidth;
    auto flatTy = VectorType::get({numElems}, elemTy);
    auto bufferTy = MemRefType::get({numElems}, elemTy);
    auto laneTy = VectorType::get({minLanes}, elemTy, {true});
    auto pairTy = VectorType::get({2 * minLanes}, elemTy, {true});
    auto laneMaskTy = laneTy.cloneWith(std::nullopt, rewriter.getI1Type());
    auto pairMaskTy = pairTy.cloneWith(std::nullopt, rewriter.getI1Type());

    auto pointerTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value stack = LLVM::StackSaveOp::create(rewriter, loc, pointerTy);
    Value input = memref::AllocaOp::create(rewriter, loc, bufferTy);
    Value output = memref::AllocaOp::create(rewriter, loc, bufferTy);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value two = arith::ConstantIndexOp::create(rewriter, loc, 2);
    Value upper = arith::ConstantIndexOp::create(rewriter, loc, rows);
    Value lanes = arith::ConstantIndexOp::create(rewriter, loc, minLanes);
    Value vscale = vector::VectorScaleOp::create(rewriter, loc);
    Value step = arith::MulIOp::create(rewriter, loc, lanes, vscale);
    Value flat =
        vector::ShapeCastOp::create(rewriter, loc, flatTy, op.getVector());
    vector::StoreOp::create(rewriter, loc, flat, input, ValueRange{zero});

    Value lanePassthru = arith::ConstantOp::create(
        rewriter, loc,
        SplatElementsAttr::get(laneTy, rewriter.getZeroAttr(elemTy)));
    Value pairPassthru = arith::ConstantOp::create(
        rewriter, loc,
        SplatElementsAttr::get(pairTy, rewriter.getZeroAttr(elemTy)));

    auto loop = scf::ForOp::create(rewriter, loc, zero, upper, step);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value index = loop.getInductionVar();
      Value remaining = arith::SubIOp::create(rewriter, loc, upper, index);
      Value laneMask = vector::CreateMaskOp::create(rewriter, loc, laneMaskTy,
                                                    ValueRange{remaining});
      Value pairCount = arith::MulIOp::create(rewriter, loc, remaining, two);
      Value pairMask = vector::CreateMaskOp::create(rewriter, loc, pairMaskTy,
                                                    ValueRange{pairCount});
      Value pairIndex = arith::MulIOp::create(rewriter, loc, index, two);

      if (rowsToColumns) {
        Value interleaved = vector::MaskedLoadOp::create(
            rewriter, loc, pairTy, input, ValueRange{pairIndex}, pairMask,
            pairPassthru);
        auto split = vector::DeinterleaveOp::create(rewriter, loc, interleaved);
        Value secondIndex = arith::AddIOp::create(rewriter, loc, upper, index);
        vector::MaskedStoreOp::create(rewriter, loc, output, ValueRange{index},
                                      laneMask, split.getRes1());
        vector::MaskedStoreOp::create(rewriter, loc, output,
                                      ValueRange{secondIndex}, laneMask,
                                      split.getRes2());
      } else {
        Value secondIndex = arith::AddIOp::create(rewriter, loc, upper, index);
        Value lhs = vector::MaskedLoadOp::create(rewriter, loc, laneTy, input,
                                                 ValueRange{index}, laneMask,
                                                 lanePassthru);
        Value rhs = vector::MaskedLoadOp::create(rewriter, loc, laneTy, input,
                                                 ValueRange{secondIndex},
                                                 laneMask, lanePassthru);
        Value interleaved =
            vector::InterleaveOp::create(rewriter, loc, lhs, rhs);
        vector::MaskedStoreOp::create(rewriter, loc, output,
                                      ValueRange{pairIndex}, pairMask,
                                      interleaved);
      }
    }

    Value result =
        vector::LoadOp::create(rewriter, loc, flatTy, output, ValueRange{zero});
    LLVM::StackRestoreOp::create(rewriter, loc, stack);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, dstTy, result);
    return success();
  }
};

struct LowerSVE : public mlir::triton::cpu::impl::LowerSVEBase<LowerSVE> {
  using LowerSVEBase::LowerSVEBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerTranspose>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::cpu {

std::unique_ptr<OperationPass<triton::FuncOp>> createLowerSVEPass() {
  return std::make_unique<LowerSVE>();
}

} // namespace mlir::triton::cpu
