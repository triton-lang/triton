#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

constexpr int kBarrierCountBitWidth = 29;
// NOTE: We only care for the parity of the phase (0: even, 1: odd), so use 1
// bit constexpr int kBarrierPhaseMask = ((1ULL << (32 - kBarrierCountBitWidth))
// - 1);
constexpr int kBarrierPhaseMask = 1;
constexpr int kInitCountPos = 32;

namespace {

struct InitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::InitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::InitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);

    auto *curBlock = rewriter.getInsertionBlock();
    auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    auto *ldsBarrierInitBlock = rewriter.createBlock(
        curBlock->getParent(), std::next(Region::iterator(curBlock)));
    rewriter.setInsertionPointToEnd(curBlock);
    auto id = getThreadId(rewriter, loc);
    auto pred = b.icmp_eq(id, b.i32_val(0));
    LLVM::CondBrOp::create(rewriter, loc, pred, ldsBarrierInitBlock, endBlock);
    rewriter.setInsertionPointToEnd(ldsBarrierInitBlock);
    // Phase changes when underflow is detected (pending count becomes
    // negative). The provided count from the user assumes that phase changes
    // when pending count reaches zero, so make the adjustment here.
    Value count = b.i64_val(op.getCount() - 1);
    Value val = b.or_(b.shl(count, b.i64_val(kInitCountPos)), count);
    b.store(val, smemObj.getBase());
    LLVM::BrOp::create(rewriter, loc, ValueRange(), endBlock);
    rewriter.setInsertionPointToStart(endBlock);
    // Synchronize the whole CTA, so all waves see the LDS barrier
    b.barrier();
    rewriter.eraseOp(op);
    return success();
  }
};

struct ArriveBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::ArriveBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::ArriveBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);
    auto count = adaptor.getCount();
    // NOTE: The LLVM intrisic expects an i64_ty for count (update value)
    // But count cannot be more than 32bits according to ISA docs.
    Value priorState =
        LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, "llvm.amdgcn.ds.atomic.barrier.arrive.rtn.b64",
            i64_ty, {smemObj.getBase(), b.i64_val(count)})
            .getResult(0);
    Value priorPhase = b.and_(
        i32_ty, b.i32_val(kBarrierPhaseMask),
        b.trunc(i32_ty, b.lshr(priorState, b.i64_val(kBarrierCountBitWidth))));
    rewriter.replaceOp(op, priorPhase);
    return success();
  }
};

struct WaitBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::WaitBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::WaitBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getAlloc(),
        typeConverter->convertType(op.getAlloc().getType().getElementType()),
        rewriter);
    Value phase = adaptor.getPhase();
    auto *curBlock = rewriter.getInsertionBlock();
    auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    auto *waitBlock = rewriter.createBlock(
        curBlock->getParent(), std::next(Region::iterator(curBlock)));
    rewriter.setInsertionPointToEnd(curBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange(), waitBlock);
    rewriter.setInsertionPointToStart(waitBlock);
    // Sleep for the minimum number of clocks. 64*SIMM16[6:0] = 64 * 1 = 64
    // clocks.
    ROCDL::SSleepOp::create(rewriter, loc, 1);
    Value curState = b.load(i64_ty, smemObj.getBase());
    Value curPhase = b.and_(
        i32_ty, b.i32_val(kBarrierPhaseMask),
        b.trunc(i32_ty, b.lshr(curState, b.i64_val(kBarrierCountBitWidth))));
    Value phaseChanged = b.icmp_ne(curPhase, phase);
    LLVM::CondBrOp::create(rewriter, loc, phaseChanged, endBlock, waitBlock);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void mlir::triton::AMD::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<InitBarrierOpConversion>(typeConverter, benefit);
  patterns.add<WaitBarrierOpConversion>(typeConverter, benefit);
  patterns.add<ArriveBarrierOpConversion>(typeConverter, benefit);
}
