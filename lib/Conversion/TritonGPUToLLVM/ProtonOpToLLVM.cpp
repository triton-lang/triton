#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton::gpu;

struct LocalRecordOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalRecordOp> {
  explicit LocalRecordOpConversion(LLVMTypeConverter &typeConverter,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalRecordOp>(typeConverter,
                                                           benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalRecordOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    const int warpsPerGroup = triton::gpu::getWarpGroupSize();
    const int wordsPerEntry = triton::gpu::getWordsPerProtonEntry();
    const int slots =
        cast<IntegerAttr>(mod->getAttr("triton_gpu.proton-slots")).getInt();
    const int numWarpgroup =
        triton::gpu::TritonGPUDialect::getNumWarps(mod) / warpsPerGroup;

    assert(op.getMetric() == triton::ProtonMetric::CYCLE);

    Value indexPtr = adaptor.getIndexPtr();
    Value dataStruct = adaptor.getData();

    auto smemPtrTy = ptr_ty(rewriter.getContext(), 3);
    Value smemDataBasePtr = extract_val(smemPtrTy, dataStruct, 0);

    Value threadId = getThreadId(rewriter, loc);
    Value warpgroupSize = i32_val(
        warpsPerGroup * triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpgroupId = udiv(threadId, warpgroupSize);
    Value isWarpgroup = icmp_eq(urem(threadId, warpgroupSize), i32_val(0));

    // Load the index from smem.
    Value curIdx = load(i32_ty, indexPtr);
    Value newIdx = add(curIdx, i32_val(wordsPerEntry));
    store(newIdx, indexPtr);

    // Compute the offset in smem.
    int numWgSlot = slots / numWarpgroup;
    Value wgSlotOffset = mul(warpgroupId, i32_val(wordsPerEntry * numWgSlot));
    Value smemTagOffset =
        add(wgSlotOffset, urem(curIdx, i32_val(wordsPerEntry * numWgSlot)));

    // Record the entry and vectorized store to smem.
    Value vecPtr = gep(smemPtrTy, i32_ty, smemDataBasePtr, smemTagOffset);
    Value tag = op.getIsStart() ? i32_val(op.getRegionId())
                                : i32_val(1 << 31 | op.getRegionId());
    Value clock = targetInfo.clock(rewriter, loc, false);

    Value valsVec = packLLVector(loc, {tag, clock}, rewriter);
    targetInfo.storeDShared(rewriter, loc, vecPtr, std::nullopt, valsVec,
                            /*pred=*/isWarpgroup);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct ProtonFinalizeOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ProtonFinalizeOp> {
  explicit ProtonFinalizeOpConversion(LLVMTypeConverter &typeConverter,
                                      const TargetInfoBase &targetInfo,
                                      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::ProtonFinalizeOp>(typeConverter,
                                                              benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ProtonFinalizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value indexPtr = adaptor.getIndexPtr();
    Value dataStruct = adaptor.getData();

    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    const int warpsPerGroup = triton::gpu::getWarpGroupSize();
    const int wordsPerEntry = triton::gpu::getWordsPerProtonEntry();
    Value threadId = getThreadId(rewriter, loc);
    Value isFirstThread = icmp_eq(threadId, i32_val(0));

    Block *prevBlock = op->getBlock();
    // Add the 'if' block.
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    auto gmemPtrTy = ptr_ty(rewriter.getContext(), 1);
    Value gmemBasePtr = adaptor.getPtr();
    auto smemPtrTy = ptr_ty(rewriter.getContext(), 3);

    // Lambda function to load a word from smem and store it to gmem.
    auto copyWord = [&](Value smemStruct, Value smemOffset, Value gmemOffset) {
      Value smemBasePtr = extract_val(smemPtrTy, smemStruct, 0);
      // Load the value from smem
      Value ptr = gep(smemPtrTy, i32_ty, smemBasePtr, smemOffset);
      Value smemLoad =
          targetInfo.loadShared(rewriter, loc, ptr, i32_ty, true_val());
      // Store the value to global memory
      Value gmemPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemOffset);
      store(smemLoad, gmemPtr);
    };

    int offset = 0;
    const int slots =
        cast<IntegerAttr>(mod->getAttr("triton_gpu.proton-slots")).getInt();
    // scratch: block id (1), sm id (1), index (1), data (slots * wordsPerEntry)
    const int scratchWordSize = 3 + slots * wordsPerEntry;
    Value pidX = targetInfo.programId(rewriter, loc, mod, 0);
    Value pidY = targetInfo.programId(rewriter, loc, mod, 1);
    Value pidZ = targetInfo.programId(rewriter, loc, mod, 2);
    Value smid = targetInfo.smId(rewriter, loc);
    Value gridDimX = rewriter.create<arith::IndexCastOp>(
        loc, i32_ty,
        rewriter.create<::mlir::gpu::GridDimOp>(loc, mlir::gpu::Dimension::x));
    Value gridDimY = rewriter.create<arith::IndexCastOp>(
        loc, i32_ty,
        rewriter.create<::mlir::gpu::GridDimOp>(loc, mlir::gpu::Dimension::y));
    Value pid =
        add(add(pidX, mul(pidY, gridDimX)), mul(pidZ, mul(gridDimX, gridDimY)));
    Value programOffset = mul(i32_val(scratchWordSize), pid);

    // Write back program id.
    Value gmemPidOffset = add(programOffset, i32_val(offset++));
    Value gmemPidPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemPidOffset);
    store(pid, gmemPidPtr);

    // Write back SM id.
    Value gmemSmOffset = add(programOffset, i32_val(offset++));
    Value gmemSmPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemSmOffset);
    store(smid, gmemSmPtr);

    // Write back the total counts.
    Value index = load(i32_ty, indexPtr);
    Value gmemIndexOffset = add(programOffset, i32_val(offset++));
    Value gmemIndexPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemIndexOffset);
    store(index, gmemIndexPtr);

    // Add the 'else' block and the condition.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, isFirstThread, ifBlock, thenBlock);

    // Write back the data.
    const int upper = wordsPerEntry * (slots - 1);
    rewriter.setInsertionPointToEnd(ifBlock);
    Value initIdx = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, 0);
    Value wbBaseOffset = add(programOffset, i32_val(offset));

    Block *writeBackBlock = rewriter.createBlock(
        op->getParentRegion(), std::next(Region::iterator(ifBlock)), {i32_ty},
        {loc});
    rewriter.setInsertionPointToStart(writeBackBlock);
    BlockArgument idx = writeBackBlock->getArgument(0);
    Value gmemWbTagOffset = add(wbBaseOffset, idx);
    Value smemTagOffset = idx;
    Value gmemWbCycleOffset = add(gmemWbTagOffset, i32_val(1));
    Value smemCycleOffset = add(smemTagOffset, i32_val(1));
    copyWord(dataStruct, smemTagOffset, gmemWbTagOffset);
    copyWord(dataStruct, smemCycleOffset, gmemWbCycleOffset);
    Value pred = icmp_slt(idx, i32_val(upper));
    Value updatedIdx = add(idx, i32_val(wordsPerEntry));
    rewriter.create<cf::CondBranchOp>(loc, pred, writeBackBlock, updatedIdx,
                                      thenBlock, ArrayRef<Value>());

    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, writeBackBlock, initIdx);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct ProtonInitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ProtonInitOp> {
  explicit ProtonInitOpConversion(LLVMTypeConverter &typeConverter,
                                  const TargetInfoBase &targetInfo,
                                  PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::ProtonInitOp>(typeConverter,
                                                          benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ProtonInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto ptrTy = ptr_ty(rewriter.getContext(), 1);
    auto indexPtr = rewriter.create<LLVM::AllocaOp>(
        loc, ptrTy, i32_ty, i32_val(1), /*alignment=*/0);
    store(i32_val(0), indexPtr);
    rewriter.replaceOp(op, indexPtr);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateProtonOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<LocalRecordOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ProtonFinalizeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ProtonInitOpConversion>(typeConverter, targetInfo, benefit);
}
