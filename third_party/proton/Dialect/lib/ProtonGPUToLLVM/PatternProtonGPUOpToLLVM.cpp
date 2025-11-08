#include "Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
namespace proton::gpu {

namespace {

Value getLinearId(Location loc, ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Note:
  // 1. We compute use i64 data type to compute and then truncate to i32
  // to support various backend intrinsics (e.g. amd).
  // 2. We avoid using the targetInfo's programId() because of its coupling
  // with cluster id in Nvidia TritonGPU's llvm lowering.
  Value pidX = arith::IndexCastOp::create(
      rewriter, loc, i64_ty,
      mlir::gpu::BlockIdOp::create(rewriter, loc, mlir::gpu::Dimension::x));
  Value pidY = arith::IndexCastOp::create(
      rewriter, loc, i64_ty,
      mlir::gpu::BlockIdOp::create(rewriter, loc, mlir::gpu::Dimension::y));
  Value pidZ = arith::IndexCastOp::create(
      rewriter, loc, i64_ty,
      mlir::gpu::BlockIdOp::create(rewriter, loc, mlir::gpu::Dimension::z));

  Value gridDimX = arith::IndexCastOp::create(
      rewriter, loc, i64_ty,
      ::mlir::gpu::GridDimOp::create(rewriter, loc, mlir::gpu::Dimension::x));
  Value gridDimY = arith::IndexCastOp::create(
      rewriter, loc, i64_ty,
      ::mlir::gpu::GridDimOp::create(rewriter, loc, mlir::gpu::Dimension::y));
  Value linearId =
      b.trunc(i32_ty, b.add(b.add(pidX, b.mul(pidY, gridDimX)),
                            b.mul(pidZ, b.mul(gridDimX, gridDimY))));
  return linearId;
}

struct ReadCounterOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::ReadCounterOp> {
  explicit ReadCounterOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::ReadCounterOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::ReadCounterOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool isClock64 = false;
    auto intType = mlir::cast<IntegerType>(op.getResult().getType());
    isClock64 = intType.getWidth() == 64;
    Value clock = targetInfo.clock(rewriter, op.getLoc(), isClock64);
    rewriter.replaceOp(op, clock);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct InitializeOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::InitializeOp> {
  explicit InitializeOpConversion(LLVMTypeConverter &typeConverter,
                                  const proton::gpu::TargetInfoBase &targetInfo,
                                  PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::InitializeOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::InitializeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    Value scratchPtr = adaptor.getScratchPtr();
    auto scratchPtrTy = mlir::cast<LLVM::LLVMPointerType>(scratchPtr.getType());

    // Header layout (total: circularHeaderSize bytes)
    //  +-------------------------------+ 0
    //  | preamble (1 word)             |
    //  +-------------------------------+ 1
    //  | program id (1 word)           |
    //  +-------------------------------+ 2
    //  | hw id (1 word)                |
    //  +-------------------------------+ 3
    //  | buffer size (1 word)          |
    //  +-------------------------------+ 4
    //  | init time                     |
    //  | (2 words)                     |
    //  +-------------------------------+ 6
    //  | pre-final time                |
    //  | (2 words)                     |
    //  +-------------------------------+ 8
    //  | post-final time               |
    //  | (2 words)                     |
    //  +-------------------------------+ 10

    Value threadId = getThreadId(rewriter, loc);
    Value isFirstThread = b.icmp_eq(threadId, b.i32_val(0));

    Block *prevBlock = op->getBlock();

    // Add the 'if' block.
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    // Write back 'preamble'.
    Value preamble = b.i32_val(0xdeadbeef);
    Value gmemPreambleOffset = b.i32_val(0);
    Value gmemPreamblePtr =
        b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemPreambleOffset);
    b.store(preamble, gmemPreamblePtr);

    // Write back 'program id'.
    Value gmemPidOffset = b.i32_val(1);
    Value gmemPidPtr = b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemPidOffset);
    Value pid = getLinearId(loc, rewriter);
    b.store(pid, gmemPidPtr);

    // Write back 'hw id'.
    Value gmemHwidOffset = b.i32_val(2);
    Value gmemHwidPtr = b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemHwidOffset);
    Value hwid = targetInfo.processorId(rewriter, loc);
    b.store(hwid, gmemHwidPtr);

    // Write back 'init time'.
    Value gmemInitTimeOffset = b.i32_val(4);
    Value gmemInitTimePtr =
        b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemInitTimeOffset);
    Value initTime = targetInfo.globalTime(rewriter, loc);
    b.store(initTime, gmemInitTimePtr);

    // Add the 'else' block and the condition.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(prevBlock);
    cf::CondBranchOp::create(rewriter, loc, isFirstThread, ifBlock, thenBlock);
    rewriter.setInsertionPointToEnd(ifBlock);
    cf::BranchOp::create(rewriter, loc, thenBlock);

    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct FinalizeOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::FinalizeOp> {
  explicit FinalizeOpConversion(LLVMTypeConverter &typeConverter,
                                const proton::gpu::TargetInfoBase &targetInfo,
                                PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::FinalizeOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::FinalizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto segmentObj =
        LLVM::SegmentObject::fromStruct(loc, adaptor.getSegment(), rewriter);
    Value scratchPtr = adaptor.getScratchPtr();

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    const int bytesPerEntry = proton::gpu::getBytesPerClockEntry();
    const int wordsPerEntry = bytesPerEntry / 4; // 1 word = 4 bytes

    int numWarps = getTotalNumWarps(mod);

    Value threadId = getRawThreadId(rewriter, loc);
    Value threadsPerWarp =
        b.i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpId = b.udiv(threadId, threadsPerWarp);
    Value laneId = b.urem(threadId, threadsPerWarp);
    Value isWarpFirstThread = b.icmp_eq(laneId, b.i32_val(0));
    Value isBlockFirstThread = b.icmp_eq(threadId, b.i32_val(0));
    auto segmentType = op.getSegment().getType();
    const int bufferSizeInWords = segmentType.getNBytes() / 4;
    const int circularHeaderWordSize = proton::gpu::getCircularHeaderSize() / 4;

    // Circular strategy memory layout (total: allocprofileScratchSize bytes)
    //  +---------------------------------------+
    //  | header (circularHeaderSize bytes)     |
    //  +---------------------------------------+
    //  | warp index (4 bytes x numWarps)       |
    //  +---------------------------------------+
    //  | profiled data (allocBufferSize bytes) |
    //  +---------------------------------------+
    const int metadataWordSize = circularHeaderWordSize + numWarps;
    auto selectIds = segmentType.getSelectIds();
    bool hasSelectIds = !selectIds.empty();
    int activeWarpCount = hasSelectIds ? selectIds.size() : numWarps;
    const int segmentWordSize = bufferSizeInWords / activeWarpCount;
    auto scratchPtrTy = mlir::cast<LLVM::LLVMPointerType>(scratchPtr.getType());
    auto segmentBaseTy =
        mlir::cast<LLVM::LLVMPointerType>(segmentObj.base.getType());

    // Control-flow outline:
    //   prevBlock
    //     └─ condbr (block leader?) -> leaderBlock / continuation
    //   leaderBlock
    //     └─ ...body...
    //     └─ br continuation
    //   continuation
    //     └─ condbr (warp leader?) -> storeBlock / afterStore
    //   storeBlock
    //     └─ ...store warp index...
    //     └─ br afterStore
    //   afterStore
    //     └─ (optional shared mem copy)
    Block *continuation =
        emitBlockLeaderPrologue(op, isBlockFirstThread, scratchPtr,
                                scratchPtrTy, bufferSizeInWords, rewriter);
    continuation = emitWarpIndexWriteback(
        op, continuation, isWarpFirstThread, warpId, scratchPtr, scratchPtrTy,
        segmentObj, circularHeaderWordSize, rewriter);
    if (segmentBaseTy.getAddressSpace() == 3) {
      // shared memory
      continuation = emitWarpCopySection(
          op, continuation, laneId, threadsPerWarp, scratchPtr, scratchPtrTy,
          segmentObj, metadataWordSize, wordsPerEntry, segmentWordSize,
          circularHeaderWordSize, segmentType.getMemorySpace(), rewriter);
    }
    emitBlockLeaderEpilogue(op, continuation, isBlockFirstThread, scratchPtr,
                            scratchPtrTy, rewriter);
    rewriter.eraseOp(op);
    return success();
  }

private:
  Block *emitBlockLeaderPrologue(mlir::triton::proton::gpu::FinalizeOp op,
                                 Value isBlockFirstThread, Value scratchPtr,
                                 LLVM::LLVMPointerType scratchPtrTy,
                                 int bufferSizeInWords,
                                 ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // Control-flow outline:
    //   prevBlock
    //     └─ condbr (block leader?) -> leaderBlock / continuation
    //   leaderBlock
    //     └─ ...body...
    //     └─ br continuation
    //   continuation
    Block *prevBlock = op->getBlock();
    Block *continuation = rewriter.splitBlock(prevBlock, op->getIterator());
    Block *leaderBlock = rewriter.createBlock(prevBlock->getParent(),
                                              Region::iterator(continuation));
    rewriter.setInsertionPointToEnd(prevBlock);
    cf::CondBranchOp::create(rewriter, loc, isBlockFirstThread, leaderBlock,
                             continuation);
    rewriter.setInsertionPointToStart(leaderBlock);

    Value gmemBufSizeOffset = b.i32_val(3);
    Value gmemBufSizePtr =
        b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemBufSizeOffset);
    Value bufferCapacityInBytes = b.i32_val(bufferSizeInWords * 4);
    b.store(bufferCapacityInBytes, gmemBufSizePtr);

    Value gmemPreFinalTimeOffset = b.i32_val(6);
    Value gmemPreFinalTimePtr =
        b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemPreFinalTimeOffset);
    Value preFinalTime = targetInfo.globalTime(rewriter, loc);
    b.store(preFinalTime, gmemPreFinalTimePtr);

    cf::BranchOp::create(rewriter, loc, continuation);
    rewriter.setInsertionPointToStart(continuation);
    return continuation;
  }

  Block *emitWarpIndexWriteback(mlir::triton::proton::gpu::FinalizeOp op,
                                Block *continuation, Value isWarpFirstThread,
                                Value warpId, Value scratchPtr,
                                LLVM::LLVMPointerType scratchPtrTy,
                                const LLVM::SegmentObject &segmentObj,
                                int circularHeaderWordSize,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Block *afterStore = rewriter.splitBlock(continuation, op->getIterator());
    Block *storeBlock = rewriter.createBlock(op->getParentRegion(),
                                             Region::iterator(afterStore));

    rewriter.setInsertionPointToEnd(continuation);
    cf::CondBranchOp::create(rewriter, loc, isWarpFirstThread, storeBlock,
                             afterStore);

    rewriter.setInsertionPointToStart(storeBlock);
    Value warpIndexOffset = b.add(warpId, b.i32_val(circularHeaderWordSize));
    Value gmemWarpIndexPtr =
        b.gep(scratchPtrTy, i32_ty, scratchPtr, warpIndexOffset);
    Value indexForStore = b.load(i32_ty, segmentObj.indexPtr);
    b.store(indexForStore, gmemWarpIndexPtr);
    cf::BranchOp::create(rewriter, loc, afterStore);

    rewriter.setInsertionPointToStart(afterStore);
    return afterStore;
  }

  Block *emitWarpCopySection(mlir::triton::proton::gpu::FinalizeOp op,
                             Block *continuation, Value laneId,
                             Value threadsPerWarp, Value scratchPtr,
                             LLVM::LLVMPointerType scratchPtrTy,
                             const LLVM::SegmentObject &segmentObj,
                             int metadataWordSize, int wordsPerEntry,
                             int segmentWordSize, int circularHeaderWordSize,
                             Attribute memSpace,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // Control-flow outline:
    //   continuation
    //     └─ br copyBlock
    //   copyBlock
    //     └─ condbr (thread can copy?) -> loopHeader / exitBlock
    //   loopHeader
    //     └─ condbr (idx < loopLimit) -> loopBody / exitBlock
    //   loopBody
    //     └─ br loopHeader (idx += threadStride)
    //   exitBlock
    Block *copyBlock = rewriter.splitBlock(continuation, op->getIterator());
    Block *exitBlock = rewriter.splitBlock(copyBlock, op->getIterator());
    Block *loopHeader = rewriter.createBlock(
        op->getParentRegion(), Region::iterator(exitBlock), {i32_ty}, {loc});
    Block *loopBody = rewriter.createBlock(
        op->getParentRegion(), Region::iterator(exitBlock), {i32_ty}, {loc});

    rewriter.setInsertionPointToEnd(continuation);
    cf::BranchOp::create(rewriter, loc, copyBlock);

    rewriter.setInsertionPointToStart(copyBlock);
    Value segmentBase = segmentObj.segmentBase;
    Value index = b.load(i32_ty, segmentObj.indexPtr);
    auto bufferBaseType = segmentObj.base.getType();
    Value maxBufferWords = b.i32_val(segmentWordSize);
    Value effectiveBufferWords =
        b.select(b.icmp_slt(index, maxBufferWords), index, maxBufferWords);
    Value hasSegment = b.icmp_sge(segmentBase, b.i32_val(0));
    Value hasData = b.icmp_sge(effectiveBufferWords, b.i32_val(wordsPerEntry));
    Value shouldCopy = b.and_(hasSegment, hasData);
    Value threadStride = b.mul(threadsPerWarp, b.i32_val(wordsPerEntry));
    Value loopUpperBound =
        b.sub(effectiveBufferWords, b.i32_val(wordsPerEntry));
    // Each lane copies records in a warp-strided pattern.
    Value laneInitIdx = b.mul(laneId, b.i32_val(wordsPerEntry));
    Value laneWithinBounds = b.icmp_sle(laneInitIdx, loopUpperBound);
    Value threadShouldCopy = b.and_(shouldCopy, laneWithinBounds);

    auto &tritonTargetInfo = targetInfo.getTritonTargetInfo();
    auto copyWord = [&](Value bufOffset, Value gmemOffset, Attribute memory) {
      // Load the value from buffer and store it to global memory.
      Value ptr = b.gep(bufferBaseType, i32_ty, segmentObj.base, bufOffset);
      Value load;
      if (mlir::isa<triton::gpu::SharedMemorySpaceAttr>(memory)) {
        load = tritonTargetInfo.loadShared(rewriter, loc, ptr, i32_ty,
                                           b.true_val());
      } else {
        llvm::report_fatal_error(
            "unsupported memory space buffer in finalize copy");
      }

      Value gmemPtr = b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemOffset);
      b.store(load, gmemPtr);
    };

    // Write back the data.
    cf::CondBranchOp::create(rewriter, loc, threadShouldCopy, loopHeader,
                             ValueRange{laneInitIdx}, exitBlock, ValueRange{});

    rewriter.setInsertionPointToStart(loopHeader);
    BlockArgument headerIdx = loopHeader->getArgument(0);
    Value continueLoop = b.icmp_sle(headerIdx, loopUpperBound);
    cf::CondBranchOp::create(rewriter, loc, continueLoop, loopBody,
                             ValueRange{headerIdx}, exitBlock, ValueRange{});

    rewriter.setInsertionPointToStart(loopBody);
    BlockArgument bodyIdx = loopBody->getArgument(0);
    Value bufTagOffset = b.add(segmentBase, bodyIdx);
    Value bufCounterOffset = b.add(bufTagOffset, b.i32_val(1));
    Value gmemBaseOffset = b.add(b.i32_val(metadataWordSize), segmentBase);
    Value gmemWbTagOffset = b.add(gmemBaseOffset, bodyIdx);
    Value gmemWbCounterOffset = b.add(gmemWbTagOffset, b.i32_val(1));
    copyWord(bufTagOffset, gmemWbTagOffset, memSpace);
    copyWord(bufCounterOffset, gmemWbCounterOffset, memSpace);
    Value nextIdx = b.add(bodyIdx, threadStride);
    cf::BranchOp::create(rewriter, loc, loopHeader, ValueRange{nextIdx});

    rewriter.setInsertionPointToStart(exitBlock);
    return exitBlock;
  }

  void emitBlockLeaderEpilogue(mlir::triton::proton::gpu::FinalizeOp op,
                               Block *thenBlock, Value isBlockFirstThread,
                               Value scratchPtr,
                               LLVM::LLVMPointerType scratchPtrTy,
                               ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // Control-flow outline:
    //   thenBlock
    //     └─ condbr (block leader?) -> leaderBlock / continuation
    //   leaderBlock
    //     └─ ...body...
    //     └─ br continuation
    //   continuation
    Block *continuation = rewriter.splitBlock(thenBlock, op->getIterator());
    Block *leaderBlock = rewriter.createBlock(thenBlock->getParent(),
                                              Region::iterator(continuation));
    rewriter.setInsertionPointToEnd(thenBlock);
    cf::CondBranchOp::create(rewriter, loc, isBlockFirstThread, leaderBlock,
                             continuation);
    rewriter.setInsertionPointToStart(leaderBlock);

    Value gmemPostFinalTimeOffset = b.i32_val(8);
    Value gmemPostFinalTimePtr =
        b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemPostFinalTimeOffset);
    Value postFinalTime = targetInfo.globalTime(rewriter, loc);
    b.store(postFinalTime, gmemPostFinalTimePtr);
    cf::BranchOp::create(rewriter, loc, continuation);
    rewriter.setInsertionPointToStart(continuation);
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct SegmentAllocOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::SegmentAllocOp> {
  explicit SegmentAllocOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::SegmentAllocOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::SegmentAllocOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    int numWarps = getTotalNumWarps(mod);

    auto segmentType = op.getResult().getType();
    auto granularity = segmentType.getGranularity();
    auto selectIds = segmentType.getSelectIds();
    bool isAllIds = selectIds.empty() ? true : false;

    if (granularity != proton::gpu::Granularity::WARP) {
      mlir::emitError(loc, "granularity must be warp for now");
      return failure();
    }

    Value curThreadId = getRawThreadId(rewriter, loc);

    Value threadsPerWarp =
        b.i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value curWarpId = b.udiv(curThreadId, threadsPerWarp);
    const int bufferSizeInBytes = op.getSegment().getType().getNBytes();

    // Specialize the segment base address calculation might bring a few cycles
    // saving per record measurement overhead.
    Value segmentBase;
    if (isAllIds) {
      if (granularity == proton::gpu::Granularity::WARP)
        segmentBase =
            allWarpSegmentAlloc(b, curWarpId, numWarps, bufferSizeInBytes);
      else
        llvm::report_fatal_error(
            "segment address specialization not implemented yet");
    } else {
      segmentBase =
          defaultSegmentAlloc(b, curWarpId, selectIds, bufferSizeInBytes);
    }

    Value buffer = adaptor.getBuffer();
    Value bufferBase;
    if (isa<LLVM::LLVMPointerType>(buffer.getType())) {
      bufferBase = buffer;
    } else {
      Type bufferBaseTy =
          mlir::cast<LLVM::LLVMStructType>(buffer.getType()).getBody()[0];
      bufferBase = b.extract_val(bufferBaseTy, buffer, 0);
    }
    auto indexPtrTy =
        ptr_ty(rewriter.getContext(), targetInfo.getIndexPtrAddrSpace());
    auto indexPtr = LLVM::AllocaOp::create(rewriter, loc, indexPtrTy, i32_ty,
                                           b.i32_val(1), /*alignment=*/0);
    b.store(b.i32_val(0), indexPtr);

    auto segmentObj = LLVM::SegmentObject(bufferBase, segmentBase, indexPtr);
    auto llvmStruct = segmentObj.getStruct(loc, rewriter);
    rewriter.replaceOp(op, llvmStruct);
    return success();
  }

private:
  Value defaultSegmentAlloc(TritonLLVMOpBuilder &b, Value curWarpId,
                            llvm::ArrayRef<int> selectedIds,
                            int bufferSize) const {
    const int segmentWordSize = bufferSize / selectedIds.size() / 4;
    int warpSegmentAlloc = 0;
    Value segmentAlloc = b.i32_val(-1);
    for (int warpId : selectedIds) {
      segmentAlloc = b.select(b.icmp_eq(curWarpId, b.i32_val(warpId)),
                              b.i32_val(warpSegmentAlloc), segmentAlloc);
      warpSegmentAlloc += segmentWordSize;
    }
    return segmentAlloc;
  }

  Value allWarpSegmentAlloc(TritonLLVMOpBuilder &b, Value curWarpId,
                            int numWarps, int bufferSize) const {
    const int segmentWordSize = bufferSize / numWarps / 4;
    return b.mul(curWarpId, b.i32_val(segmentWordSize));
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct GlobalScratchAllocOpConversion
    : public ConvertOpToLLVMPattern<proton::gpu::GlobalScratchAllocOp> {
  explicit GlobalScratchAllocOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<proton::gpu::GlobalScratchAllocOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(proton::gpu::GlobalScratchAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();
    auto &tritonTargetInfo = targetInfo.getTritonTargetInfo();

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp) {
      return failure();
    }

    ModuleOp mod = funcOp.getOperation()->getParentOfType<ModuleOp>();
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx, 1);
    assert(op->hasAttr("offset"));
    size_t offset =
        cast<IntegerAttr>(op->getAttr("offset")).getValue().getZExtValue();

    Value allocOffset = b.i32_val(offset);

    // See NOTE: [Additional Function Arguments]
    if (!LLVM::isKernel(funcOp)) {
      // Base for this function
      auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() +
                                         kProfileScratchBufferOffset);

      Value ptr = b.gep(ptrTy, i8_ty, gmemBase, allocOffset);
      rewriter.replaceOp(op, ptr);
      return success();
    }

    // Base for entire kernel
    auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() +
                                       kProfileScratchBufferOffset);
    auto allocSizeAttr = mod.getOperation()->getAttrOfType<mlir::IntegerAttr>(
        "ttg.profile_scratch_memory_size");
    assert(allocSizeAttr);

    Value linearId = getLinearId(loc, rewriter);

    auto allocSize = allocSizeAttr.getValue().getZExtValue();
    Value gmemOffset =
        b.add(allocOffset, b.mul(linearId, b.i32_val(allocSize)));

    auto ptr = b.gep(ptrTy, i8_ty, gmemBase, gmemOffset);

    rewriter.replaceOp(op, ptr);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct InitCtxOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::InitCtxOp> {
  explicit InitCtxOpConversion(LLVMTypeConverter &typeConverter,
                               const proton::gpu::TargetInfoBase &targetInfo,
                               PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::InitCtxOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::InitCtxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value scratchPtr = adaptor.getScratchPtr();
    auto scratchPtrTy = mlir::cast<LLVM::LLVMPointerType>(scratchPtr.getType());

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    int numWarps = getTotalNumWarps(mod);

    // InitCtxOp can only be called in the master warps, so using `getThreadId`
    // is fine.
    Value threadId = getThreadId(rewriter, loc);
    Value isFirstThread = b.icmp_eq(threadId, b.i32_val(0));
    const int circularHeaderWordSize = proton::gpu::getCircularHeaderSize() / 4;

    Block *prevBlock = op->getBlock();

    // Add the 'if' block.
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    // Initialize the `warp_index` section.
    for (int warpId = 0; warpId < numWarps; warpId++) {
      Value warpIndexOffset = b.i32_val(warpId + circularHeaderWordSize);
      Value gmemWarpIndexPtr =
          b.gep(scratchPtrTy, i32_ty, scratchPtr, warpIndexOffset);
      b.store(b.i32_val(0), gmemWarpIndexPtr);
    }

    // Add the 'else' block and the condition.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(prevBlock);
    cf::CondBranchOp::create(rewriter, loc, isFirstThread, ifBlock, thenBlock);
    rewriter.setInsertionPointToEnd(ifBlock);
    cf::BranchOp::create(rewriter, loc, thenBlock);

    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct RestoreCtxOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::RestoreCtxOp> {
  explicit RestoreCtxOpConversion(LLVMTypeConverter &typeConverter,
                                  const proton::gpu::TargetInfoBase &targetInfo,
                                  PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::RestoreCtxOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::RestoreCtxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto segmentObj =
        LLVM::SegmentObject::fromStruct(loc, adaptor.getSegment(), rewriter);
    Value scratchPtr = adaptor.getScratchPtr();

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int numWarps = getTotalNumWarps(mod);

    // We need to use the absolute warp id in case warp specialization is used.
    Value threadId = getRawThreadId(rewriter, loc);

    Value warpId = b.udiv(
        threadId,
        b.i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod)));
    const int circularHeaderWordSize = proton::gpu::getCircularHeaderSize() / 4;

    auto scratchPtrTy = mlir::cast<LLVM::LLVMPointerType>(scratchPtr.getType());

    // Get the `warp_index` and store it into indexPtr.
    Value warpIndexOffset = b.add(warpId, b.i32_val(circularHeaderWordSize));
    Value gmemWarpIndexPtr =
        b.gep(scratchPtrTy, i32_ty, scratchPtr, warpIndexOffset);
    Value index = b.load(i32_ty, gmemWarpIndexPtr);
    b.store(index, segmentObj.indexPtr);

    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct SaveCtxOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::SaveCtxOp> {
  explicit SaveCtxOpConversion(LLVMTypeConverter &typeConverter,
                               const proton::gpu::TargetInfoBase &targetInfo,
                               PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::SaveCtxOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::SaveCtxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value scratchPtr = adaptor.getScratchPtr();
    auto scratchPtrTy = mlir::cast<LLVM::LLVMPointerType>(scratchPtr.getType());
    auto segmentObj =
        LLVM::SegmentObject::fromStruct(loc, adaptor.getSegment(), rewriter);

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    int numWarps = getTotalNumWarps(mod);

    int numLanes = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    Value warpSize = b.i32_val(numLanes);

    // We need to use the absolute warp id in case warp specialization is used.
    Value threadId = getRawThreadId(rewriter, loc);

    Value warpId = b.udiv(threadId, warpSize);
    Value laneId = b.urem(threadId, warpSize);
    Value isWarpMaster = b.icmp_eq(laneId, b.i32_val(0));
    const int circularHeaderWordSize = proton::gpu::getCircularHeaderSize() / 4;

    Block *prevBlock = op->getBlock();

    // Add the 'if' block.
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    // Update the `warp_index` section.
    Value warpIndexOffset = b.add(warpId, b.i32_val(circularHeaderWordSize));
    Value gmemWarpIndexPtr =
        b.gep(scratchPtrTy, i32_ty, scratchPtr, warpIndexOffset);
    Value index = b.load(i32_ty, segmentObj.indexPtr);
    b.store(index, gmemWarpIndexPtr);

    // Add the 'else' block and the condition.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(prevBlock);
    cf::CondBranchOp::create(rewriter, loc, isWarpMaster, ifBlock, thenBlock);
    rewriter.setInsertionPointToEnd(ifBlock);
    cf::BranchOp::create(rewriter, loc, thenBlock);

    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

Type convertProtonGPUMemDescType(triton::gpu::MemDescType type,
                                 const TargetInfoBase &targetInfo) {
  auto ctx = type.getContext();
  // base ptr
  auto ptrType = LLVM::LLVMPointerType::get(
      ctx, targetInfo.getAddressSpace(type.getMemorySpace()));

  SmallVector<Type, 4> types;
  types.push_back(ptrType);
  auto rank = type.getRank();
  // offsets
  for (auto i = 0; i < rank; i++) {
    types.push_back(IntegerType::get(ctx, 32));
  }

  return LLVM::LLVMStructType::getLiteral(ctx, types);
}

Type convertProtonGPUSegmentType(SegmentType type,
                                 const TargetInfoBase &targetInfo) {
  auto memorySpace = targetInfo.getAddressSpace(type.getMemorySpace());
  return LLVM::SegmentObject::getStructType(type.getContext(), memorySpace,
                                            targetInfo.getIndexPtrAddrSpace());
}

} // namespace

void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit) {
  patterns.add<ReadCounterOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<InitializeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<FinalizeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<SegmentAllocOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, targetInfo,
                                               benefit);
  patterns.add<InitCtxOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<RestoreCtxOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<SaveCtxOpConversion>(typeConverter, targetInfo, benefit);
}

void populateTypeConversions(LLVMTypeConverter &typeConverter,
                             const TargetInfoBase &targetInfo) {
  typeConverter.addConversion(
      [&](triton::gpu::MemDescType type) -> std::optional<Type> {
        return convertProtonGPUMemDescType(type, targetInfo);
      });
  typeConverter.addConversion(
      [&](proton::gpu::SegmentType type) -> std::optional<Type> {
        return convertProtonGPUSegmentType(type, targetInfo);
      });
  typeConverter.addConversion(
      [&](triton::PointerType type) -> std::optional<Type> {
        auto ctx = type.getContext();
        return LLVM::LLVMPointerType::get(ctx, type.getAddressSpace());
      });
}

} // namespace proton::gpu
} // namespace mlir::triton
