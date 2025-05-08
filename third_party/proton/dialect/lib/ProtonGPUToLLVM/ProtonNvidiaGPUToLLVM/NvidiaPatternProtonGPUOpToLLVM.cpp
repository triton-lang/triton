#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/NvidiaPatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

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

    Value indexPtr = adaptor.getIndexPtr();
    Value dataStruct = adaptor.getData();
    Value scratchPtr = adaptor.getPtr();

    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    const int bytesPerEntry = proton::gpu::getBytesPerClockEntry();
    const int wordsPerEntry = bytesPerEntry / 4; // 1 word = 4 bytes

    int numWarps = mlir::triton::gpu::lookupNumWarps(mod);
    Value threadId = getThreadId(rewriter, loc);
    Value warpId = b.udiv(
        threadId,
        b.i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod)));
    Value isFirstThread = b.icmp_eq(threadId, b.i32_val(0));

    auto bufferTy =
        mlir::cast<triton::gpu::MemDescType>(op.getData().getType());
    const int bufferSizeInWords =
        mlir::ShapedType::getNumElements(bufferTy.getShape()) *
        bufferTy.getElementType().getIntOrFloatBitWidth() / 32;

    const int circularHeaderWordSize = proton::gpu::getCircularHeaderSize() / 4;

    // Header: preamble (1 word), threadblock id (1 word), SM id (1 word),
    // buffer size (1 word)

    // Circular strategy memory layout (total: allocprofileScratchSize bytes)
    //  +-----------------------------------------------+
    //  | header (circularHeaderSize bytes)             |
    //  +-----------------------------------------------+
    //  | number of events per warp (4 bytes x numWarps)|
    //  +-----------------------------------------------+
    //  | profiled data (allocBufferSize bytes)         |
    //  +-----------------------------------------------+
    const int metadataWordSize = circularHeaderWordSize + numWarps;
    const int scratchWordSize = metadataWordSize + bufferSizeInWords;

    auto &tritonTargetInfo = targetInfo.getTritonTargetInfo();

    Value hwid = targetInfo.processorId(rewriter, loc);

    auto scratchPtrTy = mlir::cast<LLVM::LLVMPointerType>(scratchPtr.getType());
    auto bufferPtrTy =
        mlir::cast<LLVM::LLVMStructType>(dataStruct.getType()).getBody()[0];
    Value bufferBasePtr = b.extract_val(bufferPtrTy, dataStruct, 0);

    // Add the `warp_index` section.
    Value warpIndexOffset = b.add(warpId, b.i32_val(circularHeaderWordSize));
    Value gmemWarpIndexPtr =
        b.gep(scratchPtrTy, i32_ty, scratchPtr, warpIndexOffset);
    Value index = b.load(i32_ty, indexPtr);
    b.store(index, gmemWarpIndexPtr);

    Block *prevBlock = op->getBlock();
    // Add the 'if' block.
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    auto copyWord = [&](Value bufOffset, Value gmemOffset, Attribute memSpace) {
      // Load the value from buffer
      Value ptr = b.gep(bufferPtrTy, i32_ty, bufferBasePtr, bufOffset);
      Value load;
      if (mlir::isa<triton::proton::gpu::StackMemorySpaceAttr>(memSpace)) {
        llvm::report_fatal_error("unimplemented");
      } else if (mlir::isa<triton::gpu::SharedMemorySpaceAttr>(memSpace)) {
        load = tritonTargetInfo.loadShared(rewriter, loc, ptr, i32_ty,
                                           b.true_val());
      } else {
        llvm::report_fatal_error(
            "unsupported memory space buffer in finalize copy");
      }

      // Store the value to global memory
      Value gmemPtr = b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemOffset);
      b.store(load, gmemPtr);
    };
    // Write back 'preamble'.
    Value preamble = b.i32_val(0xdeadbeef);
    Value gmemPreambleOffset = b.i32_val(0);
    Value gmemPreamblePtr =
        b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemPreambleOffset);
    b.store(preamble, gmemPreamblePtr);

    // Write back 'program id'.
    Value gmemPidOffset = b.i32_val(1);
    Value gmemPidPtr = b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemPidOffset);
    Value pid = mlir::triton::proton::gpu::getLinearId(loc, rewriter);
    b.store(pid, gmemPidPtr);

    // Write back 'hw id'.
    Value gmemHwidOffset = b.i32_val(2);
    Value gmemHwidPtr = b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemHwidOffset);
    b.store(hwid, gmemHwidPtr);

    // Write back 'buffer size in byte'.
    Value gmemBufSizeOffset = b.i32_val(3);
    Value gmemBufSizePtr =
        b.gep(scratchPtrTy, i32_ty, scratchPtr, gmemBufSizeOffset);
    b.store(b.i32_val(bufferSizeInWords * 4), gmemBufSizePtr);

    // Add the 'else' block and the condition.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, isFirstThread, ifBlock, thenBlock);

    // Write back the data.
    const int upper = bufferSizeInWords - wordsPerEntry;
    rewriter.setInsertionPointToEnd(ifBlock);
    Value initIdx = b.i32_val(0);
    Value wbBaseOffset = b.i32_val(metadataWordSize);

    Block *writeBackBlock = rewriter.createBlock(
        op->getParentRegion(), std::next(Region::iterator(ifBlock)), {i32_ty},
        {loc});
    rewriter.setInsertionPointToStart(writeBackBlock);
    BlockArgument idx = writeBackBlock->getArgument(0);
    Value gmemWbTagOffset = b.add(wbBaseOffset, idx);
    Value gmemWbCounterOffset = b.add(gmemWbTagOffset, b.i32_val(1));

    Value bufTagOffset = idx;
    Value bufCounterOffset = b.add(bufTagOffset, b.i32_val(1));
    auto memDescTy =
        mlir::cast<triton::gpu::MemDescType>(op.getData().getType());
    auto memSpace = memDescTy.getMemorySpace();
    copyWord(bufTagOffset, gmemWbTagOffset, memSpace);
    copyWord(bufCounterOffset, gmemWbCounterOffset, memSpace);
    Value pred = b.icmp_slt(idx, b.i32_val(upper));
    Value updatedIdx = b.add(idx, b.i32_val(wordsPerEntry));
    rewriter.create<cf::CondBranchOp>(loc, pred, writeBackBlock, updatedIdx,
                                      thenBlock, ArrayRef<Value>());

    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, writeBackBlock, initIdx);

    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

// Circular strategy memory layout of profiled data (total: N bytes).
// Assuming we record data from warp 0, 2, 7 so buffer looks like:
//  +-----------------------------------------------+
//  | warp 0 data (N/3 bytes)                       |
//  +-----------------------------------------------+
//  | warp 2 data (N/3 bytes)                       |
//  +-----------------------------------------------+
//  | warp 7 data (N/3 bytes)                       |
//  +-----------------------------------------------+

struct CircularStoreOpConversion
    : public ConvertOpToLLVMPattern<
          mlir::triton::proton::gpu::CircularStoreOp> {
  explicit CircularStoreOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<
            mlir::triton::proton::gpu::CircularStoreOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::CircularStoreOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    const int bytesPerEntry = proton::gpu::getBytesPerClockEntry();
    const int wordsPerEntry = bytesPerEntry / 4; // 1 word = 4 bytes

    Value indexPtr = adaptor.getIndexPtr();
    Value dataStruct = adaptor.getData();
    auto bufferPtrTy =
        mlir::cast<LLVM::LLVMStructType>(dataStruct.getType()).getBody()[0];

    Value bufferDataBasePtr = b.extract_val(bufferPtrTy, dataStruct, 0);

    // Update the index (could be register promoted).
    Value curIdx = b.load(i32_ty, indexPtr);
    Value newIdx = b.add(curIdx, b.i32_val(wordsPerEntry));
    b.store(newIdx, indexPtr);

    // Compute the segment size in word (4 bytes).
    auto segbaseOp =
        mlir::cast<proton::gpu::SegmentBaseOp>(op.getSeg().getDefiningOp());
    int selectedWarpNum = mlir::triton::gpu::lookupNumWarps(mod);
    auto selectedIds = segbaseOp.getSelectIdsAttr().asArrayRef();
    if (!selectedIds.empty())
      selectedWarpNum = selectedIds.size();
    auto memDescTy =
        mlir::cast<triton::gpu::MemDescType>(op.getData().getType());
    const int bufferSizeInBytes =
        mlir::ShapedType::getNumElements(memDescTy.getShape()) *
        memDescTy.getElementType().getIntOrFloatBitWidth() / 8;
    const int segmentWordSize = bufferSizeInBytes / selectedWarpNum / 4;

    // Compute the actual base offset (with urem as circular buffer).
    Value segmentBase = adaptor.getSeg();
    Value tagOffset =
        b.add(segmentBase, b.urem(curIdx, b.i32_val(segmentWordSize)));

    // Store the counter into buffer.
    Value vecPtr = b.gep(bufferPtrTy, i32_ty, bufferDataBasePtr, tagOffset);
    Value tag = op.getIsStart() ? b.i32_val(op.getScopeId())
                                : b.i32_val(1 << 31 | op.getScopeId());
    Value clock = op.getCounter();
    Value valsVec = packLLVector(loc, {tag, clock}, rewriter);

    // Compute the predicate for the writer.
    const int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    Value curThreadId = getThreadId(rewriter, loc);
    Value isWarpMaster =
        b.icmp_eq(b.urem(curThreadId, b.i32_val(warpSize)), b.i32_val(0));
    Value isWriter;

    auto granularity = segbaseOp.getGranularity();
    if (selectedIds.empty()) {
      if (granularity == proton::gpu::Granularity::WARP) {
        isWriter = isWarpMaster;
      } else {
        llvm::report_fatal_error(
            "segment address specialization not implemented yet");
      }
    } else {
      Value isCurWarpEnabled = b.icmp_ne(segmentBase, b.i32_val(-1));
      isWriter = b.and_(isCurWarpEnabled, isWarpMaster);
    }
    uint32_t AddrSpace =
        cast<LLVM::LLVMPointerType>(bufferPtrTy).getAddressSpace();
    if (AddrSpace == 1) {
      llvm::report_fatal_error("unimplemented");
    } else if (AddrSpace == 3) {
      targetInfo.getTritonTargetInfo().storeDShared(rewriter, loc, vecPtr,
                                                    std::nullopt, valsVec,
                                                    /*pred=*/isWriter);
    } else {
      llvm::report_fatal_error("unsupported address space in circular store");
    }
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

} // namespace

namespace mlir::triton::proton::gpu::NVIDIA {
void populateProtonGPUOpNvidiaPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       const TargetInfo &targetInfo,
                                       PatternBenefit benefit) {
  patterns.add<FinalizeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<CircularStoreOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::proton::gpu::NVIDIA
