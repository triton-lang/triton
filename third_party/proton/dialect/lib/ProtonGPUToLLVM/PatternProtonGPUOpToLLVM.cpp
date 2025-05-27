#include "Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/PatternMatch.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
namespace proton::gpu {

Value getLinearId(Location loc, ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Note:
  // 1. We compute use i64 data type to compute and then truncate to i32
  // to support various backend intrinsics (e.g. amd).
  // 2. We avoid using the targetInfo's programId() because of its coupling
  // with cluster id in Nvidia TritonGPU's llvm lowering.
  Value pidX = rewriter.create<arith::IndexCastOp>(
      loc, i64_ty,
      rewriter.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x));
  Value pidY = rewriter.create<arith::IndexCastOp>(
      loc, i64_ty,
      rewriter.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::y));
  Value pidZ = rewriter.create<arith::IndexCastOp>(
      loc, i64_ty,
      rewriter.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::z));

  Value gridDimX = rewriter.create<arith::IndexCastOp>(
      loc, i64_ty,
      rewriter.create<::mlir::gpu::GridDimOp>(loc, mlir::gpu::Dimension::x));
  Value gridDimY = rewriter.create<arith::IndexCastOp>(
      loc, i64_ty,
      rewriter.create<::mlir::gpu::GridDimOp>(loc, mlir::gpu::Dimension::y));
  Value linearId =
      b.trunc(i32_ty, b.add(b.add(pidX, b.mul(pidY, gridDimX)),
                            b.mul(pidZ, b.mul(gridDimX, gridDimY))));
  return linearId;
}

namespace {

struct InitBufferIndexOpConversion
    : public ConvertOpToLLVMPattern<
          mlir::triton::proton::gpu::InitBufferIndexOp> {
  explicit InitBufferIndexOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<
            mlir::triton::proton::gpu::InitBufferIndexOp>(typeConverter,
                                                          benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::InitBufferIndexOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ptrTy = ptr_ty(rewriter.getContext(), IndexPtrAddrSpace);
    auto indexPtr = rewriter.create<LLVM::AllocaOp>(
        loc, ptrTy, i32_ty, b.i32_val(1), /*alignment=*/0);
    b.store(b.i32_val(0), indexPtr);
    rewriter.replaceOp(op, indexPtr);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

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
    Value clock = targetInfo.clock(rewriter, op.getLoc(), false);
    rewriter.replaceOp(op, clock);
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

    Value indexPtr = adaptor.getIndexPtr();
    Value dataStruct = adaptor.getData();
    Value scratchPtr = adaptor.getScratchPtr();

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
    Value pid = getLinearId(loc, rewriter);
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

struct StackAllocOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::StackAllocOp> {
  explicit StackAllocOpConversion(LLVMTypeConverter &typeConverter,
                                  const proton::gpu::TargetInfoBase &targetInfo,
                                  PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::StackAllocOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::StackAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Location loc = op.getLoc();
    auto ctx = moduleOp.getContext();

    auto bufferTy =
        mlir::cast<triton::gpu::MemDescType>(op.getData().getType());

    const int bufferSizeInBytes =
        mlir::ShapedType::getNumElements(bufferTy.getShape()) *
        bufferTy.getElementType().getIntOrFloatBitWidth() / 8;

    auto bufferSizeVal = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getIntegerType(32),
        IntegerAttr::get(rewriter.getIntegerType(32), bufferSizeInBytes / 4));

    auto llvmPointerType = LLVM::LLVMPointerType::get(op->getContext());
    Type llvmInt32Type = IntegerType::get(op->getContext(), 32);
    Value arrayVal = rewriter.create<LLVM::AllocaOp>(
        loc, llvmPointerType, llvmInt32Type, bufferSizeVal);

    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // TODO(crobeck): update if we ever support multi-rank stack alloc ops
    SmallVector<Type, 4> types = {ptr_ty(ctx), llvmInt32Type};
    SmallVector<Value, 4> elems = {arrayVal,
                                   bufferSizeVal}; // i32 ptr, shape[0]

    auto structTy =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);

    Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structTy);
    for (const auto &v : llvm::enumerate(elems)) {
      llvmStruct = b.insert_val(structTy, llvmStruct, v.value(), v.index());
    }
    rewriter.replaceOp(op, llvmStruct);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

struct SegmentBaseOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::gpu::SegmentBaseOp> {
  explicit SegmentBaseOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::gpu::SegmentBaseOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::SegmentBaseOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int numWarps = mlir::triton::gpu::lookupNumWarps(mod);
    auto granularity = op.getGranularity();
    auto selectIdsAttr = op.getSelectIdsAttr();

    llvm::ArrayRef<int> selectedIds;
    bool isAllIds = false;

    if (selectIdsAttr.asArrayRef().size())
      selectedIds = selectIdsAttr.asArrayRef();
    else
      isAllIds = true;

    if (granularity != proton::gpu::Granularity::WARP) {
      mlir::emitError(loc, "granularity must be warp for now");
      return failure();
    }

    Value curThreadId = getThreadId(rewriter, loc);
    Value threadsPerWarp =
        b.i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value curWarpId = b.udiv(curThreadId, threadsPerWarp);

    auto bufferTy =
        mlir::cast<triton::gpu::MemDescType>(op.getData().getType());
    const int bufferSizeInBytes =
        mlir::ShapedType::getNumElements(bufferTy.getShape()) *
        bufferTy.getElementType().getIntOrFloatBitWidth() / 8;

    auto defaultSegmentBaseFunc = [&](int bufferSize) -> Value {
      const int segmentWordSize = bufferSize / selectedIds.size() / 4;
      int warpSegmentBase = 0;

      Value segmentBase = b.i32_val(-1);
      for (int warpId : selectedIds) {
        segmentBase = b.select(b.icmp_eq(curWarpId, b.i32_val(warpId)),
                               b.i32_val(warpSegmentBase), segmentBase);
        warpSegmentBase += segmentWordSize;
      }
      return segmentBase;
    };

    auto allWarpSegmentBaseFunc = [&](int bufferSize) -> Value {
      const int segmentWordSize = bufferSize / numWarps / 4;
      // TODO(fywkevin): assert segmentWordSize and numWarps power of 2
      Value segmentBase = b.mul(curWarpId, b.i32_val(segmentWordSize));
      return segmentBase;
    };

    // Specialize the segment base address calculation might bring a few cycles
    // saving per record measurement overhead.
    Value res;
    if (isAllIds) {
      if (granularity == proton::gpu::Granularity::WARP)
        res = allWarpSegmentBaseFunc(bufferSizeInBytes);
      else
        llvm::report_fatal_error(
            "segment address specialization not implemented yet");
    } else {
      res = defaultSegmentBaseFunc(bufferSizeInBytes);
    }

    rewriter.replaceOp(op, res);
    return success();
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
      auto gmemBase = funcOp.getArgument(
          funcOp.getNumArguments() + proton::gpu::kGlobalScratchBufferOffset);

      Value ptr = b.gep(ptrTy, i8_ty, gmemBase, allocOffset);
      rewriter.replaceOp(op, ptr);
      return success();
    }

    // Base for entire kernel
    auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() +
                                       proton::gpu::kGlobalScratchBufferOffset);
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

Type convertProtonGPUSegmentBaseType(SegmentBaseType type) {
  return IntegerType::get(type.getContext(), 32);
}

} // namespace

void populateProtonGPUOpPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit) {
  patterns.add<InitBufferIndexOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ReadCounterOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<FinalizeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<SegmentBaseOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<StackAllocOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, targetInfo,
                                               benefit);
}

void populateTypeConversions(LLVMTypeConverter &typeConverter,
                             const TargetInfoBase &targetInfo) {
  typeConverter.addConversion(
      [&](triton::gpu::MemDescType type) -> std::optional<Type> {
        return convertProtonGPUMemDescType(type, targetInfo);
      });
  typeConverter.addConversion(
      [&](proton::gpu::SegmentBaseType type) -> std::optional<Type> {
        return convertProtonGPUSegmentBaseType(type);
      });
}

} // namespace proton::gpu
} // namespace mlir::triton
