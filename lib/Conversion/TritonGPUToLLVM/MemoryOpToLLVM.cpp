#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Helper for LocalGather/ScatterOpConversion.
// For gather: storeVals is empty, returns loaded values.
// For scatter: storeVals contains values to store, returns empty.
SmallVector<Value> lowerLocalScGt(Location loc, MemDescType memDescTy,
                                  SharedMemoryObject smemObj, Type llvmElemTy,
                                  const LinearLayout &regLayout,
                                  ArrayRef<Value> idxValues, unsigned axis,
                                  ArrayRef<Value> storeVals,
                                  RewriterBase &rewriter,
                                  const TargetInfoBase &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  bool isScatter = !storeVals.empty();
  auto offsetAndBlock = computeBlockLocalOffsets(
      loc, memDescTy, regLayout, idxValues, axis, rewriter, targetInfo);
  SmallVector<LocalSharedMemoryAddress> addrs = materializeLocalAddrs(
      loc, memDescTy, smemObj, llvmElemTy, offsetAndBlock, rewriter);

  SmallVector<Value> results;
  if (!isScatter)
    results.resize(idxValues.size());

  for (auto [i, addr] : llvm::enumerate(addrs)) {
    if (isScatter) {
      targetInfo.storeDShared(rewriter, loc, addr.ptr, addr.ctaId, storeVals[i],
                              b.true_val());
    } else {
      results[i] = targetInfo.loadDShared(rewriter, loc, addr.ptr, addr.ctaId,
                                          llvmElemTy, b.true_val());
    }
  }

  return results;
}

LogicalResult lowerLocalStore(Location loc, MLIRContext *ctx, Value regVal,
                              MemDescType memDescTy, SharedMemoryObject smemObj,
                              ArrayRef<Value> inVals,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              const TargetInfoBase &targetInfo) {
  auto regTy = cast<RankedTensorType>(regVal.getType());
  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());

  auto regLayout = toLinearLayout(regTy);
  auto sharedLayout = isPaddedEncoding(memDescTy.getEncoding())
                          ? paddedLinearLayout(memDescTy)
                          : toLinearLayout(memDescTy);
  auto cvt = invertAndComposeBlockLocal(sharedLayout, regLayout);

  lowerLocalLdSt(loc, ctx, cvt, inVals, llvmElemTy, memDescTy, smemObj,
                 rewriter, targetInfo);

  return success();
}

struct GlobalScratchAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::GlobalScratchAllocOp> {
  const TargetInfoBase *targetInfo;

  GlobalScratchAllocOpConversion(LLVMTypeConverter &converter,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::GlobalScratchAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto opOffsetAttr = op->getAttrOfType<mlir::IntegerAttr>(
        "ttg.global_scratch_memory_offset");
    assert(opOffsetAttr);
    auto opOffset = opOffsetAttr.getValue().getZExtValue();

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp) {
      return failure();
    }
    Value ptr = op.getThirdPartyAllocation()
                    ? LLVM::getProfileScratchPtr(loc, rewriter, *targetInfo,
                                                 funcOp, b.i32_val(opOffset),
                                                 !op.getSharedClusterState())
                    : LLVM::getGlobalScratchPtr(loc, rewriter, *targetInfo,
                                                funcOp, b.i32_val(opOffset));

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();
    Location loc = op->getLoc();
    // Get all shared memory bases (one for non-partitioned, multiple for
    // partitioned tensors)
    SmallVector<Value> smemBases = LLVM::getSharedMemoryBases(
        loc, rewriter, targetInfo, op.getOperation());
    auto memDescTy = cast<MemDescType>(op.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = SharedMemoryObject(smemBases, llvmElemTy,
                                      memDescTy.getRank(), loc, rewriter);
    // If there is an initial tensor, store it into the shared memory.
    if (op.getSrc()) {
      auto *ctx = op.getContext();
      auto inVals = unpackTensorElements(loc, adaptor.getSrc(), rewriter,
                                         op.getSrc().getType());
      if (failed(lowerLocalStore(loc, ctx, op.getSrc(), memDescTy, smemObj,
                                 inVals, typeConverter, rewriter,
                                 targetInfo))) {
        return failure();
      }
    }
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalDeallocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalDeallocOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalDeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct LocalLoadOpConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
public:
  LocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescVal = op.getSrc();
    auto regVal = op.getResult();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto regTy = cast<RankedTensorType>(regVal.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    auto regLayout = toLinearLayout(regTy);
    auto sharedLayout = isPaddedEncoding(memDescTy.getEncoding())
                            ? paddedLinearLayout(memDescTy)
                            : toLinearLayout(memDescTy);
    auto cvt = invertAndComposeBlockLocal(sharedLayout, regLayout);

    auto outVals = lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, memDescTy,
                                  smemObj, rewriter, targetInfo, op);

    Value result =
        packTensorElements(loc, typeConverter, outVals, rewriter, regTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalStoreOp>::ConvertOpToLLVMPattern;

  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    Value regVal = op.getSrc();
    Value memDescVal = op.getDst();
    auto typeConverter = getTypeConverter();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(),
                                                         llvmElemTy, rewriter);
    auto inVals = unpackTensorElements(loc, adaptor.getSrc(), rewriter,
                                       op.getSrc().getType());
    if (failed(lowerLocalStore(loc, ctx, regVal, memDescTy, smemObj, inVals,
                               typeConverter, rewriter, targetInfo))) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

class BarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
public:
  BarrierOpConversion(const LLVMTypeConverter &converter,
                      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::BarrierOp>(converter, benefit) {}
  using OpAdaptor = typename triton::gpu::BarrierOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::gpu::BarrierOp>(op);
    return success();
  }
};

struct LocalGatherOpConversion : public ConvertOpToLLVMPattern<LocalGatherOp> {
public:
  LocalGatherOpConversion(LLVMTypeConverter &typeConverter,
                          const TargetInfoBase &targetInfo,
                          PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto memDescTy = cast<MemDescType>(op.getSrc().getType());
    // TODO: PartitionedSharedEncoding lowering will be enabled in subsequent
    // PRs.
    if (isa<triton::gpu::PartitionedSharedEncodingAttr>(
            memDescTy.getEncoding())) {
      return rewriter.notifyMatchFailure(
          op, "PartitionedSharedEncoding not yet supported in lowering");
    }
    auto regTy = cast<RankedTensorType>(op.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    SmallVector<Value> idxValues = unpackTensorElements(
        loc, adaptor.getIndices(), rewriter, op.getIndices().getType());
    auto regLayout = toLinearLayout(regTy);

    auto results = lowerLocalScGt(loc, memDescTy, smemObj, llvmElemTy,
                                  regLayout, idxValues, op.getAxis(),
                                  /*storeVals=*/{}, rewriter, targetInfo);

    Value result =
        packTensorElements(loc, typeConverter, results, rewriter, regTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalScatterOpConversion
    : public ConvertOpToLLVMPattern<LocalScatterOp> {
public:
  LocalScatterOpConversion(LLVMTypeConverter &typeConverter,
                           const TargetInfoBase &targetInfo,
                           PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto memDescTy = cast<MemDescType>(op.getDst().getType());
    // TODO: PartitionedSharedEncoding lowering will be enabled in subsequent
    // PRs.
    if (isa<triton::gpu::PartitionedSharedEncodingAttr>(
            memDescTy.getEncoding())) {
      return rewriter.notifyMatchFailure(
          op, "PartitionedSharedEncoding not yet supported in lowering");
    }
    auto valuesTy = cast<RankedTensorType>(op.getValues().getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(),
                                                         llvmElemTy, rewriter);

    SmallVector<Value> values = unpackTensorElements(
        loc, adaptor.getValues(), rewriter, op.getValues().getType());
    SmallVector<Value> idxValues = unpackTensorElements(
        loc, adaptor.getIndices(), rewriter, op.getIndices().getType());
    auto regLayout = toLinearLayout(valuesTy);

    lowerLocalScGt(loc, memDescTy, smemObj, llvmElemTy, regLayout, idxValues,
                   op.getAxis(), values, rewriter, targetInfo);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct AtomicPollOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicPollOp> {
  AtomicPollOpConversion(LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicPollOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::AtomicPollOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicPollOp");
    int numCTAs = TritonGPUDialect::getNumCTAs(moduleOp);
    if (numCTAs != 1 && !targetInfo.isCuda())
      return rewriter.notifyMatchFailure(
          op, "multi-CTA atomic_poll requires cross-CTA shared memory");

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    StringRef syncScope = targetInfo.getAtomicSyncScope(op.getScope());
    unsigned bitWidth = adaptor.getExpected().getType().getIntOrFloatBitWidth();

    // Split the block at the poll and branch only the elected thread into the
    // polling loop. All other threads skip directly to the rendezvous block.
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *doneBlock = currentBlock->splitBlock(rewriter.getInsertionPoint());
    Region *region = currentBlock->getParent();
    Block *pollInitBlock =
        rewriter.createBlock(region, Region::iterator(doneBlock));
    Block *pollLoopBlock =
        rewriter.createBlock(region, Region::iterator(doneBlock));
    Block *pollSuccessBlock =
        rewriter.createBlock(region, Region::iterator(doneBlock));
    Block *timeoutCheckBlock =
        adaptor.getTimeout()
            ? rewriter.createBlock(region, Region::iterator(doneBlock))
            : nullptr;
    BlockArgument matched = doneBlock->addArgument(i1_ty, loc);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, threadPred, pollInitBlock,
                           ValueRange{}, doneBlock, ValueRange{b.false_val()});

    rewriter.setInsertionPointToEnd(pollInitBlock);
    Value start;
    if (adaptor.getTimeout())
      start = targetInfo.getGlobalTimer(rewriter, loc);
    LLVM::BrOp::create(rewriter, loc, pollLoopBlock);

    rewriter.setInsertionPointToEnd(pollLoopBlock);
    Value loaded = LLVM::LoadOp::create(
        rewriter, loc, adaptor.getExpected().getType(), adaptor.getPtr(),
        bitWidth / 8, /*isVolatile=*/false, /*isNonTemporal=*/false,
        /*isInvariant=*/false, /*isInvariantGroup=*/false,
        LLVM::AtomicOrdering::monotonic, syncScope);
    Value pollMatched = b.icmp_eq(loaded, adaptor.getExpected());
    if (adaptor.getTimeout()) {
      LLVM::CondBrOp::create(rewriter, loc, pollMatched, pollSuccessBlock,
                             timeoutCheckBlock);

      rewriter.setInsertionPointToEnd(timeoutCheckBlock);
      Value elapsed = b.sub(targetInfo.getGlobalTimer(rewriter, loc), start);
      Value timedOut = b.icmp_uge(elapsed, adaptor.getTimeout());
      LLVM::CondBrOp::create(rewriter, loc, timedOut, doneBlock,
                             ValueRange{b.false_val()}, pollLoopBlock,
                             ValueRange{});
    } else {
      LLVM::CondBrOp::create(rewriter, loc, pollMatched, pollSuccessBlock,
                             pollLoopBlock);
    }

    rewriter.setInsertionPointToEnd(pollSuccessBlock);
    if (op.getSem() == triton::MemSemantic::ACQUIRE)
      LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::acquire,
                            syncScope);
    LLVM::BrOp::create(rewriter, loc, ValueRange{b.true_val()}, doneBlock);

    rewriter.setInsertionPointToStart(doneBlock);
    if (!adaptor.getTimeout()) {
      // Successful completion is the only possible result without a timeout,
      // so rendezvous and return true without a shared-memory broadcast.
      if (numCTAs == 1)
        targetInfo.barrier(loc, rewriter, AddrSpace::Local);
      else
        targetInfo.clusterBarrier(loc, rewriter, op);
      rewriter.replaceOp(op, b.true_val());
      return success();
    }

    // Broadcast the elected thread's result after every thread has left the
    // loop, preserving the scalar result convention used by Triton atomics.
    if (op.getResult().use_empty()) {
      if (numCTAs == 1)
        targetInfo.barrier(loc, rewriter, AddrSpace::Local);
      else
        targetInfo.clusterBarrier(loc, rewriter, op);
      rewriter.eraseOp(op);
      return success();
    }

    Value atomPtr =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    atomPtr = b.bitcast(atomPtr, ptr_ty(rewriter.getContext(),
                                        targetInfo.getSharedAddressSpace()));
    targetInfo.storeShared(rewriter, loc, atomPtr, matched, threadPred);
    if (numCTAs == 1)
      targetInfo.barrier(loc, rewriter, AddrSpace::Local);
    else
      targetInfo.clusterBarrier(loc, rewriter, op);

    Value result;
    if (numCTAs == 1) {
      result = b.load(i1_ty, atomPtr);
    } else {
      // Scalar operations are issued by CTA 0, so read CTA 0's scratch.
      result = targetInfo.loadDShared(rewriter, loc, atomPtr, b.i32_val(0),
                                      i1_ty, b.true_val());
    }
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, targetInfo,
                                               benefit);
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalDeallocOpConversion>(typeConverter, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalGatherOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalScatterOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<BarrierOpConversion>(typeConverter, benefit);
  patterns.add<AtomicPollOpConversion>(typeConverter, targetInfo, benefit);
}
