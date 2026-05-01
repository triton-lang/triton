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
SmallVector<Value>
lowerLocalScGt(Location loc, MLIRContext *ctx, MemDescType memDescTy,
               SharedMemoryObject smemObj, Type llvmElemTy,
               ArrayRef<Value> idxValues, ArrayRef<SmallVector<Value>> coords,
               unsigned axis, ArrayRef<Value> storeVals, RewriterBase &rewriter,
               const TargetInfoBase &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  bool isScatter = !storeVals.empty();
  SmallVector<Value> ptrs = computeLocalPtrs(
      loc, memDescTy, smemObj, llvmElemTy, idxValues, coords, axis, rewriter);

  SmallVector<Value> results;
  if (!isScatter)
    results.resize(coords.size());

  for (auto [i, ptr] : llvm::enumerate(ptrs)) {
    if (isScatter) {
      targetInfo.storeShared(rewriter, loc, ptr, storeVals[i], b.true_val());
    } else {
      results[i] =
          targetInfo.loadShared(rewriter, loc, ptr, llvmElemTy, b.true_val());
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
  auto cvt = regLayout.invertAndCompose(sharedLayout);

  auto kBlock = str_attr("block");
  // We could support it by removing this check if we ever want to
  if (!cvt.isTrivialOver({kBlock})) {
    return failure();
  }
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
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
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
    auto cvt = regLayout.invertAndCompose(sharedLayout);

    auto kBlock = str_attr("block");
    // We could support it by removing this check if we ever want to
    if (!cvt.isTrivialOver({kBlock})) {
      return failure();
    }

    auto outVals = lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, memDescTy,
                                  smemObj, rewriter, targetInfo, op);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, regTy);
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
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
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
    auto *ctx = op.getContext();
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

    SmallVector<Value> idxValues =
        unpackLLElements(loc, adaptor.getIndices(), rewriter);
    SmallVector<SmallVector<Value>> dstIndices =
        emitIndices(loc, rewriter, targetInfo, regTy.getEncoding(), regTy,
                    /*withCTAOffset=*/true);

    auto results = lowerLocalScGt(loc, ctx, memDescTy, smemObj, llvmElemTy,
                                  idxValues, dstIndices, op.getAxis(),
                                  /*storeVals=*/{}, rewriter, targetInfo);

    Value result = packLLElements(loc, typeConverter, results, rewriter, regTy);
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
    auto *ctx = op.getContext();
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

    SmallVector<Value> values =
        unpackLLElements(loc, adaptor.getValues(), rewriter);
    SmallVector<Value> idxValues =
        unpackLLElements(loc, adaptor.getIndices(), rewriter);
    SmallVector<SmallVector<Value>> srcIndices =
        emitIndices(loc, rewriter, targetInfo, valuesTy.getEncoding(), valuesTy,
                    /*withCTAOffset=*/true);

    lowerLocalScGt(loc, ctx, memDescTy, smemObj, llvmElemTy, idxValues,
                   srcIndices, op.getAxis(), values, rewriter, targetInfo);

    rewriter.eraseOp(op);
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
}
