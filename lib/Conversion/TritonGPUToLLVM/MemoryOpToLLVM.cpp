#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// blocked -> shared.
// Swizzling in shared memory to avoid bank conflict. Normally used for
// A/B operands of dots.
void lowerDistributedToShared(
    Location loc, Value src, Value dst, Value adaptorSrc,
    const SharedMemoryObject &smemObj, const LLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, const TargetInfoBase &targetInfo,
    std::pair<size_t, Type> *const llvmOpCount = nullptr) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto dstTy = cast<MemDescType>(dst.getType());
  auto outOrd = mlir::cast<SharedEncodingAttr>(dstTy.getEncoding()).getOrder();
  assert(srcTy.getShape().size() <= 2 ||
         (srcTy.getShape().size() == 3 && outOrd[2] == 0) &&
             "Unexpected rank of ConvertLayout(blocked->shared)");
  auto elemTy = typeConverter->convertType(srcTy.getElementType());

  auto smemBase = smemObj.getBase();
  auto dstStrides = smemObj.getStrides();
  auto inVals = unpackLLElements(loc, adaptorSrc, rewriter);
  storeDistributedToShared(dstTy, srcTy, elemTy, inVals, smemBase, dstStrides,
                           loc, rewriter, targetInfo, llvmOpCount);
}

struct GlobalScratchAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::GlobalScratchAllocOp> {
  GlobalScratchAllocOpConversion(LLVMTypeConverter &converter,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::gpu::GlobalScratchAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto opOffsetAttr = op->getAttrOfType<mlir::IntegerAttr>(
        "triton_gpu.global_scratch_memory_offset");
    assert(opOffsetAttr);
    auto opOffset = opOffsetAttr.getValue().getZExtValue();

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp) {
      return failure();
    }
    Value ptr =
        LLVM::getGlobalScratchPtr(loc, rewriter, funcOp, i32_val(opOffset));

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
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto resultTy = cast<MemDescType>(op.getType());
    auto typeConverter = getTypeConverter();
    auto sharedLayout =
        cast<triton::gpu::SharedEncodingAttr>(resultTy.getEncoding());
    auto order = sharedLayout.getOrder();
    // Workaround for 3D tensors
    // TODO: we need to modify the pipeline pass to give a proper shared
    // encoding to 3D tensors
    SmallVector<unsigned> newOrder;
    if (resultTy.getShape().size() != order.size()) {
      for (auto i = 0; i < order.size(); ++i)
        newOrder.push_back(order[i] + 1);
      newOrder.push_back(0);
    } else {
      newOrder = SmallVector<unsigned>(order.begin(), order.end());
    }

    auto llvmElemTy = typeConverter->convertType(resultTy.getElementType());
    auto shapePerCTA = getShapePerCTA(sharedLayout, resultTy.getShape());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, shapePerCTA,
                                      newOrder, loc, rewriter);
    // If there is an initial tensor, store it into the shared memory.
    if (op.getSrc()) {
      lowerDistributedToShared(loc, op.getSrc(), op.getResult(),
                               adaptor.getSrc(), smemObj, typeConverter,
                               rewriter, targetInfo);
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

  // FIXME [Dot LL]
  // Do for all DotOperandEncodingAttr once we have LLs for all of them
  static bool isSupportedDotOpLayout(MemDescType srcTy,
                                     RankedTensorType dstTy) {
    auto srcLayout = cast<SharedEncodingAttr>(srcTy.getEncoding());
    auto dstLayout = dstTy.getEncoding();
    auto bitwidth = dstTy.getElementTypeBitWidth();
    auto rank = dstTy.getRank();
    if (auto dot = dyn_cast<DotOperandEncodingAttr>(dstLayout)) {
      auto vecWidth = 32 / bitwidth;
      auto kWidth = dot.getKWidth();
      auto kOrder = dot.getOpIdx() == 0 ? rank - 1 : rank - 2;
      if (auto mma = dyn_cast<NvidiaMmaEncodingAttr>(dot.getParent())) {
        auto needTrans = kOrder != srcLayout.getOrder()[0];
        auto canUseLdmatrix =
            (bitwidth == 16 || (!needTrans)) && (kWidth == vecWidth);
        if (mma.isHopper()) {
          // I think we should be able to remove this condition, but it's here
          // as the legacy ldmatrix path does not support it
          canUseLdmatrix &= srcTy.getElementTypeBitWidth() * kWidth == 32;
        }
        // If we remove this one, ldmatrix will IMA. It can probably be relaxed
        // though
        canUseLdmatrix &=
            srcTy.getShape()[0] >= 8 && srcTy.getShape()[1] >= 4 * kWidth;
        // To be removed in https://github.com/triton-lang/triton/pull/5154
        bool legacyLoweringIsBuggy =
            (kWidth >= 8 || (kWidth == 4 && bitwidth == 32)) && mma.isAmpere();
        return (mma.isHopper() && !canUseLdmatrix) ||
               (mma.isAmpere() && legacyLoweringIsBuggy);
      }
      if (isa<AMDMfmaEncodingAttr>(dot.getParent()))
        return true;
    }
    return false;
  };

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    auto srcLayout = cast<SharedEncodingAttr>(srcTy.getEncoding());
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<SharedEncodingAttr>(srcLayout) &&
        (isa<BlockedEncodingAttr, MmaEncodingTrait, SliceEncodingAttr,
             LinearEncodingAttr>(dstLayout) ||
         isSupportedDotOpLayout(srcTy, dstTy))) {
      return lowerSharedToDistributed(op, adaptor, getTypeConverter(),
                                      rewriter);
    }
    if (isa<DotOperandEncodingAttr>(dstLayout) &&
        isa<BlockedEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      return lowerSharedToDotOpFMA(op, adaptor, getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  LogicalResult
  lowerSharedToDotOpFMA(LocalLoadOp op, LocalLoadOpAdaptor adaptor,
                        const LLVMTypeConverter *typeConverter,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType dstTy = op.getType();
    Attribute dstLayout = dstTy.getEncoding();
    auto dotLayout = cast<DotOperandEncodingAttr>(dstLayout);
    auto blockedLayout = cast<BlockedEncodingAttr>(
        cast<DotOperandEncodingAttr>(dstLayout).getParent());
    auto thread = getThreadId(rewriter, loc);
    Value res = SharedToDotOperandFMA::convertLayout(
        dotLayout.getOpIdx(), op.getSrc(), adaptor.getSrc(), blockedLayout,
        thread, loc, getTypeConverter(), rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }
  LogicalResult
  lowerSharedToDistributed(LocalLoadOp op, LocalLoadOpAdaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto dstShape = dstTy.getShape();
    auto srcSharedLayout = cast<SharedEncodingAttr>(srcTy.getEncoding());
    assert((dstShape.size() <= 2 || isSupportedDotOpLayout(srcTy, dstTy)) &&
           "Unexpected rank of ConvertLayout(shared->distributed)");

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(srcTy.getElementType()), rewriter);
    auto elemLlvmTy = typeConverter->convertType(dstTy.getElementType());

    SmallVector<Value> outVals = loadSharedToDistributed(
        dstTy, srcTy, elemLlvmTy, smemObj, loc, rewriter, targetInfo);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
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
  using BackendCallbackType =
      decltype(BackendCallbacks::localStoreOpConversion);

  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         BackendCallbackType backendCallback,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo), backendCallback(backendCallback) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value memDescVal = op.getDst();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getDst().getType().getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);

    std::pair<size_t, Type> llvmOpCount;
    lowerDistributedToShared(op.getLoc(), op.getSrc(), op.getDst(),
                             adaptor.getSrc(), smemObj, getTypeConverter(),
                             rewriter, targetInfo, &llvmOpCount);

    if (backendCallback)
      (backendCallback)(op, llvmOpCount.first, llvmOpCount.second);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
  BackendCallbackType backendCallback;
};

} // namespace

void mlir::triton::populateMemoryOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit,
    std::optional<BackendCallbacks> backendCallbacks) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, benefit);
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalDeallocOpConversion>(typeConverter, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo, benefit);

  auto backendCall =
      backendCallbacks ? backendCallbacks->localStoreOpConversion : nullptr;
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo, backendCall,
                                       benefit);
}
