#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::LLVM::AMD::storeDistributedToShared;

namespace SharedToDotOperandMFMA {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
} // namespace SharedToDotOperandMFMA

namespace SharedToDotOperandWMMA {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
} // namespace SharedToDotOperandWMMA

namespace {
// blocked -> shared.
// Swizzling in shared memory to avoid bank conflict. Normally used for
// A/B operands of dots.
void lowerDistributedToShared(Location loc, Value src, Value dst,
                              Value adaptorSrc,
                              const SharedMemoryObject &smemObj,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              const TargetInfoBase &targetInfo) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto dstTy = cast<MemDescType>(dst.getType());
  auto inOrd = mlir::cast<BlockedEncodingAttr>(srcTy.getEncoding()).getOrder();
  auto outOrd = mlir::cast<SharedEncodingAttr>(dstTy.getEncoding()).getOrder();
  assert(srcTy.getShape().size() <= 2 ||
         (srcTy.getShape().size() == 3 && outOrd[2] == 0) &&
             "Unexpected rank of ConvertLayout(blocked->shared)");
  bool crossGrain = inOrd[0] != outOrd[0];
  auto elemTy = typeConverter->convertType(srcTy.getElementType());

  auto smemBase = smemObj.getBase();
  auto dstStrides = smemObj.getStrides();
  auto inVals = unpackLLElements(loc, adaptorSrc, rewriter);
  storeDistributedToShared(dstTy, srcTy, elemTy, inVals, smemBase,
                           dstStrides, loc, rewriter, targetInfo, crossGrain);
}

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<DotOperandEncodingAttr>(dstLayout) &&
        isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      return lowerSharedToDotOperand(op, adaptor, getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  // shared -> dot_operand if the result layout is mfma
  Value lowerSharedToDotOperandMMA(
      triton::gpu::LocalLoadOp op, triton::gpu::LocalLoadOpAdaptor adaptor,
      const LLVMTypeConverter *typeConverter,
      ConversionPatternRewriter &rewriter,
      const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto llvmElemTy = typeConverter->convertType(
        cast<MemDescType>(src.getType()).getElementType());

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    Value res;
    auto dopOpParent = dotOperandLayout.getParent();
    if (!isOuter &&
        isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>(dopOpParent)) {
      auto sharedToDotConvert = isa<AMDMfmaEncodingAttr>(dopOpParent)
                                    ? SharedToDotOperandMFMA::convertLayout
                                    : SharedToDotOperandWMMA::convertLayout;
      res = sharedToDotConvert(dotOperandLayout.getOpIdx(), rewriter, loc, src,
                               dotOperandLayout, smemObj, typeConverter,
                               tid_val());
    } else {
      assert(false && "unsupported layout found");
    }
    return res;
  }

  // shared -> matrix_core_dot_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::LocalLoadOp op,
                          triton::gpu::LocalLoadOpAdaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto dstTensorTy = cast<RankedTensorType>(dst.getType());
    auto srcTensorTy = cast<MemDescType>(src.getType());
    auto dotOperandLayout =
        cast<DotOperandEncodingAttr>(dstTensorTy.getEncoding());
    auto sharedLayout = cast<SharedEncodingAttr>(srcTensorTy.getEncoding());

    bool isOuter{};
    int K{};
    if (dotOperandLayout.getOpIdx() == 0) // $a
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[0]];
    else // $b
      K = dstTensorTy.getShape()[sharedLayout.getOrder()[1]];
    isOuter = K == 1;
    Value res = lowerSharedToDotOperandMMA(op, adaptor, typeConverter, rewriter,
                                           dotOperandLayout, isOuter);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::ConvertLayoutOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(dst.getType());
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (isa<AMDMfmaEncodingAttr>(srcLayout) &&
        isa<DotOperandEncodingAttr>(dstLayout)) {
      return lowerMfmaToDotOperand(op, adaptor, rewriter);
    }
    return failure();
  }

private:
  LogicalResult
  lowerMfmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    if (isMfmaToDotShortcut(srcTy, dstTy)) {
      // vecSize is an number of sequential elements stored by one thread
      // - For MFMA encoding (encoding of the result tensor of dot
      // operation) it is 4
      // - For MFMA operand encoding it is
      // dotOperandEncoding::kWidth,
      //   which is 4 in certain cases (e.g. fp16 and bfloat16 dtypes with kpack
      //   = 1)
      //
      // For cases where these two values are equal MFMA and MFMA operand
      // layouts are the same.
      auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      Value view =
          packLLElements(loc, getTypeConverter(), vals, rewriter, dstTy);
      rewriter.replaceOp(op, view);
      return success();
    }
    return failure();
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
    Value memDescVal = op.getDst();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getDst().getType().getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);
    lowerDistributedToShared(op.getLoc(), op.getSrc(), op.getDst(),
                             adaptor.getSrc(), smemObj, getTypeConverter(),
                             rewriter, targetInfo);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};
} // namespace

namespace mlir::triton::AMD {
void populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, benefit);
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::AMD
