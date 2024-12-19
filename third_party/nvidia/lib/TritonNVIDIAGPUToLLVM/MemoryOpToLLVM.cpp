#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace SharedToDotOperandMMAv2OrV3 {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
} // namespace SharedToDotOperandMMAv2OrV3

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

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
        isa<NvidiaMmaEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      auto dot = cast<DotOperandEncodingAttr>(dstLayout);
      auto mma = cast<NvidiaMmaEncodingAttr>(dot.getParent());
      auto shared = cast<SharedEncodingAttr>(srcLayout);
      auto bitwidth = dstTy.getElementTypeBitWidth();
      auto vecWidth = 32 / bitwidth;
      auto kWidth = dot.getKWidth();
      auto rank = dstTy.getRank();
      auto kOrder = dot.getOpIdx() == 0 ? rank - 1 : rank - 2;
      auto needTrans = kOrder != shared.getOrder()[0];
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
          srcTy.getShape()[0] >= 8 &&
          srcTy.getShape()[1] >= 4 * kWidth & dstTy.getRank() <= 2;
      if (canUseLdmatrix) {
        return lowerSharedToDotOperand(op, adaptor, getTypeConverter(),
                                       rewriter);
      }
    }
    return failure();
  }

private:
  // shared -> dot_operand if the result layout is mma
  Value lowerSharedToDotOperandMMA(
      triton::gpu::LocalLoadOp op, triton::gpu::LocalLoadOpAdaptor adaptor,
      const LLVMTypeConverter *typeConverter,
      ConversionPatternRewriter &rewriter,
      const NvidiaMmaEncodingAttr &mmaLayout,
      const DotOperandEncodingAttr &dotOperandLayout) const {
    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto dst = op.getResult();
    bool isMMA = supportMMA(dst, mmaLayout.getVersionMajor());

    auto llvmElemTy =
        typeConverter->convertType(src.getType().getElementType());

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    Value res;

    if (mmaLayout.isHopper() || mmaLayout.isAmpere()) { // tensor core v2 or v3
      if (mmaLayout.isHopper())
        assert(dotOperandLayout.getOpIdx() == 0 &&
               "Operand $b in MMAv3 can only be in shared memory");

      res = SharedToDotOperandMMAv2OrV3::convertLayout(
          dotOperandLayout.getOpIdx(), rewriter, loc, src, dotOperandLayout,
          smemObj, typeConverter, getThreadId(rewriter, loc));
    } else {
      assert(false && "Unsupported mma layout found");
    }
    return res;
  };

  // shared -> mma_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::LocalLoadOp op,
                          triton::gpu::LocalLoadOpAdaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto dstEnc = cast<DotOperandEncodingAttr>(op.getType().getEncoding());
    auto sharedLayout =
        cast<SharedEncodingAttr>(op.getSrc().getType().getEncoding());

    auto mmaLayout = cast<NvidiaMmaEncodingAttr>(dstEnc.getParent());
    Value res = lowerSharedToDotOperandMMA(op, adaptor, typeConverter, rewriter,
                                           mmaLayout, dstEnc);

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    auto mmaEncoding = dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>(
        op.getSrc().getType().getEncoding());
    if (!mmaEncoding)
      return failure();
    auto sharedLayout =
        cast<triton::gpu::SharedEncodingAttr>(op.getType().getEncoding());
    if (!sharedLayout.getHasLeadingOffset())
      return failure();
    int swizzleByteSize = 0;
    if (sharedLayout.getPerPhase() == 4 && sharedLayout.getMaxPhase() == 2)
      swizzleByteSize = 32;
    else if (sharedLayout.getPerPhase() == 2 && sharedLayout.getMaxPhase() == 4)
      swizzleByteSize = 64;
    else if (sharedLayout.getPerPhase() == 1 && sharedLayout.getMaxPhase() == 8)
      swizzleByteSize = 128;
    else
      return failure();

    auto *ctx = rewriter.getContext();
    Location loc = op->getLoc();

    RankedTensorType srcTy = op.getSrc().getType();
    SmallVector<unsigned> shape =
        convertType<unsigned, int64_t>(srcTy.getShape());
    auto order = sharedLayout.getOrder();
    if (!targetInfo.canUseStMatrix(srcTy, shape, shape, order,
                                   swizzleByteSize)) {
      return failure();
    }
    auto layout = chooseStMatrixLayout(rewriter.getContext(), srcTy, shape,
                                       shape, order, swizzleByteSize);
    Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
    auto smemPtrTy = ptr_ty(ctx, 3);

    auto kRegister = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kBlock = str_attr("block");

    Value threadId = getThreadId(rewriter, loc);
    Value threadsPerWarp = i32_val(layout.getInDimSize(kLane));
    Value laneId = urem(threadId, threadsPerWarp);
    Value warpId = udiv(threadId, threadsPerWarp);

    auto regBase = applyLinearLayout(loc, rewriter, layout,
                                     {{kRegister, i32_val(0)},
                                      {kLane, laneId},
                                      {kWarp, warpId},
                                      {kBlock, i32_val(0)}})[0]
                       .second;
    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto srcVec = layout.getNumConsecutiveInOut();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    for (int i = 0; i < srcVals.size(); i += srcVec) {
      auto regIdx =
          layout.apply({{kRegister, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}})[0]
              .second;
      Value offset = xor_(regBase, i32_val(regIdx));
      auto vecAddr = gep(smemPtrTy, llvmElemTy, smemBase, offset);
      vecAddr.setInbounds(true);
      SmallVector<Value> inValsVec;
      for (int j = 0; j < srcVec; j++)
        inValsVec.push_back(srcVals[i + j]);
      Value valsVec = packLLVector(loc, inValsVec, rewriter);
      targetInfo.storeMatrixShared(rewriter, loc, vecAddr, valsVec);
    }

    auto resultTy = cast<MemDescType>(op.getType());
    auto shapePerCTA = getShapePerCTA(sharedLayout, resultTy.getShape());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, shapePerCTA,
                                      sharedLayout, loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};
} // namespace

void mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Backend optimized memory ops get higher benefit
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(typeConverter, benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
}
