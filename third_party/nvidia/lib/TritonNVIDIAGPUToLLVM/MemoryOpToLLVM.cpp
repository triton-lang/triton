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
  LocalLoadOpConversion(const LLVMTypeConverter &converter,
                        const NVIDIA::TargetInfo &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

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
      auto canUseLdmatrixLegacy =
          (bitwidth == 16 || (!needTrans)) && (kWidth == vecWidth);
      if (mma.isHopper()) {
        // I think we should be able to remove this condition, but it's here
        // as the legacy ldmatrix path does not support it
        canUseLdmatrixLegacy &= srcTy.getElementTypeBitWidth() * kWidth == 32 &&
                                dot.getOpIdx() == 0;
      }
      // If we perform swizzling, it must be done within a single ldmatrix tile
      auto maxPhase = shared.getMaxPhase();
      auto perPhase = shared.getPerPhase();
      canUseLdmatrixLegacy &=
          dstTy.getRank() <= 2 && (maxPhase / perPhase <= 8);
      auto allocShape = srcTy.getAllocShape();
      auto shape = srcTy.getShape();
      auto canUseLdmatrixLL =
          canUseLdmatrixLegacy && bitwidth == 16 && !needTrans &&
          srcTy.getShape()[0] >= 16 && srcTy.getShape()[1] >= 16 &&
          isSimpleSharedMemoryAccess(shape, allocShape, shared);
      // If we remove this one, ldmatrix will IMA. It can probably be relaxed
      // though
      canUseLdmatrixLegacy &=
          srcTy.getShape()[0] >= 8 && srcTy.getShape()[1] >= 4 * kWidth;
      // The LL path only supports ldmatrix.x4
      if (canUseLdmatrixLL) {
        return lowerSharedToDotOperandLL(op, adaptor, getTypeConverter(),
                                         rewriter);
      } else if (canUseLdmatrixLegacy) {
        return lowerSharedToDotOperandLegacy(op, adaptor, getTypeConverter(),
                                             rewriter);
      }
    }
    return failure();
  }

private:
  LogicalResult
  lowerSharedToDotOperandLegacy(triton::gpu::LocalLoadOp op,
                                triton::gpu::LocalLoadOpAdaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto dstLayout = cast<DotOperandEncodingAttr>(op.getType().getEncoding());
    auto mmaLayout = cast<NvidiaMmaEncodingAttr>(dstLayout.getParent());
    auto llvmElemTy =
        typeConverter->convertType(src.getType().getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    Value res;
    if (mmaLayout.isHopper() || mmaLayout.isAmpere()) { // tensor core v2 or v3
      if (mmaLayout.isHopper())
        assert(dstLayout.getOpIdx() == 0 &&
               "Operand $b in MMAv3 can only be in shared memory");

      res = SharedToDotOperandMMAv2OrV3::convertLayout(
          dstLayout.getOpIdx(), rewriter, loc, src, dstLayout, smemObj,
          typeConverter, getThreadId(rewriter, loc));
    } else {
      llvm_unreachable("Unsupported mma layout found");
    }
    rewriter.replaceOp(op, res);
    return success();
  }

  LogicalResult
  lowerSharedToDotOperandLL(triton::gpu::LocalLoadOp op,
                            triton::gpu::LocalLoadOpAdaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto dot = cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto shared = cast<SharedEncodingAttr>(srcTy.getEncoding());
    auto shape = dstTy.getShape();
    auto layout = chooseLdMatrixLayout(ctx, shared, dot, shape);
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    auto smemPtrTy = ptr_ty(ctx, 3);

    auto kRegister = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kBlock = str_attr("block");

    auto [laneId, warpId, blockId] =
        emitHardwareTuple(loc, rewriter, targetInfo, /*withCTAOffset=*/0,
                          layout.getInDimSize(kLane));

    auto regBase = applyLinearLayout(loc, rewriter, layout,
                                     {{kRegister, i32_val(0)},
                                      {kLane, laneId},
                                      {kWarp, warpId},
                                      {kBlock, i32_val(0)}})[0]
                       .second;
    auto numRegs = layout.getInDimSize(kRegister);
    auto vecSize = layout.getNumConsecutiveInOut();
    auto matTy =
        LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(4, i32_ty));
    SmallVector<Value> elemsI32;
    for (int i = 0; i < numRegs; i += vecSize) {
      auto regOffset =
          layout.apply({{kRegister, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}})[0]
              .second;
      Value offset = xor_(regBase, i32_val(regOffset));
      auto vecAddr = gep(smemPtrTy, llvmElemTy, smemObj.getBase(), offset);
      vecAddr.setInbounds(true);
      auto ldMatrixOp = rewriter.create<nvgpu::LoadMatrixOp>(
          loc, matTy, vecAddr, /*needTrans=*/false);
      auto resV4 = ldMatrixOp.getResult();
      elemsI32.push_back(extract_val(i32_ty, resV4, 0));
      elemsI32.push_back(extract_val(i32_ty, resV4, 1));
      elemsI32.push_back(extract_val(i32_ty, resV4, 2));
      elemsI32.push_back(extract_val(i32_ty, resV4, 3));
    }

    SmallVector<Value> elems;
    auto numElemsPerVec = 32 / llvmElemTy.getIntOrFloatBitWidth();
    auto vecTy = vec_ty(llvmElemTy, numElemsPerVec);
    for (int i = 0; i < elemsI32.size(); ++i) {
      // Unpack the 32-bit values into the final result
      auto vec = bitcast(elemsI32[i], vecTy);
      for (auto v = 0; v < numElemsPerVec; ++v)
        elems.push_back(extract_element(llvmElemTy, vec, i32_val(v)));
    }

    auto structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(elems.size(), llvmElemTy));
    auto ret = packLLElements(loc, typeConverter, elems, rewriter, structTy);
    rewriter.replaceOp(op, ret);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
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

    auto [laneId, warpId, blockId] =
        emitHardwareTuple(loc, rewriter, targetInfo, /*withCTAOffset=*/0,
                          layout.getInDimSize(kLane));

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
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo,
                                      benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
}
