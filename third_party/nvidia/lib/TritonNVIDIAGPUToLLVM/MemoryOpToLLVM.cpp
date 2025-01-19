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
      // Limitation 1: Cannot use ldmatrix if we need to transpose a non-fp16
      // matrix
      // Limitation 2: If kWidth is greater than the vector width of the dot
      // operands of MMA, we don't use ldmatrix
      // Limitation 3 [TODO: remove]: Shared memory with leading offset is not
      // supported yet
      auto canUseLdmatrixLegacy = (bitwidth == 16 || (!needTrans)) &&
                                  (kWidth == vecWidth) &&
                                  (!shared.getHasLeadingOffset());
      if (mma.isHopper()) {
        // Limitation 4 [TODO: remove]:
        // I think we should be able to remove this condition, but it's here
        // as the legacy ldmatrix path does not support it
        canUseLdmatrixLegacy &= srcTy.getElementTypeBitWidth() * kWidth == 32 &&
                                dot.getOpIdx() == 0;
      }
      // Limitation 5: If we perform swizzling, it must be done within a single
      // ldmatrix tile
      // Limitation 6 [TODO: remove]: Only support 2d matrices now but we should
      // be able to support 3D minor changes
      auto maxPhase = shared.getMaxPhase();
      auto perPhase = shared.getPerPhase();
      canUseLdmatrixLegacy &=
          dstTy.getRank() <= 2 && (maxPhase / perPhase <= 8);
      auto allocShape = srcTy.getAllocShape();
      auto shape = srcTy.getShape();
      // Limitation 7 [TODO: remove]: The LL-path doesn't support sliced shared
      // memory, transpose, or non-16-bit types
      auto canUseLdmatrixLL =
          canUseLdmatrixLegacy && bitwidth == 16 && !needTrans &&
          isSimpleSharedMemoryAccess(shape, allocShape, shared);
      if (dot.getOpIdx() == 0) {
        canUseLdmatrixLL &=
            srcTy.getShape()[0] >= 16 && srcTy.getShape()[1] >= 16;
      } else {
        // Limitation 8 [TODO: remove]: Due to the use of ldmatrix.x4, we need
        // to read 4 tiles. For opIdx=1, a single warp load four consecutive
        // tiles along the K dimension, so the minimum K size is 4 * 8 = 32.
        // The legacy path doesn't have this limitation because it reads
        // duplicated elements from shared memory and throw them away.
        // It might be better to use ldmatrix.x2 in such a case instead of
        // abandoning elements.
        canUseLdmatrixLL &=
            srcTy.getShape()[0] >= 32 && srcTy.getShape()[1] >= 16;
      }
      // Limitation 9 [TODO: remove]:
      // If we remove this one, ldmatrix will IMA. It can probably be relaxed
      // though. Remove this constraint after all other limitations have been
      // resolved
      canUseLdmatrixLegacy &=
          srcTy.getShape()[0] >= 8 && srcTy.getShape()[1] >= 4 * kWidth;
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
    for (int v = 0; v < static_cast<int>(elemsI32.size()); ++v) {
      auto vec = bitcast(elemsI32[v], vecTy);
      for (int i = 0; i < numElemsPerVec; ++i)
        elems.push_back(extract_element(llvmElemTy, vec, i32_val(i)));
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

LogicalResult lowerDistributedToSharedStmatrix(
    Location loc, TypedValue<RankedTensorType> src, MemDescType memDescType,
    Value adaptorSrc, Value smemBase, const TypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, const TargetInfoBase &targetInfo,
    std::pair<size_t, Type> *const llvmOpCount = nullptr) {
  auto mmaEncoding =
      dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>(src.getType().getEncoding());
  if (!mmaEncoding)
    return failure();
  auto sharedLayout =
      cast<triton::gpu::SharedEncodingAttr>(memDescType.getEncoding());
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

  RankedTensorType srcTy = src.getType();
  SmallVector<unsigned> shape =
      convertType<unsigned, int64_t>(srcTy.getShape());
  auto order = sharedLayout.getOrder();
  if (!targetInfo.canUseStMatrix(srcTy, shape, shape, order, swizzleByteSize)) {
    return failure();
  }

  auto *ctx = rewriter.getContext();

  auto layout =
      chooseStMatrixLayout(rewriter.getContext(), srcTy, swizzleByteSize);
  auto llvmElemTy = typeConverter->convertType(memDescType.getElementType());
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
  auto srcVals = unpackLLElements(loc, adaptorSrc, rewriter);
  auto srcVec = layout.getNumConsecutiveInOut();
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
  return success();
}

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
    MemDescType memDescType = op.getType();
    auto sharedLayout =
        cast<triton::gpu::SharedEncodingAttr>(memDescType.getEncoding());
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    Value smemBase =
        LLVM::getSharedMemoryBase(op.getLoc(), rewriter, targetInfo, op);

    if (lowerDistributedToSharedStmatrix(op.getLoc(), op.getSrc(), memDescType,
                                         adaptor.getSrc(), smemBase,
                                         typeConverter, rewriter, targetInfo)
            .failed()) {
      return failure();
    }

    auto resultTy = cast<MemDescType>(op.getType());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, resultTy.getRank(),
                                      op.getLoc(), rewriter);
    auto retVal =
        getStructFromSharedMemoryObject(op.getLoc(), smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const NVIDIA::TargetInfo &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type llvmElemTy =
        getTypeConverter()->convertType(op.getDst().getType().getElementType());
    SharedMemoryObject smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);
    MemDescType memDescType = op.getDst().getType();
    if (lowerDistributedToSharedStmatrix(
            op.getLoc(), op.getSrc(), memDescType, adaptor.getSrc(),
            smemObj.getBase(), getTypeConverter(), rewriter, targetInfo)
            .failed()) {
      return failure();
    }
    rewriter.eraseOp(op);
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
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo,
                                       benefit.getBenefit() + 1);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo,
                                      benefit.getBenefit() + 1);
  mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
}
