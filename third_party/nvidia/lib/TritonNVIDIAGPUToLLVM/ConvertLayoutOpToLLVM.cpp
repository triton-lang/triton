#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using ::mlir::LLVM::getMultiDimOffset;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getWrappedMultiDimOffset;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Forward declarations

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
      return lowerSharedToDotOperand(op, adaptor, getTypeConverter(), rewriter);
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
      const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) const {
    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto dst = op.getResult();
    bool isMMA = supportMMA(dst, mmaLayout.getVersionMajor());

    auto llvmElemTy =
        typeConverter->convertType(src.getType().getElementType());

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    Value res;

    if (isOuter) {
      assert(false && "MMA Layout does not support outer product");
      return res;
    }

    if (mmaLayout.isHopper() || mmaLayout.isAmpere()) { // tensor core v2 or v3
      if (mmaLayout.isHopper())
        assert(dotOperandLayout.getOpIdx() == 0 &&
               "MMAv3 can only have operand $b on shared memory");

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

    int K;
    if (dstEnc.getOpIdx() == 0) // $a
      K = op.getType().getShape()[sharedLayout.getOrder()[0]];
    else // $b
      K = op.getType().getShape()[sharedLayout.getOrder()[1]];
    bool isOuter = K == 1;
    auto mmaLayout = cast<NvidiaMmaEncodingAttr>(dstEnc.getParent());
    Value res = lowerSharedToDotOperandMMA(op, adaptor, typeConverter, rewriter,
                                           mmaLayout, dstEnc, isOuter);

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  ConvertLayoutOpConversion(const LLVMTypeConverter &typeConverter,
                            const NVIDIA::TargetInfo &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<MmaEncodingTrait, BlockedEncodingAttr, SliceEncodingAttr>(
            srcLayout) &&
        isa<MmaEncodingTrait, BlockedEncodingAttr, SliceEncodingAttr>(
            dstLayout)) {
      if (shouldUseDistSmem(srcLayout, dstLayout))
        return lowerDistToDistWithDistSmem(op, adaptor, rewriter, targetInfo);
    }
    if (isa<NvidiaMmaEncodingAttr>(srcLayout) &&
        isa<DotOperandEncodingAttr>(dstLayout)) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }

    return failure();
  }

private:
  LogicalResult
  lowerDistToDistWithDistSmem(triton::gpu::ConvertLayoutOp op,
                              OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter,
                              const TargetInfoBase &targetInfo) const {
    MLIRContext *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto srcLayout = srcTy.getEncoding();
    auto dstLayout = dstTy.getEncoding();
    auto srcShapePerCTA = getShapePerCTA(srcTy);
    auto srcCTAsPerCGA = triton::gpu::getCTAsPerCGA(srcLayout);
    auto srcCTAOrder = triton::gpu::getCTAOrder(srcLayout);
    unsigned rank = srcShapePerCTA.size();

    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    smemBase = bitcast(smemBase, elemPtrTy);
    auto smemShape = convertType<unsigned, int64_t>(srcShapePerCTA);

    // Store to local shared memory
    {
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      auto inIndices = emitIndices(loc, rewriter, targetInfo, srcLayout, srcTy,
                                   /*withCTAOffset*/ false);

      assert(inIndices.size() == inVals.size() &&
             "Unexpected number of indices emitted");

      for (unsigned i = 0; i < inIndices.size(); ++i) {
        Value offset = linearize(rewriter, loc, inIndices[i], smemShape);
        Value ptr = gep(elemPtrTy, llvmElemTy, smemBase, offset);
        store(inVals[i], ptr);
      }
    }

    // Cluster barrier
    rewriter.create<triton::nvidia_gpu::ClusterArriveOp>(loc, false);
    rewriter.create<triton::nvidia_gpu::ClusterWaitOp>(loc);

    // Load from remote shared memory
    {
      SmallVector<Value> srcShapePerCTACache;
      for (unsigned i = 0; i < rank; ++i)
        srcShapePerCTACache.push_back(i32_val(srcShapePerCTA[i]));

      SmallVector<Value> outVals;
      auto outIndices = emitIndices(loc, rewriter, targetInfo, dstLayout, dstTy,
                                    /*withCTAOffset*/ true);

      for (unsigned i = 0; i < outIndices.size(); ++i) {
        auto coord = outIndices[i];
        assert(coord.size() == rank && "Unexpected rank of index emitted");

        SmallVector<Value> multiDimCTAId, localCoord;
        for (unsigned d = 0; d < rank; ++d) {
          multiDimCTAId.push_back(udiv(coord[d], srcShapePerCTACache[d]));
          localCoord.push_back(urem(coord[d], srcShapePerCTACache[d]));
        }

        Value remoteCTAId =
            linearize(rewriter, loc, multiDimCTAId, srcCTAsPerCGA, srcCTAOrder);
        Value localOffset = linearize(rewriter, loc, localCoord, smemShape);

        Value ptr = gep(elemPtrTy, llvmElemTy, smemBase, localOffset);
        outVals.push_back(targetInfo.loadDShared(
            rewriter, loc, ptr, remoteCTAId, llvmElemTy, /*pred=*/true_val()));
      }

      Value result =
          packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
    }

    // Cluster barrier
    rewriter.create<triton::nvidia_gpu::ClusterArriveOp>(loc, false);
    rewriter.create<triton::nvidia_gpu::ClusterWaitOp>(loc);

    return success();
  }

  // Convert from accumulator MMA layout to 8bit dot operand layout.
  // The conversion logic is taken from:
  // https://github.com/ColfaxResearch/cutlass-kernels/blob/a9de6446c1c0415c926025cea284210c799b11f8/src/fmha-pipeline/reg2reg.h#L45
  void
  convertMMAV3To8BitsDotOperand(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto dstTy = op.getType();
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> retVals;
    for (int i = 0; i < vals.size(); i += 8) {
      Value upper = undef(vec_ty(i8_ty, 4));
      for (int j = 0; j < 4; j++) {
        upper =
            insert_element(vec_ty(i8_ty, 4), upper, vals[i + j], i32_val(j));
      }
      upper = bitcast(upper, i32_ty);
      Value lower = undef(vec_ty(i8_ty, 4));
      for (int j = 0; j < 4; j++) {
        lower = insert_element(vec_ty(i8_ty, 4), lower, vals[i + 4 + j],
                               i32_val(j));
      }
      lower = bitcast(lower, i32_ty);

      Value threadIdMod4 = urem(getThreadId(rewriter, loc), i32_val(4));
      Value cnd = or_(icmp_eq(threadIdMod4, i32_val(0)),
                      icmp_eq(threadIdMod4, i32_val(3)));
      Value selectorEx0 = select(cnd, i32_val(0x3210), i32_val(0x7654));
      Value selectorEx1 = select(cnd, i32_val(0x7654), i32_val(0x3210));
      Value selectorEx4 = select(cnd, i32_val(0x5410), i32_val(0x1054));
      Value selectorEx5 = select(cnd, i32_val(0x7632), i32_val(0x3276));

      Value isOne = icmp_eq(threadIdMod4, i32_val(1));
      Value isTwo = icmp_eq(threadIdMod4, i32_val(2));
      Value isThree = icmp_eq(threadIdMod4, i32_val(3));
      Value upperIdx = i32_val(0);
      upperIdx = select(isOne, i32_val(3), upperIdx);
      upperIdx = select(isTwo, i32_val(1), upperIdx);
      upperIdx = select(isThree, i32_val(2), upperIdx);

      Value lowerIdx = i32_val(1);
      lowerIdx = select(isOne, i32_val(2), lowerIdx);
      lowerIdx = select(isTwo, i32_val(0), lowerIdx);
      lowerIdx = select(isThree, i32_val(3), lowerIdx);

      Value upper0 =
          LLVM::NVIDIA::permute(loc, rewriter, upper, lower, selectorEx0);
      Value lower0 =
          LLVM::NVIDIA::permute(loc, rewriter, upper, lower, selectorEx1);
      Value mask = i32_val(0xFFFFFFFF);
      // Set clamp tp shuffle only within 4 lanes.
      Value clamp = i32_val(0x1C1F);
      upper0 =
          rewriter.create<NVVM::ShflOp>(loc, i32_ty, mask, upper0, upperIdx,
                                        clamp, NVVM::ShflKind::idx, UnitAttr());
      lower0 =
          rewriter.create<NVVM::ShflOp>(loc, i32_ty, mask, lower0, lowerIdx,
                                        clamp, NVVM::ShflKind::idx, UnitAttr());
      Value upper1 =
          LLVM::NVIDIA::permute(loc, rewriter, upper0, lower0, selectorEx4);
      Value vecVal = bitcast(upper1, vec_ty(i8_ty, 4));
      for (int i = 0; i < 4; i++) {
        retVals.push_back(extract_element(i8_ty, vecVal, i32_val(i)));
      }
      Value lower1 =
          LLVM::NVIDIA::permute(loc, rewriter, upper0, lower0, selectorEx5);
      vecVal = bitcast(lower1, vec_ty(i8_ty, 4));
      for (int i = 0; i < 4; i++) {
        retVals.push_back(extract_element(i8_ty, vecVal, i32_val(i)));
      }
    }
    Value result =
        packLLElements(loc, getTypeConverter(), retVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
  }

  // mma -> dot_operand
  LogicalResult
  lowerMmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    if (matchMmaV3AndDotOperandLayout(srcTy, dstTy)) {
      if (srcTy.getElementType().getIntOrFloatBitWidth() == 16) {
        rewriter.replaceOp(op, adaptor.getSrc());
        return success();
      }
      assert(srcTy.getElementType().getIntOrFloatBitWidth() == 8 &&
             "Unsupported type size.");
      convertMMAV3To8BitsDotOperand(op, adaptor, rewriter);
      return success();
    }
    return failure();
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
    auto layout = chooseStMatrixLayout(rewriter.getContext(), srcTy, shape,
                                       shape, order, swizzleByteSize);
    if (!layout.has_value())
      return failure();

    Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
    auto smemPtrTy = ptr_ty(ctx, 3);

    auto kRegister = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kBlock = str_attr("block");

    Value threadId = getThreadId(rewriter, loc);
    Value threadsPerWarp = i32_val(layout->getInDimSize(kLane));
    Value laneId = urem(threadId, threadsPerWarp);
    Value warpId = udiv(threadId, threadsPerWarp);

    auto regBase = applyLinearLayout(loc, rewriter, *layout,
                                     {{kRegister, i32_val(0)},
                                      {kLane, laneId},
                                      {kWarp, warpId},
                                      {kBlock, i32_val(0)}})[0]
                       .second;
    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto srcVec = layout->getNumConsecutiveInOut();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    for (int i = 0; i < srcVals.size(); i += srcVec) {
      auto regIdx =
          layout
              ->apply({{kRegister, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}})[0]
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
    auto shapePerCTA = getShapePerCTA(sharedLayout, resultTy.getShape());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, shapePerCTA,
                                      newOrder, loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const NVIDIA::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMOptimizedPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
}

void mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // For now give ConvertLayoutOpConversion higher benefit, I can split before
  // merging
  //
  // TODO(jlebar): lowerDistributedToDistributed does not get hit in any
  // testcases.  Is this dead code?  Does the benefit need to be increased?
  patterns.add<ConvertLayoutOpConversion>(typeConverter, targetInfo, benefit);
  // Same default benefit
  patterns.add<LocalLoadOpConversion>(typeConverter, benefit);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
