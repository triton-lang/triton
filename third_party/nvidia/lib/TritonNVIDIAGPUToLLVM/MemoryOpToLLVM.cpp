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
#include "triton/Tools/LayoutUtils.h"
namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::NVIDIA;

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
      auto dotEnc = cast<DotOperandEncodingAttr>(dstLayout);
      auto mmaEnc = cast<NvidiaMmaEncodingAttr>(dotEnc.getParent());
      auto sharedEnc = dyn_cast<SwizzledSharedEncodingAttr>(srcLayout);
      if (!sharedEnc)
        return failure();
      auto bitwidth = dstTy.getElementTypeBitWidth();
      auto vecWidth = 32 / bitwidth;
      auto kWidth = dotEnc.getKWidth();
      auto rank = dstTy.getRank();
      auto kOrder = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
      auto nonKOrder = dotEnc.getOpIdx() == 0 ? rank - 2 : rank - 1;
      auto needTrans = kOrder != sharedEnc.getOrder()[0];
      // Limitation 1 [TODO: remove]: Check LL bases to verify register and
      // address alignment
      auto canUseLdmatrix = (kWidth == vecWidth);
      canUseLdmatrix &= (sharedEnc.getMaxPhase() == 1) ||
                        (sharedEnc.getVec() * bitwidth >= 8 * 16);
      auto shape = srcTy.getShape();
      // Limitation 2 [TODO: remove]: Only support 2d matrices now but we should
      // be able to support 3D minor changes
      canUseLdmatrix &= (bitwidth == 16 || !needTrans) && shape.size() <= 2;
      // Limitation 3: Minimum tile size (8)x(8x16bits)
      canUseLdmatrix &=
          shape[kOrder] >= (8 * 16 / bitwidth) && shape[nonKOrder] >= 8;
      if (canUseLdmatrix) {
        return lowerSharedToDotOperand(op, adaptor, getTypeConverter(),
                                       rewriter);
      }
    }
    return failure();
  }

private:
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::LocalLoadOp op,
                          triton::gpu::LocalLoadOpAdaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    auto dotEnc = cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    auto sharedEnc = cast<SwizzledSharedEncodingAttr>(srcTy.getEncoding());
    auto shape = dstTy.getShape();
    auto rank = dstTy.getRank();
    auto kOrder = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
    auto nonKOrder = dotEnc.getOpIdx() == 0 ? rank - 2 : rank - 1;
    auto needTrans = kOrder != sharedEnc.getOrder()[0];

    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto ldmatrixLayout =
        chooseLdMatrixLayout(dotEnc, shape, needTrans, bitwidth);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    // Emit ldmatrix load operations for values packed in i32s
    SmallVector<Value> elemsI32;
    // Typically we load 32x8 to use ldmatrix.x4, but the minimum tile size for
    // opIdx=1 is 16x8. Therefore, we use ldmatrix.x2 instead of
    // ldmatrix.x4 in this case.
    auto shift = dotEnc.getOpIdx() == 1 && shape[kOrder] < (32 * 16 / bitwidth);
    auto maxVecElems = 8 * 16 / bitwidth;
    bool valid = emitTransferBetweenRegistersAndShared(
        ldmatrixLayout, srcTy, llvmElemTy,
        /*maxVecElems=*/maxVecElems, smemObj, loc, rewriter, targetInfo,
        [&](VectorType vecTy, Value vecAddr) {
          auto numElems = vecTy.getNumElements();
          auto numElemsI32 = (numElems * bitwidth / 32) >> shift;
          auto matTy = LLVM::LLVMStructType::getLiteral(
              ctx, SmallVector<Type>(numElemsI32, i32_ty));
          auto ldMatrixOp = rewriter.create<nvgpu::LoadMatrixOp>(
              loc, matTy, vecAddr, /*needTrans=*/needTrans);
          auto res = ldMatrixOp.getResult();
          for (auto i = 0; i < numElemsI32; ++i) {
            elemsI32.push_back(b.extract_val(i32_ty, res, i));
          }
        });
    assert(valid && "Failed to emit ldmatrix load operations");

    // Unpack i32 values to the original type
    SmallVector<Value> elems;
    auto numElemsPerVec = 32 / bitwidth;
    auto vecTy = vec_ty(llvmElemTy, numElemsPerVec);
    for (int v = 0; v < static_cast<int>(elemsI32.size()); ++v) {
      auto vec = b.bitcast(elemsI32[v], vecTy);
      for (int i = 0; i < numElemsPerVec; ++i)
        elems.push_back(b.extract_element(llvmElemTy, vec, b.i32_val(i)));
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
    Location loc, RankedTensorType tensorTy, MemDescType memDescType,
    Value adaptorSrc, Value smemBase, Type llvmElemTy,
    ConversionPatternRewriter &rewriter, const TargetInfo &targetInfo,
    std::pair<size_t, Type> *const llvmOpCount = nullptr) {
  if (!targetInfo.supportLdStMatrix())
    return failure();

  assert(llvmOpCount == nullptr && "NYI");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = tensorTy.getContext();
  auto regL = toLinearLayout(tensorTy.getShape(), tensorTy.getEncoding());
  auto memL = toLinearLayout(memDescType.getShape(), memDescType.getEncoding());
  auto cvt = minimalCvtLayout(memDescType, tensorTy);

  auto S = [ctx](StringRef v) { return StringAttr::get(ctx, v); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kBlock = S("block");
  auto kOffset = S("offset");
  auto smemPtrTy = ptr_ty(ctx, 3);

  // Just stmatrix for now
  // 1) NYI in the stmatrix lowering
  //    Pack everything into uint32_t to support bitwidths other than 16
  auto bitwidth = tensorTy.getElementTypeBitWidth();
  if (bitwidth != 16)
    return failure();

  // Inter block stmatrix is not supported
  if (cvt.hasInDim(kBlock))
    return failure();

  auto srcVals = unpackLLElements(loc, adaptorSrc, rewriter);

  // Remove broadcasting on the register dimension
  auto removeBroadcast = actionRemoveBroadcastedRegs(cvt);
  cvt = removeBroadcast.apply(cvt);
  srcVals = removeBroadcast.apply(srcVals);

  auto tile = LinearLayout::identity1D(32 / bitwidth, kReg, kOffset) *
              LinearLayout::identity1D(4, kLane, kOffset);
  // Find if there is a register permutation that allows us to divideLeft
  auto maybeAction = regPermForDivideLeft(cvt, tile);
  if (!maybeAction.has_value()) {
    return failure();
  }
  auto action = maybeAction.value();
  // Check if the action indeed allows us to divideLeft
  cvt = action.apply(cvt);
  auto maybeQuot = divideLeft(cvt, tile);
  if (!maybeQuot.has_value()) {
    return failure();
  }
  auto quot = maybeQuot.value();
  srcVals = action.apply(srcVals);
  // Map from kReg, kLane, kWarp to beginning of each tile
  auto reps = zerosLike(tile) * quot;
  assert(reps.getOutDimSize(kOffset) == cvt.getOutDimSize(kOffset));

  // Choose the 4 elements indexed by the next to bases as the vectorisation
  // factor
  auto vec = std::min(2, quot.getInDimSizeLog2(kReg));
  // 2) NYI stmatrix.x1 and stmatrix.x2
  if (vec != 2) {
    return failure();
  }

  // FIXME(Lezcano): Should we bail if any of the other 3 lane bases is zero?

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  // Compute the addresses for the 0th tile
  // Here we implement the stmatrix.x4 addressing. As per the PTX docs, the
  // threads 0-7 hold the address of the first element of the 8 columns of the
  // first submatrix, threads 8-15 for the second submatrix, etc. In general we
  // map:
  // - The lowest 3 bits of the laneId to the columns of each submatrix, which
  // is
  //   given by the 3 kLane bases of quotient that are not part of the tile
  // - The top `vec` bits of the thread id to the submatrix number, which is
  // given
  //   by the first `vec` reg bases that are not part of the tile
  std::vector<std::vector<int32_t>> laneBases;
  assert(tile.getInDimSizeLog2(kLane) == 2);
  for (int i = 0; i < 3; ++i) {
    laneBases.push_back(reps.getBasis(kLane, tile.getInDimSizeLog2(kLane) + i));
  }
  for (int i = 0; i < vec; ++i) {
    laneBases.push_back(reps.getBasis(kReg, tile.getInDimSizeLog2(kReg) + i));
  }

  LinearLayout addrLayout =
      LinearLayout({{kLane, laneBases}, {kWarp, reps.getBases().lookup(kWarp)}},
                   {{kOffset, reps.getOutDimSize(kOffset)}}, false);
  auto regBase = applyLinearLayout(loc, rewriter, addrLayout,
                                   {{kLane, laneId}, {kWarp, warpId}})[0]
                     .second;

  // Elements per op
  auto step = (1 << vec) * (32 / bitwidth);
  for (int i = 0; i < srcVals.size(); i += step) {
    auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
    Value offset = b.xor_(regBase, b.i32_val(regIdx));
    auto vecAddr = b.gep(smemPtrTy, llvmElemTy, smemBase, offset,
                         LLVM::GEPNoWrapFlags::inbounds);
    SmallVector<Value> inValsVec;
    for (int j = 0; j < step; j++)
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
    RankedTensorType srcTy = op.getSrc().getType();
    Type llvmElemTy = typeConverter->convertType(srcTy.getElementType());
    Value smemBase =
        LLVM::getSharedMemoryBase(op.getLoc(), rewriter, targetInfo, op);

    if (lowerDistributedToSharedStmatrix(op.getLoc(), srcTy, memDescType,
                                         adaptor.getSrc(), smemBase, llvmElemTy,
                                         rewriter, targetInfo)
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
    if (lowerDistributedToSharedStmatrix(op.getLoc(), op.getSrc().getType(),
                                         op.getDst().getType(),
                                         adaptor.getSrc(), smemObj.getBase(),
                                         llvmElemTy, rewriter, targetInfo)
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
