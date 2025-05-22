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
    bool transpose, Value adaptorSrc, Value smemBase, Type llvmElemTy,
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
  auto bitwidth = tensorTy.getElementTypeBitWidth();
  // In the transpose case, consecutive elements are not stored contiguously
  // so we cannot split an fp32
  // We could support bitwidth == 8, but it'd be a rather weird layout
  // so we don't do that for now
  if ((!transpose && bitwidth > 32) || (transpose && bitwidth != 16))
    return failure();
  // Inter block stmatrix is not supported
  if (cvt.hasInDim(kBlock))
    return failure();

  auto srcVals = unpackLLElements(loc, adaptorSrc, rewriter);

  // Remove broadcasting on the register dimension
  auto removeBroadcast = actionRemoveBroadcastedRegs(cvt);
  cvt = removeBroadcast.apply(cvt);
  srcVals = removeBroadcast.apply(srcVals);

  LinearLayout reps;
  if (!transpose) {
    auto tile = LinearLayout::identity1D(32 / bitwidth, kReg, kOffset) *
                LinearLayout::identity1D(4, kLane, kOffset);

    // Find if there is a register permutation that allows us to divideLeft
    // We need to pass the map from regs to offsets, as is cvt
    auto maybeAction = regPermForDivideLeft(cvt, tile);
    if (!maybeAction.has_value()) {
      return failure();
    }
    auto action = maybeAction.value();
    // Check if the action indeed allows us to divideLeft
    cvt = action.apply(cvt);
    srcVals = action.apply(srcVals);

    auto maybeQuot = divideLeft(cvt, tile);
    if (!maybeQuot.has_value()) {
      return failure();
    }
    reps = zerosLike(tile) * maybeQuot.value();
  } else {
    // Division does not quite work here. To define this properly, we would need
    // to define a different multiplication that does:
    // A *' B = [[0, A], [B, 0]] and define leftDivision for it
    // We do it ad-hoc for now, as I beleive there's not much demand for this op
    // outside of this lowering

    // Divisibility in the sense above is the same as regular divisibility
    // You need to see that the tile A is a sublayout of the matrix, and that
    // it has zeros above it and to its right.

    // In particular, offsets lanes 4, 8, 16 map to offsets 1, 2, 4...
    const auto &laneBases = cvt.getBases().find(kLane)->second;
    for (int i = 0; i < 3; ++i) {
      if (laneBases[i + 2][0] != (1 << i))
        return failure();
    }
    // ... and no other basis should depend on 1, 2, 4
    // Note that this gives us the usual alignment condition, but we have
    // translated it to checking that the matrix to the left of A is all zeros
    for (auto dim : cvt.getInDimNames()) {
      const auto &bases = cvt.getBases().find(dim)->second;
      for (auto [i, basis] : llvm::enumerate(bases)) {
        if (dim == kLane && i >= 2)
          continue;
        if (basis[0] & 0b111)
          return failure();
      }
    }

    // Hack: We are not going to use in the rest of the function reps[kLane][2:]
    // so we don't need to zero them out
    reps = cvt;
  }

  // We must have at least 2 register elements to use stmatrix.trans
  if (transpose && reps.getInDimSizeLog2(kReg) < llvm::Log2_32(32 / bitwidth)) {
    return failure();
  }

  // Choose up to 4 packs of 32-bit elements indexed by the next (at most) two
  // bases as the vectorisation factor. We don't consider the basis of the tile
  // for vectorisation so we substract them
  auto vec = std::min<int32_t>(2, reps.getInDimSizeLog2(kReg) -
                                      llvm::Log2_32(32 / bitwidth));

  // Map from kReg, kLane, kWarp to beginning of each tile
  assert(reps.getOutDimSize(kOffset) == cvt.getOutDimSize(kOffset));

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
  if (!transpose) {
    auto tileDimSizeReg = llvm::Log2_32(32 / bitwidth);
    auto tileDimSizeLane = 2;
    for (int i = 0; i < 3; ++i) {
      laneBases.push_back(reps.getBasis(kLane, tileDimSizeLane + i));
    }
    for (int i = 0; i < vec; ++i) {
      laneBases.push_back(reps.getBasis(kReg, tileDimSizeReg + i));
    }
  } else {
    // We choose the first basis of the register. In the future we could choose
    // a basis that minimises the bank conflicts
    laneBases.push_back(reps.getBasis(kReg, 0));
    laneBases.push_back(reps.getBasis(kLane, 0));
    laneBases.push_back(reps.getBasis(kLane, 1));
    for (int i = 0; i < vec; ++i) {
      laneBases.push_back(reps.getBasis(kReg, i + 1));
    }
  }

  LinearLayout addrLayout =
      LinearLayout({{kLane, laneBases}, {kWarp, reps.getBases().lookup(kWarp)}},
                   {{kOffset, reps.getOutDimSize(kOffset)}}, false);
  auto regBase = applyLinearLayout(loc, rewriter, addrLayout,
                                   {{kLane, laneId}, {kWarp, warpId}})[0]
                     .second;

  // Elements per op
  auto nVecs = 1 << vec;
  auto elemsPerVec = 32 / bitwidth;
  auto step = nVecs * elemsPerVec;
  for (int i = 0; i < srcVals.size(); i += step) {
    auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
    Value offset = b.xor_(regBase, b.i32_val(regIdx));
    auto vecAddr = b.gep(smemPtrTy, llvmElemTy, smemBase, offset,
                         LLVM::GEPNoWrapFlags::inbounds);
    // Pack into vector of i32
    SmallVector<Value> inputs;
    Type packedTy = vec_ty(llvmElemTy, 32 / bitwidth);
    for (int j = 0; j < nVecs; j++) {
      Value input = b.undef(packedTy);
      for (int k = 0; k < elemsPerVec; k++) {
        input = b.insert_element(
            packedTy, input, srcVals[i + j * elemsPerVec + k], b.i32_val(k));
      }
      inputs.push_back(b.bitcast(input, i32_ty));
    }
    rewriter.create<triton::nvgpu::StoreMatrixOp>(loc, vecAddr, inputs,
                                                  /*needTrans=*/transpose);
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

    // Try to lower transposed or not
    bool lowered = false;
    for (bool transpose : {false, true}) {
      lowered =
          lowerDistributedToSharedStmatrix(
              op.getLoc(), srcTy, memDescType, transpose, adaptor.getSrc(),
              smemBase, llvmElemTy, rewriter, targetInfo)
              .succeeded();
      if (lowered) {
        break;
      }
    }
    if (!lowered) {
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

    // Try to lower transposed or not
    bool lowered = false;
    for (bool transpose : {false, true}) {
      lowered = lowerDistributedToSharedStmatrix(
                    op.getLoc(), op.getSrc().getType(), op.getDst().getType(),
                    transpose, adaptor.getSrc(), smemObj.getBase(), llvmElemTy,
                    rewriter, targetInfo)
                    .succeeded();
      if (lowered) {
        break;
      }
    }
    if (!lowered) {
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
