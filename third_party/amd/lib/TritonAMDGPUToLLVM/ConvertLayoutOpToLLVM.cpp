#include "ConvertLayoutOpToLLVM.h"
#include "TritonGPUToLLVMBase.h"
#include "Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::linearize;

using ::AMD::ConvertTritonGPUOpToLLVMPattern;
using ::AMD::ConvertTritonGPUOpToLLVMPatternBase;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Forward declarations

namespace SharedToDotOperandMMAv1 {
using CoordTy = SmallVector<Value>;
using ValueTable = std::map<std::pair<int, int>, std::pair<Value, Value>>;

SmallVector<CoordTy> getMNCoords(Value thread, Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 ArrayRef<unsigned int> wpt,
                                 const NvidiaMmaEncodingAttr &mmaLayout,
                                 ArrayRef<int64_t> shape, bool isARow,
                                 bool isBRow, bool isAVec4, bool isBVec4);

Value convertLayout(int opIdx, Value tensor, const SharedMemoryObject &smemObj,
                    Value thread, Location loc,
                    TritonGPUToLLVMTypeConverter *typeConverter,
                    ConversionPatternRewriter &rewriter, Type resultTy);

} // namespace SharedToDotOperandMMAv1

namespace SharedToDotOperandMMAv2 {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToLLVMTypeConverter *typeConverter, Value thread);
}

#ifdef USE_ROCM
namespace SharedToDotOperandMFMA {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
} // namespace SharedToDotOperandMFMA
#endif

namespace {
#ifdef USE_ROCM
// shared -> dot_operand if the result layout is mfma
Value lowerSharedToDotOperandMFMA(
    triton::gpu::LocalLoadOp op, triton::gpu::LocalLoadOpAdaptor adaptor,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    const AMDMfmaEncodingAttr &mfmaLayout,
    const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) {
  auto loc = op.getLoc();
  Value src = op.getSrc();
  Value dst = op.getResult();
  auto llvmElemTy = typeConverter->convertType(
      src.getType().cast<MemDescType>().getElementType());

  auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                 llvmElemTy, rewriter);
  Value res;

  if (!isOuter) {
    res = SharedToDotOperandMFMA::convertLayout(
        dotOperandLayout.getOpIdx(), rewriter, loc, src, dotOperandLayout,
        smemObj, typeConverter, tid_val());
  } else {
    assert(false && "unsupported layout found");
  }
  return res;
}
#endif

// shared -> mma_operand
LogicalResult lowerSharedToDotOperand(triton::gpu::LocalLoadOp op,
                                      triton::gpu::LocalLoadOpAdaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  Value src = op.getSrc();
  Value dst = op.getResult();
  auto dstTensorTy = dst.getType().cast<RankedTensorType>();
  auto srcTensorTy = src.getType().cast<MemDescType>();
  auto dotOperandLayout =
      dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
  auto sharedLayout = srcTensorTy.getEncoding().cast<SharedEncodingAttr>();

  bool isOuter{};
  int K{};
  if (dotOperandLayout.getOpIdx() == 0) // $a
    K = dstTensorTy.getShape()[sharedLayout.getOrder()[0]];
  else // $b
    K = dstTensorTy.getShape()[sharedLayout.getOrder()[1]];
  isOuter = K == 1;

  Value res;
#ifdef USE_ROCM
  if (auto mfmaLayout = dotOperandLayout.getParent()
                            .dyn_cast_or_null<AMDMfmaEncodingAttr>()) {
    res = lowerSharedToDotOperandMFMA(op, adaptor, typeConverter, rewriter,
                                      mfmaLayout, dotOperandLayout, isOuter);
#endif
  } else if (auto blockedLayout =
                 dotOperandLayout.getParent()
                     .dyn_cast_or_null<BlockedEncodingAttr>()) {
    auto dotOpLayout = dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto thread = getThreadId(rewriter, loc);
    res = SharedToDotOperandFMA::convertLayout(
        dotOpLayout.getOpIdx(), src, adaptor.getSrc(), blockedLayout, thread,
        loc, typeConverter, rewriter);
  } else {
    assert(false && "Unsupported dot operand layout found");
  }

  rewriter.replaceOp(op, res);
  return success();
}

LogicalResult
lowerSharedToDistributed(triton::gpu::LocalLoadOp op,
                                      triton::gpu::LocalLoadOpAdaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                         ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  Value src = op.getSrc();
  Value dst = op.getResult();
  auto srcTy = src.getType().cast<MemDescType>();
  auto srcShape = srcTy.getShape();
  auto dstTy = dst.getType().cast<RankedTensorType>();
  auto dstShape = dstTy.getShape();
  assert(dstShape.size() == 2 &&
         "Unexpected rank of ConvertLayout(shared->blocked)");
  auto srcSharedLayout = srcTy.getEncoding().cast<SharedEncodingAttr>();
  auto dstLayout = dstTy.getEncoding();
  auto inOrd = getOrder(srcSharedLayout);

  auto smemObj = getSharedMemoryObjectFromStruct(
      loc, adaptor.getSrc(),
      typeConverter->convertType(srcTy.getElementType()), rewriter);
  auto elemTy = typeConverter->convertType(dstTy.getElementType());

  auto srcStrides = getStridesFromShapeAndOrder(srcShape, inOrd, loc, rewriter);
  auto dstIndices = emitIndices(loc, rewriter, dstLayout, dstTy, true);

  SmallVector<Value> outVals = loadSharedToDistributed(
      dst, dstIndices, src, smemObj, elemTy, loc, rewriter);

  Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
  rewriter.replaceOp(op, result);

  return success();
}

struct LocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  using ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerSharedToDotOperand(op, adaptor, getTypeConverter(), rewriter);
    }
    if (srcLayout.isa<SharedEncodingAttr>() &&
        isaDistributedLayout(dstLayout)) {
      return lowerSharedToDistributed(op, adaptor, getTypeConverter(),
                                      rewriter);
    }
    return failure();
  }
};

struct ConvertLayoutOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::ConvertLayoutOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    // llvm::outs() << srcTy << " " << dstTy << "\n";
    // forwarding on mma->mma shortcut, lower distributed->distributed otherwise
    if (srcLayout.isa<NvidiaMmaEncodingAttr>() &&
        dstLayout.isa<NvidiaMmaEncodingAttr>()) {
      if (isMmaToMmaShortcut(srcTy, dstTy)) {
        return lowerMmaToMma(op, adaptor, rewriter);
      }
    }
    if (isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
#ifdef USE_ROCM
    if (srcLayout.isa<AMDMfmaEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerMfmaToDotOperand(op, adaptor, rewriter);
    }
#endif
    if (srcLayout.isa<NvidiaMmaEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }
    // TODO: to be implemented
    llvm_unreachable("unsupported layout conversion");
    return failure();
  }

private:
  SmallVector<Value>
  getMultiDimOffset(Attribute layout, Location loc,
                    ConversionPatternRewriter &rewriter, unsigned elemId,
                    RankedTensorType type,
                    ArrayRef<unsigned> multiDimCTAInRepId,
                    ArrayRef<unsigned> shapePerCTATile) const {
    auto shape = type.getShape();
    unsigned rank = shape.size();
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
      auto multiDimOffsetFirstElem =
          emitBaseIndexForLayout(loc, rewriter, blockedLayout, type, false);
      SmallVector<Value> multiDimOffset(rank);
      SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
          elemId, getSizePerThread(layout), getOrder(layout));
      for (unsigned d = 0; d < rank; ++d) {
        multiDimOffset[d] =
            add(multiDimOffsetFirstElem[d],
                i32_val(multiDimCTAInRepId[d] * shapePerCTATile[d] +
                        multiDimElemId[d]));
      }
      return multiDimOffset;
    }
    if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
      unsigned dim = sliceLayout.getDim();
      auto parentEncoding = sliceLayout.getParent();
      auto parentSizePerThread = getSizePerThread(parentEncoding);
      auto parentShape = sliceLayout.paddedShape(shape);
      auto parentTy = RankedTensorType::get(parentShape, type.getElementType(),
                                            parentEncoding);
      auto offsets = emitOffsetForLayout(layout, type);
      auto parentOffset = emitOffsetForLayout(parentEncoding, parentTy);
      SmallVector<int> idxs;
      for (SmallVector<unsigned> off : offsets) {
        off.insert(off.begin() + dim, 0);
        auto it = std::find(parentOffset.begin(), parentOffset.end(), off);
        idxs.push_back(std::distance(parentOffset.begin(), it));
      }
      auto multiDimOffsetParent = getMultiDimOffset(
          parentEncoding, loc, rewriter, idxs[elemId], parentTy,
          sliceLayout.paddedShape(multiDimCTAInRepId),
          sliceLayout.paddedShape(shapePerCTATile));
      SmallVector<Value> multiDimOffset(rank);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d == dim)
          continue;
        unsigned slicedD = d < dim ? d : (d - 1);
        multiDimOffset[slicedD] = multiDimOffsetParent[d];
      }
      return multiDimOffset;
    }
    // if (auto mmaLayout = layout.dyn_cast<NvidiaMmaEncodingAttr>()) {
    //   auto shapePerCTA = getShapePerCTA(mmaLayout, shape);
    //   auto instrShape = mmaLayout.getInstrShape();
    //   SmallVector<Value> mmaColIdx(4);
    //   SmallVector<Value> mmaRowIdx(2);
    //   Value threadId = getThreadId(rewriter, loc);
    //   Value warpSize = i32_val(32);
    //   Value laneId = urem(threadId, warpSize);
    //   Value warpId = udiv(threadId, warpSize);
    //   // TODO: fix the bug in MMAEncodingAttr document
    //   SmallVector<Value> multiDimWarpId(2);
    //   auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
    //   if (mmaLayout.isHopper()) {
    //     multiDimWarpId[0] = urem(warpId, i32_val(warpsPerCTA[0]));
    //     multiDimWarpId[1] = udiv(warpId, i32_val(warpsPerCTA[0]));
    //   } else {
    //     auto order = triton::gpu::getOrder(mmaLayout);
    //     multiDimWarpId = delinearize(rewriter, loc, warpId, warpsPerCTA,
    //     order);
    //   }
    //   Value _1 = i32_val(1);
    //   Value _2 = i32_val(2);
    //   Value _4 = i32_val(4);
    //   Value _8 = i32_val(8);
    //   Value _16 = i32_val(16);
    //   if (mmaLayout.isAmpere() || mmaLayout.isHopper()) {
    //     multiDimWarpId[0] =
    //         urem(multiDimWarpId[0],
    //              i32_val(ceil<unsigned>(shapePerCTA[0], instrShape[0])));
    //     multiDimWarpId[1] =
    //         urem(multiDimWarpId[1],
    //              i32_val(ceil<unsigned>(shapePerCTA[1], instrShape[1])));

    //     Value mmaGrpId = udiv(laneId, _4);
    //     Value mmaGrpIdP8 = add(mmaGrpId, _8);
    //     Value mmaThreadIdInGrp = urem(laneId, _4);
    //     Value mmaThreadIdInGrpM2 = mul(mmaThreadIdInGrp, _2);
    //     Value mmaThreadIdInGrpM2P1 = add(mmaThreadIdInGrpM2, _1);
    //     Value rowWarpOffset = mul(multiDimWarpId[0], i32_val(instrShape[0]));
    //     mmaRowIdx[0] = add(mmaGrpId, rowWarpOffset);
    //     mmaRowIdx[1] = add(mmaGrpIdP8, rowWarpOffset);
    //     Value colWarpOffset = mul(multiDimWarpId[1], i32_val(instrShape[1]));
    //     mmaColIdx[0] = add(mmaThreadIdInGrpM2, colWarpOffset);
    //     mmaColIdx[1] = add(mmaThreadIdInGrpM2P1, colWarpOffset);
    //   } else if (mmaLayout.isVolta()) {
    //     // Volta doesn't follow the pattern here."
    //   } else {
    //     llvm_unreachable("Unexpected MMALayout version");
    //   }

    //   assert(rank == 2);
    //   SmallVector<Value> multiDimOffset(rank);
    //   if (mmaLayout.isHopper()) {
    //     unsigned elemIdRem4 = elemId % 4;
    //     unsigned nGrpId = elemId / 4;
    //     multiDimOffset[0] = elemIdRem4 < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
    //     multiDimOffset[1] = elemIdRem4 % 2 == 0 ? mmaColIdx[0] :
    //     mmaColIdx[1]; multiDimOffset[1] = add(multiDimOffset[1], i32_val(8 *
    //     nGrpId)); multiDimOffset[0] =
    //         add(multiDimOffset[0],
    //             i32_val(multiDimCTAInRepId[0] * shapePerCTATile[0]));
    //     multiDimOffset[1] =
    //         add(multiDimOffset[1],
    //             i32_val(multiDimCTAInRepId[1] * shapePerCTATile[1]));
    //   } else if (mmaLayout.isAmpere()) {
    //     multiDimOffset[0] = elemId < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
    //     multiDimOffset[1] = elemId % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
    //     multiDimOffset[0] =
    //         add(multiDimOffset[0],
    //             i32_val(multiDimCTAInRepId[0] * shapePerCTATile[0]));
    //     multiDimOffset[1] =
    //         add(multiDimOffset[1],
    //             i32_val(multiDimCTAInRepId[1] * shapePerCTATile[1]));
    //   } else if (mmaLayout.isVolta()) {
    //     auto [isARow, isBRow, isAVec4, isBVec4, _] =
    //         mmaLayout.decodeVoltaLayoutStates();
    //     auto coords = SharedToDotOperandMMAv1::getMNCoords(
    //         threadId, loc, rewriter, mmaLayout.getWarpsPerCTA(), mmaLayout,
    //         shape, isARow, isBRow, isAVec4, isBVec4);
    //     return coords[elemId];
    //   } else {
    //     llvm_unreachable("Unexpected MMALayout version");
    //   }
    //   return multiDimOffset;
    // }
#ifdef USE_ROCM
    if (auto mfmaLayout = layout.dyn_cast<AMDMfmaEncodingAttr>()) {
      auto multiDimBase =
          emitBaseIndexForLayout(loc, rewriter, layout, type, false);
      SmallVector<SmallVector<unsigned>> offsets;
      assert(rank == 2);
      SmallVector<Value> multiDimOffset(rank);
      emitMfmaOffsetForCTA(mfmaLayout, offsets, multiDimCTAInRepId[0],
                           multiDimCTAInRepId[1]);
      multiDimOffset[0] = add(multiDimBase[0], i32_val(offsets[elemId][0]));
      multiDimOffset[1] = add(multiDimBase[1], i32_val(offsets[elemId][1]));
      return multiDimOffset;
    }
#endif
    llvm_unreachable("unexpected layout in getMultiDimOffset");
  }

  SmallVector<Value>
  getWrappedMultiDimOffset(ConversionPatternRewriter &rewriter, Location loc,
                           ArrayRef<Value> multiDimOffset,
                           ArrayRef<unsigned> shape,
                           SmallVector<unsigned> shapePerCTATile,
                           SmallVector<int64_t> shapePerCTA) const {
    unsigned rank = shape.size();
    SmallVector<Value> multiDimOffsetWrapped(rank);
    for (unsigned d = 0; d < rank; ++d) {
      if (shapePerCTATile[d] > shapePerCTA[d])
        multiDimOffsetWrapped[d] = urem(multiDimOffset[d], i32_val(shape[d]));
      else
        multiDimOffsetWrapped[d] = multiDimOffset[d];
    }
    return multiDimOffsetWrapped;
  }

  // shared memory rd/st for blocked or mma layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> origRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase) const {
    auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
    auto layout = type.getEncoding();
    auto rank = type.getRank();
    auto sizePerThread = getSizePerThread(layout);
    auto accumSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<unsigned> numCTATiles(rank);
    auto shapePerCTATile = getShapePerCTATile(layout);
    auto shapePerCTA = getShapePerCTA(layout, type.getShape());
    auto order = getOrder(layout);
    for (unsigned d = 0; d < rank; ++d) {
      numCTATiles[d] = ceil<unsigned>(shapePerCTA[d], shapePerCTATile[d]);
    }
    auto elemTy = type.getElementType();
    bool isInt1 = elemTy.isInteger(1);
    bool isPtr = elemTy.isa<triton::PointerType>();
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
    if (isInt1)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);

    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
      auto multiDimCTAInRepId =
          getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
      SmallVector<unsigned> multiDimCTAId(rank);
      for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
        auto d = it.index();
        multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
      }

      auto linearCTAId =
          getLinearIndex<unsigned>(multiDimCTAId, numCTATiles, order);
      // TODO: This is actually redundant index calculation, we should
      //       consider of caching the index calculation result in case
      //       of performance issue observed.
      for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type,
                              multiDimCTAInRepId, shapePerCTATile);
        SmallVector<Value> multiDimOffsetWrapped = getWrappedMultiDimOffset(
            rewriter, loc, multiDimOffset, origRepShape, shapePerCTATile,
            shapePerCTA);
        Value offset = linearize(rewriter, loc, multiDimOffsetWrapped,
                                 paddedRepShape, outOrd);
        auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
        Value ptr = gep(elemPtrTy, llvmElemTy, smemBase, offset);
        auto vecTy = vec_ty(llvmElemTy, vec);
        ptr = bitcast(ptr, ptr_ty(rewriter.getContext(), 3));
        if (stNotRd) {
          Value valVec = undef(vecTy);
          for (unsigned v = 0; v < vec; ++v) {
            auto currVal = vals[elemId + linearCTAId * accumSizePerThread + v];
            if (isInt1)
              currVal = zext(llvmElemTy, currVal);
            else if (isPtr)
              currVal = ptrtoint(llvmElemTy, currVal);
            valVec = insert_element(vecTy, valVec, currVal, i32_val(v));
          }
          store(valVec, ptr);
        } else {
          Value valVec = load(vecTy, ptr);
          for (unsigned v = 0; v < vec; ++v) {
            Value currVal = extract_element(llvmElemTy, valVec, i32_val(v));
            if (isInt1)
              currVal = icmp_ne(currVal,
                                rewriter.create<LLVM::ConstantOp>(
                                    loc, i8_ty, rewriter.getI8IntegerAttr(0)));
            else if (isPtr)
              currVal = inttoptr(llvmElemTyOrig, currVal);
            vals[elemId + linearCTAId * accumSizePerThread + v] = currVal;
          }
        }
      }
    }
  }

  // The MMAV1's result is quite different from the existing "Replica"
  // structure, add a new simple but clear implementation for it to avoid
  // modifying the logic of the existing one.
  void processReplicaForMMAV1(Location loc, ConversionPatternRewriter &rewriter,
                              bool stNotRd, RankedTensorType type,
                              ArrayRef<unsigned> multiDimRepId, unsigned vec,
                              ArrayRef<unsigned> paddedRepShape,
                              ArrayRef<unsigned> outOrd,
                              SmallVector<Value> &vals, Value smemBase,
                              ArrayRef<int64_t> shape,
                              bool isDestMma = false) const {
    unsigned accumNumCTAsEachRep = 1;
    auto layout = type.getEncoding();
    NvidiaMmaEncodingAttr mma = layout.dyn_cast<NvidiaMmaEncodingAttr>();
    auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>();
    if (sliceLayout)
      mma = sliceLayout.getParent().cast<NvidiaMmaEncodingAttr>();

    auto order = getOrder(layout);
    auto rank = type.getRank();
    int accumSizePerThread = vals.size();

    SmallVector<unsigned> numCTAs(rank, 1);
    SmallVector<unsigned> numCTAsEachRep(rank, 1);
    SmallVector<unsigned> shapePerCTATile = getShapePerCTATile(layout, shape);
    SmallVector<int64_t> shapePerCTA = getShapePerCTA(layout, shape);
    auto elemTy = getTypeConverter()->convertType(type.getElementType());

    int ctaId = 0;

    auto multiDimCTAInRepId =
        getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
    SmallVector<unsigned> multiDimCTAId(rank);
    for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
      auto d = it.index();
      multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
    }

    std::vector<std::pair<SmallVector<Value>, Value>> coord2valT(
        accumSizePerThread);
    bool needTrans = outOrd[0] != 0;
    if (sliceLayout || isDestMma)
      needTrans = false;

    vec = needTrans ? 2 : 1;
    {
      // We need to transpose the coordinates and values here to enable vec=2
      // when store to smem.
      std::vector<std::pair<SmallVector<Value>, Value>> coord2val(
          accumSizePerThread);
      for (unsigned elemId = 0; elemId < accumSizePerThread; ++elemId) {
        // TODO[Superjomn]: Move the coordinate computation out of loop, it is
        // duplicate in Volta.
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type,
                              multiDimCTAInRepId, shapePerCTATile);
        coord2val[elemId] = std::make_pair(multiDimOffset, vals[elemId]);
      }

      if (needTrans) {
        // do transpose
        int numM = mma.getMMAv1NumOuter(shapePerCTA, 0);
        int numN = accumSizePerThread / numM;

        for (int r = 0; r < numM; r++) {
          for (int c = 0; c < numN; c++) {
            coord2valT[r * numN + c] = std::move(coord2val[c * numM + r]);
          }
        }
      } else {
        coord2valT = std::move(coord2val);
      }
    }

    // Now the coord2valT has the transposed and contiguous elements(with
    // vec=2), the original vals is not needed.
    for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
      auto coord = coord2valT[elemId].first;
      Value offset = linearize(rewriter, loc, coord, paddedRepShape, outOrd);
      auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
      Value ptr = gep(elemPtrTy, elemTy, smemBase, offset);
      auto vecTy = vec_ty(elemTy, vec);
      ptr = bitcast(ptr, ptr_ty(rewriter.getContext(), 3));
      if (stNotRd) {
        Value valVec = undef(vecTy);
        for (unsigned v = 0; v < vec; ++v) {
          auto currVal = coord2valT[elemId + v].second;
          valVec = insert_element(vecTy, valVec, currVal, i32_val(v));
        }
        store(valVec, ptr);
      } else {
        Value valVec = load(vecTy, ptr);
        for (unsigned v = 0; v < vec; ++v) {
          Value currVal = extract_element(elemTy, valVec, i32_val(v));
          vals[elemId + v] = currVal;
        }
      }
    }
  }

  LogicalResult
  lowerDistToDistWithDistSmem(triton::gpu::ConvertLayoutOp op,
                              OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();

    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto dstLayout = dstTy.getEncoding();
    auto srcShapePerCTA = getShapePerCTA(srcTy);
    auto srcCTAsPerCGA = triton::gpu::getCTAsPerCGA(srcLayout);
    auto srcCTAOrder = triton::gpu::getCTAOrder(srcLayout);
    unsigned rank = srcShapePerCTA.size();

    auto llvmElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);

    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    smemBase = bitcast(smemBase, elemPtrTy);
    auto smemShape = convertType<unsigned, int64_t>(srcShapePerCTA);

    // Store to local shared memory
    {
      auto inVals =
          unpackLLElements(loc, adaptor.getSrc(), rewriter);
      auto inIndices =
          emitIndices(loc, rewriter, srcLayout, srcTy, /*withCTAOffset*/ false);

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
      auto outIndices =
          emitIndices(loc, rewriter, dstLayout, dstTy, /*withCTAOffset*/ true);

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
        outVals.push_back(load_dsmem(ptr, remoteCTAId, llvmElemTy));
      }

      Value result =
          packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
    }

    // Cluster barrier
    rewriter.create<triton::nvidia_gpu::ClusterArriveOp>(loc, false);
    rewriter.create<triton::nvidia_gpu::ClusterWaitOp>(loc);

    return success();
  }

  // blocked/mma -> blocked/mma.
  // Data padding in shared memory to avoid bank conflict.
  LogicalResult
  lowerDistributedToDistributed(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (shouldUseDistSmem(srcLayout, dstLayout))
      return lowerDistToDistWithDistSmem(op, adaptor, rewriter);
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> numReplicates(rank);
    SmallVector<unsigned> inNumCTAsEachRep(rank);
    SmallVector<unsigned> outNumCTAsEachRep(rank);
    SmallVector<unsigned> inNumCTAs(rank);
    SmallVector<unsigned> outNumCTAs(rank);
    auto srcShapePerCTATile = getShapePerCTATile(srcLayout, srcTy.getShape());
    auto dstShapePerCTATile = getShapePerCTATile(dstLayout, shape);
    auto shapePerCTA = getShapePerCTA(srcLayout, shape);

    // For Volta, all the coords for a CTA are calculated.
    bool isSrcMmaV1{}, isDstMmaV1{};
    if (auto mmaLayout = srcLayout.dyn_cast<NvidiaMmaEncodingAttr>()) {
      isSrcMmaV1 = mmaLayout.isVolta();
    }
    if (auto sliceLayout = srcLayout.dyn_cast<SliceEncodingAttr>()) {
      isSrcMmaV1 =
          sliceLayout.getParent().isa<NvidiaMmaEncodingAttr>() &&
          sliceLayout.getParent().cast<NvidiaMmaEncodingAttr>().isVolta();
    }
    if (auto mmaLayout = dstLayout.dyn_cast<NvidiaMmaEncodingAttr>()) {
      isDstMmaV1 = mmaLayout.isVolta();
    }
    if (auto sliceLayout = dstLayout.dyn_cast<SliceEncodingAttr>()) {
      isDstMmaV1 =
          sliceLayout.getParent().isa<NvidiaMmaEncodingAttr>() &&
          sliceLayout.getParent().cast<NvidiaMmaEncodingAttr>().isVolta();
    }

    for (unsigned d = 0; d < rank; ++d) {
      unsigned inPerCTA =
          std::min<unsigned>(shapePerCTA[d], srcShapePerCTATile[d]);
      unsigned outPerCTA =
          std::min<unsigned>(shapePerCTA[d], dstShapePerCTATile[d]);
      unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
      numReplicates[d] = ceil<unsigned>(shapePerCTA[d], maxPerCTA);
      inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
      outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
      assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
      inNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], inPerCTA);
      outNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], outPerCTA);
    }
    // Potentially we need to store for multiple CTAs in this replication
    auto accumNumReplicates = product<unsigned>(numReplicates);
    auto vals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    unsigned inVec = 0;
    unsigned outVec = 0;
    auto origRepShape = getRepShapeForCvtLayout(op);
    auto paddedRepShape = getScratchConfigForCvtLayout(op, inVec, outVec);
    if (getElementTypeOrSelf(op.getType())
            .isa<mlir::Float8E4M3B11FNUZType, mlir::Float8E4M3FNType>()) {
      assert(inVec % 4 == 0 && "conversion not supported for FP8E4M3B15");
      assert(outVec % 4 == 0 && "conversion not supported for FP8E4M3B15");
    }

    unsigned outElems = getTotalElemsPerThread(dstTy);
    auto outOrd = getOrder(dstLayout);
    SmallVector<Value> outVals(outElems);

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0) {
        barrier();
      }
      if (srcLayout.isa<BlockedEncodingAttr>() ||
          srcLayout.isa<SliceEncodingAttr>() ||
#ifdef USE_ROCM
          srcLayout.isa<AMDMfmaEncodingAttr>() ||
#endif
          srcLayout.isa<NvidiaMmaEncodingAttr>()) {
        if (isSrcMmaV1)
          processReplicaForMMAV1(loc, rewriter, /*stNotRd*/ true, srcTy,
                                 multiDimRepId, inVec, paddedRepShape, outOrd,
                                 vals, smemBase, shape);
        else
          processReplica(loc, rewriter, /*stNotRd*/ true, srcTy,
                         inNumCTAsEachRep, multiDimRepId, inVec, paddedRepShape,
                         origRepShape, outOrd, vals, smemBase);
      } else {
        llvm::report_fatal_error(
            "ConvertLayout with input layout not implemented");
        return failure();
      }
      barrier();
      if (dstLayout.isa<BlockedEncodingAttr>() ||
          dstLayout.isa<SliceEncodingAttr>() ||
#ifdef USE_ROCM
          dstLayout.isa<AMDMfmaEncodingAttr>() ||
#endif
          dstLayout.isa<NvidiaMmaEncodingAttr>()) {
        if (isDstMmaV1)
          processReplicaForMMAV1(loc, rewriter, /*stNotRd*/ false, dstTy,
                                 multiDimRepId, outVec, paddedRepShape, outOrd,
                                 outVals, smemBase, shape, /*isDestMma=*/true);
        else
          processReplica(loc, rewriter, /*stNotRd*/ false, dstTy,
                         outNumCTAsEachRep, multiDimRepId, outVec,
                         paddedRepShape, origRepShape, outOrd, outVals,
                         smemBase);
      } else {
        llvm::report_fatal_error(
            "ConvertLayout with output layout not implemented");
        return failure();
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

#ifdef USE_ROCM
  LogicalResult
  lowerMfmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    if (isMfmaToDotShortcut(srcTy, dstTy)) {
      // get source values
      auto vals =
          unpackLLElements(loc, adaptor.getSrc(), rewriter);
      unsigned elems = getTotalElemsPerThread(srcTy);
      Type elemTy =
          this->getTypeConverter()->convertType(srcTy.getElementType());
      // for the destination type, we need to pack values together
      // so they can be consumed by tensor core operations
      SmallVector<Value> vecVals;
      SmallVector<Type> types;
      auto elemSize = elemTy.getIntOrFloatBitWidth();
      // TODO: Support types other than float16 and
      // bf16 (represented as int16 in llvm ir).
      assert((type::isFloat(elemTy) || type::isInt(elemTy)) && elemSize == 16);
      // vecSize is an number of sequential elements stored by one thread
      // - For MFMA (nonKDim == 32) encoding it is 4
      // - For MFMA (nonKDim == 32) operand encoding it is
      // dotOperandEndocing::kWidth,
      //   which is 4 for fp16 and bfloat16 dtypes
      //
      // For mentioned types MFMA and MFMA operand layouts are the same
      const unsigned vecSize = 4;
      Type vecTy = vec_ty(elemTy, vecSize);
      types = SmallVector<Type>(elems / vecSize, vecTy);
      for (unsigned i = 0; i < elems; i += vecSize) {
        Value packed = rewriter.create<LLVM::UndefOp>(loc, vecTy);
        for (unsigned j = 0; j < vecSize; j++)
          packed = insert_element(vecTy, packed, vals[i + j], i32_val(j));
        vecVals.push_back(packed);
      }
      Value view =
          packLLElements(loc, getTypeConverter(), vecVals, rewriter, dstTy);
      rewriter.replaceOp(op, view);
      return success();
    }
    return failure();
  }
#endif

  // mma -> dot_operand
  LogicalResult
  lowerMmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    if (matchMmaV3AndDotOperandLayout(srcTy, dstTy)) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }

    if (isMmaToDotShortcut(srcTy, dstTy)) {
      // get source values
      auto vals =
          unpackLLElements(loc, adaptor.getSrc(), rewriter);
      unsigned elems = getTotalElemsPerThread(srcTy);
      Type elemTy =
          this->getTypeConverter()->convertType(srcTy.getElementType());
      // for the destination type, we need to pack values together
      // so they can be consumed by tensor core operations
      SmallVector<Value> vecVals;
      SmallVector<Type> types;
      // For some reasons, LLVM's NVPTX backend inserts unnecessary (?) integer
      // instructions to pack & unpack sub-word integers. A workaround is to
      // store the results of ldmatrix in i32
      auto elemSize = elemTy.getIntOrFloatBitWidth();
      if (auto intTy = elemTy.dyn_cast<IntegerType>() && elemSize <= 16) {
        auto fold = 32 / elemSize;
        for (unsigned i = 0; i < elems; i += fold) {
          Value val = i32_val(0);
          for (unsigned j = 0; j < fold; j++) {
            auto ext =
                shl(i32_ty, zext(i32_ty, vals[i + j]), i32_val(elemSize * j));
            val = or_(i32_ty, val, ext);
          }
          vecVals.push_back(val);
        }
        elems = elems / (32 / elemSize);
        types = SmallVector<Type>(elems, i32_ty);
      } else {
        unsigned vecSize = std::max<unsigned>(32 / elemSize, 1);
        Type vecTy = vec_ty(elemTy, vecSize);
        types = SmallVector<Type>(elems / vecSize, vecTy);
        for (unsigned i = 0; i < elems; i += vecSize) {
          Value packed = rewriter.create<LLVM::UndefOp>(loc, vecTy);
          for (unsigned j = 0; j < vecSize; j++)
            packed = insert_element(vecTy, packed, vals[i + j], i32_val(j));
          vecVals.push_back(packed);
        }
      }

#ifndef USE_ROCM
      // This needs to be ordered the same way that
      // ldmatrix.x4 would order it
      // TODO: this needs to be refactor so we don't
      // implicitly depends on how emitOffsetsForMMAV2
      // is implemented
      SmallVector<Value> reorderedVals;
      for (unsigned i = 0; i < vecVals.size(); i += 4) {
        reorderedVals.push_back(bitcast(vecVals[i], i32_ty));
        reorderedVals.push_back(bitcast(vecVals[i + 2], i32_ty));
        reorderedVals.push_back(bitcast(vecVals[i + 1], i32_ty));
        reorderedVals.push_back(bitcast(vecVals[i + 3], i32_ty));
      }

      Value view = packLLElements(loc, getTypeConverter(), reorderedVals,
                                                      rewriter, dstTy);
      rewriter.replaceOp(op, view);
      return success();
#else
    // TODO check if this is needed
    Value view =
        packLLElements(loc, getTypeConverter(), vecVals, rewriter, dstTy);
    rewriter.replaceOp(op, view);
    return success();
#endif
    }
    return failure();
  }

  // mma -> mma
  LogicalResult lowerMmaToMma(triton::gpu::ConvertLayoutOp op,
                              OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    if (triton::gpu::getTotalElemsPerThread(srcTy) ==
        triton::gpu::getTotalElemsPerThread(dstTy)) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
    // get source values
    auto vals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> retVals;
    SmallVector<unsigned> dstElementPerThread =
        triton::gpu::getElemsPerThread(dstTy);
    SmallVector<unsigned> srcElementPerThread =
        triton::gpu::getElemsPerThread(srcTy);
    for (unsigned j = 0; j < dstElementPerThread[0]; j++) {
      for (unsigned i = 0; i < dstElementPerThread[1]; i++) {
        if (i >= srcElementPerThread[1] || j >= srcElementPerThread[0]) {
          retVals.push_back(undef(vals[0].getType()));
          continue;
        }
        unsigned index = i + j * srcElementPerThread[1];
        retVals.push_back(vals[index]);
      }
    }
    assert(retVals.size() == triton::gpu::getTotalElemsPerThread(dstTy));
    Value view =
        packLLElements(loc, getTypeConverter(), retVals, rewriter, dstTy);
    rewriter.replaceOp(op, view);
    return success();
  }

  // shared -> dot_operand if the result layout is mma
  // Value
  // lowerSharedToDotOperandMMA(triton::gpu::ConvertLayoutOp op, OpAdaptor
  // adaptor,
  //                            ConversionPatternRewriter &rewriter,
  //                            const NvidiaMmaEncodingAttr &mmaLayout,
  //                            const DotOperandEncodingAttr &dotOperandLayout,
  //                            bool isOuter) const {
  //   auto loc = op.getLoc();
  //   Value src = op.getSrc();
  //   Value dst = op.getResult();
  //   bool isMMA = supportMMA(dst, mmaLayout.getVersionMajor());

  //   auto llvmElemTy = getTypeConverter()->convertType(
  //       src.getType().cast<RankedTensorType>().getElementType());

  //   auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
  //                                                  llvmElemTy, rewriter);
  //   Value res;
  //   if (!isOuter && mmaLayout.isAmpere()) { // tensor core v2
  //     res = SharedToDotOperandMMAv2::convertLayout(
  //         dotOperandLayout.getOpIdx(), rewriter, loc, src, dotOperandLayout,
  //         smemObj, getTypeConverter(), getThreadId(rewriter, loc));
  //   } else if (!isOuter && mmaLayout.isVolta() && isMMA) { // tensor core v1
  //     bool isMMAv1Row = mmaLayout.getMMAv1IsRow(dotOperandLayout.getOpIdx());
  //     auto srcSharedLayout = src.getType()
  //                                .cast<RankedTensorType>()
  //                                .getEncoding()
  //                                .cast<SharedEncodingAttr>();

  //     // Can only convert [1, 0] to row or [0, 1] to col for now
  //     if ((srcSharedLayout.getOrder()[0] == 1 && !isMMAv1Row) ||
  //         (srcSharedLayout.getOrder()[0] == 0 && isMMAv1Row)) {
  //       llvm::errs() << "Unsupported Shared -> DotOperand[MMAv1]
  //       conversion\n"; return Value();
  //     }

  //     res = SharedToDotOperandMMAv1::convertLayout(
  //         dotOperandLayout.getOpIdx(), src, smemObj, getThreadId(rewriter,
  //         loc), loc, getTypeConverter(), rewriter, dst.getType());
  //   } else {
  //     assert(false && "Unsupported mma layout found");
  //   }
  //   return res;
  // }
};
} // namespace

namespace AMD {
void populateConvertLayoutOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, allocation,
                                          indexCacheInfo, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, benefit);
}
} // namespace AMD
