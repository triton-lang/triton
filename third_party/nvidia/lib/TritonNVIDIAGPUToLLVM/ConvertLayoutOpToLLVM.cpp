#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::linearize;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
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
                    const LLVMTypeConverter *typeConverter,
                    ConversionPatternRewriter &rewriter, Type resultTy);

} // namespace SharedToDotOperandMMAv1

namespace SharedToDotOperandMMAv2 {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
}

namespace {
struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::ConvertLayoutOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isaDistributedLayout(srcLayout) &&
        dstLayout.isa<SharedEncodingAttr>()) {
      return lowerDistributedToShared(op, adaptor, rewriter);
    }
    if (srcLayout.isa<SharedEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerSharedToDotOperand(op, adaptor, rewriter);
    }
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
    if (srcLayout.isa<NvidiaMmaEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }
    if (srcLayout.isa<SharedEncodingAttr>() &&
        isaDistributedLayout(dstLayout)) {
      return lowerSharedToDistributed(op, adaptor, rewriter);
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
    if (auto mmaLayout = layout.dyn_cast<NvidiaMmaEncodingAttr>()) {
      assert(rank == 2 ||
             (rank == 3 && mmaLayout.isAmpere()) && "Unexpected rank");
      auto shapePerCTA = getShapePerCTA(mmaLayout, shape);
      auto instrShape = mmaLayout.getInstrShape();
      SmallVector<Value> mmaColIdx(2);
      SmallVector<Value> mmaRowIdx(2);
      Value threadId = getThreadId(rewriter, loc);
      Value warpSize = i32_val(32);
      Value laneId = urem(threadId, warpSize);
      Value warpId = udiv(threadId, warpSize);
      // TODO: fix the bug in MMAEncodingAttr document
      SmallVector<Value> multiDimWarpId(2);
      auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
      if (mmaLayout.isHopper()) {
        multiDimWarpId[0] = urem(warpId, i32_val(warpsPerCTA[0]));
        multiDimWarpId[1] = udiv(warpId, i32_val(warpsPerCTA[0]));
      } else {
        auto order = triton::gpu::getOrder(mmaLayout);
        multiDimWarpId = delinearize(rewriter, loc, warpId, warpsPerCTA, order);
      }
      Value _1 = i32_val(1);
      Value _2 = i32_val(2);
      Value _4 = i32_val(4);
      Value _8 = i32_val(8);
      Value _16 = i32_val(16);
      if (mmaLayout.isAmpere() || mmaLayout.isHopper()) {
        multiDimWarpId[rank - 1] =
            urem(multiDimWarpId[rank - 1],
                 i32_val(ceil<unsigned>(shapePerCTA[rank - 1],
                                        instrShape[rank - 1])));
        multiDimWarpId[rank - 2] =
            urem(multiDimWarpId[rank - 2],
                 i32_val(ceil<unsigned>(shapePerCTA[rank - 2],
                                        instrShape[rank - 2])));

        Value mmaGrpId = udiv(laneId, _4);
        Value mmaGrpIdP8 = add(mmaGrpId, _8);
        Value mmaThreadIdInGrp = urem(laneId, _4);
        Value mmaThreadIdInGrpM2 = mul(mmaThreadIdInGrp, _2);
        Value mmaThreadIdInGrpM2P1 = add(mmaThreadIdInGrpM2, _1);
        Value rowWarpOffset =
            mul(multiDimWarpId[rank - 2], i32_val(instrShape[rank - 2]));
        mmaRowIdx[0] = add(mmaGrpId, rowWarpOffset);
        mmaRowIdx[1] = add(mmaGrpIdP8, rowWarpOffset);
        Value colWarpOffset =
            mul(multiDimWarpId[rank - 1], i32_val(instrShape[rank - 1]));
        mmaColIdx[0] = add(mmaThreadIdInGrpM2, colWarpOffset);
        mmaColIdx[1] = add(mmaThreadIdInGrpM2P1, colWarpOffset);
      } else if (mmaLayout.isVolta()) {
        // Volta doesn't follow the pattern here."
      } else {
        llvm_unreachable("Unexpected MMALayout version");
      }

      SmallVector<Value> multiDimOffset(rank);
      if (mmaLayout.isHopper()) {
        unsigned elemIdRem4 = elemId % 4;
        unsigned nGrpId = elemId / 4;
        multiDimOffset[0] = elemIdRem4 < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
        multiDimOffset[1] = elemIdRem4 % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
        multiDimOffset[1] = add(multiDimOffset[1], i32_val(8 * nGrpId));
        multiDimOffset[0] =
            add(multiDimOffset[0],
                i32_val(multiDimCTAInRepId[0] * shapePerCTATile[0]));
        multiDimOffset[1] =
            add(multiDimOffset[1],
                i32_val(multiDimCTAInRepId[1] * shapePerCTATile[1]));
      } else if (mmaLayout.isAmpere()) {
        if (rank == 3)
          multiDimOffset[0] =
              add(multiDimWarpId[0],
                  i32_val(multiDimCTAInRepId[0] * shapePerCTATile[0]));
        multiDimOffset[rank - 2] = elemId < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
        multiDimOffset[rank - 1] =
            elemId % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
        multiDimOffset[rank - 2] =
            add(multiDimOffset[rank - 2], i32_val(multiDimCTAInRepId[rank - 2] *
                                                  shapePerCTATile[rank - 2]));
        multiDimOffset[rank - 1] =
            add(multiDimOffset[rank - 1], i32_val(multiDimCTAInRepId[rank - 1] *
                                                  shapePerCTATile[rank - 1]));
      } else if (mmaLayout.isVolta()) {
        auto [isARow, isBRow, isAVec4, isBVec4, _] =
            mmaLayout.decodeVoltaLayoutStates();
        auto coords = SharedToDotOperandMMAv1::getMNCoords(
            threadId, loc, rewriter, mmaLayout.getWarpsPerCTA(), mmaLayout,
            shape, isARow, isBRow, isAVec4, isBVec4);
        return coords[elemId];
      } else {
        llvm_unreachable("Unexpected MMALayout version");
      }
      return multiDimOffset;
    }
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
    auto typeConverter = getTypeConverter();
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
    auto elemTy = typeConverter->convertType(type.getElementType());

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
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
    smemBase = bitcast(smemBase, elemPtrTy);
    auto smemShape = convertType<unsigned, int64_t>(srcShapePerCTA);

    // Store to local shared memory
    {
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
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
          packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
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
    auto typeConverter = getTypeConverter();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (shouldUseDistSmem(srcLayout, dstLayout))
      return lowerDistToDistWithDistSmem(op, adaptor, rewriter);
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
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
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
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
          srcLayout.isa<NvidiaMmaEncodingAttr>()) {
        if (isSrcMmaV1)
          processReplicaForMMAV1(loc, rewriter, /*stNotRd*/ true, srcTy,
                                 multiDimRepId, inVec, paddedRepShape, outOrd,
                                 vals, smemBase, shape);
        else if (isStMatrixCompatible(srcTy) && accumNumReplicates == 1 &&
                 outOrd[0] == 1 && paddedRepShape[1] % 8 == 0) {
          storeDistributedToSharedWithStMatrix(srcTy, vals, smemBase,
                                               paddedRepShape, origRepShape,
                                               loc, rewriter);
        } else
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

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  LogicalResult
  lowerSharedToDistributed(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto typeConverter = getTypeConverter();
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() <= 2 &&
           "Unexpected rank of ConvertLayout(shared->blocked)");
    auto srcSharedLayout = srcTy.getEncoding().cast<SharedEncodingAttr>();
    auto dstLayout = dstTy.getEncoding();
    auto inOrd = getOrder(srcSharedLayout);

    auto smemObj = getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(srcTy.getElementType()), rewriter);
    auto elemTy = typeConverter->convertType(dstTy.getElementType());

    auto srcStrides =
        getStridesFromShapeAndOrder(srcTy.getShape(), inOrd, loc, rewriter);
    auto dstIndices = emitIndices(loc, rewriter, dstLayout, dstTy, true);

    SmallVector<Value> outVals =
        loadSharedToDistributed(op.getResult(), dstIndices, op.getSrc(),
                                smemObj, elemTy, loc, rewriter);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  Value computeStMatrixAddr(Value laneId, int matStride, Location loc,
                            ConversionPatternRewriter &rewriter) const {
    Value rowInMat = urem(laneId, i32_val(8)); // row in the 8x8 matrix
    // linear index of the matrix in the 2x2 matrices
    // Decompose matIndex => s_0, s_1, that is the coordinate in 2x2 matrices in
    // a warp.
    Value matIndex = udiv(laneId, i32_val(8));
    Value s0 = urem(matIndex, i32_val(2));
    Value s1 = udiv(matIndex, i32_val(2));
    Value mIndex = add(rowInMat, mul(s0, i32_val(8)));
    int m8n8Stride = 8;
    Value offset =
        add(mul(mIndex, i32_val(matStride)), mul(s1, i32_val(m8n8Stride)));
    return offset;
  }

  void stMatrixm8n8x4(Value offset, ArrayRef<Value> vals, int indexOffset,
                      Value smemBase, Type elemTy, Location loc,
                      ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> inputs;
    auto prTy = ptr_ty(rewriter.getContext(), 3);
    // Pack the input into 2xf16
    Type packedTy = vec_ty(vals[0].getType(), 2);
    for (int i = 0; i < 4; i++) {
      Value input = undef(packedTy);
      for (int j = 0; j < 2; j++) {
        input = insert_element(packedTy, input, vals[indexOffset + i * 2 + j],
                               i32_val(j));
      }
      inputs.push_back(bitcast(input, i32_ty));
    }
    Value addr = gep(smemBase.getType(),
                     getTypeConverter()->convertType(elemTy), smemBase, offset);
    rewriter.create<triton::nvgpu::StoreMatrixOp>(loc, addr, inputs);
  }

  void storeDistributedToSharedWithStMatrix(
      RankedTensorType tensorTy, SmallVector<Value> &inVals, Value smemBase,
      ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
      Location loc, ConversionPatternRewriter &rewriter) const {
    auto shapePerCTA = getShapePerCTA(tensorTy);
    auto mmaLayout = tensorTy.getEncoding().cast<NvidiaMmaEncodingAttr>();
    auto order = triton::gpu::getOrder(mmaLayout);
    auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
    auto shapePerCTATile = getShapePerCTATile(mmaLayout);
    ArrayRef<unsigned> mmaShape = mmaLayout.getInstrShape();
    // 4xm8n8 matches exactly the size of 1 warp of wgmma layout for 16bit type
    // and has a shape of 16x16.
    int instrN = mmaShape[1] * warpsPerCTA[1];
    int instrM = mmaShape[0] * warpsPerCTA[0];
    std::array<int, 2> numRep = {ceil((int)origRepShape[0], instrM),
                                 ceil((int)origRepShape[1], instrN)};

    Value thread = getThreadId(rewriter, loc);
    Value warp = udiv(thread, i32_val(32));
    Value lane = urem(thread, i32_val(32));

    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warp, warpsPerCTA);

    // Compute the relative offset for each lane.
    Value stMatrixLaneOffset =
        computeStMatrixAddr(lane, paddedRepShape[1], loc, rewriter);
    multiDimWarpId[0] = mul(multiDimWarpId[0], i32_val(mmaShape[0]));
    multiDimWarpId[1] = mul(multiDimWarpId[1], i32_val(mmaShape[1]));
    SmallVector<Value> multiDimOffsetWrapped =
        getWrappedMultiDimOffset(rewriter, loc, multiDimWarpId, origRepShape,
                                 shapePerCTATile, shapePerCTA);
    Value relativeOffset =
        linearize(rewriter, loc, multiDimOffsetWrapped, paddedRepShape, order);
    relativeOffset = add(relativeOffset, stMatrixLaneOffset);
    int indexOffset = 0;
    int m8n8x4Stride = 16;
    int numNChunk = mmaShape[1] / m8n8x4Stride;
    for (int m = 0; m < numRep[0]; m++) {
      for (int n = 0; n < numRep[1]; n++) {
        for (int k = 0; k < numNChunk; k++) {
          Value addr =
              add(relativeOffset, i32_val(k * m8n8x4Stride + n * instrN +
                                          m * instrM * paddedRepShape[1]));
          stMatrixm8n8x4(addr, inVals, indexOffset, smemBase,
                         tensorTy.getElementType(), loc, rewriter);
          indexOffset += 8;
        }
      }
    }
  }

  bool isStMatrixCompatible(RankedTensorType tensorTy) const {
    auto mmaLayout = tensorTy.getEncoding().dyn_cast<NvidiaMmaEncodingAttr>();
    if (!mmaLayout || !mmaLayout.isHopper())
      return false;
    if (tensorTy.getElementType().getIntOrFloatBitWidth() != 16)
      return false;
    return true;
  }

  // blocked -> shared.
  // Swizzling in shared memory to avoid bank conflict. Normally used for
  // A/B operands of dots.
  LogicalResult
  lowerDistributedToShared(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto dstShapePerCTA = triton::gpu::getShapePerCTA(dstTy);
    auto srcLayout = srcTy.getEncoding();
    auto outOrd = dstTy.getEncoding().cast<SharedEncodingAttr>().getOrder();
    assert(srcTy.getShape().size() == 2 ||
           (srcTy.getShape().size() <= 3 && outOrd[2] == 0) &&
               "Unexpected rank of ConvertLayout(blocked->shared)");
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    smemBase = bitcast(smemBase, elemPtrTy);

    int32_t elemSize = elemTy.getIntOrFloatBitWidth();
    auto mmaLayout = srcLayout.dyn_cast<NvidiaMmaEncodingAttr>();
    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
    auto dstStrides =
        getStridesFromShapeAndOrder(dstShapePerCTA, outOrd, loc, rewriter);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy, false);
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    storeDistributedToShared(op.getSrc(), inVals, dstStrides, srcIndices,
                             op.getResult(), smemBase, elemTy, loc, rewriter);
    auto smemObj = SharedMemoryObject(smemBase, elemTy, dstShapePerCTA, outOrd,
                                      loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

  // shared -> mma_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto dstEnc = op.getType().getEncoding().cast<DotOperandEncodingAttr>();
    auto sharedLayout =
        op.getSrc().getType().getEncoding().cast<SharedEncodingAttr>();

    int K;
    if (dstEnc.getOpIdx() == 0) // $a
      K = op.getType().getShape()[sharedLayout.getOrder()[0]];
    else // $b
      K = op.getType().getShape()[sharedLayout.getOrder()[1]];
    bool isOuter = K == 1;

    Value res;
    if (auto mmaLayout =
            dstEnc.getParent().dyn_cast_or_null<NvidiaMmaEncodingAttr>()) {
      res = lowerSharedToDotOperandMMA(op, adaptor, rewriter, mmaLayout, dstEnc,
                                       isOuter);
    } else if (auto blockedLayout =
                   dstEnc.getParent().dyn_cast_or_null<BlockedEncodingAttr>()) {
      auto thread = getThreadId(rewriter, loc);
      res = SharedToDotOperandFMA::convertLayout(
          dstEnc.getOpIdx(), op.getSrc(), adaptor.getSrc(), blockedLayout,
          thread, loc, getTypeConverter(), rewriter);
    } else {
      assert(false && "Unsupported dot operand layout found");
    }

    rewriter.replaceOp(op, res);
    return success();
  }

  // mma -> dot_operand
  LogicalResult
  lowerMmaToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    if (matchMmaV3AndDotOperandLayout(srcTy, dstTy)) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }

    if (isMmaToDotShortcut(srcTy, dstTy)) {
      // get source values
      auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
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
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
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
  Value
  lowerSharedToDotOperandMMA(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             const NvidiaMmaEncodingAttr &mmaLayout,
                             const DotOperandEncodingAttr &dotOperandLayout,
                             bool isOuter) const {
    auto loc = op.getLoc();
    auto src = op.getSrc();
    auto dst = op.getResult();
    bool isMMA = supportMMA(dst, mmaLayout.getVersionMajor());

    auto llvmElemTy =
        getTypeConverter()->convertType(src.getType().getElementType());

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    Value res;
    if (!isOuter && mmaLayout.isAmpere()) { // tensor core v2
      res = SharedToDotOperandMMAv2::convertLayout(
          dotOperandLayout.getOpIdx(), rewriter, loc, src, dotOperandLayout,
          smemObj, getTypeConverter(), getThreadId(rewriter, loc));
    } else if (!isOuter && mmaLayout.isVolta() && isMMA) { // tensor core v1
      bool isMMAv1Row = mmaLayout.getMMAv1IsRow(dotOperandLayout.getOpIdx());
      auto srcSharedLayout =
          src.getType().getEncoding().cast<SharedEncodingAttr>();

      // Can only convert [1, 0] to row or [0, 1] to col for now
      if ((srcSharedLayout.getOrder()[0] == 1 && !isMMAv1Row) ||
          (srcSharedLayout.getOrder()[0] == 0 && isMMAv1Row)) {
        llvm::errs() << "Unsupported Shared -> DotOperand[MMAv1] conversion\n";
        return Value();
      }

      res = SharedToDotOperandMMAv1::convertLayout(
          dotOperandLayout.getOpIdx(), src, smemObj, getThreadId(rewriter, loc),
          loc, getTypeConverter(), rewriter, dst.getType());
    } else {
      assert(false && "Unsupported mma layout found");
    }
    return res;
  }
};
} // namespace

void mlir::triton::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, benefit);
}
