#include "ConvertLayoutOpToLLVM.h"
#include "DotOpHelpers.h"

using ::mlir::LLVM::DotOpFMAConversionHelper;
using ::mlir::LLVM::DotOpMmaV1ConversionHelper;
using ::mlir::LLVM::getElementsFromStruct;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::getStructFromElements;
using ::mlir::LLVM::MMA16816ConversionHelper;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;

bool isMmaToDotShortcut(MmaEncodingAttr &mmaLayout,
                        DotOperandEncodingAttr &dotOperandLayout) {
  // dot_op<opIdx=0, parent=#mma> = #mma
  // when #mma = MmaEncoding<version=2, warpsPerCTA=[..., 1]>
  return mmaLayout.getWarpsPerCTA()[1] == 1 &&
         dotOperandLayout.getOpIdx() == 0 &&
         dotOperandLayout.getParent() == mmaLayout;
}

void storeDistributedToShared(Value src, Value llSrc,
                              ArrayRef<Value> dstStrides,
                              ArrayRef<SmallVector<Value>> srcIndices,
                              Value dst, Value smemBase, Type elemTy,
                              Location loc,
                              ConversionPatternRewriter &rewriter) {
  auto srcTy = src.getType().cast<RankedTensorType>();
  auto srcShape = srcTy.getShape();
  assert(srcShape.size() == 2 && "Unexpected rank of storeDistributedToShared");
  auto dstTy = dst.getType().cast<RankedTensorType>();
  auto srcDistributedLayout = srcTy.getEncoding();
  if (auto mmaLayout = srcDistributedLayout.dyn_cast<MmaEncodingAttr>()) {
    assert((!mmaLayout.isVolta()) &&
           "ConvertLayout MMAv1->Shared is not suppported yet");
  }
  auto dstSharedLayout = dstTy.getEncoding().cast<SharedEncodingAttr>();
  auto inOrd = getOrder(srcDistributedLayout);
  auto outOrd = dstSharedLayout.getOrder();
  unsigned inVec =
      inOrd == outOrd ? getContigPerThread(srcDistributedLayout)[inOrd[0]] : 1;
  unsigned outVec = dstSharedLayout.getVec();
  unsigned minVec = std::min(outVec, inVec);
  unsigned perPhase = dstSharedLayout.getPerPhase();
  unsigned maxPhase = dstSharedLayout.getMaxPhase();
  unsigned numElems = getElemsPerThread(srcTy);
  assert(numElems == srcIndices.size());
  auto inVals = getElementsFromStruct(loc, llSrc, rewriter);
  auto wordTy = vec_ty(elemTy, minVec);
  auto elemPtrTy = ptr_ty(elemTy);
  Value outVecVal = i32_val(outVec);
  Value minVecVal = i32_val(minVec);
  Value word;
  for (unsigned i = 0; i < numElems; ++i) {
    if (i % minVec == 0)
      word = undef(wordTy);
    word = insert_element(wordTy, word, inVals[i], i32_val(i % minVec));
    if (i % minVec == minVec - 1) {
      // step 1: recover the multidim_index from the index of
      SmallVector<Value> multiDimIdx = srcIndices[i];
      SmallVector<Value> dbgVal = srcIndices[i];

      // step 2: do swizzling
      Value remained = urem(multiDimIdx[outOrd[0]], outVecVal);
      multiDimIdx[outOrd[0]] = udiv(multiDimIdx[outOrd[0]], outVecVal);
      Value off_1 = mul(multiDimIdx[outOrd[1]], dstStrides[outOrd[1]]);
      Value phaseId = udiv(multiDimIdx[outOrd[1]], i32_val(perPhase));
      phaseId = urem(phaseId, i32_val(maxPhase));
      Value off_0 = xor_(multiDimIdx[outOrd[0]], phaseId);
      off_0 = mul(off_0, outVecVal);
      remained = udiv(remained, minVecVal);
      off_0 = add(off_0, mul(remained, minVecVal));
      Value offset = add(off_1, mul(off_0, dstStrides[outOrd[0]]));

      // step 3: store
      Value smemAddr = gep(elemPtrTy, smemBase, offset);
      smemAddr = bitcast(smemAddr, ptr_ty(wordTy, 3));
      store(word, smemAddr);
    }
  }
}

struct ConvertLayoutOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::ConvertLayoutOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.src();
    Value dst = op.result();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
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
    if (isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    if (srcLayout.isa<MmaEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }
    // TODO: to be implemented
    llvm_unreachable("unsupported layout conversion");
    return failure();
  }

private:
  SmallVector<Value> getMultiDimOffset(Attribute layout, Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       unsigned elemId, ArrayRef<int64_t> shape,
                                       ArrayRef<unsigned> multiDimCTAInRepId,
                                       ArrayRef<unsigned> shapePerCTA) const {
    unsigned rank = shape.size();
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
      auto multiDimOffsetFirstElem =
          emitBaseIndexForLayout(loc, rewriter, blockedLayout, shape);
      SmallVector<Value> multiDimOffset(rank);
      SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
          elemId, getSizePerThread(layout), getOrder(layout));
      for (unsigned d = 0; d < rank; ++d) {
        multiDimOffset[d] = add(multiDimOffsetFirstElem[d],
                                idx_val(multiDimCTAInRepId[d] * shapePerCTA[d] +
                                        multiDimElemId[d]));
      }
      return multiDimOffset;
    }
    if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
      unsigned dim = sliceLayout.getDim();
      auto multiDimOffsetParent =
          getMultiDimOffset(sliceLayout.getParent(), loc, rewriter, elemId,
                            sliceLayout.paddedShape(shape),
                            sliceLayout.paddedShape(multiDimCTAInRepId),
                            sliceLayout.paddedShape(shapePerCTA));
      SmallVector<Value> multiDimOffset(rank);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d == dim)
          continue;
        unsigned slicedD = d < dim ? d : (d - 1);
        multiDimOffset[slicedD] = multiDimOffsetParent[d];
      }
      return multiDimOffset;
    }
    if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
      SmallVector<Value> mmaColIdx(4);
      SmallVector<Value> mmaRowIdx(2);
      Value threadId = getThreadId(rewriter, loc);
      Value warpSize = idx_val(32);
      Value laneId = urem(threadId, warpSize);
      Value warpId = udiv(threadId, warpSize);
      // TODO: fix the bug in MMAEncodingAttr document
      SmallVector<Value> multiDimWarpId(2);
      multiDimWarpId[0] = urem(warpId, idx_val(mmaLayout.getWarpsPerCTA()[0]));
      multiDimWarpId[1] = udiv(warpId, idx_val(mmaLayout.getWarpsPerCTA()[0]));
      Value _1 = idx_val(1);
      Value _2 = idx_val(2);
      Value _4 = idx_val(4);
      Value _8 = idx_val(8);
      Value _16 = idx_val(16);
      if (mmaLayout.isAmpere()) {
        multiDimWarpId[0] = urem(multiDimWarpId[0], idx_val(shape[0] / 16));
        multiDimWarpId[1] = urem(multiDimWarpId[1], idx_val(shape[1] / 8));
        Value mmaGrpId = udiv(laneId, _4);
        Value mmaGrpIdP8 = add(mmaGrpId, _8);
        Value mmaThreadIdInGrp = urem(laneId, _4);
        Value mmaThreadIdInGrpM2 = mul(mmaThreadIdInGrp, _2);
        Value mmaThreadIdInGrpM2P1 = add(mmaThreadIdInGrpM2, _1);
        Value rowWarpOffset = mul(multiDimWarpId[0], _16);
        mmaRowIdx[0] = add(mmaGrpId, rowWarpOffset);
        mmaRowIdx[1] = add(mmaGrpIdP8, rowWarpOffset);
        Value colWarpOffset = mul(multiDimWarpId[1], _8);
        mmaColIdx[0] = add(mmaThreadIdInGrpM2, colWarpOffset);
        mmaColIdx[1] = add(mmaThreadIdInGrpM2P1, colWarpOffset);
      } else if (mmaLayout.isVolta()) {
        multiDimWarpId[0] = urem(multiDimWarpId[0], idx_val(shape[0] / 16));
        multiDimWarpId[1] = urem(multiDimWarpId[1], idx_val(shape[1] / 16));
        Value laneIdDiv16 = udiv(laneId, _16);
        Value laneIdRem16 = urem(laneId, _16);
        Value laneIdRem2 = urem(laneId, _2);
        Value laneIdRem16Div8 = udiv(laneIdRem16, _8);
        Value laneIdRem16Div4 = udiv(laneIdRem16, _4);
        Value laneIdRem16Div4Rem2 = urem(laneIdRem16Div4, _2);
        Value laneIdRem4Div2 = udiv(urem(laneId, _4), _2);
        Value rowWarpOffset = mul(multiDimWarpId[0], _16);
        Value colWarpOffset = mul(multiDimWarpId[1], _16);
        mmaRowIdx[0] =
            add(add(mul(laneIdDiv16, _8), mul(laneIdRem16Div4Rem2, _4)),
                laneIdRem2);
        mmaRowIdx[0] = add(mmaRowIdx[0], rowWarpOffset);
        mmaRowIdx[1] = add(mmaRowIdx[0], _2);
        mmaColIdx[0] = add(mul(laneIdRem16Div8, _4), mul(laneIdRem4Div2, _2));
        mmaColIdx[0] = add(mmaColIdx[0], colWarpOffset);
        mmaColIdx[1] = add(mmaColIdx[0], _1);
        mmaColIdx[2] = add(mmaColIdx[0], _8);
        mmaColIdx[3] = add(mmaColIdx[0], idx_val(9));
      } else {
        llvm_unreachable("Unexpected MMALayout version");
      }

      assert(rank == 2);
      SmallVector<Value> multiDimOffset(rank);
      if (mmaLayout.isAmpere()) {
        multiDimOffset[0] = elemId < 2 ? mmaRowIdx[0] : mmaRowIdx[1];
        multiDimOffset[1] = elemId % 2 == 0 ? mmaColIdx[0] : mmaColIdx[1];
        multiDimOffset[0] = add(
            multiDimOffset[0], idx_val(multiDimCTAInRepId[0] * shapePerCTA[0]));
        multiDimOffset[1] = add(
            multiDimOffset[1], idx_val(multiDimCTAInRepId[1] * shapePerCTA[1]));
      } else if (mmaLayout.isVolta()) {
        // the order of elements in a thread:
        //   c0, c1, ...  c4, c5
        //   c2, c3, ...  c6, c7
        if (elemId < 2) {
          multiDimOffset[0] = mmaRowIdx[0];
          multiDimOffset[1] = mmaColIdx[elemId % 2];
        } else if (elemId >= 2 && elemId < 4) {
          multiDimOffset[0] = mmaRowIdx[1];
          multiDimOffset[1] = mmaColIdx[elemId % 2];
        } else if (elemId >= 4 && elemId < 6) {
          multiDimOffset[0] = mmaRowIdx[0];
          multiDimOffset[1] = mmaColIdx[elemId % 2 + 2];
        } else if (elemId >= 6) {
          multiDimOffset[0] = mmaRowIdx[1];
          multiDimOffset[1] = mmaColIdx[elemId % 2 + 2];
        }
        multiDimOffset[0] = add(
            multiDimOffset[0], idx_val(multiDimCTAInRepId[0] * shapePerCTA[0]));
        multiDimOffset[1] = add(
            multiDimOffset[1], idx_val(multiDimCTAInRepId[1] * shapePerCTA[1]));
      } else {
        llvm_unreachable("Unexpected MMALayout version");
      }
      return multiDimOffset;
    }
    llvm_unreachable("unexpected layout in getMultiDimOffset");
  }

  // shared memory rd/st for blocked or mma layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase) const {
    auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
    auto layout = type.getEncoding();
    auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>();
    auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>();
    auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>();
    auto rank = type.getRank();
    auto sizePerThread = getSizePerThread(layout);
    auto accumSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<unsigned> numCTAs(rank);
    auto shapePerCTA = getShapePerCTA(layout);
    auto order = getOrder(layout);
    for (unsigned d = 0; d < rank; ++d) {
      numCTAs[d] = ceil<unsigned>(type.getShape()[d], shapePerCTA[d]);
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
          getLinearIndex<unsigned>(multiDimCTAId, numCTAs, order);
      // TODO: This is actually redundant index calculation, we should
      //       consider of caching the index calculation result in case
      //       of performance issue observed.
      for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type.getShape(),
                              multiDimCTAInRepId, shapePerCTA);
        Value offset =
            linearize(rewriter, loc, multiDimOffset, paddedRepShape, outOrd);

        auto elemPtrTy = ptr_ty(llvmElemTy, 3);
        Value ptr = gep(elemPtrTy, smemBase, offset);
        auto vecTy = vec_ty(llvmElemTy, vec);
        ptr = bitcast(ptr, ptr_ty(vecTy, 3));
        if (stNotRd) {
          Value valVec = undef(vecTy);
          for (unsigned v = 0; v < vec; ++v) {
            auto currVal = vals[elemId + linearCTAId * accumSizePerThread + v];
            if (isInt1)
              currVal = zext(llvmElemTy, currVal);
            else if (isPtr)
              currVal = ptrtoint(llvmElemTy, currVal);
            valVec = insert_element(vecTy, valVec, currVal, idx_val(v));
          }
          store(valVec, ptr);
        } else {
          Value valVec = load(ptr);
          for (unsigned v = 0; v < vec; ++v) {
            Value currVal = extract_element(llvmElemTy, valVec, idx_val(v));
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

  // blocked/mma -> blocked/mma.
  // Data padding in shared memory to avoid bank conflict.
  LogicalResult
  lowerDistributedToDistributed(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.src();
    Value dst = op.result();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    auto llvmElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    auto elemPtrTy = ptr_ty(llvmElemTy, 3);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> numReplicates(rank);
    SmallVector<unsigned> inNumCTAsEachRep(rank);
    SmallVector<unsigned> outNumCTAsEachRep(rank);
    SmallVector<unsigned> inNumCTAs(rank);
    SmallVector<unsigned> outNumCTAs(rank);
    auto srcShapePerCTA = getShapePerCTA(srcLayout);
    auto dstShapePerCTA = getShapePerCTA(dstLayout);
    for (unsigned d = 0; d < rank; ++d) {
      unsigned inPerCTA = std::min<unsigned>(shape[d], srcShapePerCTA[d]);
      unsigned outPerCTA = std::min<unsigned>(shape[d], dstShapePerCTA[d]);
      unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
      numReplicates[d] = ceil<unsigned>(shape[d], maxPerCTA);
      inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
      outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
      assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
      inNumCTAs[d] = ceil<unsigned>(shape[d], inPerCTA);
      outNumCTAs[d] = ceil<unsigned>(shape[d], outPerCTA);
    }
    // Potentially we need to store for multiple CTAs in this replication
    auto accumNumReplicates = product<unsigned>(numReplicates);
    // unsigned elems = getElemsPerThread(srcTy);
    auto vals = getElementsFromStruct(loc, adaptor.src(), rewriter);
    unsigned inVec = 0;
    unsigned outVec = 0;
    auto paddedRepShape = getScratchConfigForCvtLayout(op, inVec, outVec);

    unsigned outElems = getElemsPerThread(dstTy);
    auto outOrd = getOrder(dstLayout);
    SmallVector<Value> outVals(outElems);

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0)
        barrier();
      if (srcLayout.isa<BlockedEncodingAttr>() ||
          srcLayout.isa<SliceEncodingAttr>() ||
          srcLayout.isa<MmaEncodingAttr>()) {
        processReplica(loc, rewriter, /*stNotRd*/ true, srcTy, inNumCTAsEachRep,
                       multiDimRepId, inVec, paddedRepShape, outOrd, vals,
                       smemBase);
      } else {
        assert(0 && "ConvertLayout with input layout not implemented");
        return failure();
      }
      barrier();
      if (dstLayout.isa<BlockedEncodingAttr>() ||
          dstLayout.isa<SliceEncodingAttr>() ||
          dstLayout.isa<MmaEncodingAttr>()) {
        processReplica(loc, rewriter, /*stNotRd*/ false, dstTy,
                       outNumCTAsEachRep, multiDimRepId, outVec, paddedRepShape,
                       outOrd, outVals, smemBase);
      } else {
        assert(0 && "ConvertLayout with output layout not implemented");
        return failure();
      }
    }

    SmallVector<Type> types(outElems, llvmElemTy);
    auto *ctx = llvmElemTy.getContext();
    Type structTy = struct_ty(types);
    Value result = getStructFromElements(loc, outVals, rewriter, structTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  // blocked -> shared.
  // Swizzling in shared memory to avoid bank conflict. Normally used for
  // A/B operands of dots.
  LogicalResult
  lowerDistributedToShared(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.src();
    Value dst = op.result();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto srcShape = srcTy.getShape();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShape = dstTy.getShape();
    assert(srcShape.size() == 2 &&
           "Unexpected rank of ConvertLayout(blocked->shared)");
    auto srcLayout = srcTy.getEncoding();
    auto dstSharedLayout = dstTy.getEncoding().cast<SharedEncodingAttr>();
    auto inOrd = getOrder(srcLayout);
    auto outOrd = dstSharedLayout.getOrder();
    Value smemBase = getSharedMemoryBase(loc, rewriter, dst);
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto elemPtrTy = ptr_ty(getTypeConverter()->convertType(elemTy), 3);
    smemBase = bitcast(smemBase, elemPtrTy);

    auto dstStrides =
        getStridesFromShapeAndOrder(dstShape, outOrd, loc, rewriter);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcShape);
    storeDistributedToShared(src, adaptor.src(), dstStrides, srcIndices, dst,
                             smemBase, elemTy, loc, rewriter);
    auto smemObj =
        SharedMemoryObject(smemBase, dstShape, outOrd, loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

  // shared -> mma_operand
  LogicalResult
  lowerSharedToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Value src = op.src();
    Value dst = op.result();
    auto dstTensorTy = dst.getType().cast<RankedTensorType>();
    auto srcTensorTy = src.getType().cast<RankedTensorType>();
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
    if (auto mmaLayout =
            dotOperandLayout.getParent().dyn_cast_or_null<MmaEncodingAttr>()) {
      res = lowerSharedToDotOperandMMA(op, adaptor, rewriter, mmaLayout,
                                       dotOperandLayout, isOuter);
    } else if (auto blockedLayout =
                   dotOperandLayout.getParent()
                       .dyn_cast_or_null<BlockedEncodingAttr>()) {
      auto dotOpLayout =
          dstTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
      DotOpFMAConversionHelper helper(blockedLayout);
      auto thread = getThreadId(rewriter, loc);
      if (dotOpLayout.getOpIdx() == 0) { // $a
        res = helper.loadA(src, adaptor.src(), blockedLayout, thread, loc,
                           rewriter);
      } else { // $b
        res = helper.loadB(src, adaptor.src(), blockedLayout, thread, loc,
                           rewriter);
      }
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
    auto srcTy = op.src().getType().cast<RankedTensorType>();
    auto dstTy = op.result().getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto dstLayout = dstTy.getEncoding();
    auto srcMmaLayout = srcLayout.cast<MmaEncodingAttr>();
    auto dstDotLayout = dstLayout.cast<DotOperandEncodingAttr>();
    if (isMmaToDotShortcut(srcMmaLayout, dstDotLayout)) {
      // get source values
      auto vals = getElementsFromStruct(loc, adaptor.src(), rewriter);
      unsigned elems = getElemsPerThread(srcTy);
      Type elemTy =
          this->getTypeConverter()->convertType(srcTy.getElementType());
      // for the destination type, we need to pack values together
      // so they can be consumed by tensor core operations
      unsigned vecSize =
          std::max<unsigned>(32 / elemTy.getIntOrFloatBitWidth(), 1);
      Type vecTy = vec_ty(elemTy, vecSize);
      SmallVector<Type> types(elems / vecSize, vecTy);
      SmallVector<Value> vecVals;
      for (unsigned i = 0; i < elems; i += vecSize) {
        Value packed = rewriter.create<LLVM::UndefOp>(loc, vecTy);
        for (unsigned j = 0; j < vecSize; j++)
          packed = insert_element(vecTy, packed, vals[i + j], i32_val(j));
        vecVals.push_back(packed);
      }

      // This needs to be ordered the same way that
      // ldmatrix.x4 would order it
      // TODO: this needs to be refactor so we don't
      // implicitly depends on how emitOffsetsForMMAV2
      // is implemented
      SmallVector<Value> reorderedVals;
      for (unsigned i = 0; i < vecVals.size(); i += 4) {
        reorderedVals.push_back(vecVals[i]);
        reorderedVals.push_back(vecVals[i + 2]);
        reorderedVals.push_back(vecVals[i + 1]);
        reorderedVals.push_back(vecVals[i + 3]);
      }

      // return composeValuesToDotOperandLayoutStruct(ha, numRepM, numRepK);

      Type structTy =
          LLVM::LLVMStructType::getLiteral(this->getContext(), types);
      Value view =
          getStructFromElements(loc, reorderedVals, rewriter, structTy);
      rewriter.replaceOp(op, view);
      return success();
    }
    return failure();
  }

  // shared -> dot_operand if the result layout is mma
  Value lowerSharedToDotOperandMMA(
      triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, const MmaEncodingAttr &mmaLayout,
      const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) const {
    auto loc = op.getLoc();
    Value src = op.src();
    Value dst = op.result();
    bool isHMMA = supportMMA(dst, mmaLayout.getVersionMajor());

    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.src(), rewriter);
    Value res;

    if (!isOuter && mmaLayout.isAmpere() && isHMMA) { // tensor core v2
      MMA16816ConversionHelper mmaHelper(src.getType(), mmaLayout,
                                         getThreadId(rewriter, loc), rewriter,
                                         getTypeConverter(), op.getLoc());

      if (dotOperandLayout.getOpIdx() == 0) {
        // operand $a
        res = mmaHelper.loadA(src, smemObj);
      } else if (dotOperandLayout.getOpIdx() == 1) {
        // operand $b
        res = mmaHelper.loadB(src, smemObj);
      }
    } else if (!isOuter && mmaLayout.isVolta() && isHMMA) { // tensor core v1
      DotOpMmaV1ConversionHelper helper(mmaLayout);
      bool isMMAv1Row =
          dotOperandLayout.getIsMMAv1Row().cast<BoolAttr>().getValue();
      auto srcSharedLayout = src.getType()
                                 .cast<RankedTensorType>()
                                 .getEncoding()
                                 .cast<SharedEncodingAttr>();

      // Can only convert [1, 0] to row or [0, 1] to col for now
      if ((srcSharedLayout.getOrder()[0] == 1 && !isMMAv1Row) ||
          (srcSharedLayout.getOrder()[0] == 0 && isMMAv1Row)) {
        llvm::errs() << "Unsupported Shared -> DotOperand[MMAv1] conversion\n";
        return Value();
      }

      if (dotOperandLayout.getOpIdx() == 0) { // operand $a
        // TODO[Superjomn]: transA is not available here.
        bool transA = false;
        res = helper.loadA(src, transA, smemObj, getThreadId(rewriter, loc),
                           loc, rewriter);
      } else if (dotOperandLayout.getOpIdx() == 1) { // operand $b
        // TODO[Superjomn]: transB is not available here.
        bool transB = false;
        res = helper.loadB(src, transB, smemObj, getThreadId(rewriter, loc),
                           loc, rewriter);
      }
    } else {
      assert(false && "Unsupported mma layout found");
    }
    return res;
  }
};

void populateConvertLayoutOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, allocation, smem,
                                          indexCacheInfo, benefit);
}
