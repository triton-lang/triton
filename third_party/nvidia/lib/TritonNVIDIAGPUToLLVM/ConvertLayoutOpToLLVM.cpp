#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using mlir::isLayoutMmaV1;
using ::mlir::LLVM::getMultiDimOffset;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getWrappedMultiDimOffset;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Forward declarations

namespace SharedToDotOperandMMAv1 {

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
    if (dstLayout.isa<DotOperandEncodingAttr>() &&
        dstLayout.cast<DotOperandEncodingAttr>()
            .getParent()
            .isa<NvidiaMmaEncodingAttr>()) {
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
    if (!isOuter && mmaLayout.isAmpere()) { // tensor core v2
      res = SharedToDotOperandMMAv2::convertLayout(
          dotOperandLayout.getOpIdx(), rewriter, loc, src, dotOperandLayout,
          smemObj, typeConverter, getThreadId(rewriter, loc));
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
          loc, typeConverter, rewriter, dst.getType());
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
    auto dstEnc = op.getType().getEncoding().cast<DotOperandEncodingAttr>();
    auto sharedLayout =
        op.getSrc().getType().getEncoding().cast<SharedEncodingAttr>();

    int K;
    if (dstEnc.getOpIdx() == 0) // $a
      K = op.getType().getShape()[sharedLayout.getOrder()[0]];
    else // $b
      K = op.getType().getShape()[sharedLayout.getOrder()[1]];
    bool isOuter = K == 1;
    auto mmaLayout = dstEnc.getParent().cast<NvidiaMmaEncodingAttr>();
    Value res = lowerSharedToDotOperandMMA(op, adaptor, typeConverter, rewriter,
                                           mmaLayout, dstEnc, isOuter);

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertLayoutOpOptimizedConversion
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
    // forwarding on mma->mma shortcut, lower distributed->distributed otherwise
    if (srcLayout.isa<NvidiaMmaEncodingAttr>() &&
        dstLayout.isa<NvidiaMmaEncodingAttr>()) {
      if (isMmaToMmaShortcut(srcTy, dstTy)) {
        return lowerMmaToMma(op, adaptor, rewriter);
      }
    }
    return failure();
  }

private:
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
};

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
    if (isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout)) {
      if (shouldUseDistSmem(srcLayout, dstLayout))
        return lowerDistToDistWithDistSmem(op, adaptor, rewriter);
      if (isLayoutMmaV1(srcLayout) || isLayoutMmaV1(dstLayout))
        return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    if (srcLayout.isa<NvidiaMmaEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerMmaToDotOperand(op, adaptor, rewriter);
    }

    return failure();
  }

private:
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

      if (isLayoutMmaV1(srcLayout))
        processReplicaForMMAV1(loc, rewriter, /*stNotRd*/ true, srcTy,
                               multiDimRepId, inVec, paddedRepShape, outOrd,
                               vals, smemBase, shape);
      else
        processReplica(loc, rewriter, /*stNotRd*/ true, srcTy, inNumCTAsEachRep,
                       multiDimRepId, inVec, paddedRepShape, origRepShape,
                       outOrd, vals, smemBase);

      barrier();

      if (isLayoutMmaV1(dstLayout))
        processReplicaForMMAV1(loc, rewriter, /*stNotRd*/ false, dstTy,
                               multiDimRepId, outVec, paddedRepShape, outOrd,
                               outVals, smemBase, shape, /*isDestMma=*/true);
      else
        processReplica(loc, rewriter, /*stNotRd*/ false, dstTy,
                       outNumCTAsEachRep, multiDimRepId, outVec, paddedRepShape,
                       origRepShape, outOrd, outVals, smemBase);
    }

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

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
};
} // namespace

void mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMOptimizedPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpOptimizedConversion>(typeConverter, benefit);
}

void mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // For now give ConvertLayoutOpConversion higher benefit, I can split before
  // merging
  patterns.add<ConvertLayoutOpConversion>(typeConverter, benefit);
  // Same default benefit
  patterns.add<LocalLoadOpConversion>(typeConverter, benefit);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
