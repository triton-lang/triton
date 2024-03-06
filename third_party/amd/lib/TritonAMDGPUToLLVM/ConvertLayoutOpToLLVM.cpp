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
namespace SharedToDotOperandMFMA {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);
} // namespace SharedToDotOperandMFMA

namespace {
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

// shared -> mfma_operand
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
  if (auto mfmaLayout = dotOperandLayout.getParent()
                            .dyn_cast_or_null<AMDMfmaEncodingAttr>()) {
    res = lowerSharedToDotOperandMFMA(op, adaptor, typeConverter, rewriter,
                                      mfmaLayout, dotOperandLayout, isOuter);
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

    if (isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    if (srcLayout.isa<AMDMfmaEncodingAttr>() &&
        dstLayout.isa<DotOperandEncodingAttr>()) {
      return lowerMfmaToDotOperand(op, adaptor, rewriter);
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
          srcLayout.isa<AMDMfmaEncodingAttr>() ||
          srcLayout.isa<NvidiaMmaEncodingAttr>()) {
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
          dstLayout.isa<AMDMfmaEncodingAttr>() ||
          dstLayout.isa<NvidiaMmaEncodingAttr>()) {
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
