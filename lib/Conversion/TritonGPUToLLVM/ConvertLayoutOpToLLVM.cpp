#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace {

using ::mlir::LLVM::getMultiDimOffset;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::getWrappedMultiDimOffset;
using ::mlir::LLVM::linearize;

using namespace mlir::triton::gpu;

// XXX(Keren): A temporary knob to control the use of legacy MMA conversion
// because LinearLayout seems to have some performance issues.
constexpr bool useLegacyMMAConversion = false;

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
public:
  ConvertLayoutOpConversion(LLVMTypeConverter &typeConverter,
                            const TargetInfoBase &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isSupported(srcLayout, dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter, targetInfo);
    }
    return failure();
  }

private:
  bool isSupported(Attribute srcLayout, Attribute dstLayout) const {
    return isa<BlockedEncodingAttr, MmaEncodingTrait, SliceEncodingAttr>(
               srcLayout) &&
           isa<BlockedEncodingAttr, MmaEncodingTrait, SliceEncodingAttr>(
               dstLayout);
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
    bool isPtr = isa<triton::PointerType>(elemTy);
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
            getMultiDimOffset(layout, loc, rewriter, targetInfo, elemId, type,
                              multiDimCTAInRepId, shapePerCTATile);
        SmallVector<Value> multiDimOffsetWrapped = getWrappedMultiDimOffset(
            rewriter, loc, multiDimOffset, origRepShape, shapePerCTATile,
            shapePerCTA);
        Value offset = linearize(rewriter, loc, multiDimOffsetWrapped,
                                 paddedRepShape, outOrd);
        auto elemPtrTy = smemBase.getType();
        Value ptr = gep(elemPtrTy, llvmElemTy, smemBase, offset);
        auto vecTy = vec_ty(llvmElemTy, vec);
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
  lowerDistributedToDistributed(ConvertLayoutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter,
                                const TargetInfoBase &targetInfo) const {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    if (product(srcTy.getShape()) == 1) {
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      SmallVector<Value> outVals(getTotalElemsPerThread(dstTy), inVals[0]);
      Value result =
          packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
      rewriter.replaceOp(op, result);
      return success();
    }

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> numReplicates(rank);
    SmallVector<unsigned> inNumCTAsEachRep(rank);
    SmallVector<unsigned> outNumCTAsEachRep(rank);
    SmallVector<unsigned> inNumCTAs(rank);
    SmallVector<unsigned> outNumCTAs(rank);
    auto srcShapePerCTATile = getShapePerCTATile(srcLayout);
    auto dstShapePerCTATile = getShapePerCTATile(dstLayout);
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
    auto scratchConfig = getScratchConfigForCvt(srcTy, dstTy);
    unsigned inVec = scratchConfig.inVec;
    unsigned outVec = scratchConfig.outVec;
    const auto &paddedRepShape = scratchConfig.paddedRepShape;
    const auto &origRepShape = scratchConfig.repShape;

    unsigned outElems = getTotalElemsPerThread(dstTy);
    auto outOrd = getOrder(dstLayout);
    SmallVector<Value> outVals(outElems);

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0) {
        barrier();
      }
      processReplica(loc, rewriter, /*stNotRd*/ true, srcTy, inNumCTAsEachRep,
                     multiDimRepId, inVec, paddedRepShape, origRepShape, outOrd,
                     vals, smemBase);
      barrier();
      processReplica(loc, rewriter, /*stNotRd*/ false, dstTy, outNumCTAsEachRep,
                     multiDimRepId, outVec, paddedRepShape, origRepShape,
                     outOrd, outVals, smemBase);
    }

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct ConvertLayoutOpBlockedToDotOpShortcutConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  const TargetInfoBase &targetInfo;
  explicit ConvertLayoutOpBlockedToDotOpShortcutConversion(
      LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    const auto &shape = op.getType().getShape();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto dstDotEncoding = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
    if (!dstDotEncoding)
      return failure();
    if (!isa<BlockedEncodingAttr>(srcTy.getEncoding()) ||
        !isa<BlockedEncodingAttr>(dstDotEncoding.getParent()))
      return failure();
    if (cvtNeedsSharedMemory(srcTy, dstTy))
      return failure();
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

struct ConvertLayoutOpUsingLinearLayoutsConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  const TargetInfoBase &targetInfo;

  // Set benefit to 2 so that this pattern applies before other convert-layout
  // conversions.  TODO(jlebar): Eventually we want this to be the only pattern.
  explicit ConvertLayoutOpUsingLinearLayoutsConversion(
      LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
      PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    const auto &shape = op.getType().getShape();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    auto conversion = minimalCvtLayout(srcTy, dstTy);
    if (!conversion.has_value()) {
      return rewriter.notifyMatchFailure(
          op, "NYI. srcTy and/or dstTy don't implement LLs yet");
    }
    LinearLayout srcLayout =
        *toLinearLayout(srcTy.getShape(), srcTy.getEncoding());
    LinearLayout dstLayout =
        *toLinearLayout(dstTy.getShape(), dstTy.getEncoding());

    StringAttr kBlock = str_attr("block");
    StringAttr kWarp = str_attr("warp");
    StringAttr kLane = str_attr("lane");
    StringAttr kRegister = str_attr("register");

    assert(to_vector(conversion->getInDimNames()) ==
           to_vector(conversion->getOutDimNames()));
    auto dims = conversion->getInDimNames();
    if (llvm::is_contained(dims, kBlock)) {
      // Case 1: Transfer between values in different CTAs.
      //          This requires moving values through distributed shared memory.
      return rewriter.notifyMatchFailure(
          op, "NYI: Transfer between different CTAs");
    } else if (llvm::is_contained(dims, kWarp)) {
      // Case 2: Transfer between values in the same CTA, in which case we move
      //         values through shared memory.
      return transferWithinBlock(op, srcLayout, dstLayout, adaptor, rewriter);
    } else if (llvm::is_contained(dims, kLane)) {
      // Case 3. Transfer between values in the same warp, in which case we try
      //         to move values using warp shuffles, though if the pattern is
      //         complicated enough we may fall back to using shared memory
      transferWithinWarp(op, *conversion, adaptor, rewriter);
      return success();
      //return transferWithinBlock(op, srcLayout, dstLayout, adaptor, rewriter);
    } else if (llvm::is_contained(dims, kRegister)) {
      // Case 4. Transfer between values in the same thread, in which case we
      //         simply reorder the elements of adaptor.getSrc().
      return transferWithinThread(op, *conversion, adaptor, rewriter);
    } else {
      // Cast 5. The two layouts are equivalent. We should probably remove
      // these in RemoveLayoutConversion.
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
  }

  LogicalResult
  transferWithinThread(ConvertLayoutOp op, const LinearLayout &conversion,
                       OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    StringAttr kRegister = str_attr("register");
    assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> outVals(conversion.getInDimSize(kRegister));
    for (int i = 0; i < outVals.size(); i++) {
      auto srcIdx = conversion.apply({{kRegister, i}}).begin()->second;
      outVals[i] = inVals[srcIdx];
    }
    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult transferWithinBlock(ConvertLayoutOp op,
                                    const LinearLayout &srcLayout,
                                    const LinearLayout &dstLayout,
                                    OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    // TODO (Keren): Currently, we handle general mma/blocked/slice/dot(ampere)
    // -> mma/blocked/slice/dot(ampere) conversions. The following tasks must be
    // completed before we can remove the layoutIsOK check:
    // 1. Support for AMD's WMMA dot operand
    std::function<bool(Attribute)> layoutIsOK = [&](Attribute layout) {
      if (isa<MmaEncodingTrait>(layout)) {
        return !useLegacyMMAConversion;
      }
      if (auto dotOperand = dyn_cast<DotOperandEncodingAttr>(layout)) {
        if (isa<NvidiaMmaEncodingAttr, AMDMfmaEncodingAttr>(
                dotOperand.getParent())) {
          return !useLegacyMMAConversion;
        }
        return false;
      }
      if (isa<BlockedEncodingAttr, LinearEncodingAttr>(layout)) {
        return true;
      }
      if (auto slice = dyn_cast<SliceEncodingAttr>(layout)) {
        return layoutIsOK(slice.getParent());
      }
      return false;
    };
    if (!layoutIsOK(srcTy.getEncoding()) || !layoutIsOK(dstTy.getEncoding())) {
      return failure();
    }
    // FIXME [Dot LL] Remove this once we implement this trick in LLs
    if (matchMmaV3AndDotOperandLayout(srcTy, dstTy)) {
      return failure();
    }

    // The following check can be removed when generalized warp shuffle
    // conversions are ready:
    if (matchMFMAAndDotOperandShuffleCase(srcTy, dstTy)) {
      return failure();
    }

    assert(cvtNeedsSharedMemory(srcTy, dstTy));

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    assert(!inVals.empty());

    // We munge the input values by converting i<n> (n<8) elements to i8 and
    // pointers to i64. This is necessary because TargetInfo::loadDShared and
    // storeDShared can't handle vectors of pointers or sub-byte elements.
    auto elemTy = srcTy.getElementType();
    auto isSubByteInt =
        elemTy.isInteger() && elemTy.getIntOrFloatBitWidth() < 8;
    auto isPtr = isa<triton::PointerType>(elemTy);
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
    if (isSubByteInt)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);
    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    // Munge input values
    for (const auto &it : llvm::enumerate(inVals)) {
      if (isSubByteInt) {
        inVals[it.index()] = zext(llvmElemTy, it.value());
      } else if (isPtr) {
        inVals[it.index()] = ptrtoint(llvmElemTy, it.value());
      }
    }

    // Pretty sure this is the identity function ATM
    // It'd be better to simply call `quotient({kBlock})` and
    // remove kBlock from transferWithinBlockImpl
    auto srcLayoutWithinBlock = getLayoutWithinBlock(srcLayout);
    auto dstLayoutWithinBlock = getLayoutWithinBlock(dstLayout);
    SmallVector<Value> outVals =
        transferWithinBlockImpl(inVals, op, srcLayoutWithinBlock,
                                dstLayoutWithinBlock, adaptor, rewriter);

    // Unmunge output values
    for (const auto &it : llvm::enumerate(outVals)) {
      if (isSubByteInt) {
        outVals[it.index()] = trunc(llvmElemTyOrig, it.value());
      } else if (isPtr) {
        outVals[it.index()] = inttoptr(llvmElemTyOrig, it.value());
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  // Use warp shuffles to implement a layout conversion where data only needs to
  // be moved within warps.
  void transferWithinWarp(ConvertLayoutOp op, const LinearLayout &conversion,
                          OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const;

  SmallVector<Value>
  transferWithinBlockImpl(ArrayRef<Value> inVals, ConvertLayoutOp op,
                          const LinearLayout &srcLayout,
                          const LinearLayout &dstLayout, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    StringAttr kOffset = str_attr("offset");
    StringAttr kIteration = str_attr("iteration");

    Value threadId = getThreadId(rewriter, loc);
    Value threadsPerWarp = i32_val(srcLayout.getInDimSize(kLane));
    Value laneId = urem(threadId, threadsPerWarp);
    Value warpId = udiv(threadId, threadsPerWarp);

    auto scratchConfig =
        getScratchConfigForCvt(op.getSrc().getType(), op.getType());
    auto tensorShapePerCTA = convertType<unsigned, int64_t>(getShapePerCTA(
        op.getSrc().getType().getEncoding(), op.getType().getShape()));
    // Input dims: [offset, iteration, block]
    // Output dims: dimN-1, dimN-2, ..., dim0, where N is obtained from repShape
    LinearLayout sharedLayout = chooseShemLayoutForRegToRegConversion(
        ctx, tensorShapePerCTA, scratchConfig.repShape, scratchConfig.order);

    // Layout for the store from registers to shared memory.
    //
    // Note: If two threads in the same warp write to the same shmem offset, the
    // hardware resolves that without a stall or a bank conflict.  Therefore we
    // don't need to avoid duplicate writes.
    // Input dims: [reg, lane, warp]
    // Output dims: [offset, iteration]
    bool isStMatrix = targetInfo.canUseStMatrix(
        op.getSrc().getType(), scratchConfig.repShape,
        scratchConfig.paddedRepShape, scratchConfig.order,
        /*swizzleByteSize=*/0);
    LinearLayout shmemStoreLayout =
        isStMatrix ? chooseStMatrixLayout(
                         ctx, op.getSrc().getType(), scratchConfig.repShape,
                         scratchConfig.paddedRepShape, scratchConfig.order,
                         /*swizzleByteSize=*/0)
                   : srcLayout.invertAndCompose(sharedLayout);

    const int shmemAllocatedNumElems =
        getNumScratchElements(scratchConfig.paddedRepShape);
    assert(shmemStoreLayout.getOutDimSize(kOffset) <= shmemAllocatedNumElems);

    // Layout for the load from shmem to registers.
    LinearLayout shmemLoadLayout = dstLayout.invertAndCompose(sharedLayout);

    // Check that the `register` fully determines the `iteration`.  That is,
    // each thread does exactly the same reads and writes to shmem on each
    // iteration, just with different input/output registers.
    assert(
        shmemStoreLayout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));
    assert(
        shmemLoadLayout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));

    // iteration -> registers
    SmallVector<SmallVector<int>> inRegsForIter =
        collectRegsForIter(ctx, shmemStoreLayout);
    SmallVector<SmallVector<int>> outRegsForIter =
        collectRegsForIter(ctx, shmemLoadLayout);

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto sharedPtrTy = smemBase.getType();
    Type elemTy = inVals[0].getType();
    auto outSize = shmemLoadLayout.getInDimSize(kRegister);
    auto iterations = sharedLayout.getInDimSize(kIteration);
    assert(scratchConfig.inVec * iterations <= inVals.size());
    assert(scratchConfig.outVec * iterations <= outSize);

    // Check only one dimension has been padded.
    // This means the difference between the padded shape and the original shape
    // should only be in one dimension, specifically in
    // `scratchConfig.order[0]`.
    auto rank = scratchConfig.repShape.size();
    for (auto i = 0; i < rank; i++) {
      if (i == scratchConfig.order[0]) {
        continue;
      }
      assert(scratchConfig.repShape[i] == scratchConfig.paddedRepShape[i]);
    }
    auto paddedStride = scratchConfig.repShape[scratchConfig.order[0]];
    auto paddedSize =
        scratchConfig.paddedRepShape[scratchConfig.order[0]] - paddedStride;

    // Linear layout function is split in two parts below:
    //
    // L(r, t, w, b) = L(0, t, w, b) xor L(r, 0, 0, 0)
    //   offset      =    regBase   xor    regIdx
    //
    // It is the same hack as what we've done in the emitIndices function to get
    // around performance issues on AMD GPUs
    auto getVecAddr = [&](LinearLayout &layout, Value &regBase,
                          int regSlice) -> Value {
      auto regIdx = layout
                        .apply({{kRegister, regSlice},
                                {kLane, 0},
                                {kWarp, 0},
                                {kBlock, 0}})[0]
                        .second;
      Value offset = xor_(regBase, i32_val(regIdx));
      if (paddedSize > 0) {
        assert(llvm::isPowerOf2_32(paddedStride));
        assert(llvm::isPowerOf2_32(paddedSize));
        auto rshiftVal = llvm::Log2_32(paddedStride);
        auto lshiftVal = llvm::Log2_32(paddedSize);
        offset = add(shl(lshr(offset, i32_val(rshiftVal)), i32_val(lshiftVal)),
                     offset);
      }
      auto vecAddr = gep(sharedPtrTy, elemTy, smemBase, offset);
      vecAddr.setInbounds(true);
      return vecAddr;
    };

    auto storeBase = applyLinearLayout(loc, rewriter, shmemStoreLayout,
                                       {{kRegister, i32_val(0)},
                                        {kLane, laneId},
                                        {kWarp, warpId},
                                        {kBlock, i32_val(0)}})[0]
                         .second;
    auto loadBase = applyLinearLayout(loc, rewriter, shmemLoadLayout,
                                      {{kRegister, i32_val(0)},
                                       {kLane, laneId},
                                       {kWarp, warpId},
                                       {kBlock, i32_val(0)}})[0]
                        .second;
    // register idx -> Value
    llvm::MapVector<int, Value> outVals;
    for (int i = 0; i < iterations; i++) {
      if (i != 0)
        barrier();

      auto &inRegs = inRegsForIter[i];
      auto &outRegs = outRegsForIter[i];

      // When using `stmatrix`, we can store `inVec` elements even if they are
      // not contiguous
      auto inVec = isStMatrix ? shmemStoreLayout.getNumConsecutiveInOut()
                              : scratchConfig.inVec;
      for (int j = 0; j < inVals.size() / iterations; j += inVec) {
        auto inRegSlice = inRegs[j];
        Value vecAddr = getVecAddr(shmemStoreLayout, storeBase, inRegSlice);
        SmallVector<Value> inValsVec;
        for (int k = 0; k < inVec; k++)
          inValsVec.push_back(inVals[inRegSlice + k]);
        Value valsVec = packLLVector(loc, inValsVec, rewriter);
        if (isStMatrix) {
          targetInfo.storeMatrixShared(rewriter, loc, vecAddr, valsVec);
        } else {
          targetInfo.storeDShared(rewriter, loc, vecAddr, std::nullopt, valsVec,
                                  /*pred=*/true_val());
        }
      }

      barrier();

      for (int j = 0; j < outSize / iterations; j += scratchConfig.outVec) {
        auto outRegSlice = outRegs[j];
        auto vecAddr = getVecAddr(shmemLoadLayout, loadBase, outRegSlice);
        Value valsVec =
            targetInfo.loadDShared(rewriter, loc, vecAddr, std::nullopt,
                                   vec_ty(elemTy, scratchConfig.outVec),
                                   /*pred=*/true_val());
        for (Value v : unpackLLVector(loc, valsVec, rewriter))
          outVals[outRegSlice++] = v;
      }
    }

    SmallVector<Value> outValsVec;
    for (size_t i = 0; i < outVals.size(); i++)
      outValsVec.push_back(outVals[i]);
    return outValsVec;
  }

  // Determine which registers are read/written in which iteration of the shmem
  // transfer specified by `layout`.
  SmallVector<SmallVector<int> /*registers*/>
  collectRegsForIter(MLIRContext *ctx, const LinearLayout &layout) const {
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    StringAttr kIteration = str_attr("iteration");

    // The choice of iteration should be determined only by the register.  That
    // is, it should be correct to split the register dimension into iterations.
    assert(layout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));

    LinearLayout sublayout = layout.sublayout({kRegister}, {kIteration});
    SmallVector<SmallVector<int>> ret(sublayout.getOutDimSize(kIteration));
    for (int reg = 0; reg < sublayout.getInDimSize(kRegister); reg++) {
      auto idx = sublayout.apply({{kRegister, reg}});
      ret[idx.begin()->second].push_back(reg);
    }
    return ret;
  }
};

} // namespace

static LinearLayout combineLinearLayouts(const LinearLayout &lhs,
                                         const LinearLayout &rhs) {
  LinearLayout::BasesT result;
  assert(lhs.getNumInDims() == rhs.getNumInDims() &&
         lhs.getNumOutDims() == rhs.getNumOutDims() &&
         "linear layouts must have the same shape");
  for (auto [inDimName, lhsInDimBases, rhsInDimBases] :
       llvm::zip(llvm::make_first_range(lhs.getBases()),
                 llvm::make_second_range(lhs.getBases()),
                 llvm::make_second_range(rhs.getBases()))) {
    assert(lhsInDimBases.size() == rhsInDimBases.size() &&
           "linear layouts must have the same shape");
    auto &resultInDimBases = result[inDimName];
    for (auto [lhsBases, rhsBases] : llvm::zip(lhsInDimBases, rhsInDimBases)) {
      std::vector<int32_t> resultBases;
      for (auto [lhs, rhs] : llvm::zip(lhsBases, rhsBases)) {
        resultBases.push_back(lhs ^ rhs);
      }
      resultInDimBases.push_back(std::move(resultBases));
    }
  }

  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  for (StringAttr outDim : lhs.getOutDimNames()) {
    outDims.emplace_back(outDim, lhs.getOutDimSize(outDim));
  }
  return LinearLayout(result, outDims, /*requiresSurjective=*/false);
}

static bool linearLayoutIsZero(const LinearLayout &ll) {
  for (auto [inDim, inDimBases] : ll.getBases()) {
    for (auto basis : inDimBases) {
      if (!llvm::all_of(basis, [](int32_t b) { return b == 0; })) {
        return false;
      }
    }
  }
  return true;
}

static bool linearLayoutIs1DSubPermutation(const LinearLayout &ll) {
  assert(ll.getNumInDims() == 1 && ll.getNumOutDims() == 1);
  assert(ll.getBases().size() == 1);
  StringAttr dim = *ll.getInDimNames().begin();
  assert(ll.getInDimSize(dim) == ll.getOutDimSize(dim));
  int32_t mask = 0;
  for (ArrayRef<int32_t> bases : ll.getBases().front().second) {
    assert(bases.size() == 1);
    int32_t basis = bases.front();
    if (!basis)
      continue;
    if (!llvm::isPowerOf2_32(basis))
      return false;
    if (mask & basis) // check if this bit is already set
      return false;
    mask |= basis;
  }
  return true; // missing bits are allowed in subpermutation
}

static LinearLayout linearLayoutZeros1D(StringAttr inDim, int32_t inDimSize,
                                        StringAttr outDim, int32_t outDimSize) {
  LinearLayout::BasesT bases;
  auto &inDimBases = bases[inDim];
  inDimBases.assign(llvm::Log2_32(inDimSize), {0});
  return LinearLayout(std::move(bases), {{outDim, outDimSize}},
                      /*requiresSurjective=*/false);
}

static bool linearLayoutIs1DIdentityWithZeros(const LinearLayout &ll) {
  assert(ll.getNumInDims() == 1 && ll.getNumOutDims() == 1);
  assert(ll.getBases().size() == 1);
  StringAttr dim = *ll.getInDimNames().begin();
  assert(ll.getInDimSize(dim) == ll.getOutDimSize(dim));
  for (auto [i, bases] : llvm::enumerate(ll.getBases().front().second)) {
    assert(bases.size() == 1);
    int32_t basis = bases.front();
    if (basis && basis != (1 << i))
      return false;
  }
  return true;
}

static LinearLayout concatIns(const LinearLayout &lhs,
                              const LinearLayout &rhs) {
  LinearLayout::BasesT result;
  for (auto &bases : lhs.getBases())
    result.insert(bases);
  for (auto &bases : rhs.getBases())
    result.insert(bases);
  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  for (StringAttr outDim : lhs.getOutDimNames())
    outDims.emplace_back(outDim, lhs.getOutDimSize(outDim));
  return LinearLayout(result, outDims, /*requiresSurjective=*/false);
}

static LinearLayout concatOuts(const LinearLayout &lhs,
                               const LinearLayout &rhs) {
  LinearLayout::BasesT result;
  for (auto [lhsBases, rhsBases] : llvm::zip(lhs.getBases(), rhs.getBases())) {
    auto &resultBases = result[lhsBases.first];
    assert(lhsBases.first == rhsBases.first);
    for (auto [lhsBasis, rhsBasis] :
         llvm::zip(lhsBases.second, rhsBases.second)) {
      std::vector<int32_t> resultBasis;
      llvm::append_range(resultBasis, lhsBasis);
      llvm::append_range(resultBasis, rhsBasis);
      resultBases.push_back(std::move(resultBasis));
    }
  }
  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  for (StringAttr outDim : lhs.getOutDimNames())
    outDims.emplace_back(outDim, lhs.getOutDimSize(outDim));
  for (StringAttr outDim : rhs.getOutDimNames())
    outDims.emplace_back(outDim, rhs.getOutDimSize(outDim));
  return LinearLayout(result, outDims, /*requiresSurjective=*/false);
}

static LinearLayout stripZeroBasesAlongDim(const LinearLayout &ll,
                                           StringAttr stripDim) {
  LinearLayout::BasesT result;
  for (auto &[inDim, inDimBases] : ll.getBases()) {
    auto &newInDimBases = result[inDim];
    if (inDim != stripDim) {
      newInDimBases = inDimBases;
      continue;
    }
    for (auto &basis : inDimBases) {
      if (llvm::any_of(basis, [](int32_t val) { return val != 0; })) {
        newInDimBases.push_back(basis);
      }
    }
  }
  return LinearLayout(std::move(result), llvm::to_vector(ll.getOutDimNames()));
}

void ConvertLayoutOpUsingLinearLayoutsConversion::transferWithinWarp(
    ConvertLayoutOp op, const LinearLayout &conversion, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));

  // We have already checked that data movement is only required within a warp,
  // thus we can discard the block and warp dimensions.
  LinearLayout C = conversion.sublayout({kLane, kRegister}, {kLane, kRegister});

  // Get the source register values and prepare the outputs.
  SmallVector<Value> inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> outVals(conversion.getInDimSize(kRegister));

  // `C` is map from `(dst_lane, dst_reg) -> (src_lane, src_reg)`. From the
  // perspetive of the destination lane, it tells us which register from which
  // lane to get the value. Since the source and destination layouts are
  // permutation matrices, the overall transformation amounts to permuting data
  // around (plus broadcasting, if necessary).
  //
  // Warp shuffles allow indexing into another lane, but does not allowing
  // selecting the register. Suppose we decompose `C` into `C = P2 ∘ W ∘ P1`,
  // where `W` is a warp shuffle and `P1` and `P2` are (lane-dependent) register
  // permutations within a lane. Start from `C` and work backwards.
  //
  // Given any `C`, is it possible that for a given register, two destination
  // lanes map to different registers in the same source lane. This is
  // impossible to represent using a shuffle. This happens when, with respect to
  // the identity layout, a register base is swapped with a lane base (when the
  // destination lane changes, the source register changes but the lane does
  // not).
  //
  // The goal of `P2` is to permute registers within a thread so that this does
  // not happen. Specifically, pick `P2` such that bases in
  // `(P2^-1 ∘ C).sublayout(kLane, {kLane, kRegister})` has non-zero lane
  // components when the register components are non-zero.
  //
  // P2 can only change the register mapping within a thread. Constrain P2 as:
  //
  //   P2(x) = [ I P ] [ reg  ] = [ reg + P(lane)  ]
  //           [ 0 I ] [ lane ]   [ lane           ]
  //
  // Then `P2^-1 ∘ C` is:
  //
  //   [ I  0 ] [ C(r,r) C(r,l) ] = [ C(r,r)             C(r,l)           ]
  //   [ P' I ] [ C(l,r) C(l,l) ]   [ P'*C(r,r)+C(l,r)   P'*C(r,l)+C(l,l) ]
  //
  // We can see that P' selects rows (i.e. bases) from the upper half (register)
  // and combines them with the lower half (lane). Because the goal is to select
  // register bases `i` where C(r,l)[i] != 0, we know P'*C(r,r) = 0. Note that
  // solutions for P' do not always exist (no register permutation will
  // decompose C to make the warp shuffle possible), and this happens when there
  // aren't enough non-zero bases in C(r,l).

  // Find the indices of the missing lane bases: rows in the lower half where
  // the register component is non-zero but the lane component is zero.
  SmallVector<int> missingLaneRows;
  for (int i = 0, e = C.getInDimSizeLog2(kLane); i != e; ++i) {
    ArrayRef<int32_t> /*C(l,(r,l))[i]*/ lowerHalfRow = C.getBasis(kLane, i);
    assert(lowerHalfRow.size() == 2);
    if (/*C(l,r)*/ lowerHalfRow[0] != 0) {
      assert(/*C(l,l)[i]*/ lowerHalfRow[1] == 0);
      missingLaneRows.push_back(i);
    }
  }

  // Find rows in the upperhalf that can be selected by P' to make the lane
  // components in the lower half non-zero.
  std::vector<std::vector<int32_t>> PPrimeLaneBases(C.getInDimSizeLog2(kLane),
                                                    {0});
  for (int i = 0, e = C.getInDimSizeLog2(kRegister); i != e; ++i) {
    ArrayRef<int32_t> /*C(r,(r,l))[i]*/ upperHalfRow = C.getBasis(kRegister, i);
    assert(upperHalfRow.size() == 2);
    if (/*C(r,l)[i]*/ upperHalfRow[1] != 0) {
      int32_t laneBase = upperHalfRow[1];
      assert(/*C(r,r)[i]*/ upperHalfRow[0] == 0);
      if (!missingLaneRows.empty()) {
        // Select row i into row j from the missing rows. The order in which the
        // missing rows are selected doesn't really matter.
        PPrimeLaneBases[missingLaneRows.pop_back_val()][0] |= (1 << i);
      }
    } else {
      assert(upperHalfRow[0] != 0);
    }
  }
  if (!missingLaneRows.empty()) {
    llvm::report_fatal_error("decomposition failed: no solution for P'");
  }
  // P' outputs the destination register.
  LinearLayout PPrime({{kLane, std::move(PPrimeLaneBases)}},
                      {{kRegister, C.getInDimSize(kRegister)}},
                      /*requiresSurjective=*/false);

  // Form P2^-1 from P'.
  LinearLayout top = concatOuts(
      LinearLayout::identity1D(C.getInDimSize(kRegister), kRegister, kRegister),
      linearLayoutZeros1D(kRegister, C.getInDimSize(kRegister), kLane,
                          C.getInDimSize(kLane)));
  LinearLayout bottom = concatOuts(
      PPrime, LinearLayout::identity1D(C.getInDimSize(kLane), kLane, kLane));
  LinearLayout P2inv = concatIns(top, bottom);
  LinearLayout Cp = P2inv.compose(C);

  // Now we have C' = P2^-1 ∘ C = W ∘ P1. W is considerably easier to compute.
  // A warp shuffle is a function from `(lane, register) -> (lane)`, i.e.
  //
  //   W = [ I 0 ] [ reg  ] = [ reg              ]
  //       [ R L ] [ lane ]   [ R(reg) + L(lane) ]
  //
  // `W^-1 ∘ C'` will be
  //
  //   [ I R ] [ C'(r,r) C'(r,l) ] = [ ... C'(r,l) + R*C'(l,l) ]
  //   [ 0 L ] [ C'(l,r) C'(l,l) ] = [ ... L*C'(l,l)           ]
  //
  // Since P1 cannot change lanes, we know that
  //
  //   W^-1 ∘ C' = [ ... 0 ]
  //               [ ... 1 ]
  //
  // Thus L = C'(l,l)^-1, and A = -C'(r,l) * C'(l,l)^-1. (0 - LL) = LL in GF(2).
  // We know that C'(l,l) has a suitable pseudo-inverse.
  LinearLayout L = Cp.sublayout(kLane, kLane).invert();
  LinearLayout R = Cp.sublayout(kRegister, kLane).compose(L);

  // Now form W^-1.
  LinearLayout WinvLeft =
      concatIns(LinearLayout::identity1D(Cp.getInDimSize(kRegister), kRegister,
                                         kRegister),
                linearLayoutZeros1D(kLane, Cp.getInDimSize(kLane), kRegister,
                                    Cp.getInDimSize(kRegister)));
  LinearLayout Winv = concatOuts(WinvLeft, concatIns(R, L));

  // Check that Winv was formed correctly. P1 is just what's left over.
  LinearLayout P1 = Winv.compose(Cp);
  assert(P1.sublayoutIsZero(kRegister, kLane));
  assert(linearLayoutIs1DIdentityWithZeros(P1.sublayout(kLane, kLane)));

  // Grab the source elements and prepare the outputs of just the shuffles.
  SmallVector<Value> srcValues =
      unpackLLElements(loc, adaptor.getSrc(), rewriter);
  SmallVector<Value> shflOuts(conversion.getInDimSize(kRegister));

  Value threadId = getThreadId(rewriter, loc);
  Value threadsPerWarp = i32_val(conversion.getInDimSize(kLane));
  Value laneId = urem(threadId, threadsPerWarp);

  // To minimize the number of selects emitted on the source side, determine the
  // minimum set of registers that could be selected from each thread.
  // InstCombine *might* be able to crush this, but if the sizePerThread is
  // large, it's truly a huge number of selects that get emitted.
  P1 = P1.sublayout({kLane, kRegister}, kRegister);
  Cp = Cp.sublayout({kLane, kRegister}, kLane);
  // If reducedP1 is trivial, then we will emit
  // shflSrc = select(i == i, src[i], undef) and this will get trivially folded,
  // so don't worry about this case.
  LinearLayout reducedP1 = stripZeroBasesAlongDim(P1, kLane);

  // Emit one shuffle per destination register.
  for (int i = 0, e = shflOuts.size(); i != e; ++i) {
    // 'Cp' maps a (dst_lane, dst_reg) -> (src_lane, src_reg), and we know that
    // for a register, it does not map to different registers in the same lane.
    // At the same time, for each register, P1 returns the source value index
    // to provide as the shuffle value.
    auto out = applyLinearLayout(loc, rewriter, P1,
                                 {{kLane, laneId}, {kRegister, i32_val(i)}});
    assert(out.size() == 1);
    Value srcRegIdx = out.front().second;
    // The size of the input lane dimension is the number of selects to emit.
    // TODO(jeff): For dtypes smaller than i32, we can use byte permutes and
    // shuffle multiple values at a time.
    Value shflSrc = undef(srcValues.front().getType());
    for (unsigned j = 0, e = reducedP1.getInDimSize(kLane); j != e; ++j) {
      int32_t check =
          reducedP1.apply({{kLane, j}, {kRegister, i}}).front().second;
      shflSrc =
          select(icmp_eq(srcRegIdx, i32_val(check)), srcValues[j], shflSrc);
    }

    out = applyLinearLayout(loc, rewriter, Cp,
                            {{kLane, laneId}, {kRegister, i32_val(i)}});
    assert(out.size() == 1);
    Value shflIdx = out.front().second;
    shflOuts[i] = targetInfo.shuffleIdx(rewriter, loc, shflSrc, shflIdx);
  }

  // Finally, we just need to apply P2 to the shflOuts to permute the registers
  // into their final form. Use the same trick to reduce the number of emitted
  // selects.
  P2inv = P2inv.sublayout({kLane, kRegister}, {kRegister});
  LinearLayout reducedP2 = stripZeroBasesAlongDim(P2inv, kLane);
  SmallVector<Value> results(shflOuts.size());
  for (int i = 0, e = results.size(); i != e; ++i) {
    Value result = undef(srcValues.front().getType());

    auto out = applyLinearLayout(loc, rewriter, P2inv,
                                 {{kLane, laneId}, {kRegister, i32_val(i)}});
    Value resultIdx = out.front().second;
    for (unsigned j = 0, e = reducedP2.getInDimSize(kLane); j != e; ++j) {
      int32_t check =
          reducedP2.apply({{kLane, j}, {kRegister, i}}).front().second;
      result = select(icmp_eq(resultIdx, i32_val(check)), shflOuts[j], result);
    }
    results[i] = result;
  }

  Value result =
      packLLElements(loc, getTypeConverter(), results, rewriter, op.getType());
  rewriter.replaceOp(op, result);
}

void mlir::triton::populateConvertLayoutOpUsingLinearLayoutsToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpUsingLinearLayoutsConversion>(
      typeConverter, targetInfo, benefit);
}

void mlir::triton::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // We prefer using the linear layout conversion, so it gets a higher benefit.
  // Eventually the LL conversion will subsume all of the others and be the only
  // one left.
  mlir::triton::populateConvertLayoutOpUsingLinearLayoutsToLLVMPattern(
      typeConverter, targetInfo, patterns, benefit.getBenefit() + 1);
  patterns.add<ConvertLayoutOpBlockedToDotOpShortcutConversion>(
      typeConverter, targetInfo, benefit);
  patterns.add<ConvertLayoutOpConversion>(typeConverter, targetInfo, benefit);
}
