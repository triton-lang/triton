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

using ::mlir::isLayoutMmaV1;
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
               dstLayout) &&
           !isLayoutMmaV1(srcLayout) && !isLayoutMmaV1(dstLayout);
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

  // For some reasons, LLVM's NVPTX backend inserts unnecessary (?) integer
  // instructions to pack & unpack sub-word integers.  A workaround is to
  // store the results of tensors with dot operand encodings in i32 to
  // facilitate instructions such as `ldmatrix`.
  //
  // TODO: Confirm if the problem is still there.
  bool needsPacking(RankedTensorType srcTy) const {
    auto srcLayout = srcTy.getEncoding();
    if (auto dotOpEnc = dyn_cast<DotOperandEncodingAttr>(srcLayout)) {
      auto mmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(dotOpEnc.getParent());
      if (mmaEnc && mmaEnc.getVersionMajor() < 3) {
        return true;
      }
    }
    return false;
  }

  bool needsUnpacking(RankedTensorType dstTy) const {
    auto dstLayout = dstTy.getEncoding();
    if (auto dotOpEnc = dyn_cast<DotOperandEncodingAttr>(dstLayout)) {
      auto mmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(dotOpEnc.getParent());
      if (mmaEnc && mmaEnc.getVersionMajor() < 3) {
        return true;
      }
    }
    return false;
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
      // TODO(Keren): implement warp shuffle instead of using the general
      // approach that uses shared memory
      return transferWithinBlock(op, srcLayout, dstLayout, adaptor, rewriter);
    } else if (llvm::is_contained(dims, kRegister) ||
               dstLayout.getInDimSize(kRegister) !=
                   srcLayout.getInDimSize(kRegister)) {
      // Case 4. Transfer between values in the same thread, in which case we
      //         simply reorder the elements of adaptor.getSrc().
      return transferWithinThread(
          op, dstLayout.getFreeVariableMasks()[kRegister],
          dstLayout.getInDimSize(kRegister), *conversion, adaptor, rewriter);
    } else {
      // Cast 5. The two layouts are equivalent. We should probably remove
      // these in RemoveLayoutConversion.
      if (needsPacking(dstTy) || needsUnpacking(srcTy)) {
        auto inVals = unpackLLElements(op.getLoc(), adaptor.getSrc(), rewriter);
        if (needsUnpacking(srcTy))
          inVals =
              unpackI32(inVals, srcTy.getElementType(), rewriter, op.getLoc());
        if (needsPacking(dstTy))
          inVals =
              packI32(inVals, dstTy.getElementType(), rewriter, op.getLoc());
        auto res = packLLElements(op.getLoc(), getTypeConverter(), inVals,
                                  rewriter, op.getType());
        rewriter.replaceOp(op, res);
      } else {
        rewriter.replaceOp(op, adaptor.getSrc());
      }
      return success();
    }
  }

  LogicalResult
  transferWithinThread(ConvertLayoutOp op, int32_t regMasks, int32_t numRegs,
                       const LinearLayout &conversion, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    StringAttr kRegister = str_attr("register");
    assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (needsUnpacking(srcTy))
      inVals = unpackI32(inVals, srcTy.getElementType(), rewriter, loc);
    SmallVector<Value> outVals(numRegs);
    for (int i = 0; i < numRegs; i++) {
      // Remove free masks from the register index
      // For example, if idx = 0b00111, and masks = 0b00100, then we get
      // 0b00011. It means that register 7 (0b111) has the same value as
      // register 3 (0b011).
      auto idx = i & (~regMasks);
      auto srcIdx = conversion.hasInDim(kRegister)
                        ? conversion.apply({{kRegister, idx}}).begin()->second
                        : idx;
      outVals[i] = inVals[srcIdx];
    }
    if (needsPacking(dstTy))
      outVals = packI32(outVals, dstTy.getElementType(), rewriter, loc);
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

    // TODO (Keren): Currently, we handle general mma/blocked/slice ->
    // mma/blocked/slice conversions.
    // The following tasks must be completed before we can remove the layoutIsOK
    // check:
    // 1. Support for AMD's MFMA and WMMA
    std::function<bool(Attribute)> layoutIsOK = [&](Attribute layout) {
      if (auto nvidiaMma = dyn_cast<NvidiaMmaEncodingAttr>(layout)) {
        if (useLegacyMMAConversion) {
          return false;
        }
        return true;
      }
      if (auto dotOperand = dyn_cast<DotOperandEncodingAttr>(layout)) {
        if (auto nvidiaMma =
                dyn_cast<NvidiaMmaEncodingAttr>(dotOperand.getParent())) {
          if (useLegacyMMAConversion) {
            return false;
          }
          // FIXME [Dot LL]
          // Enabling LL path for buggy kWidth path
          bool largeKWidth =
              dotOperand.getKWidth() * dstTy.getElementTypeBitWidth() > 64;
          return largeKWidth && nvidiaMma.isAmpere();
        }
        return false;
      }
      if (isa<BlockedEncodingAttr>(layout)) {
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
    if (needsUnpacking(srcTy))
      inVals = unpackI32(inVals, dstTy.getElementType(), rewriter, loc);

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

    if (needsPacking(dstTy))
      outVals = packI32(outVals, dstTy.getElementType(), rewriter, loc);
    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

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
    std::optional<LinearLayout> shmemStoreLayout =
        chooseStMatrixLayout(ctx, op.getSrc().getType(), scratchConfig.repShape,
                             scratchConfig.paddedRepShape, scratchConfig.order,
                             /*swizzleByteSize=*/0);
    bool isStMatrix = shmemStoreLayout.has_value();
    if (!isStMatrix) {
      shmemStoreLayout = srcLayout.invertAndCompose(sharedLayout);
    }
    assert(shmemStoreLayout.has_value());

    const int shmemAllocatedNumElems =
        getNumScratchElements(scratchConfig.paddedRepShape);
    assert(shmemStoreLayout->getOutDimSize(kOffset) <= shmemAllocatedNumElems);

    // Layout for the load from shmem to registers.
    LinearLayout shmemLoadLayout = dstLayout.invertAndCompose(sharedLayout);

    // Check that the `register` fully determines the `iteration`.  That is,
    // each thread does exactly the same reads and writes to shmem on each
    // iteration, just with different input/output registers.
    assert(shmemStoreLayout->sublayoutIsZero({kLane, kWarp, kBlock},
                                             {kIteration}));
    assert(
        shmemLoadLayout.sublayoutIsZero({kLane, kWarp, kBlock}, {kIteration}));

    // iteration -> registers
    SmallVector<SmallVector<int>> inRegsForIter =
        collectRegsForIter(ctx, *shmemStoreLayout);
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

    auto storeBase = applyLinearLayout(loc, rewriter, *shmemStoreLayout,
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
      auto inVec = isStMatrix ? shmemStoreLayout->getNumConsecutiveInOut()
                              : scratchConfig.inVec;
      for (int j = 0; j < inVals.size() / iterations; j += inVec) {
        auto inRegSlice = inRegs[j];
        Value vecAddr = getVecAddr(*shmemStoreLayout, storeBase, inRegSlice);
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
