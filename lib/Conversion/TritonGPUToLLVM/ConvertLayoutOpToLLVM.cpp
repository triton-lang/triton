#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace mlir::triton::gpu {
namespace {

using ::mlir::isLayoutMmaV1;
using ::mlir::LLVM::getMultiDimOffset;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::getWrappedMultiDimOffset;
using ::mlir::LLVM::linearize;

struct LocalLoadOpConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
public:
  LocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    // TODO: do we need to check if src is shared ?
    if (isa<SharedEncodingAttr>(srcLayout) && isaDistributedLayout(dstLayout)) {
      return lowerSharedToDistributed(op, adaptor, getTypeConverter(),
                                      rewriter);
    }
    if (isa<DotOperandEncodingAttr>(dstLayout) &&
        isa<BlockedEncodingAttr>(
            cast<DotOperandEncodingAttr>(dstLayout).getParent())) {
      return lowerSharedToDotOpFMA(op, adaptor, getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  LogicalResult
  lowerSharedToDotOpFMA(LocalLoadOp op, LocalLoadOpAdaptor adaptor,
                        const LLVMTypeConverter *typeConverter,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType dstTy = op.getType();
    Attribute dstLayout = dstTy.getEncoding();
    auto dotLayout = cast<DotOperandEncodingAttr>(dstLayout);
    auto blockedLayout = cast<BlockedEncodingAttr>(
        cast<DotOperandEncodingAttr>(dstLayout).getParent());
    auto thread = getThreadId(rewriter, loc);
    Value res = SharedToDotOperandFMA::convertLayout(
        dotLayout.getOpIdx(), op.getSrc(), adaptor.getSrc(), blockedLayout,
        thread, loc, getTypeConverter(), rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }
  LogicalResult
  lowerSharedToDistributed(LocalLoadOp op, LocalLoadOpAdaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() <= 2 &&
           "Unexpected rank of ConvertLayout(shared->blocked)");
    auto srcSharedLayout = cast<SharedEncodingAttr>(srcTy.getEncoding());
    auto dstLayout = dstTy.getEncoding();
    auto inOrd = getOrder(srcSharedLayout);

    auto smemObj = getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(srcTy.getElementType()), rewriter);
    auto elemLlvmTy = typeConverter->convertType(dstTy.getElementType());

    SmallVector<Value> outVals = loadSharedToDistributed(
        dstTy, srcTy, elemLlvmTy, smemObj, loc, rewriter, targetInfo);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

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
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    return failure();
  }

private:
  bool isSupported(Attribute srcLayout, Attribute dstLayout) const {
    return isaDistributedLayout(srcLayout) && isaDistributedLayout(dstLayout) &&
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
  lowerDistributedToDistributed(ConvertLayoutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
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

    unsigned outElems = getTotalElemsPerThread(dstTy);
    auto outOrd = getOrder(dstLayout);
    SmallVector<Value> outVals(outElems);

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0) {
        barrier();
      }
      auto successful = targetInfo.processReplicaUsingStMatrix(
          rewriter, loc, smemBase, vals, srcTy,
          getTypeConverter()->convertType(srcTy.getElementType()),
          paddedRepShape, origRepShape, outOrd, accumNumReplicates);
      if (!successful) {
        processReplica(loc, rewriter, /*stNotRd*/ true, srcTy, inNumCTAsEachRep,
                       multiDimRepId, inVec, paddedRepShape, origRepShape,
                       outOrd, vals, smemBase);
      }
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
    std::optional<LinearLayout> srcLayout =
        gpu::toLinearLayout(shape, op.getSrc().getType().getEncoding());
    std::optional<LinearLayout> dstLayout =
        gpu::toLinearLayout(shape, op.getType().getEncoding());
    if (!srcLayout.has_value() || !dstLayout.has_value()) {
      return failure();
    }
    // llvm::errs() << "XXX Handling conversion op: " << op << "\n";

    // There are four cases to handle.
    //
    //  1. Transfer between values in the same thread, in which case we simply
    //     reorder the elements of adaptor.getSrc().
    //  2. Transfer between values in the same warp, in which case we try to
    //     move values using warp shuffles, though if the pattern is complicated
    //     enough we may fall back to using shared memory (case 3).
    //  3. Transfer between values in the same CTA, in which case we move values
    //     through shared memory.
    //  4. Transfer between values in different CTAs, in which case we move
    //     values through distributed shared memory.
    //
    // We can tell which case we're in by examining `conversion`.  If e.g. the
    // block -> block mapping is {1, 2, 4, ...} then there's no movement between
    // data in different CTAs and we know we're not in case 4.
    LinearLayout conversion = srcLayout->invertAndCompose(*dstLayout);

    int numLanes = conversion.getInDimSize(str_attr("lane"));
    int numWarps = conversion.getInDimSize(str_attr("warp"));
    int numBlocks = conversion.getInDimSize(str_attr("block"));

    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");

    // TODO(jlebar): These checks are overly-restrictive.  For example, we can
    // transfer by shuffling registers (case 1) if and only if all of the bases
    // for `register` have 0s for lane, warp, and block.  But the check below is
    // stronger than this, checking also that the choice of lane/warp/block does
    // not affect the permutation of registers.  If we allow different
    // lane/warp/blocks to have different permutations, we can generalize this.
    if (std::optional<LinearLayout> c = conversion.divideRight(
            LinearLayout::identity1D(numLanes, kLane, kLane) *
            LinearLayout::identity1D(numWarps, kWarp, kWarp) *
            LinearLayout::identity1D(numBlocks, kBlock, kBlock));
        c.has_value()) {
      return transferWithinThread(*c, op, adaptor, rewriter);
    }

    if (std::optional<LinearLayout> c = conversion.divideRight(
            LinearLayout::identity1D(numWarps, kWarp, kWarp) *
            LinearLayout::identity1D(numBlocks, kBlock, kBlock));
        c.has_value()) {
      return transferWithinLane(*c, op, adaptor, rewriter);
    }

    return transferWithinBlockOrGroup(conversion, op, *srcLayout, *dstLayout,
                                      adaptor, rewriter);
  }

  LogicalResult
  transferWithinThread(const LinearLayout &conversion, ConvertLayoutOp op,
                       OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    StringAttr kRegister = str_attr("register");

    assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));
    assert(ArrayRef(to_vector(conversion.getInDimNames())) ==
           ArrayRef{kRegister});
    assert(ArrayRef(to_vector(conversion.getOutDimNames())) ==
           ArrayRef{kRegister});

    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> outVals(conversion.getOutDimSize(kRegister));
    for (int i = 0; i < conversion.getInDimSize(kRegister); i++) {
      auto dstIdx = conversion.apply({{kRegister, i}});
      outVals[dstIdx.begin()->second] = inVals[i];
    }
    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult transferWithinLane(const LinearLayout &conversion,
                                   ConvertLayoutOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    // TODO(jlebar): Implement me.
    return failure();
  }

  LogicalResult
  transferWithinBlockOrGroup(const LinearLayout &conversion, ConvertLayoutOp op,
                             const LinearLayout &srcLayout,
                             const LinearLayout &dstLayout, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();

    assert(cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));

    // TODO(jlebar): For now we handle only blocked/slice -> blocked/slice
    // conversions.  Once we have ldmatrix support in
    // load/storeDistributedToShared, we can remove this constraint.
    std::function<bool(Attribute)> layoutIsOK = [&](Attribute layout) {
      if (isa<BlockedEncodingAttr>(layout)) {
        return true;
      }
      if (auto slice = dyn_cast<SliceEncodingAttr>(layout)) {
        return layoutIsOK(slice.getParent());
      }
      return false;
    };
    if (!layoutIsOK(op.getSrc().getType().getEncoding()) ||
        !layoutIsOK(op.getType().getEncoding())) {
      return failure();
    }

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    assert(!inVals.empty());
    Type origTy = inVals[0].getType();

    // Munge inVals into the right type.
    assert(!inVals.empty());
    for (auto &v : inVals) {
      if (isa<LLVM::LLVMPointerType>(v.getType())) {
        v = ptrtoint(int_ty(64), v);
      }
      // Currently can't handle sub-byte stores to shmem.
      int bitwidth = v.getType().getIntOrFloatBitWidth();
      if (bitwidth < 8) {
        v = zext(int_ty(8), bitcast(v, int_ty(bitwidth)));
      }
    }

    SmallVector<Value> outVals = transferWithinBlockOrGroupImpl(
        inVals, conversion, op, srcLayout, dstLayout, adaptor, rewriter);

    // Un-munge dstVals.
    for (auto &v : outVals) {
      if (isa<LLVM::LLVMPointerType>(origTy)) {
        v = inttoptr(origTy, v);
      }
      int bitwidth = origTy.getIntOrFloatBitWidth();
      if (bitwidth < 8) {
        v = bitcast(trunc(int_ty(bitwidth), v), origTy);
      }
    }

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  SmallVector<Value> transferWithinBlockOrGroupImpl(
      ArrayRef<Value> inVals, const LinearLayout &conversion,
      ConvertLayoutOp op, const LinearLayout &srcLayout,
      const LinearLayout &dstLayout, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();

    auto sharedPtrTy = ptr_ty(ctx, /*addressSpace=*/3);

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    StringAttr kOffset = str_attr("offset");
    StringAttr kDim = str_attr("dim");

    Value threadId = getThreadId(rewriter, loc);
    Value threadsPerWarp = i32_val(srcLayout.getInDimSize(kLane));
    Value laneId = urem(threadId, threadsPerWarp);
    Value warpId = udiv(threadId, threadsPerWarp);

    // llvm::errs() << "XXX srcLayout: " << srcLayout << "\n";
    // llvm::errs() << "XXX dstLayout: " << dstLayout << "\n";

    LinearLayout sharedLayout =
        chooseShemLayoutForRegToRegConversion(srcLayout, dstLayout);
    // llvm::errs() << "XXX sharedLayout: " << sharedLayout << "\n";
    const int shmemAllocatedNumElems = product(getRepShapeForCvtLayout(op));

    // Layout for the store from registers to shared memory.
    //
    // Note: If two threads in the same warp write to the same shmem offset, the
    // hardware resolves that without a stall or a bank conflict.  Therefore we
    // don't need to avoid duplicate writes.
    LinearLayout shmemStoreLayout = srcLayout.invertAndCompose(sharedLayout);
    // llvm::errs() << "XXX shmemStoreLayout: " << shmemStoreLayout << "\n";
    int storeVec = shmemStoreLayout.getNumConsecutiveInOut();
    assert(shmemStoreLayout.getOutDimSize(kOffset) <= shmemAllocatedNumElems);

    // Layout for the load from shmem to registers.
    //
    // TODO(jlebar): A rearrangement of the registers (which is free!) might
    // lead to better vectorization.  Same for stores.
    LinearLayout shmemLoadLayout = dstLayout.invertAndCompose(sharedLayout);
    int loadVec = shmemLoadLayout.getNumConsecutiveInOut();

    Type elemTy = inVals[0].getType();
    int bitwidth = elemTy.getIntOrFloatBitWidth();

    // Suppose we have 8 registers per thread in the input and output layouts.
    // The naive approach would be to write all 8 registers to shared memory,
    // and then read them all in the new layout.  This works, but it requires
    // enough shared memory for all 8 values across all warps in the block.
    //
    // At the other extreme, we could write one register, then read it in, and
    // repeat that 8 times.  This uses 8x less shared memory, but potentially it
    // robs us of the opportunity to do vectoried loads/stores to shmem.
    //
    // We compromise by writing one vector's worth of registers per stage.
    int maxVecElems = std::min(
        std::max(loadVec, storeVec),
        std::max(1, targetInfo.getMaxSharedAccessBitwidth() / bitwidth));
    if (shmemStoreLayout.getInDimSize(kRegister) > maxVecElems &&
        shmemLoadLayout.getInDimSize(kRegister) > maxVecElems) {
      // By construction, the first N input registers are adjacent in shared
      // memory.
      // XXX XXX
    }

    bool isCrossCTA = isCrossCTAConversion(conversion);

    // Avoid emitting a call to getClusterCTAId if we know it's always 0.
    std::optional<Value> ctaId;
    if (conversion.getInDimSize(kBlock) > 1) {
      ctaId = targetInfo.getClusterCTAId(rewriter, loc);
    }

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, op.getOperation());

    if (isCrossCTA) {
      rewriter.create<nvidia_gpu::ClusterArriveOp>(loc, /*relaxed=*/false);
      rewriter.create<nvidia_gpu::ClusterWaitOp>(loc);
    } else {
      barrier();
    }

    // XXX: CSE applyLinearLayout.
    for (int i = 0; i < inVals.size(); i += storeVec) {
      auto offsetAndBlock = llvm::to_vector(llvm::make_second_range(
          applyLinearLayout(loc, rewriter, shmemStoreLayout,
                            {{kRegister, i32_val(i)},
                             {kLane, laneId},
                             {kWarp, warpId},
                             {kBlock, ctaId.value_or(i32_val(0))}})));
      assert(offsetAndBlock.size() == 2);
      Value offset = offsetAndBlock[0];
      std::optional<Value> block =
          isCrossCTA ? std::make_optional(offsetAndBlock[1]) : std::nullopt;

      auto vecAddr = gep(sharedPtrTy, elemTy, smemBase, offset);
      vecAddr.setInbounds(true);

      Value valsVec =
          packLLVector(loc, ArrayRef(inVals).slice(i, storeVec), rewriter);
      targetInfo.storeDShared(rewriter, loc, vecAddr, block, valsVec,
                              /*pred=*/true_val());
    }

    // XXX: Move into TargetInfo.
    if (isCrossCTA) {
      rewriter.create<nvidia_gpu::ClusterArriveOp>(loc, /*relaxed=*/false);
      rewriter.create<nvidia_gpu::ClusterWaitOp>(loc);
    } else {
      barrier();
    }

    SmallVector<Value> outVals;
    for (int i = 0; i < dstLayout.getInDimSize(kRegister); i += loadVec) {
      auto offsetAndBlock = llvm::to_vector(llvm::make_second_range(
          applyLinearLayout(loc, rewriter, shmemLoadLayout,
                            {{kRegister, i32_val(i)},
                             {kLane, laneId},
                             {kWarp, warpId},
                             {kBlock, ctaId.value_or(i32_val(0))}})));
      assert(offsetAndBlock.size() == 2);
      Value offset = offsetAndBlock[0];
      std::optional<Value> block =
          isCrossCTA ? std::make_optional(offsetAndBlock[1]) : std::nullopt;

      auto vecAddr = gep(sharedPtrTy, elemTy, smemBase, offset);
      vecAddr.setInbounds(true);
      Value valsVec = targetInfo.loadDShared(rewriter, loc, vecAddr, block,
                                             vec_ty(elemTy, loadVec),
                                             /*pred=*/true_val());

      for (Value v : unpackLLVector(loc, valsVec, rewriter)) {
        outVals.push_back(v);
      }
    }

    // XXX: Move to TargetInfo
    if (isCrossCTA) {
      rewriter.create<nvidia_gpu::ClusterArriveOp>(loc, /*relaxed=*/false);
      rewriter.create<nvidia_gpu::ClusterWaitOp>(loc);
    } else {
      barrier();
    }

    return outVals;
  }
};

} // namespace
} // namespace mlir::triton::gpu

void mlir::triton::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // We prefer using the linear layout conversion, so it gets a higher benefit.
  // Eventually the LL conversion will subsume all of the others and be the only
  // one left.
  patterns.add<gpu::ConvertLayoutOpUsingLinearLayoutsConversion>(
      typeConverter, targetInfo, benefit.getBenefit() + 1);
  patterns.add<gpu::ConvertLayoutOpConversion>(typeConverter, targetInfo,
                                               benefit);
  patterns.add<gpu::LocalLoadOpConversion>(typeConverter, targetInfo, benefit);
}
