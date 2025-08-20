#include "Analysis/AMDGPUAllocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::ConvertLayoutOp;
using ::triton::gpu::LinearEncodingAttr;

namespace {

// Match MFMA->Linear Layout conversion
static bool matchMFMAAndLinearLayoutCase(RankedTensorType srcTy,
                                         RankedTensorType dstTy) {
  auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(srcTy.getEncoding());
  auto linearLayout = dyn_cast<LinearEncodingAttr>(dstTy.getEncoding());
  if (!mfmaLayout || !linearLayout)
    return false;

  std::optional<LinearLayout> storeLL =
      mlir::triton::gpu::chooseMfmaLikeStoreLayout(srcTy);
  return linearLayout.getLinearLayout() ==
         storeLL.value_or(LinearLayout::empty());
};

class ConvertLayoutOpMFMAToLinearConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  ConvertLayoutOpMFMAToLinearConversion(LLVMTypeConverter &typeConverter,
                                        const TargetInfoBase &targetInfo,
                                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    auto dstType = cast<RankedTensorType>(op.getType());

    if (!matchMFMAAndLinearLayoutCase(srcType, dstType))
      return failure();

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (inVals.empty() || inVals.size() % 8 != 0)
      return failure();

    auto mfmaLayout = dyn_cast<AMDMfmaEncodingAttr>(srcType.getEncoding());
    auto mDim = mfmaLayout.getMDim();
    auto nDim = mfmaLayout.getNDim();
    assert((mDim == 32 || mDim == 16) && mDim == nDim &&
           "Expected MFMA size 32 or 16");
    assert(triton::gpu::lookupThreadsPerWarp(rewriter) == 64 &&
           "Expected warp size 64 for MFMA");

    auto elemTy = srcType.getElementType();
    auto vecTy = vec_ty(elemTy, 2);

    SmallVector<Value> outVals;
    auto idx0 = b.i32_val(0);
    auto idx1 = b.i32_val(1);
    auto intrinsicName = mDim == 32 ? "llvm.amdgcn.permlane32.swap"
                                    : "llvm.amdgcn.permlane16.swap";
    // Convert MFMA layout to a MFMA-like linear layout where each thread
    // holds 8 consecutive elements
    for (size_t idx = 0; idx < inVals.size(); idx += 8) {
      SmallVector<Value, 4> inVecs;
      for (size_t vIdx = 0; vIdx < 4; vIdx++) {
        Value vec = b.undef(vecTy);
        vec = b.insert_element(vecTy, vec, inVals[idx + vIdx * 2 + 0], idx0);
        vec = b.insert_element(vecTy, vec, inVals[idx + vIdx * 2 + 1], idx1);
        inVecs.push_back(vec);
      }

      Value resVec0, resVec1, resVec2, resVec3;

      // Swap the row 2 and 3 of vec0 and the row 0 and 1 of vec2
      MLIRContext *ctx = rewriter.getContext();
      Type retType = struct_ty({i32_ty, i32_ty});
      Value falseVal = b.false_val();
      Value perm =
          LLVM::createLLVMIntrinsicCallOp(
              rewriter, loc, intrinsicName, retType,
              ValueRange{b.bitcast(inVecs[0], i32_ty),
                         b.bitcast(inVecs[2], i32_ty), falseVal, falseVal})
              ->getResult(0);
      resVec0 = b.bitcast(b.extract_val(i32_ty, perm, 0), vecTy);
      resVec2 = b.bitcast(b.extract_val(i32_ty, perm, 1), vecTy);

      // Swap the row 2 and 3 of vec1 and the row 0 and 1 of vec3
      perm = LLVM::createLLVMIntrinsicCallOp(
                 rewriter, loc, intrinsicName, retType,
                 ValueRange{b.bitcast(inVecs[1], i32_ty),
                            b.bitcast(inVecs[3], i32_ty), falseVal, falseVal})
                 ->getResult(0);
      resVec1 = b.bitcast(b.extract_val(i32_ty, perm, 0), vecTy);
      resVec3 = b.bitcast(b.extract_val(i32_ty, perm, 1), vecTy);

      for (Value res : {resVec0, resVec1, resVec2, resVec3})
        for (Value idx : {idx0, idx1})
          outVals.push_back(b.extract_element(elemTy, res, idx));
    }

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

class ConvertLayoutForcedPadding
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
public:
  ConvertLayoutForcedPadding(LLVMTypeConverter &typeConverter,
                             const TargetInfoBase &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
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

  SmallVector<Value> transferWithinBlockImpl(ArrayRef<Value> inVals,
                                             triton::gpu::ConvertLayoutOp op,
                                             const LinearLayout &srcLayout,
                                             const LinearLayout &dstLayout,
                                             RewriterBase &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    StringAttr kOffset = str_attr("offset");
    StringAttr kIteration = str_attr("iteration");

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

    auto scratchConfig = triton::AMD::getScratchConfigForCvt(
        op.getSrc().getType(), op.getType());
    auto tensorShapePerCTA =
        convertType<unsigned, int64_t>(triton::gpu::getShapePerCTA(
            op.getSrc().getType().getEncoding(), op.getType().getShape()));
    // Input dims: [offset, iteration, block]
    // Output dims: dimN-1, dimN-2, ..., dim0, where N is obtained from repShape
    LinearLayout sharedLayout =
        triton::gpu::chooseShemLayoutForRegToRegConversion(
            ctx, tensorShapePerCTA, scratchConfig.repShape,
            scratchConfig.order);

    // Layout for the store from registers to shared memory.
    //
    // Note: If two threads in the same warp write to the same shmem offset, the
    // hardware resolves that without a stall or a bank conflict.  Therefore we
    // don't need to avoid duplicate writes.
    // Input dims: [reg, lane, warp]
    // Output dims: [offset, iteration]
    LinearLayout shmemStoreLayout = srcLayout.invertAndCompose(sharedLayout);

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
      Value offset = b.xor_(regBase, b.i32_val(regIdx));
      if (paddedSize > 0) {
        assert(llvm::isPowerOf2_32(paddedStride));
        assert(llvm::isPowerOf2_32(paddedSize));
        auto rshiftVal = llvm::Log2_32(paddedStride);
        auto lshiftVal = llvm::Log2_32(paddedSize);
        offset = b.add(
            b.shl(b.lshr(offset, b.i32_val(rshiftVal)), b.i32_val(lshiftVal)),
            offset);
      }
      auto vecAddr = b.gep(sharedPtrTy, elemTy, smemBase, offset,
                           LLVM::GEPNoWrapFlags::inbounds);
      return vecAddr;
    };

    auto storeBase = applyLinearLayout(loc, rewriter, shmemStoreLayout,
                                       {{kRegister, b.i32_val(0)},
                                        {kLane, laneId},
                                        {kWarp, warpId},
                                        {kBlock, b.i32_val(0)}})[0]
                         .second;
    auto loadBase = applyLinearLayout(loc, rewriter, shmemLoadLayout,
                                      {{kRegister, b.i32_val(0)},
                                       {kLane, laneId},
                                       {kWarp, warpId},
                                       {kBlock, b.i32_val(0)}})[0]
                        .second;
    // register idx -> Value
    llvm::MapVector<int, Value> outVals;
    for (int i = 0; i < iterations; i++) {
      if (i != 0)
        b.barrier();

      auto &inRegs = inRegsForIter[i];
      auto &outRegs = outRegsForIter[i];

      // When using `stmatrix`, we can store `inVec` elements even if they are
      // not contiguous
      auto inVec = scratchConfig.inVec;
      for (int j = 0; j < inVals.size() / iterations; j += inVec) {
        auto inRegSlice = inRegs[j];
        Value vecAddr = getVecAddr(shmemStoreLayout, storeBase, inRegSlice);
        SmallVector<Value> inValsVec;
        for (int k = 0; k < inVec; k++)
          inValsVec.push_back(inVals[inRegSlice + k]);
        Value valsVec = packLLVector(loc, inValsVec, rewriter);
        targetInfo.storeDShared(rewriter, loc, vecAddr, std::nullopt, valsVec,
                                /*pred=*/b.true_val());
      }

      b.barrier();

      for (int j = 0; j < outSize / iterations; j += scratchConfig.outVec) {
        auto outRegSlice = outRegs[j];
        auto vecAddr = getVecAddr(shmemLoadLayout, loadBase, outRegSlice);
        Value valsVec =
            targetInfo.loadDShared(rewriter, loc, vecAddr, std::nullopt,
                                   vec_ty(elemTy, scratchConfig.outVec),
                                   /*pred=*/b.true_val());
        for (Value v : unpackLLVector(loc, valsVec, rewriter))
          outVals[outRegSlice++] = v;
      }
    }

    SmallVector<Value> outValsVec;
    for (size_t i = 0; i < outVals.size(); i++)
      outValsVec.push_back(outVals[i]);
    return outValsVec;
  }

  /// Converts ConverLayoutOp to llvm using padded pattern.
  /// This pattern adds unused memory locations after every rows of tensor
  /// fastest changing dimension:
  /// e0 e1 e2 e3 p p \
  /// e4 e5 e6 e7 p p \
  /// ...
  /// e e e e p p
  /// Dimension order is chosen in order to use wide output reads.
  ///
  /// \param op operation to convert
  /// \param src llvm structure containing operation input
  /// \param targetInfo
  /// \param typeConverter
  /// \param rewriter
  /// \returns llvm structure containing converted output
  Value transferWithinBlockPadding(triton::gpu::ConvertLayoutOp op, Value src,
                                   RewriterBase &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto typeConverter = getTypeConverter();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    // Remove the kBlock dimension from the layout as it's the identity in the
    // cvt
    auto srcLayout = triton::gpu::toLinearLayout(srcTy);
    auto dstLayout = triton::gpu::toLinearLayout(dstTy);

    SmallVector<Value> inVals = unpackLLElements(loc, src, rewriter);
    assert(!inVals.empty());

    // We munge the input values by converting i<n> (n<8) elements to i8 and
    // pointers to i64. This is necessary because TargetInfo::loadDShared and
    // storeDShared can't handle vectors of pointers or sub-byte elements.
    auto elemTy = srcTy.getElementType();
    auto isSubByteInt =
        elemTy.isInteger() && elemTy.getIntOrFloatBitWidth() < 8;
    auto isPtr = isa<triton::PointerType>(elemTy);
    auto llvmElemTyOrig = typeConverter->convertType(elemTy);
    if (isSubByteInt)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);
    auto llvmElemTy = typeConverter->convertType(elemTy);

    // Munge input values
    for (const auto &it : llvm::enumerate(inVals)) {
      if (isSubByteInt) {
        inVals[it.index()] = b.zext(llvmElemTy, it.value());
      } else if (isPtr) {
        inVals[it.index()] = b.ptrtoint(llvmElemTy, it.value());
      }
    }

    // Pretty sure this is the identity function ATM
    // It'd be better to simply call `quotient({kBlock})` and
    // remove kBlock from transferWithinBlockImpl
    auto srcLayoutWithinBlock = triton::gpu::getLayoutWithinBlock(srcLayout);
    auto dstLayoutWithinBlock = triton::gpu::getLayoutWithinBlock(dstLayout);
    SmallVector<Value> outVals = transferWithinBlockImpl(
        inVals, op, srcLayoutWithinBlock, dstLayoutWithinBlock, rewriter);

    // Unmunge output values
    for (const auto &it : llvm::enumerate(outVals)) {
      if (isSubByteInt) {
        outVals[it.index()] = b.trunc(llvmElemTyOrig, it.value());
      } else if (isPtr) {
        outVals[it.index()] = b.inttoptr(llvmElemTyOrig, it.value());
      }
    }

    Value result =
        packLLElements(loc, typeConverter, outVals, rewriter, op.getType());
    return result;
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->hasAttr(mlir::triton::AMD::AttrSharedMemPadded))
      return failure();
    auto srcType = op.getSrc().getType();
    auto dstType = op.getType();
    if (!cvtNeedsSharedMemory(srcType, dstType))
      return failure();

    auto result = transferWithinBlockPadding(op, adaptor.getSrc(), rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};
} // namespace

void mlir::triton::AMD::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpMFMAToLinearConversion>(typeConverter, targetInfo,
                                                      benefit);
  patterns.add<ConvertLayoutForcedPadding>(typeConverter, targetInfo, benefit);
  // No need to convert when ForcedSwizzling as it's already the default
  // lowering
}
