#include "Analysis/AMDGPUAllocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"

using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::ConvertLayoutOp;
using ::triton::gpu::LinearEncodingAttr;

namespace {

class ConvertLayoutOpPermlaneSwap
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  ConvertLayoutOpPermlaneSwap(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &amdTargInfo =
        static_cast<const mlir::triton::AMD::TargetInfo &>(targetInfo);
    if (amdTargInfo.getISAFamily() != AMD::ISAFamily::CDNA4)
      return failure();

    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = cast<RankedTensorType>(op.getType());
    if (!cvtNeedsWarpShuffle(srcTy, dstTy))
      return failure();

    MLIRContext *ctx = op.getContext();
    StringAttr kReg = str_attr("register");
    StringAttr kLane = str_attr("lane");

    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    int bitwidth = elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : 64;
    auto factors = getWarpLayoutConvertDecomposition(srcTy, dstTy, bitwidth);
    auto &[pReg, pLane, mixedTranspositions, nPack] = factors;

    if (mixedTranspositions.size() != 1)
      return failure();
    auto t = mixedTranspositions[0];
    auto [rBit, lBit] = t.transposition;

    // Following `transferWithinWarp` and `getWarpLayoutConvertDecomposition`,
    // an intra-warp layout conversion can be described as a permutation of
    // hardware index bits. The `permlane_swap` instructions can be used to
    // effect transpositions (r_i l4) and (r_i l5) more cheaply than in the
    // general pathway, where `l4` and `l5` are lane index bits and `r_i` is
    // a register index bit, or 'basis vector' in the language of LinearLayouts.
    //
    // Certain layout conversions which benefit from using `permlane_swap` are
    // produced during chained matrix multiplication kernels, namely the MFMA to
    // DotOp conversion and the epilogue StoreOp vectorization optimization.
    // This was the initial motivation for the pattern, but the implementation
    // itself is entirely general.
    //
    // At the moment, we handle lane-register bit transpositions as above and
    // 3-cycles involving both `l4` and `l5` bits such as (r_i l4 l5). In both
    // cases, we require that `i >= nPack`, where `nPack` indicates the number
    // of intra-register index bits (i.e., the degree of register packing), and
    // that there are no intra-register element permutations prescribed by the
    // general decomposition algorithm.
    if (!(rBit >= nPack && t.topPreSel == 0x3210 && t.topPostSel == 0x3210 &&
          (lBit == 4 || lBit == 5))) {
      return failure();
    }

    bool isSingleTransposition =
        mlir::triton::squareSublayoutIsIdentity(pLane, kLane);

    const auto &laneBases = pLane.getBases().lookup(kLane);
    auto next = [&](size_t b) { return llvm::Log2_32(laneBases[b][0]); };
    for (size_t b = 0; b < laneBases.size(); ++b) {
      if (b == 4 || b == 5)
        continue;
      if (next(b) != b)
        return failure();
    }
    bool isThreeCycle = (next(4) == 5 && next(5) == 4);

    if (!(isSingleTransposition || isThreeCycle))
      return failure();

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    const char *instr0 = lBit == 5 ? "llvm.amdgcn.permlane32.swap"
                                   : "llvm.amdgcn.permlane16.swap";
    Type retTy = struct_ty({i32_ty, i32_ty});
    Value f = b.false_val();
    auto permlaneSwap = [&](Value v0, Value v1, auto instr) {
      SmallVector<Value> ret;
      Value args[] = {v0, v1, f, f};
      auto out =
          LLVM::createLLVMIntrinsicCallOp(rewriter, loc, instr, retTy, args)
              ->getResult(0);
      ret.push_back(b.extract_val(i32_ty, out, 0));
      ret.push_back(b.extract_val(i32_ty, out, 1));
      return ret;
    };

    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    // Handle broadcasting in registers.
    auto srcLL = triton::gpu::toLinearLayout(srcTy);
    auto rmSrc = actionRemoveBroadcastedRegs(srcLL);
    inVals = rmSrc.apply(inVals);
    // The input values may require broadcasting so that the conversion can be
    // described as a permutation. This does not cost anything for simple cases.
    int regDim = inVals.size();
    int pRegDim = pReg.getInDimSize(kReg);
    if (pRegDim > regDim) {
      SmallVector<Value> original(inVals.begin(), inVals.end());
      inVals.clear();
      inVals.reserve(pRegDim);
      while (inVals.size() < pRegDim)
        inVals.append(original.begin(), original.end());
      regDim = pRegDim;
    }

    // Apply pReg.
    SmallVector<Value> newInVals(regDim);
    for (const auto &[i, v] : llvm::enumerate(inVals))
      newInVals[pReg.apply({{kReg, i}})[0].second] = v;
    inVals = std::move(newInVals);

    // Handle register packing.
    int elemsPerVec = 1 << nPack;
    int bitsPerVecElem = 32 / elemsPerVec;
    if (elemsPerVec > 1) {
      SmallVector<Value> packedVals;
      packedVals.reserve(regDim / elemsPerVec);
      if (bitwidth < bitsPerVecElem) {
        llvm::for_each(inVals, [&](Value &v) {
          if (elemTy != int_ty(bitwidth))
            v = b.bitcast(v, int_ty(bitwidth));
          v = b.zext(int_ty(bitsPerVecElem), v);
        });
      }
      for (int i = 0; i < regDim; i += elemsPerVec) {
        auto slice = ArrayRef<Value>(inVals).slice(i, elemsPerVec);
        Value v = packLLVector(loc, slice, rewriter);
        v = b.bitcast(v, i32_ty);
        packedVals.emplace_back(v);
      }
      inVals = std::move(packedVals);
    } else {
      // Handle non-integer and 64-bit types.
      llvm::for_each(inVals, [&](Value &v) {
        if (isa<LLVM::LLVMPointerType>(elemTy))
          v = b.ptrtoint(i64_ty, v);
        if (!isa<IntegerType>(elemTy))
          v = b.bitcast(v, int_ty(bitwidth));
        if (bitwidth < 32)
          v = b.zext(i32_ty, v);
      });
      if (bitwidth == 64) {
        SmallVector<Value> half0;
        SmallVector<Value> half1;
        Type vecTy = vec_ty(i32_ty, 2);
        llvm::for_each(inVals, [&](Value v) {
          auto vec = b.bitcast(v, vecTy);
          half0.push_back(b.extract_element(i32_ty, vec, b.i32_val(0)));
          half1.push_back(b.extract_element(i32_ty, vec, b.i32_val(1)));
        });
        inVals = llvm::to_vector(llvm::concat<Value>(half0, half1));
      }
    }

    // Apply `permlane_swap`s.
    SmallVector<Value> outVals(inVals.size());

    int rIdx = rBit - nPack;
    int tileSize = 1 << (rIdx + 1);
    int numTiles = inVals.size() / tileSize;
    for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
      int baseIdx = tileIdx * tileSize;
      for (int i = 0; i < tileSize / 2; ++i) {
        int r0 = baseIdx + i;
        int r1 = r0 + (1 << rIdx);
        auto swapped0 = permlaneSwap(inVals[r0], inVals[r1], instr0);
        outVals[r0] = swapped0[0];
        outVals[r1] = swapped0[1];
        if (isThreeCycle) {
          // E.g., we factor (r_i l5 l4) = (r_i l4)(r_i l5), read right to left.
          const char *instr1 = lBit == 5 ? "llvm.amdgcn.permlane16.swap"
                                         : "llvm.amdgcn.permlane32.swap";
          auto swapped1 = permlaneSwap(outVals[r0], outVals[r1], instr1);
          outVals[r0] = swapped1[0];
          outVals[r1] = swapped1[1];
        }
      }
    }

    // Unpack registers.
    if (elemsPerVec > 1) {
      SmallVector<Value> unpackedVals;
      unpackedVals.reserve(regDim);
      auto packedTy =
          bitwidth < bitsPerVecElem ? int_ty(bitsPerVecElem) : elemTy;
      auto vecTy = vec_ty(packedTy, elemsPerVec);
      auto unpackVal = [&](Value v) {
        v = b.bitcast(v, vecTy);
        return unpackLLVector(loc, v, rewriter);
      };
      for (auto v : outVals) {
        auto unpacked = unpackVal(v);
        unpackedVals.append(unpacked.begin(), unpacked.end());
      }
      if (bitwidth < bitsPerVecElem) {
        llvm::for_each(unpackedVals, [&](Value &v) {
          v = b.trunc(int_ty(bitwidth), v);
          if (elemTy != int_ty(bitwidth))
            v = b.bitcast(v, elemTy);
        });
      }
      outVals = std::move(unpackedVals);
    } else {
      // Rebuild 64-bit types and restore original element type.
      if (bitwidth == 64) {
        int shift = outVals.size() / 2;
        SmallVector<Value> newOutVals(shift);
        auto vecTy = vec_ty(i32_ty, 2);
        for (int i = 0; i < shift; ++i) {
          Value vec = b.undef(vecTy);
          vec = b.insert_element(vecTy, vec, outVals[i], b.i32_val(0));
          vec = b.insert_element(vecTy, vec, outVals[i + shift], b.i32_val(1));
          newOutVals[i] = b.bitcast(vec, i64_ty);
        }
        outVals = std::move(newOutVals);
      }
      llvm::for_each(outVals, [&](Value &v) {
        if (isa<LLVM::LLVMPointerType>(elemTy))
          v = b.inttoptr(elemTy, v);
        if (bitwidth < 32)
          v = b.trunc(int_ty(bitwidth), v);
        if (!isa<IntegerType>(elemTy))
          v = b.bitcast(v, elemTy);
      });
    }

    // Handle broadcasting in registers.
    // The `factors` produce output values which may contain broadcasting.
    // This needs to be removed before using `broadcastAs` to get the correct
    // broadcasting as expected by the original destination layout.
    auto dstLL = triton::gpu::toLinearLayout(dstTy);
    auto rmDst = actionRemoveBroadcastedRegs(dstLL);
    auto strippedDst = rmDst.apply(dstLL);
    outVals.resize(strippedDst.getInDimSize(kReg));

    if (!rmDst.isIdentity())
      outVals = broadcastAs(outVals, dstLL);

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
  patterns.add<ConvertLayoutOpPermlaneSwap>(typeConverter, targetInfo, benefit);
  patterns.add<ConvertLayoutForcedPadding>(typeConverter, targetInfo, benefit);
  // No need to convert when ForcedSwizzling as it's already the default
  // lowering
}
