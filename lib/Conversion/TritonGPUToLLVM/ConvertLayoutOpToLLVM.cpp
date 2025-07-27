#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
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
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SmallSet.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace {

using namespace mlir;
using namespace mlir::triton::gpu;

constexpr int kPtrBitWidth = 64;
struct ConvertLayoutOpUsingLinearLayoutsConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  const TargetInfoBase &targetInfo;

  // Set benefit to 2 so that this pattern applies before other convert-layout
  // conversions.  TODO(jlebar): Eventually we want this to be the only pattern.
  explicit ConvertLayoutOpUsingLinearLayoutsConversion(
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

    LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
    LinearLayout srcLayout = toLinearLayout(srcTy);
    LinearLayout dstLayout = toLinearLayout(dstTy);

    StringAttr kBlock = str_attr("block");
    StringAttr kWarp = str_attr("warp");
    StringAttr kLane = str_attr("lane");
    StringAttr kRegister = str_attr("register");

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    auto dims = conversion.getInDimNames();
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
      //         expensive enough we fall back to using shared memory
      if (cvtNeedsWarpShuffle(srcTy, dstTy))
        return transferWithinWarp(op, adaptor, rewriter);

      // TODO: Since data is only transferred within a warp over shared memory,
      // we should use `bar.warp.sync` instead of `barrier`, which will improve
      // latency when warps issue barriers on different cycles.
      return transferWithinBlock(op, srcLayout, dstLayout, adaptor, rewriter);
    } else if (llvm::is_contained(dims, kRegister)) {
      // Case 4. Transfer between values in the same thread, in which case we
      //         simply reorder the elements of adaptor.getSrc().
      return transferWithinThread(op, conversion, adaptor, rewriter);
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
    assert(cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));

    // Try to use swizzling to implement the conversion
    if (succeeded(transferWithinBlockSwizzling(op, adaptor.getSrc(), targetInfo,
                                               getTypeConverter(), rewriter))) {
      return success();
    }

    Value result = transferWithinBlockPadding(op, adaptor.getSrc(), targetInfo,
                                              getTypeConverter(), rewriter);

    rewriter.replaceOp(op, result);
    return success();
  }

  // Use warp shuffles to implement a layout conversion where data only needs to
  // be moved within warps.
  LogicalResult transferWithinWarp(ConvertLayoutOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    StringAttr kReg = str_attr("register");
    StringAttr kLane = str_attr("lane");

    auto factors = getWarpLayoutConvertDecomposition(srcTy, dstTy);
    auto &[pReg, pLane, mixedTranspositions] = factors;
    int m = mixedTranspositions.size();
    bool pLaneIsTrivial = squareSublayoutIsIdentity(pLane, kLane);
    assert((m > 0 || !pLaneIsTrivial) && "Shuffles not needed for conversion");

    // The desired layout conversion can be expressed as a permutation P of
    // hardware index bits for the `kLane` and `kReg` dimensions. The `factors`
    // of P describe a decomposition
    //
    //                 P = P_mixed \circ P_lane \circ P_reg,
    //
    // where P_reg and P_lane are permutations involving only register or only
    // lane index bits and P_mixed is a product of disjoint transpositions of
    // register index bits with lane index bits. Our goal is to implement P
    // using predicated selects and warp-shuffles. We have two tools for this:
    //  - An out-of-place `Ship` method which implements one mixed transposition
    //    at a time using 1.5 * R selects and .5 * R shuffles each.
    //  - An in-place `Swap` method which can simultaneously implement P_lane
    //    and multiple mixed transpositions at a time using 2 * m * R selects
    //    and either (1 - (1/2)^m) * R shuffles if `pLaneIsTrivial` or R
    //    shuffles otherwise.
    // Here, R denotes the number of 32-bit registers in use after packing (or
    // splitting, if applied to 64-bit types or pointers), and in the `Swap`
    // method, `m` denotes the number of mixed transpositions passed in.
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    // To avoid unnecessary data movement, we remove any broadcasting in the
    // register dimension from the `inVals`.
    auto srcLayout = toLinearLayout(srcTy);
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    inVals = removeBroadcastSrc.apply(inVals);

    // If the target layout has a larger register dimension than the source
    // layout, then we broadcast along the register dimension to match size. The
    // removal of broadcasting above and introduction here is expected by the
    // `factors`.
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

    // The `mixedTranspositions` and `pReg` apply to register indices before any
    // packing (i.e., register indices should be read as element indices). To
    // ensure that only elements which end up in the same destination lane are
    // packed into a common register, we swap any 'low' register bits out with
    // unused higher index register bits in the list of `mixedTranspositions`
    // and apply their effects to `inVals` before packing.
    //
    // The fraction of elements in a lane that must be moved to another lane
    // under the layout conversion is 1 - (1/2)^m. The remaining fraction can be
    // packed into 32-bit registers so long as they fit.
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    int bitwidth =
        elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : kPtrBitWidth;
    int nPackPrelim = llvm::Log2_32(std::clamp(32 / bitwidth, 1, 4));
    int nReg = pReg.getTotalInDimSizeLog2();
    int nPack = std::min(nPackPrelim, nReg - m);

    // Determine any needed register bit conjugations.
    SmallVector<std::pair<int32_t, int32_t>> regConjugations;
    llvm::SmallSet<int32_t, 6> usedRegBits;
    if (nPack > 0) {
      // Any `regBitIdx` not originally in `mixedTranspositions` and `>= nPack`
      // can be used to swap out the original 'low' bit index.
      for (auto [regBitIdx, laneBitIdx] : mixedTranspositions)
        usedRegBits.insert(regBitIdx);
      int potentialHighIdx = nPack;
      for (auto &[regBitIdx, laneBitIdx] : mixedTranspositions) {
        if (regBitIdx < nPack) {
          while (usedRegBits.contains(potentialHighIdx))
            ++potentialHighIdx;
          regConjugations.emplace_back(regBitIdx, potentialHighIdx);
          regBitIdx = potentialHighIdx++;
        }
      }
    }

    // Apply pReg and any conjugations.
    SmallVector<Value> newInVals(regDim);
    auto swapBits = [](const auto &p, int num) {
      int bit0 = (num >> p.first) & 1;
      int bit1 = (num >> p.second) & 1;
      if (bit0 != bit1)
        num ^= ((1 << p.first) | (1 << p.second));
      return num;
    };
    auto applyConj = [&](int idx) {
      for (const auto &p : regConjugations)
        idx = swapBits(p, idx);
      return idx;
    };
    for (const auto &[i, v] : llvm::enumerate(inVals))
      newInVals[applyConj(pReg.apply({{kReg, i}})[0].second)] = v;
    inVals = std::move(newInVals);

    // Pack registers if possible.
    int elemsPerVec = 1 << nPack;
    int bitsPacked = elemsPerVec * std::max(bitwidth, 8);
    auto packedIntTy = int_ty(bitsPacked);
    bool padTo32 = bitsPacked < 32;
    if (elemsPerVec > 1) {
      SmallVector<Value> packedVals;
      packedVals.reserve(regDim / elemsPerVec);
      if (bitwidth < 8)
        llvm::for_each(inVals, [&](Value &v) { v = b.zext(i8_ty, v); });
      for (int i = 0; i < regDim; i += elemsPerVec) {
        auto slice = ArrayRef<Value>(inVals).slice(i, elemsPerVec);
        Value v = packLLVector(loc, slice, rewriter);
        v = b.bitcast(v, packedIntTy);
        if (padTo32)
          v = b.zext(i32_ty, v);
        packedVals.emplace_back(v);
      }
      inVals = std::move(packedVals);
    }

    SmallVector<Value> outVals;
    if (m == 1 && pLaneIsTrivial) {
      outVals = transferWithinWarpShipImpl(loc, rewriter, inVals, nPack,
                                           mixedTranspositions[0]);
    } else {
      outVals = transferWithinWarpSwapImpl(loc, rewriter, inVals, nPack, pLane,
                                           pLaneIsTrivial, mixedTranspositions);
    }

    // Unpack registers if needed.
    if (elemsPerVec > 1) {
      SmallVector<Value> unpackedVals;
      unpackedVals.reserve(regDim);
      auto vecTy = vec_ty(bitwidth < 8 ? i8_ty : elemTy, elemsPerVec);
      auto unpackVal = [&](Value v) {
        if (padTo32)
          v = b.trunc(packedIntTy, v);
        v = b.bitcast(v, vecTy);
        return unpackLLVector(loc, v, rewriter);
      };
      for (auto v : outVals) {
        auto unpacked = unpackVal(v);
        unpackedVals.append(unpacked.begin(), unpacked.end());
      }
      if (bitwidth < 8)
        llvm::for_each(unpackedVals, [&](Value &v) { v = b.trunc(elemTy, v); });
      outVals = std::move(unpackedVals);
    }

    // Perform the second half of any prescribed register bit conjugations.
    if (!regConjugations.empty()) {
      SmallVector<Value> newOutVals(regDim);
      for (const auto &[i, v] : llvm::enumerate(outVals))
        newOutVals[applyConj(i)] = v;
      outVals = std::move(newOutVals);
    }

    // If `dstLayout` has a smaller `kReg` dimension than `srcLayout` after
    // broadcasting is removed, then drop the extra registers from `outVals`.
    auto dstLayout = toLinearLayout(dstTy);
    auto removeBroadcastDst = actionRemoveBroadcastedRegs(dstLayout);
    auto strippedDstLayout = removeBroadcastDst.apply(dstLayout);
    outVals.resize(strippedDstLayout.getInDimSize(kReg));

    // Introduce broadcasting in registers if expected by `dstLayout`.
    if (!removeBroadcastDst.isIdentity())
      outVals = broadcastAs(outVals, dstLayout);

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  SmallVector<Value> transferWithinWarpSwapImpl(
      Location loc, ConversionPatternRewriter &rewriter, ArrayRef<Value> inVals,
      int nPack, const LinearLayout &pLane, bool pLaneIsTrivial,
      ArrayRef<std::pair<int, int>> mixedTranspositions) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    StringAttr kReg = str_attr("register");
    StringAttr kLane = str_attr("lane");

    SmallVector<Value> vals(inVals.begin(), inVals.end());
    int m = mixedTranspositions.size();
    int numRegs = inVals.size();
    // A single mixed transposition (r_i l_j) which swaps the i-th register
    // index bit and the j-th lane index bit of an element applies a tiled 2x2
    // block transpose with block size (1 << i) by (1 << j) to the data. This
    // can be realized as:
    //
    //             [ A B ] selp [ A D ] shfl [ A D ] selp [ A C ]
    //             [ C D ] ---> [ C B ] ---> [ B C ] ---> [ B D ].
    //
    // In linear-algebraic terms, this is the factorization over GF(2):
    //
    //   1. r_i ^= l_j (selp)                     selp    shfl    selp
    //   2. l_j ^= r_i (shfl)        [ 0 1 ]     [ 1 1 ] [ 1 0 ] [ 1 1 ]
    //   3. r_i ^= l_j (selp),       [ 1 0 ]  =  [ 0 1 ] [ 1 1 ] [ 0 1 ],
    //
    // where we pass in bits as column vectors [r_i, l_j].
    //
    // When the transpositions are all disjoint, we can group the three stages
    // of each transposition together. The two combined `selp` stages each use
    // `numRegs` selects per transposition, while the `shfl` stage only requires
    // code emission when at least one of the `r_i` bits is on, resulting in
    // `(1 - (1/2)^m) * numRegs` shuffles in total. If `pLane` is nontrivial,
    // then we can conjugate its effects through the first two stages and fuse
    // it with the second stage, resulting in `numRegs` shuffles instead.
    Value laneId = getLaneId(rewriter, loc);

    // Implement r_i ^= l_j using `numRegs` independent selects.
    auto applyConditionalSwap = [&](int rBitIdx, int lBitIdx) {
      SmallVector<Value> newVals(numRegs);
      Value lBitVal = b.and_(laneId, b.i32_val(1 << lBitIdx));
      Value lBitOff = b.icmp_eq(lBitVal, b.i32_val(0));

      int tileSize = 1 << (rBitIdx + 1);
      int numTiles = numRegs / tileSize;
      for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        int baseIdx = tileIdx * tileSize;
        for (int i = 0; i < tileSize / 2; ++i) {
          int idx = baseIdx + i;
          int partnerIdx = idx + (1 << rBitIdx);
          Value val = vals[idx];
          Value partnerVal = vals[partnerIdx];
          newVals[idx] = b.select(lBitOff, val, partnerVal);
          newVals[partnerIdx] = b.select(lBitOff, partnerVal, val);
        }
      }
      return newVals;
    };

    auto pLaneInv = pLane.invert();
    const auto &pLInvBases = pLaneInv.getBases().lookup(kLane);

    // Perform r_i ^= l_{pLaneInv(j)}.
    for (auto [origRBitIdx, origLBitIdx] : mixedTranspositions) {
      int rBitIdx = origRBitIdx - nPack;
      int lBitIdx = llvm::Log2_32(pLInvBases[origLBitIdx][0]);
      vals = applyConditionalSwap(rBitIdx, lBitIdx);
    }
    // Perform l_{pLaneInv(j)} ^= r_i and apply pLane.
    Value laneIdPerm;
    if (pLaneIsTrivial) {
      laneIdPerm = laneId;
    } else {
      laneIdPerm = triton::gpu::matrixVectorProd(b, pLaneInv, laneId);
    }
    for (int r = 0; r < numRegs; ++r) {
      int mask = 0;
      for (auto [origRBitIdx, lBitIdx] : mixedTranspositions) {
        int rBitIdx = origRBitIdx - nPack;
        if (r & (1 << rBitIdx)) {
          mask |= pLInvBases[lBitIdx][0];
        }
      }
      if (!pLaneIsTrivial || mask > 0) {
        Value srcIdx = b.xor_(laneIdPerm, b.i32_val(mask));
        vals[r] = targetInfo.shuffleIdx(rewriter, loc, vals[r], srcIdx);
      }
    }
    // Perform r_i ^= l_j.
    for (auto [origRBitIdx, lBitIdx] : mixedTranspositions) {
      int rBitIdx = origRBitIdx - nPack;
      vals = applyConditionalSwap(rBitIdx, lBitIdx);
    }
    return vals;
  }

  SmallVector<Value>
  transferWithinWarpShipImpl(Location loc, ConversionPatternRewriter &rewriter,
                             ArrayRef<Value> inVals, int nPack,
                             std::pair<int, int> mixedTransposition) const {
    // Implements the effects of a single mixed transposition as in
    // `transferWithinWarpSwapImpl`, but uses auxiliary registers to hold the
    // values to be shuffled, resulting in fewer emitted instructions.
    int numRegs = inVals.size();
    auto [origRBitIdx, lBitIdx] = mixedTransposition;
    int rBitIdx = origRBitIdx - nPack;
    int tileSize = 1 << (rBitIdx + 1);
    int numTiles = numRegs / tileSize;

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value laneId = getLaneId(rewriter, loc);
    Value lBitVal = b.and_(laneId, b.i32_val(1 << lBitIdx));
    Value lBitOff = b.icmp_eq(lBitVal, b.i32_val(0));
    SmallVector<Value> outVals(numRegs);

    for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
      int baseIdx = tileIdx * tileSize;
      for (int i = 0; i < tileSize / 2; ++i) {
        int idx = baseIdx + i;
        int partnerIdx = idx + (1 << rBitIdx);
        Value valToShip = b.select(lBitOff, inVals[partnerIdx], inVals[idx]);
        Value shippedVal =
            targetInfo.shuffleXor(rewriter, loc, valToShip, (1 << lBitIdx));
        outVals[idx] = b.select(lBitOff, inVals[idx], shippedVal);
        outVals[partnerIdx] = b.select(lBitOff, shippedVal, inVals[partnerIdx]);
      }
    }
    return outVals;
  }
};

} // namespace

void mlir::triton::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpUsingLinearLayoutsConversion>(
      typeConverter, targetInfo, benefit);
}
