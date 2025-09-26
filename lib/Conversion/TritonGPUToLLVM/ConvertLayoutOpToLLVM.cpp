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

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace {

using namespace mlir;
using namespace mlir::triton::gpu;
using TranspositionInfo = DecomposedWarpConversion::TranspositionInfo;

constexpr int kPtrBitWidth = 64;
struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  const TargetInfoBase &targetInfo;

  // Set benefit to 2 so that this pattern applies before other convert-layout
  // conversions.  TODO(jlebar): Eventually we want this to be the only pattern.
  explicit ConvertLayoutOpConversion(LLVMTypeConverter &typeConverter,
                                     const TargetInfoBase &targetInfo,
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
      transferWithinBlockSwizzling(op, adaptor.getSrc(), rewriter);
      return success();
    } else if (llvm::is_contained(dims, kLane)) {
      // Case 3. Transfer between values in the same warp, in which case we try
      //         to move values using warp shuffles, though if the pattern is
      //         expensive enough we fall back to using shared memory
      if (cvtNeedsWarpShuffle(srcTy, dstTy))
        return transferWithinWarp(op, adaptor, rewriter);

      transferWithinBlockSwizzling(op, adaptor.getSrc(), rewriter);
      return success();
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

  SmallVector<Value> transferWithinBlockSwizzlingImpl(
      Location loc, ConversionPatternRewriter &rewriter,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      ArrayRef<Value> inVals, Type llvmElemTy, Value smemBase) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // We handle transformations recursively as they all need a preprocessing
    // and a postprocessing step.

    // Handle pointer types as 64-bit integers
    if (isa<LLVM::LLVMPointerType>(llvmElemTy)) {
      auto llvmElemTyPtr = i64_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(inVals, [&](Value v) {
        return b.ptrtoint(llvmElemTyPtr, v).getResult();
      }));
      auto outVals =
          transferWithinBlockSwizzlingImpl(loc, rewriter, srcLayout, dstLayout,
                                           newInVals, llvmElemTyPtr, smemBase);
      for (auto &v : outVals) {
        v = b.inttoptr(llvmElemTy, v);
      }
      return outVals;
    }

    // Handle sub-byte elements like i1
    if (llvmElemTy.getIntOrFloatBitWidth() < 8) {
      // Upcast to i8
      auto i8ElemTy = i8_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(
          inVals, [&](Value v) { return b.zext(i8ElemTy, v).getResult(); }));
      auto outVals = transferWithinBlockSwizzlingImpl(
          loc, rewriter, srcLayout, dstLayout, newInVals, i8ElemTy, smemBase);
      for (auto &v : outVals) {
        v = b.trunc(llvmElemTy, v);
      }
      return outVals;
    }

    // Remove broadcasting in src
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    if (!removeBroadcastSrc.isIdentity()) {
      auto prmtSrc = removeBroadcastSrc.apply(srcLayout);
      auto newInVals = removeBroadcastSrc.apply(inVals);
      return transferWithinBlockSwizzlingImpl(loc, rewriter, prmtSrc, dstLayout,
                                              newInVals, llvmElemTy, smemBase);
    }

    // Remove broadcasting in dst
    auto removeBroadcastDst = actionRemoveBroadcastedRegs(dstLayout);
    if (!removeBroadcastDst.isIdentity()) {
      auto prmtDst = removeBroadcastDst.apply(dstLayout);
      auto outVals = transferWithinBlockSwizzlingImpl(
          loc, rewriter, srcLayout, prmtDst, inVals, llvmElemTy, smemBase);
      return broadcastAs(outVals, dstLayout);
    }

    // At this point we have a type that's at least 8-bit
    // and we don't have broadcasting in the registers
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto smem = optimalSwizzlingLdSt(srcLayout, dstLayout, bitwidth);

    // Extract reps from smem
    auto kReg = str_attr("register");
    auto kReps = str_attr("reps");
    auto nReps = smem.getInDimSize(kReps);
    auto reps = LinearLayout::identity1D(nReps, kReg, kReps);

    auto totalStoreCvt = srcLayout.invertAndCompose(smem);
    auto totalLoadCvt = dstLayout.invertAndCompose(smem);

    // The permutation exists by construction of the reps dimension in
    // optimalSwizzling
    auto permStore =
        regPermForDivide(totalStoreCvt, reps, /*left=*/false).value();
    totalStoreCvt = permStore.apply(totalStoreCvt);
    auto permutedInVals = permStore.apply(inVals);
    auto permLoad =
        regPermForDivide(totalLoadCvt, reps, /*left=*/false).value();
    totalLoadCvt = permLoad.apply(totalLoadCvt);

    // Remove the reps and flatten into offset
    auto storeCvt = *divideRight(totalStoreCvt, reps);
    auto loadCvt = *divideRight(totalLoadCvt, reps);
    auto kOffset = str_attr("offset");
    storeCvt = storeCvt.reshapeOuts({{kOffset, storeCvt.getTotalOutDimSize()}});
    loadCvt = loadCvt.reshapeOuts({{kOffset, loadCvt.getTotalOutDimSize()}});

    auto tileSize = storeCvt.getInDimSize(kReg);

    assert(permutedInVals.size() == tileSize * nReps);
    SmallVector<Value> outVals;
    auto affineOffset = b.i32_val(0);
    auto maskSpanAffineOffset = 0;
    auto noPaddingOffset = [](Value v) { return v; };

    bool isWarpSync = mlir::isCvtWarpSync(srcLayout, dstLayout);
    for (int i = 0; i < nReps; ++i) {
      if (i > 0)
        targetInfo.barrier(loc, rewriter, isWarpSync);

      auto tileInVals =
          ArrayRef<Value>(permutedInVals).slice(i * tileSize, tileSize);
      // Store
      lowerLdStShared(loc, ctx, storeCvt, tileInVals, llvmElemTy, smemBase,
                      noPaddingOffset, affineOffset, maskSpanAffineOffset,
                      rewriter, targetInfo);
      targetInfo.barrier(loc, rewriter, isWarpSync);
      // Load
      SmallVector<Value> tileOutVals = lowerLdStShared(
          loc, ctx, loadCvt, {}, llvmElemTy, smemBase, noPaddingOffset,
          affineOffset, maskSpanAffineOffset, rewriter, targetInfo);
      llvm::append_range(outVals, tileOutVals);
    }

    // Undo the permLoad used to divideRight
    outVals = permLoad.inverse().apply(outVals);
    return outVals;
  }

  void transferWithinBlockSwizzling(ConvertLayoutOp op, Value src,
                                    ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    // Remove the kBlock dimension from the layout as it's the identity in the
    // cvt
    auto srcLayout = toLinearLayout(srcTy);
    auto dstLayout = toLinearLayout(dstTy);
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    srcLayout = srcLayout.sublayout({kReg, kLane, kWarp},
                                    to_vector(srcLayout.getOutDimNames()));
    dstLayout = dstLayout.sublayout({kReg, kLane, kWarp},
                                    to_vector(dstLayout.getOutDimNames()));

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto inVals = unpackLLElements(loc, src, rewriter);
    auto outVals = transferWithinBlockSwizzlingImpl(
        loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, smemBase);

    Value result =
        packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);
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
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    int bitwidth =
        elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : kPtrBitWidth;

    auto factors = getWarpLayoutConvertDecomposition(srcTy, dstTy, bitwidth);
    auto &[pReg, pLane, mixedTranspositions, nPack] = factors;
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
    //    at a time using 1.5 * R selects/permutes and .5 * R shuffles each.
    //  - An in-place `Swap` method which can simultaneously implement P_lane
    //    and multiple mixed transpositions at a time using 2 * m * R selects/
    //    permutes and either (1 - (1/2)^m) * R shuffles if `pLaneIsTrivial` and
    //    R shuffles otherwise.
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

    // Apply pReg.
    SmallVector<Value> newInVals(regDim);
    for (const auto &[i, v] : llvm::enumerate(inVals))
      newInVals[pReg.apply({{kReg, i}})[0].second] = v;
    inVals = std::move(newInVals);

    // Pack registers if possible.
    int elemsPerVec = 1 << nPack;
    int bitsPerVecElem = 32 / elemsPerVec;
    if (elemsPerVec > 1) {
      SmallVector<Value> packedVals;
      packedVals.reserve(regDim / elemsPerVec);
      if (bitwidth == 8 && bitsPerVecElem == 16) {
        // TODO: Can remove `if` part of `if-else` once ptxas bugfix lands.
        for (int i = 0; i < regDim; i += elemsPerVec) {
          Value x0 = b.zext(i32_ty, b.bitcast(inVals[i], int_ty(bitwidth)));
          Value x1 = b.zext(i32_ty, b.bitcast(inVals[i + 1], int_ty(bitwidth)));
          x1 = b.shl(x1, b.i32_val(16));
          packedVals.emplace_back(b.or_(x0, x1));
        }
      } else {
        if (bitwidth < bitsPerVecElem) {
          for (Value &v : inVals) {
            if (elemTy != int_ty(bitwidth))
              v = b.bitcast(v, int_ty(bitwidth));
            v = b.zext(int_ty(bitsPerVecElem), v);
          }
        }
        for (int i = 0; i < regDim; i += elemsPerVec) {
          auto slice = ArrayRef<Value>(inVals).slice(i, elemsPerVec);
          Value v = packLLVector(loc, slice, rewriter);
          v = b.bitcast(v, i32_ty);
          packedVals.emplace_back(v);
        }
      }
      inVals = std::move(packedVals);
    }

    auto isShippable = [](const TranspositionInfo &t) {
      // The `Ship` method cannot mix elements from different registers in the
      // same lane, so we are restricted to cycles like (l0 r1), (l0 r2), and
      // (l0 r0 r1) which do not use both high and low register bits.
      return t.topPreSel == t.topPostSel ||
             (t.topPreSel == 0x5140 && t.topPostSel == 0x6240) ||
             (t.topPreSel == 0x6420 && t.topPostSel == 0x5410) ||
             (t.topPreSel == 0x3210 && t.topPostSel == 0x3120);
    };

    SmallVector<Value> outVals;
    if (m == 1 && pLaneIsTrivial && isShippable(mixedTranspositions[0])) {
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
        for (Value &v : unpackedVals) {
          v = b.trunc(int_ty(bitwidth), v);
          if (elemTy != int_ty(bitwidth))
            v = b.bitcast(v, elemTy);
        }
      }
      outVals = std::move(unpackedVals);
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
      ArrayRef<TranspositionInfo> mixedTranspositions) const {
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
    auto pLaneInv = pLane.invert();
    const auto &pLInvBases = pLaneInv.getBases().lookup(kLane);

    // Implement r_i ^= l_j using `numRegs` independent selects or permutes.
    auto applySwap = [&](TranspositionInfo t, bool preShuf) {
      int rIdx = t.transposition.first - nPack;
      int origLIdx = t.transposition.second;
      int lIdx = preShuf ? llvm::Log2_32(pLInvBases[origLIdx][0]) : origLIdx;
      uint16_t topSel = preShuf ? t.topPreSel : t.topPostSel;
      uint16_t botSel = preShuf ? t.botPreSel : t.botPostSel;

      SmallVector<Value> newVals(numRegs);
      Value lBitVal = b.and_(laneId, b.i32_val(1 << lIdx));
      Value lBitOff = b.icmp_eq(lBitVal, b.i32_val(0));

      int tileSize = 1 << (rIdx + 1);
      int numTiles = numRegs / tileSize;
      for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        int baseIdx = tileIdx * tileSize;
        for (int i = 0; i < tileSize / 2; ++i) {
          int r0 = baseIdx + i;
          int r1 = r0 + (1 << rIdx);
          Value v0 = vals[r0];
          Value v1 = vals[r1];
          if (topSel == 0x3210 && botSel == 0x7654) {
            newVals[r0] = b.select(lBitOff, v0, v1);
            newVals[r1] = b.select(lBitOff, v1, v0);
          } else {
            Value sel00 = b.i32_val(topSel);
            Value sel01 = b.i32_val(preShuf ? botSel : (topSel ^ 0x4444));
            Value sel10 = b.i32_val(botSel);
            Value sel11 = b.i32_val(preShuf ? topSel : (botSel ^ 0x4444));
            Value sel1 = b.select(lBitOff, sel00, sel01);
            Value sel2 = b.select(lBitOff, sel10, sel11);
            newVals[r0] = targetInfo.permute(rewriter, loc, v0, v1, sel1);
            newVals[r1] = targetInfo.permute(rewriter, loc, v0, v1, sel2);
          }
        }
      }
      return newVals;
    };

    // Stage 1 (selp/prmt)
    for (const auto &t : mixedTranspositions)
      vals = applySwap(t, /*preShuf=*/true);
    // Stage 2 (shfl)
    Value laneIdPerm;
    if (!pLaneIsTrivial)
      laneIdPerm = triton::gpu::matrixVectorProd(b, pLaneInv, laneId);
    for (int r = 0; r < numRegs; ++r) {
      int mask = 0;
      for (const auto &t : mixedTranspositions) {
        int rIdx = t.transposition.first - nPack;
        int lIdx = t.transposition.second;
        if (r & (1 << rIdx)) {
          mask |= pLInvBases[lIdx][0];
        }
      }
      if (pLaneIsTrivial) {
        if (mask != 0)
          vals[r] = targetInfo.shuffleXor(rewriter, loc, vals[r], mask);
      } else {
        Value srcIdx = b.xor_(laneIdPerm, b.i32_val(mask));
        vals[r] = targetInfo.shuffleIdx(rewriter, loc, vals[r], srcIdx);
      }
    }
    // Stage 3 (selp/prmt)
    for (const auto &t : mixedTranspositions)
      vals = applySwap(t, /*preShuf=*/false);
    return vals;
  }

  SmallVector<Value>
  transferWithinWarpShipImpl(Location loc, ConversionPatternRewriter &rewriter,
                             ArrayRef<Value> inVals, int nPack,
                             TranspositionInfo t) const {
    // Implements the effects of a single mixed transposition as in
    // `transferWithinWarpSwapImpl`, but uses auxiliary registers to hold the
    // values to be shuffled, resulting in fewer emitted instructions.
    int numRegs = inVals.size();
    int rIdx = t.transposition.first - nPack;
    int lIdx = t.transposition.second;
    int tileSize = 1 << (rIdx + 1);
    int numTiles = numRegs / tileSize;

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value laneId = getLaneId(rewriter, loc);
    Value lBitVal = b.and_(laneId, b.i32_val(1 << lIdx));
    Value lBitOff = b.icmp_eq(lBitVal, b.i32_val(0));
    SmallVector<Value> outVals(numRegs);

    auto shipDiagSels = [](auto postSel) {
      if (postSel == 0x3120)
        return std::pair{0x7564, 0x7564};
      auto high = (postSel & 0x4444) >> 2;
      auto sel10 = postSel ^ ((postSel & 0x1000) ? high << 1 : high);
      return std::pair{sel10, sel10 ^ 0x4444};
    };

    for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
      int baseIdx = tileIdx * tileSize;
      for (int i = 0; i < tileSize / 2; ++i) {
        int r0 = baseIdx + i;
        int r1 = r0 + (1 << rIdx);
        Value v0 = inVals[r0];
        Value v1 = inVals[r1];
        if (t.topPreSel == 0x3210 && t.topPostSel == 0x3210) {
          Value valToShip = b.select(lBitOff, v1, v0);
          Value shippedVal =
              targetInfo.shuffleXor(rewriter, loc, valToShip, (1 << lIdx));
          outVals[r0] = b.select(lBitOff, v0, shippedVal);
          outVals[r1] = b.select(lBitOff, shippedVal, v1);
        } else {
          Value shipSel =
              b.select(lBitOff, b.i32_val(t.botPreSel), b.i32_val(t.topPreSel));
          Value valToShip = targetInfo.permute(rewriter, loc, v0, v1, shipSel);
          Value shippedVal =
              targetInfo.shuffleXor(rewriter, loc, valToShip, (1 << lIdx));
          Value sel00 = b.i32_val(t.topPostSel);
          Value sel01 = b.i32_val(shipDiagSels(t.topPostSel).second);
          Value sel10 = b.i32_val(shipDiagSels(t.topPostSel).first);
          Value sel11 = b.i32_val(t.botPostSel ^ 0x4444);
          Value sel1 = b.select(lBitOff, sel00, sel01);
          Value sel2 = b.select(lBitOff, sel10, sel11);
          outVals[r0] = targetInfo.permute(rewriter, loc, v0, shippedVal, sel1);
          outVals[r1] = targetInfo.permute(rewriter, loc, v1, shippedVal, sel2);
        }
      }
    }
    return outVals;
  }
};

} // namespace

void mlir::triton::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, targetInfo, benefit);
}
