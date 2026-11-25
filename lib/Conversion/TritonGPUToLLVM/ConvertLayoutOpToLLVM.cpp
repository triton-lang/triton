#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include <optional>

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

struct ConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  const TargetInfoBase &targetInfo;

  explicit ConvertLayoutOpConversion(LLVMTypeConverter &typeConverter,
                                     const TargetInfoBase &targetInfo,
                                     PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    auto kRegister = str_attr("register");

    auto srcLayout = toLinearLayout(srcTy).removeZeroBasesAlongDim(kRegister);
    auto dstLayout = toLinearLayout(dstTy).removeZeroBasesAlongDim(kRegister);
    LinearLayout conversion = minimalCvtLayout(srcLayout, dstLayout);

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    if (conversion.getNumInDims() == 0) {
      assert(adaptor.getSrc().getType() ==
             getTypeConverter()->convertType(dstTy));
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
    if (conversion.getNumInDims() == 1)
      return transferWithinThread(op, conversion, adaptor, rewriter);

    if (cvtNeedsWarpShuffle(op))
      return transferWithinWarp(op, srcLayout, dstLayout, adaptor, rewriter);

    transferSwizzlingLocalMem(op, adaptor.getSrc(), srcLayout, dstLayout,
                              rewriter);
    return success();
  }

  LogicalResult
  transferWithinThread(ConvertLayoutOp op, const LinearLayout &conversion,
                       OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    auto kRegister = str_attr("register");
    auto dstTy = op.getType();
    auto dims = to_vector(conversion.getInDimNames());
    assert(dims.size() == 1 && dims.front() == kRegister &&
           "expected a register-only conversion");

    auto inVals = unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> outVals(conversion.getInDimSize(kRegister));
    for (int i = 0; i < outVals.size(); i++) {
      auto srcIdx = conversion.apply({{kRegister, i}}).begin()->second;
      assert(srcIdx < inVals.size());
      outVals[i] = inVals[srcIdx];
    }
    Value result = packUniqueTensorElements(loc, getTypeConverter(), outVals,
                                            rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }

  void transferSwizzlingLocalMem(ConvertLayoutOp op, Value src,
                                 const LinearLayout &srcLayout,
                                 const LinearLayout &dstLayout,
                                 ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto inVals = unpackUniqueTensorElements(loc, src, rewriter);
    auto outVals = mlir::convertLayoutViaSharedMemory(
        loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, {0}, op,
        targetInfo);

    Value result = packUniqueTensorElements(loc, getTypeConverter(),
                                            outVals.front(), rewriter, dstTy);
    rewriter.replaceOp(op, result);
  }

  // Use warp shuffles to implement a layout conversion where data only needs to
  // be moved within warps.
  LogicalResult transferWithinWarp(ConvertLayoutOp op,
                                   const LinearLayout &srcLayout,
                                   const LinearLayout &dstLayout,
                                   OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto elemTy = getTypeConverter()->convertType(srcTy.getElementType());
    int bitwidth = getIntOrFloatOrPtrBitWidth(elemTy);

    auto [pReg, shuffleMap, mixedTranspositions, nPack] =
        getWarpLayoutConvertDecomposition(srcLayout, dstLayout, bitwidth);
    int m = mixedTranspositions.size();
    bool isXorShuffle = squareSublayoutIsIdentity(shuffleMap, kLane);

    // In permutation cases, the desired layout conversion can be expressed as
    // a permutation P of hardware index bits for the `kLane` and `kReg`
    // dimensions. The `factors` of P describe a decomposition
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
    //    permutes and either (1 - (1/2)^m) * R shuffles if `isXorShuffle` and
    //    R shuffles otherwise.
    // Here, R denotes the number of 32-bit registers in use after packing (or
    // splitting, if applied to 64-bit types or pointers), and in the `Swap`
    // method, `m` denotes the number of mixed transpositions passed in.
    auto inVals = unpackUniqueTensorElements(loc, adaptor.getSrc(), rewriter);

    // If the target layout has a larger register dimension than the source
    // layout, then we broadcast along the register dimension to match size. The
    // removal of broadcasting above and introduction here is expected by the
    // `factors`.
    int regDim = pReg.getInDimSize(kReg);
    SmallVector<Value> newInVals(regDim);
    for (int r = 0; r < regDim; ++r)
      newInVals[pReg.apply({{kReg, r}})[0].second] = inVals[r % inVals.size()];
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
    if (m == 1 && isXorShuffle && isShippable(mixedTranspositions[0])) {
      outVals = transferWithinWarpShipImpl(loc, rewriter, inVals, nPack,
                                           mixedTranspositions[0]);
    } else {
      outVals =
          transferWithinWarpSwapImpl(loc, rewriter, inVals, nPack, shuffleMap,
                                     isXorShuffle, mixedTranspositions);
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
          v = b.trunc(int_ty(bitwidth), v, LLVM::IntegerOverflowFlags::nuw);
          if (elemTy != int_ty(bitwidth))
            v = b.bitcast(v, elemTy);
        }
      }
      outVals = std::move(unpackedVals);
    }

    // If `dstLayout` has a smaller `kReg` dimension than `srcLayout` after
    // broadcasting is removed, then drop the extra registers from `outVals`.
    outVals.resize(dstLayout.getInDimSize(kReg));

    Value result = packUniqueTensorElements(loc, getTypeConverter(), outVals,
                                            rewriter, dstTy);
    rewriter.replaceOp(op, result);
    return success();
  }

  SmallVector<Value> transferWithinWarpSwapImpl(
      Location loc, ConversionPatternRewriter &rewriter, ArrayRef<Value> inVals,
      int nPack, const LinearLayout &shuffleMap, bool isXorShuffle,
      ArrayRef<TranspositionInfo> mixedTranspositions) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kBlock = str_attr("block");

    SmallVector<Value> vals(inVals.begin(), inVals.end());
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
    // `(1 - (1/2)^m) * numRegs` shuffles in total. If `P_lane` is nontrivial,
    // then we can conjugate its effects through the first two stages and fuse
    // it with the second stage, resulting in `numRegs` shuffles instead.
    Value laneId, warpId;
    std::tie(laneId, warpId) = getLaneAndWarpId(rewriter, loc);
    // Implement r_i ^= l_j using `numRegs` independent selects or permutes.
    auto applySwap = [&](TranspositionInfo t, bool preShuf) {
      int rIdx = t.regBit - nPack;
      int lIdx = preShuf ? t.srcLane : t.dstLane;
      uint16_t topSel = preShuf ? t.topPreSel : t.topPostSel;
      uint16_t botSel = preShuf ? t.botPreSel : t.botPostSel;

      SmallVector<Value> newVals(numRegs);
      Value lBitOff = b.true_val();
      if (lIdx >= 0)
        lBitOff = b.icmp_eq(b.and_(laneId, b.i32_val(1 << lIdx)), b.i32_val(0));

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
    if (!isXorShuffle) {
      SmallVector<std::pair<StringAttr, Value>> indices{{kLane, laneId}};
      if (!shuffleMap.sublayoutIsZero(kWarp, kLane))
        indices.push_back({kWarp, warpId});
      if (!shuffleMap.sublayoutIsZero(kBlock, kLane))
        indices.push_back({kBlock, targetInfo.getClusterCTAId(rewriter, loc)});
      auto dims = to_vector(llvm::make_first_range(indices));
      laneIdPerm = applyLinearLayout(loc, rewriter,
                                     shuffleMap.sublayout(dims, kLane), indices)
                       .front()
                       .second;
    }
    auto registerToLane = shuffleMap.sublayout(kReg, kLane);
    for (int r = 0; r < numRegs; ++r) {
      int mask = registerToLane.apply({{kReg, r << nPack}})[0].second;
      if (isXorShuffle) {
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
    int rIdx = t.regBit - nPack;
    int lIdx = t.dstLane;
    int tileSize = 1 << (rIdx + 1);
    int numTiles = numRegs / tileSize;

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value laneId = getLaneId(rewriter, loc);
    Value lBitVal = b.and_(laneId, b.i32_val(1 << lIdx));
    Value lBitOff = b.icmp_eq(lBitVal, b.i32_val(0));
    SmallVector<Value> outVals(numRegs);

    auto postShipSel = [](uint16_t preSel, uint16_t postSel, bool rBitOn) {
      // When selecting non-shipped values, we need to compose with the
      // preshuffle masks since the values were not previously permuted.
      uint16_t ret = 0;
      for (size_t k = 0; k < 4; ++k) {
        uint16_t postNib = (postSel >> (4 * k)) & 0xF;
        uint16_t nib = postNib;
        if (postNib < 4) {
          uint16_t preNib = (preSel >> (4 * postNib)) & 0xF;
          nib = rBitOn ? (preNib - 4) : preNib;
        }
        ret |= nib << (4 * k);
      }
      return ret;
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
          uint16_t sel00 =
              postShipSel(t.topPreSel, t.topPostSel, /*rBitOn=*/false);
          uint16_t sel01 =
              postShipSel(t.botPreSel, t.topPostSel ^ 0x4444, /*rBitOn=*/false);
          uint16_t sel10 =
              postShipSel(t.topPreSel, t.botPostSel, /*rBitOn=*/true);
          uint16_t sel11 =
              postShipSel(t.botPreSel, t.botPostSel ^ 0x4444, /*rBitOn=*/true);
          Value sel1 = b.select(lBitOff, b.i32_val(sel00), b.i32_val(sel01));
          Value sel2 = b.select(lBitOff, b.i32_val(sel10), b.i32_val(sel11));
          outVals[r0] = targetInfo.permute(rewriter, loc, v0, shippedVal, sel1);
          outVals[r1] = targetInfo.permute(rewriter, loc, v1, shippedVal, sel2);
        }
      }
    }
    return outVals;
  }
};

struct SharedTransfer {
  LinearLayout storeCvt, loadCvt;
  ColumnAction permLoad;
  SmallVector<Value> inputs;
  int nReps;
};

SharedTransfer prepareTransfer(Location loc, RewriterBase &rewriter,
                               const LinearLayout &srcLayout,
                               const LinearLayout &dstLayout,
                               const SharedMemoryPack &pack,
                               ArrayRef<SmallVector<Value>> inVals,
                               const TargetInfoBase &targetInfo) {
  auto *ctx = rewriter.getContext();
  int bitwidth = pack.type.getElementTypeBitWidth();
  int numBanks = targetInfo.getSharedMemoryBanks();
  int32_t vecBitwidth =
      triton::gpu::getVecBitwidthLdSt(srcLayout, dstLayout, bitwidth);
  auto [dstTile, srcTile] = targetInfo.getSharedLdStTiles(vecBitwidth);
  auto smem = optimalSwizzlingLdSt(srcLayout, dstLayout, bitwidth, numBanks,
                                   srcTile, dstTile);

  // Extract reps from smem
  auto kReg = str_attr("register");
  auto kBlock = str_attr("block");
  auto kReps = str_attr("reps");
  auto nReps = smem.getInDimSize(kReps);
  auto reps = LinearLayout::identity1D(nReps, kReg, kReps);

  auto totalStoreCvt = srcLayout.invertAndCompose(smem);
  auto totalLoadCvt = invertAndComposeLocal(smem, dstLayout, {kBlock});

  // The permutation exists by construction of the reps dimension in
  // optimalSwizzling
  auto permStore =
      regPermForDivide(totalStoreCvt, reps, /*left=*/false).value();
  totalStoreCvt = permStore.apply(totalStoreCvt);
  auto permLoad = regPermForDivide(totalLoadCvt, reps, /*left=*/false).value();
  totalLoadCvt = permLoad.apply(totalLoadCvt);

  // Remove the reps and flatten into offset
  auto storeCvt = *divideRight(totalStoreCvt, reps);
  auto loadCvt = *divideRight(totalLoadCvt, reps);
  auto kOffset = str_attr("offset");
  auto nBlock = storeCvt.getInDimSize(kBlock);
  storeCvt = storeCvt.reshapeOuts(
      {{kOffset, storeCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});
  loadCvt = loadCvt.reshapeOuts(
      {{kOffset, loadCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});

  // Apply one swizzle and register order to all fields, then interleave words.
  unsigned count = pack.type.getNumElements();
  if (count > 1) {
    auto fields = LinearLayout::identity1D(count, kReg, kOffset);
    storeCvt = fields * storeCvt;
    loadCvt = fields * loadCvt;
  }
  SmallVector<Value> inputs(storeCvt.getInDimSize(kReg) * nReps);
  for (const auto &field : pack.fields) {
    auto values = permStore.apply(inVals[field.index]);
    for (auto [r, value] : llvm::enumerate(values))
      llvm::copy(pack.encode(loc, rewriter, field, value),
                 inputs.begin() + r * count + field.firstWord);
  }
  return {std::move(storeCvt), std::move(loadCvt), std::move(permLoad),
          std::move(inputs), nReps};
}

} // namespace

namespace mlir {

using namespace mlir::triton::gpu;

// Form power-of-two packs up to maxPackBits using the narrowest field's
// width as the word size. A field wider than the limit forms its own pack.
SmallVector<SharedMemoryPack> getSharedMemoryPacks(TritonLLVMOpBuilder &b,
                                                   Value smem,
                                                   ArrayRef<Type> elemTypes,
                                                   ArrayRef<unsigned> offsets,
                                                   unsigned maxPackBits) {
  auto widths = llvm::map_to_vector(elemTypes, [](Type ty) -> unsigned {
    return std::max(8, getIntOrFloatOrPtrBitWidth(ty));
  });
  auto *ctx = smem.getContext();
  auto makePack = [&](ArrayRef<unsigned> fields) {
    unsigned wordBits = widths[fields.back()];
    auto wordTy = IntegerType::get(ctx, wordBits);
    SharedMemoryPack pack;
    unsigned numWords = 0;
    for (unsigned i : fields) {
      unsigned words = widths[i] / wordBits;
      pack.fields.push_back({i, elemTypes[i], numWords, words});
      numWords += words;
    }
    pack.type = vec_ty(wordTy, numWords);
    pack.base = smem;
    if (unsigned offset = offsets[fields.front()])
      pack.base = b.gep(smem.getType(), IntegerType::get(ctx, 8), smem,
                        LLVM::GEPArg(offset));
    return pack;
  };
  if (elemTypes.size() == 1)
    return {makePack({0})};

  // The allocator lays out fields widest first. Each pack reuses consecutive
  // scratch slices starting at its first field's offset.
  auto indices = llvm::to_vector(llvm::seq<unsigned>(elemTypes.size()));
  llvm::sort(indices,
             [&](unsigned i, unsigned j) { return offsets[i] < offsets[j]; });
  SmallVector<SharedMemoryPack> packs;
  ArrayRef<unsigned> remaining = indices;
  while (!remaining.empty()) {
    unsigned bits = 0, count = 1;
    for (auto [i, field] : llvm::enumerate(remaining)) {
      bits += widths[field];
      if (bits > maxPackBits)
        break;
      if (llvm::isPowerOf2_32(bits))
        count = i + 1;
    }
    packs.push_back(makePack(remaining.take_front(count)));
    remaining = remaining.drop_front(count);
  }
  return packs;
}

SmallVector<Value> SharedMemoryPack::encode(Location loc,
                                            RewriterBase &rewriter,
                                            const Field &field,
                                            Value value) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type wordTy = type.getElementType();
  if (isa<LLVM::LLVMPointerType>(field.type))
    value = b.ptrtoint(i64_ty, value);
  else if (field.type.getIntOrFloatBitWidth() < 8)
    return {b.zext(wordTy, value)};
  if (field.numWords > 1)
    return unpackLLVector(loc, b.bitcast(value, vec_ty(wordTy, field.numWords)),
                          rewriter);
  if (isa<FloatType>(field.type))
    value = b.bitcast(value, wordTy);
  return {value};
}

Value SharedMemoryPack::decode(Location loc, RewriterBase &rewriter,
                               const Field &field,
                               ArrayRef<Value> words) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  words = words.slice(field.firstWord, field.numWords);
  bool isPtr = isa<LLVM::LLVMPointerType>(field.type);
  Type scalarTy = isPtr ? i64_ty : field.type;
  if (scalarTy.getIntOrFloatBitWidth() < 8)
    return b.trunc(scalarTy, words.front(), LLVM::IntegerOverflowFlags::nuw);
  Value value =
      words.size() == 1 ? words.front() : packLLVector(loc, words, rewriter);
  if (value.getType() != scalarTy)
    value = b.bitcast(value, scalarTy);
  if (isPtr)
    return b.inttoptr(field.type, value);
  return value;
}

SmallVector<SmallVector<Value>> convertLayoutViaSharedMemory(
    Location loc, ConversionPatternRewriter &rewriter,
    const LinearLayout &srcLayout, const LinearLayout &dstLayout,
    ArrayRef<SmallVector<Value>> inVals, ArrayRef<Type> elemTypes,
    ArrayRef<unsigned> offsets, Operation *sourceOp,
    const TargetInfoBase &targetInfo, Value storePred) {
  auto *ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (!storePred)
    storePred = b.true_val();

  Value smem = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, sourceOp);
  // Only common register bits left out of the vector can become repetitions.
  // Reserving room for all of them makes every multi-field pack single-round.
  unsigned maxPackBits = 0;
  if (elemTypes.size() > 1) {
    auto common =
        getCommonRegBases(srcLayout.flattenOuts(), dstLayout.flattenOuts());
    maxPackBits = 128u >> common.size();
  }
  auto packs = getSharedMemoryPacks(b, smem, elemTypes, offsets, maxPackBits);
  SmallVector<SharedTransfer, 1> transfers;
  transfers.reserve(packs.size());
  int maxReps = 0;
  for (const auto &pack : packs) {
    transfers.push_back(prepareTransfer(loc, rewriter, srcLayout, dstLayout,
                                        pack, inVals, targetInfo));
    maxReps = std::max(maxReps, transfers.back().nReps);
  }

  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");
  bool isWarpSync = mlir::isCvtDimSync(srcLayout, dstLayout, kWarp);
  bool isBlockSync = mlir::isCvtDimSync(srcLayout, dstLayout, kBlock);
  auto emitBarrier = [&]() {
    if (isWarpSync) {
      targetInfo.warpSync(loc, rewriter);
    } else if (isBlockSync) {
      targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
    } else {
      targetInfo.clusterBarrier(loc, rewriter, sourceOp);
    }
  };

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  auto zero = b.i32_val(0);
  SmallVector<SmallVector<Value>> outVals(elemTypes.size());
  for (int rep = 0; rep < maxReps; ++rep) {
    if (rep > 0)
      emitBarrier(); // Finish the previous loads before reusing scratch.
    for (auto [transfer, pack] : llvm::zip_equal(transfers, packs)) {
      if (rep >= transfer.nReps)
        continue;
      auto tileSize = transfer.inputs.size() / transfer.nReps;
      auto tile = ArrayRef(transfer.inputs).slice(rep * tileSize, tileSize);
      lowerLdSt(loc, ctx, transfer.storeCvt, tile, pack.type.getElementType(),
                pack.base,
                /*paddingShifts=*/{}, zero, 0,
                /*affineBlockOffset=*/{}, 0, laneId, warpId, rewriter,
                targetInfo, /*maybeMaxVecElems=*/{},
                makeSharedStoreEmitter(targetInfo, storePred));
    }
    emitBarrier();
    for (auto [transfer, pack] : llvm::zip_equal(transfers, packs)) {
      if (rep >= transfer.nReps)
        continue;
      auto tile = lowerLdSt(
          loc, ctx, transfer.loadCvt, {}, pack.type.getElementType(), pack.base,
          /*paddingShifts=*/{}, zero, 0, /*affineBlockOffset=*/{}, 0, laneId,
          warpId, rewriter, targetInfo,
          /*maybeMaxVecElems=*/{}, makeSharedLoadEmitter(targetInfo));
      for (unsigned i = 0; i < tile.size(); i += pack.type.getNumElements()) {
        auto words = ArrayRef(tile).slice(i, pack.type.getNumElements());
        for (const auto &field : pack.fields)
          outVals[field.index].push_back(
              pack.decode(loc, rewriter, field, words));
      }
    }
  }

  for (auto [transfer, pack] : llvm::zip_equal(transfers, packs)) {
    auto order = transfer.permLoad.inverse();
    for (const auto &field : pack.fields)
      outVals[field.index] = order.apply(outVals[field.index]);
  }
  return outVals;
}

} // namespace mlir

void mlir::triton::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpConversion>(typeConverter, targetInfo, benefit);
}
