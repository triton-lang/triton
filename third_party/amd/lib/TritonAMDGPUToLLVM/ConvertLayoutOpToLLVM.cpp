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
    if (!(amdTargInfo.getISAFamily() == AMD::ISAFamily::CDNA4 ||
          amdTargInfo.getISAFamily() == AMD::ISAFamily::GFX1250))
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
} // namespace

void mlir::triton::AMD::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ConvertLayoutOpPermlaneSwap>(typeConverter, targetInfo, benefit);
  // No need to convert when ForcedSwizzling as it's already the default
  // lowering
}
