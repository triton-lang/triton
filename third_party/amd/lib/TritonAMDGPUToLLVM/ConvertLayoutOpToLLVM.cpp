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
    if (!amdTargInfo.supportsPermlaneSwap())
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

class ConvertLayoutOpInThreadSwap
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {

  static constexpr int regBytes = 4;

public:
  ConvertLayoutOpInThreadSwap(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  struct ByteLocation {
    int regIdx;
    int byteIdx;
  };

  // Creates v_perm operation:
  // It copies 4 bytes in dst value from two provided registers in accordance
  // with indexes in shuffleIds.
  //
  // index | copied contents
  //   0   | v2 & 0xff
  //   1   | (v2 >> 8) & 0xff
  //   2   | (v2 >> 16) & 0xff
  //   3   | (v2 >> 24) & 0xff
  //   4   | v1 & 0xff
  //   5   | (v1 >> 8) & 0xff
  //   6   | (v1 >> 16) & 0xff
  //   7   | (v1 >> 24) & 0xff
  static Value createVPerm(TritonLLVMOpBuilder &b, Value v1, Value v2,
                           ArrayRef<int> shuffleIds) {
    auto loc = b.loc;
    auto &rewriter = *b.builder;
    std::string intrinsic = "llvm.amdgcn.perm";
    int encodedIndices = 0;
    for (int i = 0; i < shuffleIds.size(); ++i) {
      assert(shuffleIds[i] >= 0 && shuffleIds[i] < 8);
      encodedIndices += shuffleIds[i] << (i * 8);
    }
    auto encodedIndicesVal = b.int_val(32, encodedIndices);
    return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, {i32_ty},
                                           {v1, v2, encodedIndicesVal})
        .getResult(0);
  }

  static int getLinearByteLoc(const ByteLocation &l) {
    return l.regIdx * regBytes + l.byteIdx;
  }

  static bool mergeablePairs(const std::array<ByteLocation, 2> &p1,
                             const std::array<ByteLocation, 2> &p2) {
    for (int i = 0; i < 2; ++i)
      if (p1[i].regIdx != p2[0].regIdx && p1[i].regIdx != p2[1].regIdx)
        return false;
    return true;
  }

  // generates full mapping from destination value index to related source value
  // index
  static std::vector<int> getFullLayout(const LinearLayout &conversion,
                                        mlir::MLIRContext *ctx) {
    auto numValues = conversion.getTotalInDimSize();
    std::vector<int> fullLayout(numValues);
    StringAttr kRegister = str_attr("register");
    for (int dstIdx = 0; dstIdx < numValues; ++dstIdx) {
      auto srcIdx = conversion.apply({{kRegister, dstIdx}}).begin()->second;
      fullLayout[dstIdx] = srcIdx;
    }
    return fullLayout;
  }

  // Converts full layout mapping to more convenient form.
  // Mapping each output register byte to some input byte:
  // regContents[output reg no][output reg byte no] -> ByteLocation(input
  // regiser no, byte index in this register)
  static std::vector<std::array<ByteLocation, regBytes>>
  generateDstRegContents(const std::vector<int> &fullLayout) {
    int numRegs = fullLayout.size() / regBytes;
    // mapping for dst register bytes to source bytes
    std::vector<std::array<ByteLocation, regBytes>> dstRegContents;
    for (int r = 0; r < numRegs; ++r) {
      std::array<ByteLocation, regBytes> regContents;
      for (int byteIdx = 0; byteIdx < regBytes; ++byteIdx) {
        regContents[byteIdx].regIdx =
            fullLayout[r * regBytes + byteIdx] / regBytes;
        regContents[byteIdx].byteIdx =
            fullLayout[r * regBytes + byteIdx] % regBytes;
      }
      dstRegContents.push_back(regContents);
    }
    return dstRegContents;
  }

  // Unpacks input values and packs them in int32 values, like they will be
  // stored in actual registers
  static std::vector<Value>
  repackInputToRegisters(Location loc, OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter) {
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto numRegs = inVals.size() / regBytes;
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    std::vector<Value> srcRegs(numRegs);
    for (int i = 0; i < numRegs; ++i) {
      SmallVector<Value> regComponents;
      for (int elem = 0; elem < regBytes; ++elem)
        regComponents.push_back(inVals[i * regBytes + elem]);
      auto vectorizedReg = packLLVector(loc, regComponents, rewriter);
      srcRegs[i] = b.bitcast(vectorizedReg, int_ty(32));
    }
    return srcRegs;
  }

  static void processOneWayDependencies(
      const std::vector<Value> &srcRegs, std::vector<Value> &dstRegs,
      llvm::ArrayRef<std::array<ByteLocation, regBytes>> dstRegContents,
      TritonLLVMOpBuilder &b) {
    auto numRegs = srcRegs.size();
    for (int i = 0; i < numRegs; ++i) {
      assert(!dstRegs[i]);
      bool needBytePermute = false;
      bool multipleDeps = false;
      int singleRegSrcIdx = dstRegContents[i][0].regIdx;
      for (int byteIdx = 0; byteIdx < regBytes; ++byteIdx) {
        needBytePermute |= (dstRegContents[i][byteIdx].byteIdx != byteIdx);
        multipleDeps |= (dstRegContents[i][byteIdx].regIdx != singleRegSrcIdx);
      }
      if (multipleDeps)
        continue;
      if (needBytePermute) {
        SmallVector<int> permute(regBytes);
        llvm::transform(dstRegContents[i], permute.begin(),
                        [](ByteLocation loc) { return loc.byteIdx; });
        dstRegs[i] = createVPerm(b, srcRegs[singleRegSrcIdx],
                                 srcRegs[singleRegSrcIdx], permute);
      } else {
        dstRegs[i] = srcRegs[singleRegSrcIdx];
      }
    }
  }

  static void processTwoWayDependencies(
      const std::vector<Value> &srcRegs, std::vector<Value> &dstRegs,
      llvm::ArrayRef<std::array<ByteLocation, regBytes>> dstRegContents,
      TritonLLVMOpBuilder &b) {
    auto numRegs = srcRegs.size();
    for (int i = 0; i < numRegs; ++i) {
      if (dstRegs[i])
        continue;
      std::set<int> srcRegSet;
      for (int byteIdx = 0; byteIdx < regBytes; ++byteIdx) {
        srcRegSet.insert(dstRegContents[i][byteIdx].regIdx);
      }
      if (srcRegSet.size() != 2)
        continue;
      int reg1 = *srcRegSet.begin();
      int reg2 = *(++srcRegSet.begin());

      SmallVector<int> permute(regBytes);
      for (int byteIdx = 0; byteIdx < regBytes; ++byteIdx) {
        if (dstRegContents[i][byteIdx].regIdx == reg1) {
          permute[byteIdx] = dstRegContents[i][byteIdx].byteIdx;
        } else {
          assert(dstRegContents[i][byteIdx].regIdx == reg2);
          permute[byteIdx] = dstRegContents[i][byteIdx].byteIdx + regBytes;
        }
      }
      dstRegs[i] = createVPerm(b, srcRegs[reg2], srcRegs[reg1], permute);
    }
  }

  struct BytePairInfo {
    std::vector<std::array<ByteLocation, 2>> pairCombinations;
    std::vector<int> srcByteToPairMap;
    std::vector<bool> pairMerged;
  };

  struct ByteQuadInfo {
    std::vector<std::array<ByteLocation, 4>> quadCombinations;
    std::vector<int> srcByteToQuadMap;
  };

  static BytePairInfo assembleBytePairs(
      const std::vector<Value> &dstRegs,
      llvm::ArrayRef<std::array<ByteLocation, regBytes>> dstRegContents) {
    int numRegs = dstRegs.size();
    int numBytes = numRegs * regBytes;
    BytePairInfo info;
    info.srcByteToPairMap.resize(numBytes, -1);
    for (int i = 0; i < numRegs; ++i) {
      if (dstRegs[i])
        continue;
      for (int pair = 0; pair < 2; ++pair) {
        int newId = info.pairCombinations.size();
        const auto &dstReg = dstRegContents[i];
        info.pairCombinations.push_back(
            {dstReg[pair * 2], dstReg[pair * 2 + 1]});
        info.srcByteToPairMap[getLinearByteLoc(dstReg[pair * 2])] = newId;
        info.srcByteToPairMap[getLinearByteLoc(dstReg[pair * 2 + 1])] = newId;
      }
    }
    info.pairMerged.resize(info.pairCombinations.size());
    return info;
  }

  static ByteQuadInfo assembleByteQuads(int numBytes, BytePairInfo &pairInfo) {
    ByteQuadInfo info;
    info.srcByteToQuadMap.resize(numBytes, -1);
    for (int i = 0; i < pairInfo.pairCombinations.size(); ++i) {
      if (pairInfo.pairMerged[i])
        continue;
      int srcRegNo = pairInfo.pairCombinations[i][0].regIdx;
      int srcByteNo = pairInfo.pairCombinations[i][0].byteIdx;
      for (int byteIdx = 0; byteIdx < regBytes; ++byteIdx) {
        // skip same byte
        if (srcByteNo == byteIdx)
          continue;
        int candidateSecondPair =
            pairInfo.srcByteToPairMap[getLinearByteLoc({srcRegNo, byteIdx})];
        if (candidateSecondPair < 0)
          continue;
        if (mergeablePairs(pairInfo.pairCombinations[candidateSecondPair],
                           pairInfo.pairCombinations[i]) &&
            !pairInfo.pairMerged[candidateSecondPair]) {
          std::array<ByteLocation, 4> quad;
          llvm::copy(pairInfo.pairCombinations[candidateSecondPair],
                     quad.begin());
          llvm::copy(pairInfo.pairCombinations[i], quad.begin() + 2);
          info.quadCombinations.push_back(quad);
          for (int byteIdx = 0; byteIdx < regBytes; ++byteIdx) {
            info.srcByteToQuadMap[getLinearByteLoc(quad[byteIdx])] =
                info.quadCombinations.size() - 1;
          }
          pairInfo.pairMerged[candidateSecondPair] = true;
          pairInfo.pairMerged[i] = true;
          break;
        }
      }
    }
    return info;
  }

  static std::vector<Value> materializePairs(const std::vector<Value> &srcRegs,
                                             const BytePairInfo &pairInfo,
                                             TritonLLVMOpBuilder &b) {
    std::vector<Value> materializedPairs(pairInfo.pairCombinations.size());
    for (int i = 0; i < pairInfo.pairCombinations.size(); ++i) {
      if (pairInfo.pairMerged[i])
        continue;
      const auto &p = pairInfo.pairCombinations[i];
      SmallVector<int> permute(regBytes);
      permute[0] = p[0].byteIdx;
      permute[1] = p[1].byteIdx + regBytes;
      materializedPairs[i] =
          createVPerm(b, srcRegs[p[1].regIdx], srcRegs[p[0].regIdx], permute);
    }
    return materializedPairs;
  }

  static std::vector<Value> materializeQuads(const std::vector<Value> &srcRegs,
                                             const ByteQuadInfo &quadInfo,
                                             TritonLLVMOpBuilder &b) {
    std::vector<Value> materializedQuads;
    for (int i = 0; i < quadInfo.quadCombinations.size(); ++i) {
      const auto &q = quadInfo.quadCombinations[i];
      SmallVector<int> permute(regBytes);
      int firstRegNo = q[0].regIdx;
      Value firstReg = srcRegs[firstRegNo];
      Value secondReg;
      for (int byteIdx = 0; byteIdx < regBytes; ++byteIdx) {
        if (q[byteIdx].regIdx != firstRegNo) {
          secondReg = srcRegs[q[byteIdx].regIdx];
          permute[byteIdx] = q[byteIdx].byteIdx + regBytes;
        } else {
          permute[byteIdx] = q[byteIdx].byteIdx;
        }
      }
      assert(secondReg);
      materializedQuads.push_back(createVPerm(b, secondReg, firstReg, permute));
    }
    return materializedQuads;
  }

  static void combineDstRegsFromPairsAndQuads(
      std::vector<Value> &dstRegs,
      llvm::ArrayRef<std::array<ByteLocation, regBytes>> dstRegContents,
      const std::vector<Value> &pairs, const std::vector<Value> &quads,
      const BytePairInfo &pairInfo, const ByteQuadInfo &quadInfo,
      TritonLLVMOpBuilder &b) {
    int numRegs = dstRegs.size();
    for (int i = 0; i < numRegs; ++i) {
      if (dstRegs[i])
        continue;
      SmallVector<int> permute(regBytes);
      Value firstReg;
      Value secondReg;
      for (int dstByteIdx = 0; dstByteIdx < regBytes; ++dstByteIdx) {
        int linearSrcPos = getLinearByteLoc(dstRegContents[i][dstByteIdx]);
        Value v;
        int bytePos;
        int quadNo = quadInfo.srcByteToQuadMap[linearSrcPos];
        if (quadNo >= 0) {
          v = quads[quadNo];
          for (int vByteIdx = 0; vByteIdx < regBytes; ++vByteIdx) {
            if (getLinearByteLoc(quadInfo.quadCombinations[quadNo][vByteIdx]) ==
                linearSrcPos) {
              bytePos = vByteIdx;
              break;
            }
          }
        } else {
          int pairNo = pairInfo.srcByteToPairMap[linearSrcPos];
          v = pairs[pairNo];
          assert(!pairInfo.pairMerged[pairNo]);
          for (int vByteIdx = 0; vByteIdx < 2; ++vByteIdx) {
            if (getLinearByteLoc(pairInfo.pairCombinations[pairNo][vByteIdx]) ==
                linearSrcPos) {
              bytePos = vByteIdx;
              break;
            }
          }
        }
        if (!firstReg)
          firstReg = v;
        if (v != firstReg) {
          if (!secondReg) {
            secondReg = v;
          }
          bytePos += regBytes;
        }
        permute[dstByteIdx] = bytePos;
      }
      dstRegs[i] = createVPerm(b, secondReg, firstReg, permute);
    }
  }

  // Algorithm overview
  //
  // 1. For each dst register, combine bytes from src register in pairs
  // according contents of low and high bytes of destination register.
  // 2. Combine these pairs of bytes into quads, so one quad is formed from only
  // two src registers
  // 3. Use these byte quads to generate each destination register.
  //
  // Stage 1 and 2 will be materialize in a layer of temporary registers
  // containing halves of dst register. Since we generated halves for every dst
  // register, we can make them out of tmp registers with one v_perm.
  //
  // Example
  //
  // src0 = [0, 1, 2, 3]
  // src1 = [4, 5, 6, 7]
  // src2 = [8, 9, 10, 11]
  // src3 = [12, 13, 14, 15]
  //
  // dst1 = (src0, byte0), (src1, byte1), (src2, byte2), (src3, byte3)
  // dst2 = (src0, byte1), (src1, byte0), (src2, byte3), (src3, byte2)
  //
  // dst1 expected value = [0, 5, 10, 15]
  // dst2 expected value = [0, 4, 11, 14]
  //
  // Pairs are
  //   (src0, byte0)+(src1, byte1), (src2, byte2)+(src3, byte3),
  //   (src0, byte1)+(src1, byte0), (src2, byte3)+(src3, byte2)
  //
  // Quads are (src0, byte0)+(src1, byte1)+(src0, byte1)+(src1, byte0),
  //           (src2, byte2)+(src3, byte3)+(src2, byte3)+(src3, byte2)
  //
  // These quads will be materialized in two v_perm instructions:
  // tmp1 = v_perm src0, src1 // combines bytes 0 and 1 from both sources
  //       tmp1 = [0, 5, 1, 4]
  // tmp2 = v_perm src2, src3 // combines bytes 2 and 3 from both sources
  //       tmp2 = [10, 15, 11, 14]
  // Then we can combine these temporary registers into dst:
  // dst1 = v_perm tmp1, tmp2 // combines bytes 0 and 1 from sources
  //       dst1 = [0, 5, 10, 15]
  // dst2 = v_perm tmp1, tmp2 // combines bytes 2 and 3 from sources
  //       dst2 = [0, 4, 11, 14]
  static void processFourWayDependencies(
      const std::vector<Value> &srcRegs, std::vector<Value> &dstRegs,
      llvm::ArrayRef<std::array<ByteLocation, regBytes>> dstRegContents,
      TritonLLVMOpBuilder &b) {
    int numRegs = srcRegs.size();
    int numBytes = numRegs * regBytes;

    auto pairInfo = assembleBytePairs(dstRegs, dstRegContents);
    auto quadInfo = assembleByteQuads(numBytes, pairInfo);

    auto materializedPairs = materializePairs(srcRegs, pairInfo, b);
    auto materializedQuads = materializeQuads(srcRegs, quadInfo, b);

    combineDstRegsFromPairsAndQuads(dstRegs, dstRegContents, materializedPairs,
                                    materializedQuads, pairInfo, quadInfo, b);
  }

  Value
  repackRegisterValuesInStructure(const std::vector<Value> &dstRegs,
                                  ConvertLayoutOp op, Type llvmElemType,
                                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    int numRegs = dstRegs.size();
    SmallVector<Value> outVals(numRegs * regBytes);
    auto vectorRegType = vec_ty(llvmElemType, regBytes);
    for (int regIdx = 0; regIdx < numRegs; ++regIdx) {
      auto vectorizedReg = LLVM::BitcastOp::create(rewriter, loc, vectorRegType,
                                                   dstRegs[regIdx]);
      auto unpacked = unpackLLVector(loc, vectorizedReg, rewriter);
      for (int elem = 0; elem < regBytes; elem++) {
        outVals[regIdx * regBytes + elem] = unpacked[elem];
      }
    }
    return packLLElements(loc, getTypeConverter(), outVals, rewriter,
                          op.getType());
  }

  void transferWithVPerm(ConvertLayoutOp op, const LinearLayout &conversion,
                         OpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto numValues = conversion.getTotalInDimSize();
    auto ctx = rewriter.getContext();
    auto fullLayout = getFullLayout(conversion, ctx);
    constexpr int regBytes = 4;
    // Non-trivial "in register" layout_conversion permutes at least 4 values.
    assert(numValues % regBytes == 0);
    int numRegs = numValues / regBytes;
    // Mapping for dst register bytes to source bytes
    auto dstRegContents = generateDstRegContents(fullLayout);
    std::vector<Value> srcRegs = repackInputToRegisters(loc, adaptor, rewriter);
    TritonLLVMOpBuilder b(loc, rewriter);

    std::vector<Value> dstRegs(numRegs);

    // Process dst registers that depend only on one src register
    processOneWayDependencies(srcRegs, dstRegs, dstRegContents, b);
    // Process dst registers that depend on two src registers
    processTwoWayDependencies(srcRegs, dstRegs, dstRegContents, b);
    // Process dst registers that depend on four src registers
    processFourWayDependencies(srcRegs, dstRegs, dstRegContents, b);

    // All destination registers should be materialized at this point
    assert(std::all_of(dstRegs.begin(), dstRegs.end(),
                       [](Value v) { return bool(v); }));

    auto llvmElemType =
        getTypeConverter()->convertType(op.getSrc().getType().getElementType());
    // Pack dst values to structure and finalize conversion
    Value result =
        repackRegisterValuesInStructure(dstRegs, op, llvmElemType, rewriter);
    rewriter.replaceOp(op, result);
  }

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &amdTargInfo =
        static_cast<const mlir::triton::AMD::TargetInfo &>(targetInfo);

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);

    auto ctx = op.getContext();
    StringAttr kRegister = mlir::StringAttr::get(ctx, "register");

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    auto dims = conversion.getInDimNames();
    if (!(conversion.getNumInDims() == 1 &&
          llvm::is_contained(dims, kRegister)))
      return failure();

    // TODO: v_perm could be useful for fp16 tensors.
    if (srcTy.getElementType().getIntOrFloatBitWidth() != 8)
      return failure();
    // TODO: broadcasting is not supported at the moment.
    if (!conversion.isInvertible())
      return failure();
    transferWithVPerm(op, conversion, adaptor, rewriter);
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
  patterns.add<ConvertLayoutOpInThreadSwap>(typeConverter, targetInfo, benefit);
  // No need to convert when ForcedSwizzling as it's already the default
  // lowering
}
