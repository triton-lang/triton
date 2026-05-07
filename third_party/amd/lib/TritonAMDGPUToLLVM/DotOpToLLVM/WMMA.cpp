/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/Support/LLVM.h"

#include "../PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUTransforms/WmmaGroup.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/TypeSwitch.h"
namespace mlir::triton::AMD {
namespace {

static std::optional<mlir::triton::LinearLayout>
wmmaRepLayoutForTensor(const mlir::triton::LinearLayout &wholeTileLL,
                       const mlir::triton::LinearLayout &tileLL) {
  llvm::SmallDenseMap<StringAttr, int64_t> shape;
  for (auto outDim : wholeTileLL.getOutDimNames())
    shape[outDim] = wholeTileLL.getOutDimSize(outDim);

  // Clamp the tileLL to the tensor's output dims so that divideLeft succeeds
  // when the tensor is smaller than the WMMA instruction shape.
  auto clampedTileLL = ensureLayoutNotLargerThan(tileLL, shape);

  auto quot = divideLeft(wholeTileLL, clampedTileLL);
  if (quot.has_value())
    return zerosLike(clampedTileLL) * *quot;
  return {};
}

#define S(v) StringAttr::get(ctx, (v))
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::LinearEncodingAttr;

Value prepareOperands(ConversionPatternRewriter &rewriter, Value rawElems,
                      Type type, int wmmaVer, int kBase, Location loc,
                      bool isScale = false) {
  Value convertedElems;
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  if (type.isF32() || type.isF16()) {
    convertedElems = rawElems;
  } else if (type.isBF16()) {
    convertedElems = rawElems;
    // Before wmma v3, bf16 is converted to i16
    if (wmmaVer < 3)
      convertedElems = tb.bitcast(rawElems, vec_ty(i16_ty, kBase));
  } else {
    // When scaleFactor == 16, scales are stored in i64
    // This only applies to scale operands, not regular dot operands
    Type targetTy = (isScale && kBase == 8) ? i64_ty : i32_ty;
    auto elems =
        kBase * type.getIntOrFloatBitWidth() / targetTy.getIntOrFloatBitWidth();
    assert(elems >= 1 && "unexpected number of elements");
    if (elems == 1)
      convertedElems = tb.bitcast(rawElems, targetTy);
    else
      convertedElems = tb.bitcast(rawElems, vec_ty(targetTy, elems));
  }
  return convertedElems;
}

// Returns the number of zero-padded registers when tensorK < wmma instr K.
static int computeKPadding(int kBase, int64_t tensorK,
                           DotOperandEncodingAttr dotEnc, unsigned warpSize) {
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(dotEnc.getParent());
  auto mnkDim = wmmaLayout.getInstrShape();
  int kInstrSize = mnkDim.back();

  if (tensorK >= kInstrSize)
    return 0;

  // Wmma operand layouts for narrow dtypes have multiple K repetitions per
  // lane, e.g. reg0-3 hold k[0-16] and reg 4-7 hold k[32-47] for lane0. This
  // means if tensorK is smaller than one k repetition we get broadcasts in the
  // lane dimension so we have tensorK valid elements per nonKRepeat subtile.
  constexpr int wmmaTileDim = 16;
  int nonKDim = wmmaLayout.getOperandNonKDim(dotEnc.getOpIdx());
  int nonKRepeat = nonKDim / wmmaTileDim;
  int lanesInKDim = warpSize / wmmaTileDim;
  int elemsPerKRep = lanesInKDim * dotEnc.getKWidth();
  if (tensorK < elemsPerKRep)
    return kBase - tensorK * nonKRepeat;

  // If tensorK is at least one k repetition tile, pad full out-of-bounds tiles.
  return kBase - kBase * tensorK / kInstrSize;
}

// Returns a bitmask of the lane bits that will move in K direction. For WMMA
// v2+ operand layouts the first nonKDim(=16) lanes (log2(nonKDim) bits) walk
// the non-K dimension and then wrap around moving in K dimension.
static int computeKLaneBitsMask(DotOperandEncodingAttr dotEnc) {
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(dotEnc.getParent());
  if (wmmaLayout.getVersion() < 2)
    return 0;
  constexpr int wmmaTileDim = 16;
  return ~(wmmaTileDim - 1);
}

// When the operand tensor K is smaller than the WMMA instruction K, the layout
// will have broadcasting lanes. Broadcasting lanes which would move in K
// direction need to be masked out (to zero) so they don't incorrectly
// contribute stale K to the WMMA result.
static Value maskRepeatedKLanes(ConversionPatternRewriter &rewriter,
                                Location loc, LinearLayout dotLayout,
                                DotOperandEncodingAttr dotEnc, Value operand,
                                unsigned warpSize) {
  auto ctx = dotLayout.getOutDimNames().begin()->getContext();
  StringAttr kLane = StringAttr::get(ctx, "lane");
  int32_t laneFreeMask = dotLayout.getFreeVariableMasks().lookup(kLane);
  // Only mask out lane bits that will move in K direction
  int32_t kFreeMask = laneFreeMask & computeKLaneBitsMask(dotEnc);
  if (kFreeMask == 0)
    return operand;
  TritonLLVMOpBuilder tb(loc, rewriter);
  Value laneId = tb.and_(getThreadId(rewriter, loc), tb.i32_val(warpSize - 1));
  Value freeBits = tb.and_(laneId, tb.i32_val(kFreeMask));
  Value isCanonical = tb.icmp_eq(freeBits, tb.i32_val(0));
  Value zero = tb.null(operand.getType());
  return tb.select(isCanonical, operand, zero);
}

Value getOperandVals(ConversionPatternRewriter &rewriter,
                     const LLVMTypeConverter *typeConverter,
                     LinearLayout dotLayout, Value value, int opIdx, int rank,
                     int batch, int nonK, int kIdx, int kInstrSize, int kBase,
                     int64_t kDimTensor, DotOperandEncodingAttr dotEnc,
                     unsigned warpSize, int *opSel, Type type, Location loc,
                     bool isScale = false) {
  auto ctx = dotLayout.getOutDimNames().begin()->getContext();

  const StringAttr dim0 = StringAttr::get(ctx, "dim0");
  const StringAttr dim1 = StringAttr::get(ctx, "dim1");
  const StringAttr dim2 = StringAttr::get(ctx, "dim2");

  TritonLLVMOpBuilder tb(loc, rewriter);

  auto elems = unpackLLElements(loc, value, rewriter);

  Type elemTy = typeConverter->convertType(type);
  Type vecTy = vec_ty(elemTy, kBase);

  Value rawElems = tb.undef(vecTy);

  // kIdx is expressed in "instructions"; convert to element indexing.
  const int kElemIdx = kIdx * kInstrSize;

  // Choose which output dimension gets nonK vs K depending on opIdx.
  const int mCoord = (opIdx == 0) ? nonK : kElemIdx;
  const int nCoord = (opIdx == 0) ? kElemIdx : nonK;

  // Compute registers via pseudoinverse
  SmallVector<std::pair<StringAttr, int32_t>> outCoords;
  outCoords.reserve(rank);

  if (rank == 3) {
    outCoords.push_back({dim0, batch});
    outCoords.push_back({dim1, mCoord});
    outCoords.push_back({dim2, nCoord});
  } else {
    outCoords.push_back({dim0, mCoord});
    outCoords.push_back({dim1, nCoord});
  }

  auto inDims = dotLayout.pseudoinvert().apply(outCoords);

  const int startReg = inDims[0].second; // "register"
  const int lane = inDims[1].second;     // "lane"

  if (isScale && lane != 0) {
    assert(opSel && "opSel must be provided when isScale is true");
    *opSel = 1;
  }

  // ---- Fill vector, padding tail with zeros ----
  const int kPadding =
      isScale ? 0 : computeKPadding(kBase, kDimTensor, dotEnc, warpSize);
  const int validK = kBase - kPadding;

  Value zero;
  if (kPadding > 0) {
    zero = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                    rewriter.getZeroAttr(elemTy));
  }

  // The register layout splits kBase into nonKRepeat subtiles of kPerSubtile
  // registers:
  //   regs [0, kPerSubtile)             -> M=0..15
  //   regs [kPerSubtile, 2*kPerSubtile) -> M=16..31
  // Padding must be applied per-subtile so each nonK half is padded
  // independently.
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(dotEnc.getParent());
  int nonKDim = wmmaLayout.getOperandNonKDim(dotEnc.getOpIdx());
  const int nonKTileDim = 16;
  int nonKRepeat = nonKDim / nonKTileDim;
  int kPerSubtile = kBase / nonKRepeat;
  int validKPerSubTile = validK / nonKRepeat;

  for (int k = 0; k < kBase; ++k) {
    int subTilePos = k % kPerSubtile;
    Value elem = (subTilePos < validKPerSubTile) ? elems[startReg + k] : zero;
    rawElems = tb.insert_element(vecTy, rawElems, elem, tb.i32_val(k));
  }

  return rawElems;
}

static inline int32_t getWmmaF8F6F4MatrixFormat(Type t) {
  return llvm::TypeSwitch<Type, int32_t>(t)
      .Case<Float8E4M3FNType>([](Type) { return 0; })
      .Case<Float8E5M2Type>([](Type) { return 1; })
      .Case<Float6E2M3FNType>([](Type) { return 2; })
      .Case<Float6E3M2FNType>([](Type) { return 3; })
      .Case<Float4E2M1FNType>([](Type) { return 4; })
      .Default([](Type) { return -1; });
}

Value generateWMMAIntrinsic(ConversionPatternRewriter &rewriter, Location loc,
                            int wmmaVer, Value valA, Value valB, Value valC,
                            Type aElType, Type bElType, Type dElType,
                            StringRef name, std::optional<bool> tiedLower) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  SmallVector<Value> operands;

  if (wmmaVer == 1 || wmmaVer == 2) {
    // arguments for v1 and v2:
    // int:   %A_sign, %A, %B_sign, %B, %C, [%clamp]
    // float: %A, %B, %C, [%tied_to_high]
    if (aElType.isInteger())
      operands.push_back(b.int_val(1, !aElType.isUnsignedInteger()));
    operands.push_back(valA);

    if (bElType.isInteger())
      operands.push_back(b.int_val(1, !bElType.isUnsignedInteger()));
    operands.push_back(valB);

    operands.push_back(valC);

    if (tiedLower.has_value() || 32 / dElType.getIntOrFloatBitWidth() > 1 ||
        dElType.isInteger(32))
      operands.push_back(b.int_val(1, tiedLower.value_or(false)));
  } else {
    assert(wmmaVer == 3 && "unexpected wmma version");
    // LLVM AMDGPU intrinsics use one of:
    // - AMDGPUWmmaIntrinsicModsC (f16/bf16/f32 acc, fp8 packed int, etc.):
    //   %A, %B, %C_mod, %C, matrix_a_reuse, matrix_b_reuse
    // - AMDGPUWmmaIntrinsicModsABClamp (e.g. i32_16x16x64_iu8):
    //   %A_mod, %A, %B_mod, %B, %C, reuse, reuse, clamp
    const bool isIntDot = aElType.isInteger() && bElType.isInteger();
    if (isIntDot) {
      operands.push_back(b.int_val(1, !aElType.isUnsignedInteger()));
      operands.push_back(valA);
      operands.push_back(b.int_val(1, !bElType.isUnsignedInteger()));
      operands.push_back(valB);
      operands.push_back(valC);
      operands.push_back(b.i1_val(0));
      operands.push_back(b.i1_val(0));
      operands.push_back(b.i1_val(0));
    } else {
      operands.push_back(valA);
      operands.push_back(valB);
      operands.push_back(b.int_val(16, 0));
      operands.push_back(valC);
      operands.push_back(b.i1_val(0));
      operands.push_back(b.i1_val(0));
    }
  }

  auto wmmaIntrinsic = LLVM::createLLVMIntrinsicCallOp(
      rewriter, loc, name, valC.getType(), operands);
  return wmmaIntrinsic.getResult(0);
}

static inline int32_t getWmmaScaleDataType(Type scaleElemType) {
  // Data Type of block-scale
  // 0: E8M0
  // 1: E5M3
  // 2: E4M3
  return llvm::TypeSwitch<Type, int32_t>(scaleElemType)
      .Case<IntegerType>([](Type) { return 0; })
      .Case<Float8E4M3FNType>([](Type) { return 2; })
      .Default([](Type) { return -1; });
}

Value generateScaledWMMAIntrinsic(ConversionPatternRewriter &rewriter,
                                  Location loc, Value valA, Value valScaleA,
                                  Value valB, Value valScaleB, Value valC,
                                  Type aElType, Type aScaleElType, Type bElType,
                                  Type bScaleElType, Type dElType,
                                  int opSelScaleA, int opSelScaleB,
                                  StringRef name) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> operands;

  // F4 only intrinsic does not need matrix format args
  bool isF4 = name.ends_with(".f4");
  if (!isF4) {
    operands.push_back(b.i32_val(getWmmaF8F6F4MatrixFormat(aElType)));
  }
  operands.push_back(valA);
  if (!isF4) {
    operands.push_back(b.i32_val(getWmmaF8F6F4MatrixFormat(bElType)));
  }
  operands.push_back(valB);
  // C_mod is unused. Should be set to 0
  Value modC = b.i16_val(0);
  operands.push_back(modC);
  operands.push_back(valC);
  // Set scale_opsel bit. 0: Use scales in 0..15 lanes; 1: Use scales in 16..31
  // lanes
  operands.push_back(b.i32_val(opSelScaleA));
  int32_t scaleDTypeA = getWmmaScaleDataType(aScaleElType);
  assert(scaleDTypeA != -1);
  operands.push_back(b.i32_val(scaleDTypeA));
  operands.push_back(valScaleA);
  // Set scale_opsel bit.
  operands.push_back(b.i32_val(opSelScaleB));
  int32_t scaleDTypeB = getWmmaScaleDataType(bScaleElType);
  assert(scaleDTypeB != -1);
  operands.push_back(b.i32_val(scaleDTypeB));
  operands.push_back(valScaleB);
  // Set "Reuse matrix A" and "Reuse matrix B" to 0.
  operands.push_back(b.i1_val(0));
  operands.push_back(b.i1_val(0));
  auto wmmaIntrinsic = LLVM::createLLVMIntrinsicCallOp(
      rewriter, loc, name, valC.getType(), operands);
  return wmmaIntrinsic.getResult(0);
}

Value generateWMMAOp(ConversionPatternRewriter &rewriter, Location loc,
                     int version, Value valA, Value valB, Value valC,
                     Type aElType, Type bElType, Type dElType,
                     StringRef intrinsicName, std::optional<bool> tiedLower) {
  // Independent of wmma version because builtin functions are backward
  // compatible
  return generateWMMAIntrinsic(rewriter, loc, version, valA, valB, valC,
                               aElType, bElType, dElType, intrinsicName,
                               tiedLower);
}

static uint64_t packMN(uint32_t m, uint32_t n) {
  return (uint64_t(m) << 32) | uint64_t(n);
}

std::optional<int> findNextM(LinearLayout repLayout, int &reg, int elemsPerVec,
                             int m, int rank) {
  auto ctx = repLayout.getOutDimNames().begin()->getContext();

  const StringAttr kRegister = S("register");
  const StringAttr kLane = S("lane");
  const StringAttr kWarp = S("warp");
  const StringAttr kBlock = S("block");

  const int mOutIdx = (rank == 3) ? 1 : 0;
  auto getMForReg = [&](int r) -> int {
    auto out =
        repLayout.apply({{kRegister, r}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    return out[mOutIdx].second;
  };

  const int regLimit = repLayout.getInDimSize(kRegister);

  int nextM = getMForReg(reg);
  while (reg < regLimit && nextM == m) {
    reg += elemsPerVec;
    nextM = getMForReg(reg);
  }

  if (nextM == m)
    return std::nullopt;
  return nextM;
}

// Conduct the Dot conversion.
LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter,
                         const LLVMTypeConverter *typeConverter) {
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());
  int wmmaVer = wmmaLayout.getVersion();
  auto ctx = op.getContext();
  auto mnkDim = wmmaLayout.getInstrShape();

  auto loc = op.getLoc();
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  Value a = op.getA();
  Value b = op.getB();
  Value d = op.getD();
  auto aTensorTy = cast<RankedTensorType>(a.getType());
  auto bTensorTy = cast<RankedTensorType>(b.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  auto aElemTy = aTensorTy.getElementType();
  auto bElemTy = bTensorTy.getElementType();
  auto dElemTy = dTensorTy.getElementType();

  std::string intrinsicName;
  FailureOr<WmmaIntrinsic> maybeWmmaIntrinsic =
      wmmaLayout.getIsTransposed()
          ? WmmaIntrinsic::get(wmmaVer, mnkDim[1], mnkDim[0], mnkDim[2],
                               bElemTy, aElemTy, dElemTy)
          : WmmaIntrinsic::get(wmmaVer, mnkDim[0], mnkDim[1], mnkDim[2],
                               aElemTy, bElemTy, dElemTy);
  if (failed(maybeWmmaIntrinsic)) {
    return op.emitError("no matching matrix core intrinsic ")
           << "for wmma version " << wmmaVer << " with instruction shape ["
           << mnkDim[0] << ", " << mnkDim[1] << ", " << mnkDim[2]
           << "] and element types A=" << aElemTy << ", B=" << bElemTy
           << ", D=" << dElemTy << ". Check whether the wmma version,"
           << " instruction shape, and data types "
           << "are supported on the current AMD GPU architecture.";
  }

  unsigned kInstrSize = maybeWmmaIntrinsic->kDim;

  intrinsicName = maybeWmmaIntrinsic->name;

  auto resShape = dTensorTy.getShape();
  auto rank = resShape.size();
  auto K = aTensorTy.getShape()[rank - 1];

  auto tile = wmmaLayout.getTileLayout(rank);
  auto wmmaLL = triton::gpu::toLinearLayout(resShape, wmmaLayout);
  auto repLayout = wmmaRepLayoutForTensor(wmmaLL, tile);
  if (!repLayout.has_value()) {
    return op.emitError("failed to divide wmma layout by tile layout");
  }
  const unsigned numRepK = std::max(static_cast<unsigned>(K / kInstrSize), 1u);

  Value loadedA = adaptor.getA();
  Value loadedB = adaptor.getB();
  Value loadedC = adaptor.getC();
  auto aLayout = triton::gpu::toLinearLayout(aTensorTy);
  auto bLayout = triton::gpu::toLinearLayout(bTensorTy);

  auto aEnc = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  auto bEnc = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
  auto kBase = maybeWmmaIntrinsic->kBase;
  auto kDimTensorA = aTensorTy.getShape().back();
  auto kDimTensorB = bTensorTy.getShape()[rank - 2];

  auto dstElemTy = dTensorTy.getElementType();
  auto fc = unpackLLElements(loc, loadedC, rewriter);

  unsigned warpSize = gpu::lookupThreadsPerWarp(rewriter);
  constexpr unsigned vgprElemBitWidth = 32;
  unsigned paddedOutputElemSize =
      wmmaVer == 1 ? vgprElemBitWidth / dstElemTy.getIntOrFloatBitWidth() : 1;
  // compute number of output elements that each thread holds for one WMMA
  // instruction.
  auto elemsPerVec = mnkDim[0] * mnkDim[1] * paddedOutputElemSize / warpSize;
  auto dElemsToStorePerThread = mnkDim[0] * mnkDim[1] / warpSize;
  auto vecTy = vec_ty(dstElemTy, elemsPerVec);

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");
  StringAttr kBlock = S("block");

  llvm::DenseSet<uint64_t> mnProcessed;
  int tiedGroup = 1;

  for (int reg = 0; reg < repLayout->getInDimSize(kRegister);
       reg += dElemsToStorePerThread) {
    auto repIndices = repLayout->apply(
        {{kRegister, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    int batchIdx = (rank == 3 ? repIndices[0].second : 0);
    int m = repIndices[rank == 3 ? 1 : 0].second;
    int n = repIndices[rank == 3 ? 2 : 1].second;

    int nextMReg = reg + dElemsToStorePerThread;
    std::optional<int> nextM;
    if (paddedOutputElemSize == 2) {
      if (mnProcessed.count(packMN((uint32_t)m, (uint32_t)n))) {
        continue;
      }
      nextM = findNextM(*repLayout, nextMReg, dElemsToStorePerThread, m, rank);
      if (nextM.has_value()) {
        tiedGroup = 2;
        mnProcessed.insert(packMN((uint32_t)m, (uint32_t)n));
        mnProcessed.insert(packMN((uint32_t)nextM.value(), (uint32_t)n));
        intrinsicName += ".tied";
      }
    }

    Value acc = tb.undef(vecTy);
    auto selectRegValue = [&](int subTied) {
      return (subTied == 0) ? reg : nextMReg;
    };

    for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
      for (int subTied = 0; subTied < tiedGroup; ++subTied) {
        acc = tb.insert_element(vecTy, acc, fc[selectRegValue(subTied) + v],
                                tb.i32_val(v * paddedOutputElemSize + subTied));
      }
    }
    for (size_t k = 0; k < numRepK; ++k) {
      auto ha = getOperandVals(
          rewriter, typeConverter, aLayout, loadedA,
          /*opIdx*/ 0, rank, batchIdx, m, k, kInstrSize, kBase, kDimTensorA,
          aEnc, warpSize, /*opScale*/ nullptr, aTensorTy.getElementType(), loc);
      ha = prepareOperands(rewriter, ha, aTensorTy.getElementType(), wmmaVer,
                           kBase, loc);
      ha = maskRepeatedKLanes(rewriter, loc, aLayout, aEnc, ha, warpSize);

      auto hb = getOperandVals(
          rewriter, typeConverter, bLayout, loadedB,
          /*opIdx*/ 1, rank, batchIdx, n, k, kInstrSize, kBase, kDimTensorB,
          bEnc, warpSize, /*opScale*/ nullptr, bTensorTy.getElementType(), loc);
      hb = prepareOperands(rewriter, hb, bTensorTy.getElementType(), wmmaVer,
                           kBase, loc);
      hb = maskRepeatedKLanes(rewriter, loc, bLayout, bEnc, hb, warpSize);

      Value haNext;
      if (tiedGroup == 2) {
        haNext = getOperandVals(rewriter, typeConverter, aLayout, loadedA,
                                /*opIdx*/ 0, rank, batchIdx, nextM.value(), k,
                                kInstrSize, kBase, kDimTensorA, aEnc, warpSize,
                                nullptr, aTensorTy.getElementType(), loc);

        haNext = prepareOperands(rewriter, haNext, aTensorTy.getElementType(),
                                 wmmaVer, kBase, loc);
        haNext =
            maskRepeatedKLanes(rewriter, loc, aLayout, aEnc, haNext, warpSize);
      }

      for (int subTied = 0; subTied < tiedGroup; ++subTied) {
        auto optTied =
            tiedGroup == 2 ? std::optional<bool>(subTied != 0) : std::nullopt;
        auto aValue = subTied == 0 ? ha : haNext;
        acc = wmmaLayout.getIsTransposed()
                  ? generateWMMAOp(rewriter, loc, wmmaVer, hb, aValue, acc,
                                   bTensorTy.getElementType(),
                                   aTensorTy.getElementType(), dstElemTy,
                                   intrinsicName, optTied)
                  : generateWMMAOp(rewriter, loc, wmmaVer, aValue, hb, acc,
                                   aTensorTy.getElementType(),
                                   bTensorTy.getElementType(), dstElemTy,
                                   intrinsicName, optTied);
      }
    }
    for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
      for (int subTied = 0; subTied < tiedGroup; ++subTied) {
        fc[selectRegValue(subTied) + v] = tb.extract_element(
            dstElemTy, acc, tb.i32_val(v * paddedOutputElemSize + subTied));
      }
    }
  }

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      wmmaLayout.getContext(), SmallVector<Type>(fc.size(), dstElemTy));
  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

  rewriter.replaceOp(op, res);
  return success();
}

// For asymmetric WMMA (e.g. 32x16) with isTransposed=true the dot's operand A
// is 16xK and operand B is Kx32, but the intrinsic expects A as 32xK and B as
// Kx16, so we swap A and B and adjust the layouts for the operands and scale
// accordingly:
//  - layout adjustment uses AMDWmmaEncodingAttr::getOperandNonKDim which
//    returns the swapped nonK dim for asymmetric transposed WMMA
//  - WmmaScaleIntrinsic::get returns swapped kBaseA and kBaseB for
//    asymmetric transposed WMMA
//  - Intrinsic call created with (hb, sb, ha, sa) matches the expected LLVM
//  types
LogicalResult convertScaledDot(triton::DotScaledOp op,
                               triton::DotScaledOp::Adaptor adaptor,
                               ConversionPatternRewriter &rewriter,
                               const LLVMTypeConverter *typeConverter) {
  auto ctx = op.getContext();
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());
  int wmmaVer = wmmaLayout.getVersion();
  assert(wmmaVer == 3 && "Scaled dot not supported for wmma1/wmma2");
  auto mnkDim = wmmaLayout.getInstrShape();

  auto loc = op.getLoc();
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  Value a = op.getA();
  Value b = op.getB();
  Value aScale = op.getAScale();
  Value bScale = op.getBScale();
  Value d = op.getD();
  auto aTensorTy = cast<RankedTensorType>(a.getType());
  auto aScaleTensorTy = cast<RankedTensorType>(aScale.getType());
  auto bTensorTy = cast<RankedTensorType>(b.getType());
  auto bScaleTensorTy = cast<RankedTensorType>(bScale.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  const auto rank = aTensorTy.getShape().size();

  unsigned kInstrSize = mnkDim[2];

  auto aElemType = op.getAElemType();
  bool isFp4A = aElemType == triton::ScaleDotElemType::E2M1;
  int kDimA = isFp4A ? kInstrSize / 2 : kInstrSize;

  auto bElemType = op.getBElemType();
  bool isFp4B = bElemType == triton::ScaleDotElemType::E2M1;
  int kDimB = isFp4B ? kInstrSize / 2 : kInstrSize;

  unsigned scaleFactor = op.deduceScaleFactor();
  int kDimScale = kInstrSize / scaleFactor;

  bool isFp6A = (aElemType == triton::ScaleDotElemType::E2M3) ||
                (aElemType == triton::ScaleDotElemType::E3M2);
  bool isFp6B = (bElemType == triton::ScaleDotElemType::E2M3) ||
                (bElemType == triton::ScaleDotElemType::E3M2);
  if (isFp6A || isFp6B)
    return op.emitError("NYI: FP6 scaled dot");

  Type scaledAElemType =
      LLVM::AMD::scaleDotElemTypeToMLIRType(op.getContext(), op.getAElemType());
  Type scaledBElemType =
      LLVM::AMD::scaleDotElemTypeToMLIRType(op.getContext(), op.getBElemType());
  auto KBaseScale = scaleFactor == 32 ? 4 : 8;

  FailureOr<WmmaScaleIntrinsic> maybeWmmaScaleIntrinsic =
      WmmaScaleIntrinsic::get(wmmaVer, mnkDim[0], mnkDim[1], scaledAElemType,
                              scaledBElemType, dTensorTy.getElementType(),
                              (scaleFactor == 16) /*isScale16*/,
                              wmmaLayout.getIsTransposed());
  if (failed(maybeWmmaScaleIntrinsic)) {
    return op.emitError("no matching wmma scale intrinsic ")
           << "for wmma version " << wmmaVer << " with instruction shape ["
           << mnkDim[0] << ", " << mnkDim[1]
           << "] and element types A=" << aElemType << ", B=" << bElemType
           << ", D=" << dTensorTy.getElementType()
           << ". Check whether the wmma version, instruction shape, and data "
              "types are supported on the current AMD GPU architecture.";
  }

  auto kBaseA = maybeWmmaScaleIntrinsic->kBaseA;
  auto kBaseB = maybeWmmaScaleIntrinsic->kBaseB;

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");
  StringAttr kBlock = S("block");

  auto K = aTensorTy.getShape()[rank - 1];
  auto resShape = dTensorTy.getShape();

  auto tile = wmmaLayout.getTileLayout(rank);
  auto wmmaLL = triton::gpu::toLinearLayout(resShape, wmmaLayout);
  auto repLayout = wmmaRepLayoutForTensor(wmmaLL, tile);
  if (!repLayout.has_value()) {
    return op.emitError("failed to divide wmma layout by tile layout");
  }

  Value loadedA = adaptor.getA();
  Value loadedAScale = adaptor.getAScale();
  Value loadedB = adaptor.getB();
  Value loadedBScale = adaptor.getBScale();
  Value loadedC = adaptor.getC();
  const unsigned numRepK = std::max(static_cast<unsigned>(K / kDimA), 1u);

  auto aLayout = triton::gpu::toLinearLayout(aTensorTy);
  auto bLayout = triton::gpu::toLinearLayout(bTensorTy);
  auto aScaleLayout = triton::gpu::toLinearLayout(aScaleTensorTy);
  auto bScaleLayout = triton::gpu::toLinearLayout(bScaleTensorTy);

  auto dstElemTy = dTensorTy.getElementType();
  auto fc = unpackLLElements(loc, loadedC, rewriter);

  unsigned warpSize = gpu::lookupThreadsPerWarp(rewriter);

  auto aEnc = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  auto bEnc = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());

  auto kDimTensorA = aTensorTy.getShape().back();
  auto kDimTensorB = bTensorTy.getShape()[rank - 2];

  // compute number of output elements that each thread holds for one WMMA
  // instruction.
  auto elemsPerVec = mnkDim[0] * mnkDim[1] / warpSize;
  auto dElemsToStorePerThread = mnkDim[0] * mnkDim[1] / warpSize;
  auto vecTy = vec_ty(dstElemTy, elemsPerVec);

  for (int reg = 0; reg < repLayout->getInDimSize(kRegister);
       reg += elemsPerVec) {

    auto repIndices = repLayout->apply(
        {{kRegister, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    int batchIdx = (rank == 3 ? repIndices[0].second : 0);
    int m = repIndices[rank == 3 ? 1 : 0].second;
    int n = repIndices[rank == 3 ? 2 : 1].second;

    Value acc = tb.undef(vecTy);
    for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
      acc = tb.insert_element(vecTy, acc, fc[reg + v], tb.i32_val(v));
    }
    for (size_t k = 0; k < numRepK; k++) {
      int scaleOpSelA = 0;
      int scaleOpSelB = 0;

      auto ha = getOperandVals(rewriter, typeConverter, aLayout, loadedA,
                               /*opIdx*/ 0, rank, batchIdx, m, k, kDimA, kBaseA,
                               kDimTensorA, aEnc, warpSize, /*opSel*/ nullptr,
                               aTensorTy.getElementType(), loc);
      ha = prepareOperands(rewriter, ha, aTensorTy.getElementType(), wmmaVer,
                           kBaseA, loc);

      auto hb = getOperandVals(rewriter, typeConverter, bLayout, loadedB,
                               /*opIdx*/ 1, rank, batchIdx, n, k, kDimB, kBaseB,
                               kDimTensorB, bEnc, warpSize, /*opSel*/ nullptr,
                               bTensorTy.getElementType(), loc);
      hb = prepareOperands(rewriter, hb, bTensorTy.getElementType(), wmmaVer,
                           kBaseB, loc);

      ha = maskRepeatedKLanes(rewriter, loc, aLayout, aEnc, ha, warpSize);
      hb = maskRepeatedKLanes(rewriter, loc, bLayout, bEnc, hb, warpSize);

      auto sa = getOperandVals(
          rewriter, typeConverter, aScaleLayout, loadedAScale,
          /*opIdx*/ 0, rank, batchIdx, m, k, kDimScale, KBaseScale,
          /*kDimTensor*/ kDimScale, aEnc, warpSize, &scaleOpSelA,
          aScaleTensorTy.getElementType(), loc,
          /*isScale*/ true);
      sa = prepareOperands(rewriter, sa, aScaleTensorTy.getElementType(),
                           wmmaVer, KBaseScale, loc, /*isScale=*/true);

      auto sb = getOperandVals(
          rewriter, typeConverter, bScaleLayout, loadedBScale,
          /*opIdx*/ 0, rank, batchIdx, n, k, kDimScale, KBaseScale,
          /*kDimTensor*/ kDimScale, bEnc, warpSize, &scaleOpSelB,
          bScaleTensorTy.getElementType(), loc,
          /*isScale*/ true);
      sb = prepareOperands(rewriter, sb, bScaleTensorTy.getElementType(),
                           wmmaVer, KBaseScale, loc, /*isScale=*/true);

      StringRef intrinsicName = maybeWmmaScaleIntrinsic->name;
      acc = wmmaLayout.getIsTransposed()
                ? generateScaledWMMAIntrinsic(
                      rewriter, loc, hb, sb, ha, sa, acc, scaledBElemType,
                      bScaleTensorTy.getElementType(), scaledAElemType,
                      aScaleTensorTy.getElementType(), dstElemTy, scaleOpSelB,
                      scaleOpSelA, intrinsicName)
                : generateScaledWMMAIntrinsic(
                      rewriter, loc, ha, sa, hb, sb, acc, scaledAElemType,
                      aScaleTensorTy.getElementType(), scaledBElemType,
                      bScaleTensorTy.getElementType(), dstElemTy, scaleOpSelA,
                      scaleOpSelB, intrinsicName);
    }
    for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
      fc[reg + v] = tb.extract_element(dstElemTy, acc, tb.i32_val(v));
    }
  }

  Type structTy = LLVM::LLVMStructType::getLiteral(
      wmmaLayout.getContext(), SmallVector<Type>(fc.size(), dstElemTy));
  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

  rewriter.replaceOp(op, res);
  return success();
}

} // namespace

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return cast<RankedTensorType>(tensor.getType());
  };

  assert(isa<DotOperandEncodingAttr>(rankedTType(op.getA()).getEncoding()) &&
         isa<DotOperandEncodingAttr>(rankedTType(op.getB()).getEncoding()) &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(isa<AMDWmmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support $c with a wmma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  return convertDot(op, adaptor, rewriter, typeConverter);
}

LogicalResult convertScaledWMMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter) {
  assert(isa<LinearEncodingAttr>(op.getAScale().getType().getEncoding()) &&
         isa<LinearEncodingAttr>(op.getBScale().getType().getEncoding()) &&
         "Both LhsScale and RhsScale should be linear layout.");

  auto cTensorTy = op.getC().getType();
  auto dTensorTy = op.getD().getType();
  assert(isa<AMDWmmaEncodingAttr>(cTensorTy.getEncoding()) &&
         "Currently, we only support C with a wmma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's C operand should pass the same number of values as D.");

  return convertScaledDot(op, adaptor, rewriter, typeConverter);
}
} // namespace mlir::triton::AMD
