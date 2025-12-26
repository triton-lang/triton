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

#define S(v) StringAttr::get(ctx, (v))
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::LinearEncodingAttr;

Value prepareOperands(ConversionPatternRewriter &rewriter, Value rawElems,
                      Type type, int wmmaVer, int kBase, Location loc) {
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
    auto elems =
        kBase * type.getIntOrFloatBitWidth() / i32_ty.getIntOrFloatBitWidth();
    assert(elems >= 1 && "unexpected number of elements");
    if (elems == 1)
      convertedElems = tb.bitcast(rawElems, i32_ty);
    else
      convertedElems = tb.bitcast(rawElems, vec_ty(i32_ty, elems));
  }
  return convertedElems;
}

Value getOperandVals(ConversionPatternRewriter &rewriter,
                     const LLVMTypeConverter *typeConverter,
                     LinearLayout dotLayout, Value value, int opIdx, int rank,
                     int batch, int nonK, int kIdx, int kInstSize, int kBase,
                     int kPadding, int *opSel, Type type, Location loc,
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
  const int kElemIdx = kIdx * kInstSize;

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
  const int validK = kBase - kPadding;

  Value zero;
  if (kPadding > 0) {
    zero = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                    rewriter.getZeroAttr(elemTy));
  }

  for (int k = 0; k < kBase; ++k) {
    Value elem = (k < validK) ? elems[startReg + k] : zero;
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

  LLVM::FastmathFlagsAttr defaultFlags{};
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
    // arguments for v3:
    // int:          %A_mod, %A, %B_mod, %B, %C, %A_reuse, %B_reuse
    // f32/f16/bf16: %A_mod, %A, %B_mod, %B, %C_mod, %C, %A_reuse, %B_reuse
    // f8/bf8:       %A, %B, %C_mod, %C, %A_reuse, %B_reuse
    if (aElType.isInteger())
      operands.push_back(b.int_val(1, !aElType.isUnsignedInteger()));
    else if (aElType.isFloat(16) || aElType.isF32())
      operands.push_back(b.int_val(1, 0));
    operands.push_back(valA);

    if (bElType.isInteger())
      operands.push_back(b.int_val(1, !bElType.isUnsignedInteger()));
    else if (bElType.isFloat(16) || bElType.isF32())
      operands.push_back(b.int_val(1, 0));
    operands.push_back(valB);

    if (bElType.isFloat(16) || bElType.isF32() || aElType.isFloat(8))
      operands.push_back(b.int_val(16, 0));
    operands.push_back(valC);

    operands.push_back(b.i1_val(0));
    operands.push_back(b.i1_val(0));
  }

  auto wmmaIntrinsic = LLVM::createLLVMIntrinsicCallOp(
      rewriter, loc, name, valC.getType(), operands);
  return wmmaIntrinsic.getResult(0);
}

Value generateScaledWMMAIntrinsic(ConversionPatternRewriter &rewriter,
                                  Location loc, Value valA, Value valScaleA,
                                  Value valB, Value valScaleB, Value valC,
                                  Type aElType, Type bElType, Type dElType,
                                  int scaleKWidth, int opSelScaleA,
                                  int opSelScaleB) {
  assert(scaleKWidth == 4 || scaleKWidth == 8);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  std::string name = "llvm.amdgcn.wmma.scale";
  if (scaleKWidth == 8) {
    name += "16";
  }
  name += ".f32.16x16x128.f8f6f4";

  LLVM::FastmathFlagsAttr defaultFlags{};
  SmallVector<Value> operands;

  // Reference: llvm/include/llvm/IR/IntrinsicsAMDGPU.td,
  // int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4
  Value fmtA = b.i32_val(getWmmaF8F6F4MatrixFormat(aElType));
  operands.push_back(fmtA);
  operands.push_back(valA);
  Value fmtB = b.i32_val(getWmmaF8F6F4MatrixFormat(bElType));
  operands.push_back(fmtB);
  operands.push_back(valB);
  // C_mod is unused. Should be set to 0
  Value modC = b.i16_val(0);
  operands.push_back(modC);
  operands.push_back(valC);
  // Set scale_opsel bit. 0: Use scales in 0..15 lanes; 1: Use scales in 16..31
  // lanes
  operands.push_back(b.i32_val(opSelScaleA));
  // Set a_scale_fmt to 0 = E8M0
  operands.push_back(b.i32_val(0));
  operands.push_back(valScaleA);
  // Set scale_opsel bit.
  operands.push_back(b.i32_val(opSelScaleB));
  // Set b_scale fmt to 0 = E8M0
  operands.push_back(b.i32_val(0));
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

  const auto kDimOperandSize = aTensorTy.getShape().back();

  std::string intrinsicName;
  FailureOr<WmmaIntrinsic> maybeWmmaIntrinsic =
      wmmaLayout.getIsTransposed()
          ? WmmaIntrinsic::get(wmmaVer, mnkDim[1], mnkDim[0], mnkDim[2],
                               bElemTy, aElemTy, dElemTy)
          : WmmaIntrinsic::get(wmmaVer, mnkDim[0], mnkDim[1], mnkDim[2],
                               aElemTy, bElemTy, dElemTy);
  if (failed(maybeWmmaIntrinsic)) {
    return op.emitError(
        "no matching matrix core intrinsic due to unsupported element type");
  }

  unsigned kDim = maybeWmmaIntrinsic->kDim;

  auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
  intrinsicName = maybeWmmaIntrinsic->name;

  auto resShape = dTensorTy.getShape();
  auto rank = resShape.size();
  auto K = aTensorTy.getShape()[rank - 1];

  auto tile = wmmaLayout.getTileLayout(rank);
  auto wmmaLL = triton::gpu::toLinearLayout(resShape, wmmaLayout);
  auto quot = divideLeft(wmmaLL, tile).value();
  auto repLayout = zerosLike(tile) * quot;
  const unsigned numRepK = std::max(static_cast<unsigned>(K / kDim), 1u);

  Value loadedA = adaptor.getA();
  Value loadedB = adaptor.getB();
  Value loadedC = adaptor.getC();
  auto aLayout = triton::gpu::toLinearLayout(aTensorTy);
  auto bLayout = triton::gpu::toLinearLayout(bTensorTy);

  // If kDim > kDimTensor, we need add zeros to the kBase vector. The amount of
  // zeros is determined by kBase * (1 - kDimTensor / kDim)
  auto kBase = maybeWmmaIntrinsic->kBase;
  auto kDimTensor = aTensorTy.getShape().back();
  auto paddingFactor = kDim > kDimTensor ? (kDim / kDimTensor) : 1;
  auto kPadding = kBase - kBase / paddingFactor;

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

  for (int reg = 0; reg < repLayout.getInDimSize(kRegister);
       reg += dElemsToStorePerThread) {
    auto repIndices = repLayout.apply(
        {{kRegister, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    int b = (rank == 3 ? repIndices[0].second : 0);
    int m = repIndices[rank == 3 ? 1 : 0].second;
    int n = repIndices[rank == 3 ? 2 : 1].second;

    int nextMReg = reg + dElemsToStorePerThread;
    std::optional<int> nextM;
    if (paddedOutputElemSize == 2) {
      if (mnProcessed.count(packMN((uint32_t)m, (uint32_t)n))) {
        continue;
      }
      nextM = findNextM(repLayout, nextMReg, dElemsToStorePerThread, m, rank);
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
      auto ha =
          getOperandVals(rewriter, typeConverter, aLayout, loadedA,
                         /*opIdx*/ 0, rank, b, m, k, kDim, kBase, kPadding,
                         /*opScale*/ nullptr, aTensorTy.getElementType(), loc);
      ha = prepareOperands(rewriter, ha, aTensorTy.getElementType(), wmmaVer,
                           kBase, loc);

      auto hb =
          getOperandVals(rewriter, typeConverter, bLayout, loadedB,
                         /*opIdx*/ 1, rank, b, n, k, kDim, kBase, kPadding,
                         /*opScale*/ nullptr, bTensorTy.getElementType(), loc);
      hb = prepareOperands(rewriter, hb, bTensorTy.getElementType(), wmmaVer,
                           kBase, loc);

      Value haNext;
      if (tiedGroup == 2) {
        haNext =
            getOperandVals(rewriter, typeConverter, aLayout, loadedA,
                           /*opIdx*/ 0, rank, b, nextM.value(), k, kDim, kBase,
                           kPadding, nullptr, aTensorTy.getElementType(), loc);

        haNext = prepareOperands(rewriter, haNext, aTensorTy.getElementType(),
                                 wmmaVer, kBase, loc);
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
  auto elemTy = aTensorTy.getElementType();
  const auto rank = aTensorTy.getShape().size();

  unsigned kDim = mnkDim[2];
  unsigned kBase = 64;

  bool isFp4A = op.getAElemType() == triton::ScaleDotElemType::E2M1;
  int kBaseA = isFp4A ? kBase / 2 : kBase;
  int kDimA = isFp4A ? kDim / 2 : kDim;
  int scaleFactorA = isFp4A ? 16 : 32;

  bool isFp4B = op.getBElemType() == triton::ScaleDotElemType::E2M1;
  int kBaseB = isFp4B ? kBase / 2 : kBase;
  int kDimB = isFp4B ? kDim / 2 : kDim;
  int scaleFactorB = isFp4B ? 16 : 32;

  bool isFp6A = (op.getAElemType() == triton::ScaleDotElemType::E2M3) ||
                (op.getAElemType() == triton::ScaleDotElemType::E3M2);
  bool isFp6B = (op.getBElemType() == triton::ScaleDotElemType::E2M3) ||
                (op.getBElemType() == triton::ScaleDotElemType::E3M2);
  if (isFp6A || isFp6B)
    return op.emitError("NYI: FP6 scaled dot");

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");
  StringAttr kBlock = S("block");

  auto K = aTensorTy.getShape()[rank - 1];
  auto resShape = dTensorTy.getShape();
  SmallVector<unsigned> repA;
  SmallVector<unsigned> repB;

  auto tile = wmmaLayout.getTileLayout(rank);
  auto wmmaLL = triton::gpu::toLinearLayout(resShape, wmmaLayout);
  auto quot = divideLeft(wmmaLL, tile).value();
  auto repLayout = zerosLike(tile) * quot;

  Value loadedA = adaptor.getA();
  Value loadedAScale = adaptor.getAScale();
  Value loadedB = adaptor.getB();
  Value loadedBScale = adaptor.getBScale();
  Value loadedC = adaptor.getC();
  const unsigned numRepK = std::max(static_cast<unsigned>(K / kDimA), 1u);

  // If kDim > kDimTensor, we need add zeros to the kBase vector. The amount of
  // zeros is determined by kBase * (1 - kDimTensor / kDim)
  auto kDimTensorA = aTensorTy.getShape().back();
  auto paddingFactor = kDimA > kDimTensorA ? (kDimA / kDimTensorA) : 1;
  auto kPaddingA = kBaseA - kBaseA / paddingFactor;
  auto kPaddingB = kBaseB - kBaseB / paddingFactor;
  auto KBaseScale = 4;
  auto aLayout = triton::gpu::toLinearLayout(aTensorTy);
  auto bLayout = triton::gpu::toLinearLayout(bTensorTy);
  auto aScaleLayout = triton::gpu::toLinearLayout(aScaleTensorTy);
  auto bScaleLayout = triton::gpu::toLinearLayout(bScaleTensorTy);

  auto dstElemTy = dTensorTy.getElementType();
  auto fc = unpackLLElements(loc, loadedC, rewriter);

  Type scaledAElemType =
      LLVM::AMD::scaleDotElemTypeToMLIRType(op.getContext(), op.getAElemType());
  Type scaledBElemType =
      LLVM::AMD::scaleDotElemTypeToMLIRType(op.getContext(), op.getBElemType());

  unsigned warpSize = gpu::lookupThreadsPerWarp(rewriter);
  // compute number of output elements that each thread holds for one WMMA
  // instruction.
  auto elemsPerVec = mnkDim[0] * mnkDim[1] / warpSize;
  auto dElemsToStorePerThread = mnkDim[0] * mnkDim[1] / warpSize;
  auto vecTy = vec_ty(dstElemTy, elemsPerVec);

  for (int reg = 0; reg < repLayout.getInDimSize(kRegister);
       reg += elemsPerVec) {

    auto repIndices = repLayout.apply(
        {{kRegister, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    int b = (rank == 3 ? repIndices[0].second : 0);
    int m = repIndices[rank == 3 ? 1 : 0].second;
    int n = repIndices[rank == 3 ? 2 : 1].second;

    Value acc = tb.undef(vecTy);
    for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
      acc = tb.insert_element(vecTy, acc, fc[reg + v], tb.i32_val(v));
    }
    for (size_t k = 0; k < numRepK; k++) {
      int scaleOpSelA = 0;
      int scaleOpSelB = 0;

      auto ha =
          getOperandVals(rewriter, typeConverter, aLayout, loadedA,
                         /*opIdx*/ 0, rank, b, m, k, kDimA, kBaseA, kPaddingA,
                         /*opSel*/ nullptr, aTensorTy.getElementType(), loc);
      ha = prepareOperands(rewriter, ha, aTensorTy.getElementType(), wmmaVer,
                           kBaseA, loc);

      auto hb =
          getOperandVals(rewriter, typeConverter, bLayout, loadedB,
                         /*opIdx*/ 1, rank, b, n, k, kDimB, kBaseB, kPaddingB,
                         /*opSel*/ nullptr, bTensorTy.getElementType(), loc);
      hb = prepareOperands(rewriter, hb, bTensorTy.getElementType(), wmmaVer,
                           kBaseB, loc);

      auto sa = getOperandVals(
          rewriter, typeConverter, aScaleLayout, loadedAScale,
          /*opIdx*/ 0, rank, b, m, k, kDimA / scaleFactorA, KBaseScale,
          /*padding*/ 0, &scaleOpSelA, aScaleTensorTy.getElementType(), loc,
          /*isScale*/ true);
      sa = prepareOperands(rewriter, sa, aScaleTensorTy.getElementType(),
                           wmmaVer, KBaseScale, loc);

      auto sb = getOperandVals(
          rewriter, typeConverter, bScaleLayout, loadedBScale,
          /*opIdx*/ 0, rank, b, n, k, kDimB / scaleFactorB, KBaseScale,
          /*padding*/ 0, &scaleOpSelB, bScaleTensorTy.getElementType(), loc,
          /*isScale*/ true);
      sb = prepareOperands(rewriter, sb, bScaleTensorTy.getElementType(),
                           wmmaVer, KBaseScale, loc);

      acc =
          wmmaLayout.getIsTransposed()
              ? generateScaledWMMAIntrinsic(rewriter, loc, hb, sb, ha, sa, acc,
                                            scaledBElemType, scaledAElemType,
                                            dstElemTy, KBaseScale, scaleOpSelB,
                                            scaleOpSelA)
              : generateScaledWMMAIntrinsic(rewriter, loc, ha, sa, hb, sb, acc,
                                            scaledAElemType, scaledBElemType,
                                            dstElemTy, KBaseScale, scaleOpSelA,
                                            scaleOpSelB);
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

  auto loc = op.getLoc();
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());
  return convertScaledDot(op, adaptor, rewriter, typeConverter);
}
} // namespace mlir::triton::AMD
