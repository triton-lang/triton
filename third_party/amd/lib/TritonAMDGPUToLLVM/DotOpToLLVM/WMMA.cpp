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

#include "../PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUTransforms/WmmaGroup.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::AMD {
namespace {

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::LinearEncodingAttr;

using ValueTable = std::map<std::tuple<unsigned, unsigned, unsigned>, Value>;

ValueTable getValuesFromDotOperandLayoutStruct(
    ConversionPatternRewriter &rewriter, const LLVMTypeConverter *typeConverter,
    int wmmaVer, Value value, int batch, int n0, int n1, int kBase, Type type,
    Location loc) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto elems = unpackLLElements(loc, value, rewriter);
  ValueTable vals;
  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < n0; i++) {
      for (int j = 0; j < n1; j++) {
        Type elemTy = typeConverter->convertType(type);
        Type ty = vec_ty(elemTy, kBase);
        Value rawElems = tb.undef(ty);
        for (int k = 0; k < kBase; ++k) {
          rawElems = tb.insert_element(
              ty, rawElems,
              elems[n0 * n1 * kBase * b + kBase * (n1 * i + j) + k],
              tb.i32_val(k));
        }

        Value convertedElems;
        if (type.isF32() || type.isF16()) {
          convertedElems = rawElems;
        } else if (type.isBF16()) {
          convertedElems = rawElems;
          // Before wmma v3, bf16 is converted to i16
          if (wmmaVer < 3)
            convertedElems = tb.bitcast(rawElems, vec_ty(i16_ty, kBase));
        } else if (kBase == 4 && type.getIntOrFloatBitWidth() == 8) {
          convertedElems = tb.bitcast(rawElems, i32_ty);
        } else {
          convertedElems = tb.bitcast(
              rawElems, vec_ty(i32_ty, kBase * type.getIntOrFloatBitWidth() /
                                           i32_ty.getIntOrFloatBitWidth()));
        }
        vals[{b, i, j}] = convertedElems;
      }
    }
  }
  return vals;
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
                                  int scaleKWidth) {
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
  // Set a_scale mantissa to zero as use E8M0 format (no mantissa bits)
  operands.push_back(b.i32_val(0));
  // Set a_scale_fmt to 0 = E8M0
  operands.push_back(b.i32_val(0));
  operands.push_back(valScaleA);
  // Set b_scale mantissa to zero as we use E8M0 format (no mantissa bits)
  operands.push_back(b.i32_val(0));
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

// Conduct the Dot conversion.
LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter,
                         const LLVMTypeConverter *typeConverter) {
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());
  int wmmaVer = wmmaLayout.getVersion();
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();
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
  FailureOr<WmmaIntrinsic> maybeWmmaIntrinsic = WmmaIntrinsic::get(
      wmmaVer, mnkDim[0], mnkDim[1], mnkDim[2], aElemTy, bElemTy, dElemTy);
  if (failed(maybeWmmaIntrinsic)) {
    return op.emitError(
        "no matching matrix core intrinsic due to unsupported element type");
  }

  unsigned kDim = maybeWmmaIntrinsic->kDim;

  auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
  intrinsicName = maybeWmmaIntrinsic->name;

  auto repA = wmmaLayout.getRepForOperand(aTensorTy.getShape(), kDim, 0);
  auto repB = wmmaLayout.getRepForOperand(bTensorTy.getShape(), kDim, 1);

  assert(repA[2] == repB[1]);

  Value loadedA = adaptor.getA();
  Value loadedB = adaptor.getB();
  Value loadedC = adaptor.getC();
  auto numRepM = repA[1];
  auto numRepN = repB[2];
  auto numRepK = repA[2];
  auto numRepB = repA[0];

  int kBase = maybeWmmaIntrinsic->kBase;
  ValueTable ha = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, wmmaVer, loadedA, numRepB, numRepM, numRepK,
      kBase, aTensorTy.getElementType(), loc);
  ValueTable hb = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, wmmaVer, loadedB, numRepB, numRepN, numRepK,
      kBase, aTensorTy.getElementType(), loc);
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

  bool tied = numRepM % 2 == 0 && paddedOutputElemSize == 2;
  int tiedGroup = tied ? 2 : 1;
  if (tied)
    intrinsicName += ".tied";

  for (int b = 0; b < numRepB; ++b) {
    for (int m = 0; m < numRepM / tiedGroup; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        auto batchOffIdx = b * numRepM * numRepN * dElemsToStorePerThread;
        auto nRepOffId = n * dElemsToStorePerThread;
        auto nBatchOffSum = nRepOffId + batchOffIdx;

        Value acc = tb.undef(vecTy);
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          for (int subTied = 0; subTied < tiedGroup; ++subTied) {
            auto mRepOffId =
                (m * tiedGroup + subTied) * numRepN * dElemsToStorePerThread;
            auto fcThreadOffIdx = nBatchOffSum + mRepOffId;
            acc = tb.insert_element(
                vecTy, acc, fc[fcThreadOffIdx + v],
                tb.i32_val(v * paddedOutputElemSize + subTied));
          }
        }
        for (size_t k = 0; k < numRepK; ++k) {
          for (int subTied = 0; subTied < tiedGroup; ++subTied) {
            auto optTied =
                tied ? std::optional<bool>(subTied != 0) : std::nullopt;
            acc = wmmaLayout.getIsTransposed()
                      ? generateWMMAOp(rewriter, loc, wmmaVer, hb[{b, n, k}],
                                       ha[{b, m * tiedGroup + subTied, k}], acc,
                                       bTensorTy.getElementType(),
                                       aTensorTy.getElementType(), dstElemTy,
                                       intrinsicName, optTied)
                      : generateWMMAOp(rewriter, loc, wmmaVer,
                                       ha[{b, m * tiedGroup + subTied, k}],
                                       hb[{b, n, k}], acc,
                                       aTensorTy.getElementType(),
                                       bTensorTy.getElementType(), dstElemTy,
                                       intrinsicName, optTied);
          }
        }
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          for (int subTied = 0; subTied < tiedGroup; ++subTied) {
            auto mRepOffId =
                (m * tiedGroup + subTied) * numRepN * dElemsToStorePerThread;
            auto fcThreadOffIdx = nBatchOffSum + mRepOffId;
            fc[fcThreadOffIdx + v] = tb.extract_element(
                dstElemTy, acc, tb.i32_val(v * paddedOutputElemSize + subTied));
          }
        }
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
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());
  int wmmaVer = wmmaLayout.getVersion();
  assert(wmmaVer == 3 && "Scaled dot not supported for wmma1/wmma2");
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();
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

  unsigned kDim = mnkDim[2];
  unsigned kBase = 64;

  bool isFp4A = op.getAElemType() == triton::ScaleDotElemType::E2M1;
  int kBaseA = isFp4A ? kBase / 2 : kBase;
  int kDimA = isFp4A ? kDim / 2 : kDim;

  bool isFp4B = op.getBElemType() == triton::ScaleDotElemType::E2M1;
  int kBaseB = isFp4B ? kBase / 2 : kBase;
  int kDimB = isFp4B ? kDim / 2 : kDim;

  auto repA = wmmaLayout.getRepForOperand(aTensorTy.getShape(), kDimA, 0);
  auto repB = wmmaLayout.getRepForOperand(bTensorTy.getShape(), kDimB, 1);

  assert(repA[2] == repB[1]);

  Value loadedA = adaptor.getA();
  Value loadedAScale = adaptor.getAScale();
  Value loadedB = adaptor.getB();
  Value loadedBScale = adaptor.getBScale();
  Value loadedC = adaptor.getC();
  auto numRepM = repA[1];
  auto numRepN = repB[2];
  auto numRepK = repA[2];
  auto numRepB = repA[0];

  auto scaleShapeA = aScaleTensorTy.getShape();
  constexpr int scaleKWidthA = 4;
  auto scaleShapeB = bScaleTensorTy.getShape();
  constexpr int scaleKWidthB = 4;

  ValueTable ha = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, wmmaVer, loadedA, numRepB, numRepM, numRepK,
      kBaseA, aTensorTy.getElementType(), loc);
  ValueTable hb = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, wmmaVer, loadedB, numRepB, numRepN, numRepK,
      kBaseB, bTensorTy.getElementType(), loc);
  ValueTable sa = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, wmmaVer, loadedAScale, numRepB, numRepM, numRepK,
      scaleKWidthA, aScaleTensorTy.getElementType(), loc);
  ValueTable sb = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, wmmaVer, loadedBScale, numRepB, numRepN, numRepK,
      scaleKWidthB, bScaleTensorTy.getElementType(), loc);
  auto dstElemTy = dTensorTy.getElementType();
  auto fc = unpackLLElements(loc, loadedC, rewriter);

  Type scaledAElemType =
      LLVM::AMD::scaleDotElemTypeToMLIRType(op.getContext(), op.getAElemType());
  Type scaledBElemType =
      LLVM::AMD::scaleDotElemTypeToMLIRType(op.getContext(), op.getBElemType());

  unsigned warpSize = gpu::lookupThreadsPerWarp(rewriter);
  constexpr unsigned vgprElemBitWidth = 32;
  // compute number of output elements that each thread holds for one WMMA
  // instruction.
  auto elemsPerVec = mnkDim[0] * mnkDim[1] / warpSize;
  auto dElemsToStorePerThread = mnkDim[0] * mnkDim[1] / warpSize;
  auto vecTy = vec_ty(dstElemTy, elemsPerVec);
  for (int b = 0; b < numRepB; ++b) {
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        auto batchOffIdx = b * numRepM * numRepN * dElemsToStorePerThread;
        auto mRepOffId = m * numRepN * dElemsToStorePerThread;
        auto nRepOffId = n * dElemsToStorePerThread;
        auto fcThreadOffIdx = batchOffIdx + mRepOffId + nRepOffId;

        Value acc = tb.undef(vecTy);
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          acc = tb.insert_element(vecTy, acc, fc[fcThreadOffIdx + v],
                                  tb.i32_val(v));
        }
        for (size_t k = 0; k < numRepK; k++) {
          acc = wmmaLayout.getIsTransposed()
                    ? generateScaledWMMAIntrinsic(
                          rewriter, loc, hb[{b, n, k}], sb[{b, n, k}],
                          ha[{b, m, k}], sa[{b, m, k}], acc, scaledBElemType,
                          scaledAElemType, dstElemTy, scaleKWidthA)
                    : generateScaledWMMAIntrinsic(
                          rewriter, loc, ha[{b, m, k}], sa[{b, m, k}],
                          hb[{b, n, k}], sb[{b, n, k}], acc, scaledAElemType,
                          scaledBElemType, dstElemTy, scaleKWidthB);
        }
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          fc[fcThreadOffIdx + v] =
              tb.extract_element(dstElemTy, acc, tb.i32_val(v));
        }
      }
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
