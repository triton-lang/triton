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

namespace mlir::triton::AMD {
namespace {

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;

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
        if (type.isF16() || (wmmaVer == 3 && type.isBF16())) {
          convertedElems = rawElems;
        } else if (type.isBF16()) {
          convertedElems = tb.bitcast(rawElems, vec_ty(i16_ty, kBase));
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

std::string getTypeStr(Type ty) {
  std::string scalarName;
  if (ty.isF32()) {
    scalarName = "f32";
  } else if (ty.isF16()) {
    scalarName = "f16";
  } else if (ty.isBF16()) {
    scalarName = "bf16";
  } else if (llvm::isa<Float8E4M3FNType>(ty)) {
    scalarName = "fp8";
  } else if (llvm::isa<Float8E5M2Type>(ty)) {
    scalarName = "bf8";
  } else if (ty.isInteger(32)) {
    scalarName = "i32";
  } else if (ty.isInteger(16)) {
    scalarName = "i16";
  } else if (ty.isInteger(8)) {
    scalarName = "iu8";
  } else if (ty.isInteger(4)) {
    scalarName = "iu4";
  } else if (auto vecTy = dyn_cast<VectorType>(ty)) {
    auto elemType = vecTy.getElementType();
    auto numElems = vecTy.getNumElements();
    scalarName = "v" + std::to_string(numElems) + getTypeStr(elemType);
  } else {
    llvm::report_fatal_error("WMMA data type not supported");
  }
  return scalarName;
}

std::string addInstructionSuffix(std::string intrinsicName, unsigned kBase,
                                 unsigned elemsPerVec, Type aElTy, Type bElTy,
                                 Type dElTy, bool tied) {
  if (tied) {
    intrinsicName += ".tied";
  } else {
    if (isa<FloatType>(aElTy) && aElTy.getIntOrFloatBitWidth() == 8)
      intrinsicName += "." + getTypeStr(bElTy);
    intrinsicName += ".v" + std::to_string(elemsPerVec) + getTypeStr(dElTy);
    intrinsicName += ".v" + std::to_string(kBase) + getTypeStr(aElTy);
  }

  return intrinsicName;
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
    // int:       %A_mod, %A, %B_mod, %B, %C, %A_reuse, %B_reuse
    // fp16/bf16: %A_mod, %A, %B_mod, %B, %C_mod, %C, %A_reuse, %B_reuse
    // fp8/bf8:   %A, %B, %C_mod, %C, %A_reuse, %B_reuse
    if (aElType.isInteger())
      operands.push_back(b.int_val(1, !aElType.isUnsignedInteger()));
    else if (aElType.isBF16() || aElType.isF16())
      operands.push_back(b.int_val(1, 0));
    operands.push_back(valA);

    if (bElType.isInteger())
      operands.push_back(b.int_val(1, !bElType.isUnsignedInteger()));
    else if (bElType.isBF16() || bElType.isF16())
      operands.push_back(b.int_val(1, 0));
    operands.push_back(valB);

    if ((bElType.isBF16() || bElType.isF16()) || aElType.isInteger())
      operands.push_back(b.int_val(16, 0));
    operands.push_back(valC);

    operands.push_back(b.i1_val(0));
    operands.push_back(b.i1_val(0));
  }

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
  FailureOr<WmmaIntrinsic> maybeWmmaIntrinsic =
      WmmaIntrinsic::selectFor(wmmaVer, mnkDim[0], mnkDim[1], kDimOperandSize,
                               aElemTy, bElemTy, dElemTy);
  if (failed(maybeWmmaIntrinsic)) {

    return op.emitError(
        "no matching matrix core intrinsic due to unsupported element type");
  }

  unsigned kDim = maybeWmmaIntrinsic->kDim;

  auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
  intrinsicName = maybeWmmaIntrinsic->name;

  auto repA = wmmaLayout.getRepForOperand(aTensorTy.getShape(), aElemTy, 0);
  auto repB = wmmaLayout.getRepForOperand(bTensorTy.getShape(), bElemTy, 1);

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

  intrinsicName = addInstructionSuffix(intrinsicName, kBase, elemsPerVec,
                                       aElemTy, bElemTy, dElemTy, tied);
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
} // namespace mlir::triton::AMD
