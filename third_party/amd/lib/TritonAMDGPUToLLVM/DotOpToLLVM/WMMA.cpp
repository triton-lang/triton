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
#include "../TritonAMDGPUToLLVM/SchedInstructions.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::AMD {
namespace {

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;

enum class WMMAInstrType : uint8_t {
  // D = AB + C;
  // typeof(D) == typeof(C)
  // typeof(A) == typeof(B)
  // typeof(D), typeof(A):
  FP32_FP16,
  FP32_BF16,
  FP16_FP16,
  BF16_BF16,
  I32_I8,
  I32_I4,
  NOT_APPLICABLE,
};

using ValueTable = std::map<std::tuple<unsigned, unsigned, unsigned>, Value>;

ValueTable
getValuesFromDotOperandLayoutStruct(ConversionPatternRewriter &rewriter,
                                    const LLVMTypeConverter *typeConverter,
                                    Value value, int batch, int n0, int n1,
                                    int kWidth, Type type, Location loc) {
  auto elems = unpackLLElements(loc, value, rewriter);
  ValueTable vals;
  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < n0; i++) {
      for (int j = 0; j < n1; j++) {
        Type elemTy = typeConverter->convertType(type);
        Type ty = vec_ty(elemTy, kWidth);
        Value rawElems = undef(ty);
        for (int k = 0; k < kWidth; ++k) {
          rawElems = insert_element(
              ty, rawElems,
              elems[n0 * n1 * kWidth * b + kWidth * (n1 * i + j) + k],
              i32_val(k));
        }

        Value convertedElems;
        if (type.isF16()) {
          convertedElems = rawElems;
        } else if (type.isBF16()) {
          convertedElems = bitcast(rawElems, vec_ty(i16_ty, kWidth));
        } else {
          convertedElems = bitcast(
              rawElems, vec_ty(i32_ty, kWidth * type.getIntOrFloatBitWidth() /
                                           i32_ty.getIntOrFloatBitWidth()));
        }
        vals[{b, i, j}] = convertedElems;
      }
    }
  }
  return vals;
}

static WMMAInstrType getWMMAInstrTypeFromDot(DotOp op) {
  auto aOperandTy = op.getA().getType();
  auto aTensorTy = cast<RankedTensorType>(aOperandTy);
  auto aElemTy = aTensorTy.getElementType();
  auto bOperandTy = op.getB().getType();
  auto bTensorTy = cast<RankedTensorType>(bOperandTy);
  auto bElemTy = bTensorTy.getElementType();
  assert(aElemTy == bElemTy);
  auto cOperandTy = op.getC().getType();
  auto cTensorTy = cast<RankedTensorType>(cOperandTy);
  auto cElemTy = cTensorTy.getElementType();
  auto dOperandTy = op.getD().getType();
  auto dTensorTy = cast<RankedTensorType>(dOperandTy);
  auto dElemTy = dTensorTy.getElementType();
  assert(cElemTy == dElemTy);

  if (dElemTy.isF32() && aElemTy.isF16())
    return WMMAInstrType::FP32_FP16;
  if (dElemTy.isF32() && aElemTy.isBF16())
    return WMMAInstrType::FP32_BF16;
  if (dElemTy.isF16() && aElemTy.isF16())
    return WMMAInstrType::FP16_FP16;
  if (dElemTy.isBF16() && aElemTy.isBF16())
    return WMMAInstrType::BF16_BF16;
  if (dElemTy.isInteger(32) && aElemTy.isInteger(8))
    return WMMAInstrType::I32_I8;
  if (dElemTy.isInteger(32) && aElemTy.isInteger(4))
    return WMMAInstrType::I32_I4;

  return WMMAInstrType::NOT_APPLICABLE;
}

Value generateROCDLOp(ConversionPatternRewriter &rewriter, Location loc,
                      WMMAInstrType wmmaType, Value valA, Value valB,
                      Value valC, Type aElType, Type bElType) {
  auto resType = valC.getType();
  Value falseFlag = int_val(1, false);
  switch (wmmaType) {
  case WMMAInstrType::FP32_FP16:
    return rewriter.create<ROCDL::wmma_f32_16x16x16_f16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC});
  case WMMAInstrType::FP32_BF16:
    return rewriter.create<ROCDL::wmma_f32_16x16x16_bf16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC});
  case WMMAInstrType::FP16_FP16:
    return rewriter.create<ROCDL::wmma_f16_16x16x16_f16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
  case WMMAInstrType::BF16_BF16:
    return rewriter.create<ROCDL::wmma_bf16_16x16x16_bf16>(
        loc, TypeRange{resType}, ValueRange{valA, valB, valC, falseFlag});
  case WMMAInstrType::I32_I8:
    return rewriter.create<ROCDL::wmma_i32_16x16x16_iu8>(
        loc, TypeRange{resType},
        ValueRange{int_val(1, !aElType.isUnsignedInteger()), valA,
                   int_val(1, !bElType.isUnsignedInteger()), valB, valC,
                   falseFlag});
  case WMMAInstrType::I32_I4:
    return rewriter.create<ROCDL::wmma_i32_16x16x16_iu4>(
        loc, TypeRange{resType},
        ValueRange{int_val(1, !aElType.isUnsignedInteger()), valA,
                   int_val(1, !bElType.isUnsignedInteger()), valB, valC,
                   falseFlag});
  default:
    llvm::report_fatal_error("WMMA data type not supported");
  }
  return Value();
}

std::string getTypeStr(Type ty) {
  std::string scalarName;
  if (ty.isF32()) {
    scalarName = "f32";
  } else if (ty.isF16()) {
    scalarName = "f16";
  } else if (ty.isBF16()) {
    scalarName = "bf16";
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

StringRef getWmmaIntrinsicName(Type aElTy, Type bElTy, Type dElTy, Type valATy,
                               Type valCTy) {
  static llvm::SmallDenseMap<llvm::hash_code, std::string> intrinsics;
  using MapInfo = llvm::DenseMapInfo<Type>;
  llvm::hash_code h = llvm::hash_combine(
      MapInfo::getHashValue(aElTy), MapInfo::getHashValue(bElTy),
      MapInfo::getHashValue(dElTy), MapInfo::getHashValue(valATy),
      MapInfo::getHashValue(valCTy));
  if (!intrinsics.contains(h)) {
    std::string name = "llvm.amdgcn.wmma.";
    name += getTypeStr(dElTy);
    name += ".16x16x16."; // TODO support 16x16x32 for i4 operands
    name += getTypeStr(aElTy);
    if (isa<FloatType>(aElTy) && aElTy.getIntOrFloatBitWidth() == 8)
      name += '.' + getTypeStr(bElTy);
    name += '.' + getTypeStr(valCTy) + "." + getTypeStr(valATy);
    intrinsics[h] = name;
  }
  return intrinsics[h];
}

Value generateWMMAIntrinsic(ConversionPatternRewriter &rewriter, Location loc,
                            WMMAInstrType wmmaType, Value valA, Value valB,
                            Value valC, Type aElType, Type bElType,
                            Type dElType) {
  auto name = getWmmaIntrinsicName(aElType, bElType, dElType, valA.getType(),
                                   valC.getType());
  LLVM::FastmathFlagsAttr defaultFlags{};
  SmallVector<Value> operands;
  if (aElType.isInteger())
    operands.push_back(int_val(1, !aElType.isUnsignedInteger()));
  operands.push_back(valA);
  if (bElType.isInteger())
    operands.push_back(int_val(1, !bElType.isUnsignedInteger()));
  operands.push_back(valB);
  operands.push_back(valC);
  // Flag for using low bits in registers. Result could be already packed to
  // int32. Set low bits by default for now.
  if (32 / dElType.getIntOrFloatBitWidth() > 1 || dElType.isInteger(32)) {
    operands.push_back(int_val(1, false));
  }
  auto wmmaIntrinsic = LLVM::createLLVMIntrinsicCallOp(
      rewriter, loc, name, valC.getType(), operands);
  return wmmaIntrinsic.getResult(0);
}

Value generateWMMAOp(ConversionPatternRewriter &rewriter, Location loc,
                     WMMAInstrType wmmaType, Value valA, Value valB, Value valC,
                     Type aElType, Type bElType, Type dElType, int version) {
  if (version == 1) {
    return generateROCDLOp(rewriter, loc, wmmaType, valA, valB, valC, aElType,
                           bElType);
  } else {
    assert(version == 2);
    return generateWMMAIntrinsic(rewriter, loc, wmmaType, valA, valB, valC,
                                 aElType, bElType, dElType);
  }
}

// Conduct the Dot conversion.
LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor,
                         ConversionPatternRewriter &rewriter,
                         const LLVMTypeConverter *typeConverter) {
  auto wmmaLayout = cast<AMDWmmaEncodingAttr>(
      cast<RankedTensorType>(op.getResult().getType()).getEncoding());
  int wmmaVer = wmmaLayout.getVersion();
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();
  auto mnkDim = AMDWmmaEncodingAttr::getMNKDimPerInstr();
  auto wmmaInstrType = getWMMAInstrTypeFromDot(op);

  auto loc = op.getLoc();
  Value a = op.getA();
  Value b = op.getB();
  Value d = op.getD();
  auto aTensorTy = cast<RankedTensorType>(a.getType());
  auto bTensorTy = cast<RankedTensorType>(b.getType());
  auto dTensorTy = cast<RankedTensorType>(d.getType());
  auto elemTy = aTensorTy.getElementType();

  auto aEncoding = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
  auto bEncoding = cast<DotOperandEncodingAttr>(bTensorTy.getEncoding());
  int kWidth = aEncoding.getKWidth();

  auto repA =
      wmmaLayout.getRepForOperand(aTensorTy.getShape(), elemTy, kWidth, 0);
  auto repB =
      wmmaLayout.getRepForOperand(bTensorTy.getShape(), elemTy, kWidth, 1);

  assert(repA[2] == repB[1]);

  Value loadedA = adaptor.getA();
  Value loadedB = adaptor.getB();
  Value loadedC = adaptor.getC();
  auto numRepM = repA[1];
  auto numRepN = repB[2];
  auto numRepK = repA[2];
  auto numRepB = repA[0];

  ValueTable ha = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedA, numRepB, numRepM, numRepK, kWidth,
      aTensorTy.getElementType(), loc);
  ValueTable hb = getValuesFromDotOperandLayoutStruct(
      rewriter, typeConverter, loadedB, numRepB, numRepN, numRepK, kWidth,
      aTensorTy.getElementType(), loc);
  auto dstElemTy = dTensorTy.getElementType();
  auto fc = unpackLLElements(loc, loadedC, rewriter);

  unsigned warpSize = triton::gpu::getWarpSize(wmmaLayout);
  constexpr unsigned vgprElemBitWidth = 32;
  unsigned paddedOutputElemSize =
      wmmaVer == 1 ? vgprElemBitWidth / dstElemTy.getIntOrFloatBitWidth() : 1;
  // compute number of output elements that each thread holds for one WMMA
  // instruction.
  auto elemsPerVec = mnkDim[0] * mnkDim[1] * paddedOutputElemSize / warpSize;
  auto dElemsToStorePerThread = mnkDim[0] * mnkDim[1] / warpSize;
  auto vecTy = vec_ty(dstElemTy, elemsPerVec);
  for (int b = 0; b < numRepB; ++b) {
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        auto batchOffIdx = b * numRepM * numRepN * dElemsToStorePerThread;
        auto mRepOffId = m * numRepN * dElemsToStorePerThread;
        auto nRepOffId = n * dElemsToStorePerThread;
        auto fcThreadOffIdx = batchOffIdx + mRepOffId + nRepOffId;

        Value acc = undef(vecTy);
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          acc = insert_element(vecTy, acc, fc[fcThreadOffIdx + v],
                               i32_val(v * paddedOutputElemSize));
        }
        for (size_t k = 0; k < numRepK; k++) {
          acc = generateWMMAOp(rewriter, loc, wmmaInstrType, ha[{b, m, k}],
                               hb[{b, n, k}], acc, aTensorTy.getElementType(),
                               bTensorTy.getElementType(), dstElemTy, wmmaVer);
        }
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          fc[fcThreadOffIdx + v] = extract_element(
              dstElemTy, acc, i32_val(v * paddedOutputElemSize));
        }
      }
    }
  }

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      wmmaLayout.getContext(), SmallVector<Type>(fc.size(), dstElemTy));
  Value res = packLLElements(loc, typeConverter, fc, rewriter, structTy);

  const size_t mmaCount = numRepB * numRepM * numRepN * numRepK;
  setNumGeneratedMMAs(op, mmaCount, mnkDim[0], mnkDim[1], mnkDim[2], elemTy);

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
