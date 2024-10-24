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
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::AMD {
namespace {

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;

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
                               Type valCTy, bool tied) {
  static llvm::SmallDenseMap<llvm::hash_code, std::string> intrinsics;
  using MapInfo = llvm::DenseMapInfo<Type>;
  llvm::hash_code h = llvm::hash_combine(
      MapInfo::getHashValue(aElTy), MapInfo::getHashValue(bElTy),
      MapInfo::getHashValue(dElTy), MapInfo::getHashValue(valATy),
      MapInfo::getHashValue(valCTy), llvm::hash_value(tied));
  if (!intrinsics.contains(h)) {
    std::string name = "llvm.amdgcn.wmma.";
    name += getTypeStr(dElTy);
    name += ".16x16x16."; // TODO support 16x16x32 for i4 operands
    name += getTypeStr(aElTy);
    if (tied) {
      name += ".tied";
    } else {
      if (isa<FloatType>(aElTy) && aElTy.getIntOrFloatBitWidth() == 8)
        name += '.' + getTypeStr(bElTy);
      name += '.' + getTypeStr(valCTy) + "." + getTypeStr(valATy);
    }
    intrinsics[h] = name;
  }
  return intrinsics[h];
}

Value generateWMMAIntrinsic(ConversionPatternRewriter &rewriter, Location loc,
                            Value valA, Value valB, Value valC, Type aElType,
                            Type bElType, Type dElType,
                            std::optional<bool> tiedLower) {
  auto name = getWmmaIntrinsicName(aElType, bElType, dElType, valA.getType(),
                                   valC.getType(), tiedLower.has_value());
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
  if (tiedLower.has_value() || 32 / dElType.getIntOrFloatBitWidth() > 1 ||
      dElType.isInteger(32)) {
    operands.push_back(int_val(1, tiedLower.value_or(false)));
  }
  auto wmmaIntrinsic = LLVM::createLLVMIntrinsicCallOp(
      rewriter, loc, name, valC.getType(), operands);
  return wmmaIntrinsic.getResult(0);
}

Value generateWMMAOp(ConversionPatternRewriter &rewriter, Location loc,
                     Value valA, Value valB, Value valC, Type aElType,
                     Type bElType, Type dElType,
                     std::optional<bool> tiedLower) {
  // Independent of wmma version because builtin functions are backward
  // compatible
  return generateWMMAIntrinsic(rewriter, loc, valA, valB, valC, aElType,
                               bElType, dElType, tiedLower);
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
  bool tied = numRepM % 2 == 0 && paddedOutputElemSize == 2;
  int mGroup = tied ? 2 : 1;
  for (int b = 0; b < numRepB; ++b) {
    for (int m = 0; m < numRepM / mGroup; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        auto batchOffIdx = b * numRepM * numRepN * dElemsToStorePerThread;
        auto nRepOffId = n * dElemsToStorePerThread;
        auto nBatchOffSum = nRepOffId + batchOffIdx;

        Value acc = undef(vecTy);
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          for (int subM = 0; subM < mGroup; ++subM) {
            auto mRepOffId =
                (m * mGroup + subM) * numRepN * dElemsToStorePerThread;
            auto fcThreadOffIdx = nBatchOffSum + mRepOffId;
            acc = insert_element(vecTy, acc, fc[fcThreadOffIdx + v],
                                 i32_val(v * paddedOutputElemSize + subM));
          }
        }
        for (size_t k = 0; k < numRepK; ++k) {
          for (int subM = 0; subM < mGroup; ++subM) {
            acc = generateWMMAOp(
                rewriter, loc, ha[{b, m * mGroup + subM, k}], hb[{b, n, k}],
                acc, aTensorTy.getElementType(), bTensorTy.getElementType(),
                dTensorTy.getElementType(),
                tied ? std::optional<bool>(subM != 0) : std::nullopt);
          }
        }
        for (unsigned v = 0; v < dElemsToStorePerThread; ++v) {
          for (int subM = 0; subM < mGroup; ++subM) {
            auto mRepOffId =
                (m * mGroup + subM) * numRepN * dElemsToStorePerThread;
            auto fcThreadOffIdx = nBatchOffSum + mRepOffId;
            fc[fcThreadOffIdx + v] = extract_element(
                dstElemTy, acc, i32_val(v * paddedOutputElemSize + subM));
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
