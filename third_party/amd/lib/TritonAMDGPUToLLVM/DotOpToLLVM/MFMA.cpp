/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
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
#ifdef USE_ROCM

#include "../DotOpToLLVM.h"
#include "Utility.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::AMD::TritonGPUToLLVMTypeConverter;
using ::mlir::LLVM::shflSync;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MfmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

enum class MatrixCoreType : uint8_t {
  // D = AB + C
  FP32_FP8_FP8_FP32,
  FP32_FP8_BF8_FP32,
  FP32_BF8_FP8_FP32,
  FP32_BF8_BF8_FP32,
  FP32_FP16_FP16_FP32,
  FP32_BF16_BF16_FP32,
  FP32_BF16_BF16_FP32_1K,
  FP32_FP32_FP32_FP32,
  FP64_FP64_FP64_FP64,
  INT32_INT8_INT8_INT32,
  INT32_INT8_INT8_INT32_CDNA3,
  NOT_APPLICABLE,
};

struct MFMAInstrDescr {
  MatrixCoreType coreType;
  unsigned size;
};

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

struct DotOpMFMAConversionHelper {
  MfmaEncodingAttr mfmaLayout;

  ConversionPatternRewriter &rewriter;
  TritonGPUToLLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConversionHelper(
      MfmaEncodingAttr mfmaLayout, ConversionPatternRewriter &rewriter,
      TritonGPUToLLVMTypeConverter *typeConverter, Location loc)
      : mfmaLayout(mfmaLayout), rewriter(rewriter),
        typeConverter(typeConverter), loc(loc), ctx(mfmaLayout.getContext()) {}

  Value getThreadId() const {
    auto llvmIndexTy = typeConverter->getIndexType();
    auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::TruncIOp>(loc, i32_ty, tid);
  }

  Value generateMFMA32Op(MatrixCoreType coreType, Value valA, Value valB,
                         Value valC) const {
    auto resType = valC.getType();
    Value zeroFlag = i32_val(0);
    switch (coreType) {
    case MatrixCoreType::FP32_FP8_FP8_FP32:
      return rewriter.create<ROCDL::mfma_f32_32x32x16_fp8_fp8>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_FP8_BF8_FP32:
      return rewriter.create<ROCDL::mfma_f32_32x32x16_fp8_bf8>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF8_FP8_FP32:
      return rewriter.create<ROCDL::mfma_f32_32x32x16_bf8_fp8>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF8_BF8_FP32:
      return rewriter.create<ROCDL::mfma_f32_32x32x16_bf8_bf8>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_FP16_FP16_FP32:
      return rewriter.create<ROCDL::mfma_f32_32x32x8f16>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32:
      return rewriter.create<ROCDL::mfma_f32_32x32x4bf16>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32_1K:
      return rewriter.create<ROCDL::mfma_f32_32x32x8bf16_1k>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_FP32_FP32_FP32:
      return rewriter.create<ROCDL::mfma_f32_32x32x2f32>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::INT32_INT8_INT8_INT32:
      return rewriter.create<ROCDL::mfma_i32_32x32x8i8>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::INT32_INT8_INT8_INT32_CDNA3:
      return rewriter.create<ROCDL::mfma_i32_32x32x16_i8>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP64_FP64_FP64_FP64:
      return rewriter.create<ROCDL::mfma_f64_16x16x4f64>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    default:
      llvm::report_fatal_error("MFMA 32x32 data type not supported");
    }
  }

  Value generateMFMA16Op(MatrixCoreType coreType, Value valA, Value valB,
                         Value valC) const {
    auto resType = valC.getType();
    Value zeroFlag = i32_val(0);
    switch (coreType) {
    case MatrixCoreType::FP32_FP16_FP16_FP32:
      return rewriter.create<ROCDL::mfma_f32_16x16x16f16>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32:
      return rewriter.create<ROCDL::mfma_f32_16x16x8bf16>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32_1K:
      return rewriter.create<ROCDL::mfma_f32_16x16x16bf16_1k>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_FP32_FP32_FP32:
      return rewriter.create<ROCDL::mfma_f32_16x16x4f32>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::INT32_INT8_INT8_INT32:
      return rewriter.create<ROCDL::mfma_i32_16x16x16i8>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP64_FP64_FP64_FP64:
      return rewriter.create<ROCDL::mfma_f64_16x16x4f64>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    default:
      llvm::report_fatal_error("MFMA data type not supported");
    }
  }

  Value generateMFMA4Op(MatrixCoreType coreType, Value valA, Value valB,
                        Value valC) const {
    auto resType = valC.getType();
    Value zeroFlag = i32_val(0);
    switch (coreType) {
    case MatrixCoreType::FP32_FP16_FP16_FP32:
      return rewriter.create<ROCDL::mfma_f32_4x4x4f16>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32:
      return rewriter.create<ROCDL::mfma_f32_4x4x2bf16>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32_1K:
      return rewriter.create<ROCDL::mfma_f32_4x4x4bf16_1k>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_FP32_FP32_FP32:
      return rewriter.create<ROCDL::mfma_f32_4x4x1f32>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::INT32_INT8_INT8_INT32:
      return rewriter.create<ROCDL::mfma_i32_4x4x4i8>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    default:
      llvm::report_fatal_error("MFMA4 data type not supported");
    }
  }

  Value generateMFMAOp(MFMAInstrDescr mfmaDescr, Value valA, Value valB,
                       Value valC) const {
    switch (mfmaDescr.size) {
    case 32:
      return generateMFMA32Op(mfmaDescr.coreType, valA, valB, valC);
      break;
    case 16:
      return generateMFMA16Op(mfmaDescr.coreType, valA, valB, valC);
      break;
    case 4:
      return generateMFMA4Op(mfmaDescr.coreType, valA, valB, valC);
    default:
      llvm::report_fatal_error("MFMA nonkDim size is not supported");
    }
    return Value();
  }

  int getNumSubmatrices(Type elementType, int nonKDim) const {
    switch (nonKDim) {
    case 32:
    case 16:
      return 1;
      break;
    case 4:
      assert(elementType.getIntOrFloatBitWidth() <= 32 &&
             "fp64 is not supported yet");
      assert(elementType.getIntOrFloatBitWidth() != 8 ||
             elementType.isInteger(8) && "fp8 is not supported yet");
      return 16;
      break;
    default:
      llvm::report_fatal_error("unsupported nonKDim in MFMA dot");
    }
    return -1;
  }

  // TODO unify this function with Utility.cpp:supportMFMATypes
  static MatrixCoreType getMatrixCoreTypeFromDot(DotOp op) {
    auto aOperandTy = op.getA().getType();
    auto aTensorTy = aOperandTy.cast<RankedTensorType>();
    auto aElemTy = aTensorTy.getElementType();
    auto bOperandTy = op.getB().getType();
    auto bTensorTy = bOperandTy.cast<RankedTensorType>();
    auto bElemTy = bTensorTy.getElementType();

    auto dotOpEncoding = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto mfmaEncoding = dotOpEncoding.getParent().cast<MfmaEncodingAttr>();
    if (aElemTy.isFloat8E4M3FNUZ() && bElemTy.isFloat8E4M3FNUZ())
      return MatrixCoreType::FP32_FP8_FP8_FP32;
    if (aElemTy.isFloat8E4M3FNUZ() && bElemTy.isFloat8E5M2FNUZ())
      return MatrixCoreType::FP32_FP8_BF8_FP32;
    if (aElemTy.isFloat8E5M2FNUZ() && bElemTy.isFloat8E4M3FNUZ())
      return MatrixCoreType::FP32_BF8_FP8_FP32;
    if (aElemTy.isFloat8E5M2FNUZ() && bElemTy.isFloat8E5M2FNUZ())
      return MatrixCoreType::FP32_BF8_BF8_FP32;
    if (aElemTy.isF16())
      return MatrixCoreType::FP32_FP16_FP16_FP32;
    if (aElemTy.isF32())
      return MatrixCoreType::FP32_FP32_FP32_FP32;
    if (aElemTy.isBF16()) {
      auto nonKDim = mfmaEncoding.getNonKDim();
      auto kWidth = dotOpEncoding.getKWidth();
      if ((nonKDim == 32 || nonKDim == 16 || nonKDim == 4) && kWidth == 4) {
        return MatrixCoreType::FP32_BF16_BF16_FP32_1K;
      } else {
        assert((nonKDim == 32 && kWidth == 2) ||
               (nonKDim == 16 && kWidth == 2) || (nonKDim == 4 && kWidth == 2));
        return MatrixCoreType::FP32_BF16_BF16_FP32;
      }
    }
    if (aElemTy.isInteger(8)) {
      auto nonKDim = mfmaEncoding.getNonKDim();
      auto kWidth = dotOpEncoding.getKWidth();
      if ((nonKDim == 32 || nonKDim == 16 || nonKDim == 4) && kWidth == 8) {
        return MatrixCoreType::INT32_INT8_INT8_INT32_CDNA3;
      } else {
        assert((nonKDim == 32 || nonKDim == 16 || nonKDim == 4) && kWidth == 4);
        return MatrixCoreType::INT32_INT8_INT8_INT32;
      }
    }
    if (aElemTy.isF64())
      return MatrixCoreType::FP64_FP64_FP64_FP64;
    return MatrixCoreType::NOT_APPLICABLE;
  }

  static MFMAInstrDescr getMatrixInstrDescr(DotOp op) {
    MFMAInstrDescr descr;
    auto tensorTy = op.getD().getType().cast<RankedTensorType>();
    auto encoding = tensorTy.getEncoding().cast<MfmaEncodingAttr>();
    descr.coreType = getMatrixCoreTypeFromDot(op);
    descr.size = encoding.getNonKDim();
    return descr;
  }

  Value processSubBlocks(int numSubBlocks, Value acc, bool reduceSubBlocks,
                         bool zeroSubBlocks) const {
    assert((numSubBlocks & (numSubBlocks - 1)) == 0 &&
           "numSubBlocks in not pow 2!");
    if (numSubBlocks == 1)
      return acc;
    constexpr int waveSize = 64;
    int subBlockSize = waveSize / numSubBlocks;
    Value laneId = getThreadId();
    laneId = and_(laneId, i32_val(waveSize - 1));
    auto vecTy = dyn_cast<VectorType>(acc.getType());
    auto elemType = vecTy.getElementType();
    assert(elemType.getIntOrFloatBitWidth() == 32);
    int numScalars = vecTy.getNumElements();
    std::vector<Value> accScalar(numScalars);
    for (int i = 0; i < numScalars; ++i)
      accScalar[i] = extract_element(elemType, acc, i32_val(i));

    if (reduceSubBlocks) {
      while (subBlockSize < waveSize) {
        for (int i = 0; i < numScalars; ++i) {
          Value other_acc = shflSync(loc, rewriter, accScalar[i], subBlockSize);
          if (elemType.isInteger(32))
            accScalar[i] = add(accScalar[i], other_acc);
          else
            accScalar[i] = fadd(accScalar[i], other_acc);
        }
        subBlockSize *= 2;
      }
    }
    if (zeroSubBlocks) {
      Value zero;
      if (elemType.isInteger(32))
        zero = i32_val(0);
      else
        zero = f32_val(0.0);
      auto cond = icmp_ult(laneId, i32_val(subBlockSize));
      for (int i = 0; i < numScalars; ++i)
        accScalar[i] = select(cond, accScalar[i], zero);
    }

    Value reducedAcc = undef(vecTy);
    for (int i = 0; i < numScalars; ++i)
      reducedAcc = insert_element(vecTy, reducedAcc, accScalar[i], i32_val(i));
    return reducedAcc;
  }

  /// @brief MFMA 4x4 is computes 16 matrix mupliplications, this functions adds
  /// these 16 matrices to get final 4x4 matrix
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value reduceSubBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, true, false);
  }

  /// @brief Zeroes out redundant values in all sub-blocks except first one
  ///
  /// Every wave in mfma 4x4 layout holds only 4 unique values(scalar or
  /// vectors) in blocks of 4 consecutive threads, There are 16 copies of these
  /// 4 values across all threads of the wave. Need to zero out 15 copies to use
  /// accumulator between dot operations.
  /// @param numSubBlocks
  /// @param acc
  /// @return
  Value zeroAuxiliarBlocks(int numSubBlocks, Value acc) const {
    return processSubBlocks(numSubBlocks, acc, false, true);
  }

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto nonKDim = mfmaLayout.getNonKDim();
    assert(nonKDim == 32 || nonKDim == 16 || nonKDim == 4);
    auto mfmaInstrDescr = getMatrixInstrDescr(op);

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = a.getType().cast<RankedTensorType>();
    auto bTensorTy = b.getType().cast<RankedTensorType>();
    auto dTensorTy = d.getType().cast<RankedTensorType>();
    auto elemTy = aTensorTy.getElementType();

    auto aEncoding = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto bEncoding = bTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    int kWidth = aEncoding.getKWidth();

    auto repA = mfmaLayout.getMFMARepForOperands(aTensorTy.getShape(), elemTy,
                                                 kWidth, 0);
    auto repB = mfmaLayout.getMFMARepForOperands(bTensorTy.getShape(), elemTy,
                                                 kWidth, 1);

    assert(repA[1] == repB[0]);

    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();

    auto numRepM = repA[0];
    auto numRepN = repB[1];
    auto numRepK = repA[1];

    ValueTable ha = getValuesFromDotOperandLayoutStruct(
        loadedA, numRepM, numRepK, aTensorTy.getElementType());
    ValueTable hb = getValuesFromDotOperandLayoutStruct(
        loadedB, numRepN, numRepK, aTensorTy.getElementType());
    auto dstElemTy = dTensorTy.getElementType();
    auto fc = typeConverter->unpackLLElements(loc, loadedC, rewriter);

    unsigned warpSize = triton::gpu::getWarpSize(mfmaLayout);
    // compute number of output elements that each thread holds for one MFMA
    // instruction. subBlocks
    const int subBlocks =
        getNumSubmatrices(aTensorTy.getElementType(), nonKDim);
    auto elemsPerVec = nonKDim * nonKDim * subBlocks / warpSize;

    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        Value acc = undef(vecTy);
        for (unsigned v = 0; v < elemsPerVec; ++v) {
          acc = insert_element(
              vecTy, acc, fc[m * numRepN * elemsPerVec + n * elemsPerVec + v],
              i32_val(v));
        }
        acc = zeroAuxiliarBlocks(subBlocks, acc);
        for (size_t k = 0; k < numRepK; k++) {
          acc =
              mfmaLayout.getIsTransposed()
                  ? generateMFMAOp(mfmaInstrDescr, hb[{n, k}], ha[{m, k}], acc)
                  : generateMFMAOp(mfmaInstrDescr, ha[{m, k}], hb[{n, k}], acc);
        }
        acc = reduceSubBlocks(subBlocks, acc);
        for (unsigned v = 0; v < elemsPerVec; ++v) {
          fc[m * numRepN * elemsPerVec + n * elemsPerVec + v] =
              extract_element(dstElemTy, acc, i32_val(v));
        }
      }
    }

    // replace with new packed result
    Type structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(fc.size(), dstElemTy));
    Value res = typeConverter->packLLElements(loc, fc, rewriter, structTy);
    rewriter.replaceOp(op, res);

    return success();
  }

  ValueTable getValuesFromDotOperandLayoutStruct(Value value, int n0, int n1,
                                                 Type type) const {
    auto elems = typeConverter->unpackLLElements(loc, value, rewriter);
    ValueTable vals;
    for (int i = 0; i < n0; i++) {
      for (int j = 0; j < n1; j++) {
        vals[{i, j}] = elems[n1 * i + j];
      }
    }
    return vals;
  }
};

} // namespace

namespace AMD {
LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto rankedTType = [](Value tensor) {
    return tensor.getType().cast<RankedTensorType>();
  };

  assert(rankedTType(op.getA()).getEncoding().isa<DotOperandEncodingAttr>() &&
         rankedTType(op.getB()).getEncoding().isa<DotOperandEncodingAttr>() &&
         "Both $a and %b should be DotOperand layout.");

  auto cTensorTy = rankedTType(op.getC());
  auto dTensorTy = rankedTType(op.getD());
  assert(cTensorTy.getEncoding().isa<MfmaEncodingAttr>() &&
         "Currently, we only support $c with a mfma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto loc = op.getLoc();
  auto mfmaLayout = op.getResult()
                        .getType()
                        .cast<RankedTensorType>()
                        .getEncoding()
                        .cast<MfmaEncodingAttr>();

  DotOpMFMAConversionHelper helper(mfmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}
} // namespace AMD

#endif // ifdef USE_ROCM
