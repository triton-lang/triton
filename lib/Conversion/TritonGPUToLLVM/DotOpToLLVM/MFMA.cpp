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
#include "../Utility.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MfmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

enum class MatrixCoreType : uint8_t {
  // D = AB + C
  FP32_FP16_FP16_FP32 = 0, // default
  FP32_BF16_BF16_FP32,
  FP32_BF16_BF16_FP32_1K,
  FP32_FP32_FP32_FP32,
  FP64_FP64_FP64_FP64,
  INT32_INT8_INT8_INT32,
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

  Value generateMFMAOp(MFMAInstrDescr mfmaDescr, Value valA, Value valB,
                       Value valC) const {
    auto resType = valC.getType();
    Value zeroFlag = i32_val(0);
    switch (mfmaDescr.coreType) {
    case MatrixCoreType::FP32_FP16_FP16_FP32:
      if (mfmaDescr.size == 16) {
        return rewriter.create<ROCDL::mfma_f32_16x16x16f16>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
      } else {
        return rewriter.create<ROCDL::mfma_f32_32x32x8f16>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
      }
    case MatrixCoreType::FP32_BF16_BF16_FP32:
      if (mfmaDescr.size == 16) {
        return rewriter.create<ROCDL::mfma_f32_16x16x8bf16>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
      } else {
        return rewriter.create<ROCDL::mfma_f32_32x32x4bf16>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
      }
    case MatrixCoreType::FP32_BF16_BF16_FP32_1K:
      if (mfmaDescr.size == 16) {
        return rewriter.create<ROCDL::mfma_f32_16x16x16bf16_1k>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
      } else {
        assert(mfmaDescr.size == 32);
        return rewriter.create<ROCDL::mfma_f32_32x32x8bf16_1k>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
      }
    case MatrixCoreType::FP32_FP32_FP32_FP32:
      if (mfmaDescr.size == 16) {
        return rewriter.create<ROCDL::mfma_f32_16x16x4f32>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
      } else {
        assert(mfmaDescr.size == 32);
        return rewriter.create<ROCDL::mfma_f32_32x32x2f32>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
      }
    case MatrixCoreType::INT32_INT8_INT8_INT32:
      if (mfmaDescr.size == 16) {
        return rewriter.create<ROCDL::mfma_i32_16x16x16i8>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
      } else {
        return rewriter.create<ROCDL::mfma_i32_32x32x8i8>(
            loc, TypeRange{resType},
            ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
      }
    case MatrixCoreType::FP64_FP64_FP64_FP64:
      assert(mfmaDescr.size == 16);
      return rewriter.create<ROCDL::mfma_f64_16x16x4f64>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    default:
      llvm::report_fatal_error("MFMA data type not supported");
    }
  }

  static MatrixCoreType getMatrixCoreTypeFromDot(DotOp op) {
    auto aOperandTy = op.getA().getType();
    auto tensorTy = aOperandTy.cast<RankedTensorType>();
    auto elemTy = tensorTy.getElementType();
    auto dotOpEncoding = tensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto mfmaEncoding = dotOpEncoding.getParent().cast<MfmaEncodingAttr>();
    if (elemTy.isF16())
      return MatrixCoreType::FP32_FP16_FP16_FP32;
    if (elemTy.isF32())
      return MatrixCoreType::FP32_FP32_FP32_FP32;
    if (elemTy.isBF16()) {
      auto nonKDim = mfmaEncoding.getNonKDim();
      auto kWidth = dotOpEncoding.getKWidth();
      if ((nonKDim == 32 && kWidth == 4) || (nonKDim == 16 && kWidth == 4)) {
        return MatrixCoreType::FP32_BF16_BF16_FP32_1K;
      } else {
        assert((nonKDim == 32 && kWidth == 2) ||
               (nonKDim == 16 && kWidth == 2));
        return MatrixCoreType::FP32_BF16_BF16_FP32;
      }
    }
    if (elemTy.isInteger(8))
      return MatrixCoreType::INT32_INT8_INT8_INT32;
    if (elemTy.isF64())
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

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
    auto nonKDim = mfmaLayout.getNonKDim();
    assert(nonKDim == 32 || nonKDim == 16);
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

    auto repA = aEncoding.getMFMARep(aTensorTy.getShape(), elemTy);
    auto repB = bEncoding.getMFMARep(bTensorTy.getShape(), elemTy);

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
    auto fc =
        typeConverter->unpackLLElements(loc, loadedC, rewriter, dstElemTy);

    unsigned warpSize = triton::gpu::getWarpSize(mfmaLayout);
    // compute number of output elements that each thread holds for one MFMA
    // instruction
    auto elemsPerVec = nonKDim * nonKDim / warpSize;

    auto vecTy = vec_ty(dstElemTy, elemsPerVec);
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        Value acc = undef(vecTy);
        for (unsigned v = 0; v < elemsPerVec; ++v) {
          acc = insert_element(
              vecTy, acc, fc[m * numRepN * elemsPerVec + n * elemsPerVec + v],
              i32_val(v));
        }

        for (size_t k = 0; k < numRepK; k++) {
          acc =
              mfmaLayout.getIsTransposed()
                  ? generateMFMAOp(mfmaInstrDescr, hb[{n, k}], ha[{m, k}], acc)
                  : generateMFMAOp(mfmaInstrDescr, ha[{m, k}], hb[{n, k}], acc);
        }
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
    auto elems = typeConverter->unpackLLElements(loc, value, rewriter, type);
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

#endif // ifdef USE_ROCM
