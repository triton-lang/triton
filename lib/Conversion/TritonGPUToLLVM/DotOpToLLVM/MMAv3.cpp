#ifdef USE_ROCM

#include "../DotOpToLLVM.h"
#include "../Utility.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

enum class MatrixCoreType : uint8_t {
  // D = AB + C
  FP32_FP16_FP16_FP32 = 0, // default
  FP32_BF16_BF16_FP32,
  FP32_FP32_FP32_FP32,
  FP64_FP64_FP64_FP64,
  INT32_INT8_INT8_INT32,
  NOT_APPLICABLE,
};

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

struct DotOpMFMAConversionHelper {
  MmaEncodingAttr mmaLayout;

  ConversionPatternRewriter &rewriter;
  TritonGPUToLLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConversionHelper(
      MmaEncodingAttr mmaLayout, ConversionPatternRewriter &rewriter,
      TritonGPUToLLVMTypeConverter *typeConverter, Location loc)
      : mmaLayout(mmaLayout), rewriter(rewriter), typeConverter(typeConverter),
        loc(loc), ctx(mmaLayout.getContext()) {}

  Value getThreadId() const {
    auto llvmIndexTy = typeConverter->getIndexType();
    auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::TruncIOp>(loc, i32_ty, tid);
  }

  Value generateMFMAOp(MatrixCoreType mfmaTy, Value valA, Value valB,
                       Value valC) const {
    auto resType = valC.getType();
    Value zeroFlag = i32_val(0);
    switch (mfmaTy) {
    case MatrixCoreType::FP32_FP16_FP16_FP32:
      return rewriter.create<ROCDL::mfma_f32_32x32x8f16>(
          loc, TypeRange{resType},
          ValueRange{valA, valB, valC, zeroFlag, zeroFlag, zeroFlag});
    case MatrixCoreType::FP32_BF16_BF16_FP32:
      return rewriter.create<ROCDL::mfma_f32_32x32x4bf16>(
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
    case MatrixCoreType::FP64_FP64_FP64_FP64:
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
    if (elemTy.isF16())
      return MatrixCoreType::FP32_FP16_FP16_FP32;
    if (elemTy.isF32())
      return MatrixCoreType::FP32_FP32_FP32_FP32;
    if (elemTy.isBF16())
      return MatrixCoreType::FP32_BF16_BF16_FP32;
    if (elemTy.isInteger(8))
      return MatrixCoreType::INT32_INT8_INT8_INT32;
    if (elemTy.isF64())
      return MatrixCoreType::FP64_FP64_FP64_FP64;
    return MatrixCoreType::NOT_APPLICABLE;
  }

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {
    auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
    auto mfmaTy = getMatrixCoreTypeFromDot(op);

    Value a = op.getA();
    Value b = op.getB();
    Value d = op.getD();
    auto aTensorTy = a.getType().cast<RankedTensorType>();
    auto bTensorTy = b.getType().cast<RankedTensorType>();
    auto dTensorTy = d.getType().cast<RankedTensorType>();
    auto elemTy = aTensorTy.getElementType();

    auto aEncoding = aTensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    auto bEncoding = bTensorTy.getEncoding().cast<DotOperandEncodingAttr>();

    auto repA = aEncoding.getMMAv3Rep(aTensorTy.getShape(), elemTy);
    auto repB = bEncoding.getMMAv3Rep(bTensorTy.getShape(), elemTy);

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

    auto vecTy = vec_ty(dstElemTy, 16);
    for (int m = 0; m < numRepM; ++m) {
      for (int n = 0; n < numRepN; ++n) {
        Value acc = undef(vecTy);
        for (unsigned v = 0; v < 16; ++v) {
          acc = insert_element(vecTy, acc, fc[m * numRepN * 16 + n * 16 + v],
                               i32_val(v));
        }

        for (size_t k = 0; k < numRepK; k++) {
          acc = generateMFMAOp(mfmaTy, ha[{m, k}], hb[{n, k}], acc);
        }
        for (unsigned v = 0; v < 16; ++v) {
          fc[m * numRepN * 16 + n * 16 + v] =
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
  assert(cTensorTy.getEncoding().isa<MmaEncodingAttr>() &&
         "Currently, we only support $c with a mma layout.");

  assert(cTensorTy.getShape()[0] == dTensorTy.getShape()[0] &&
         cTensorTy.getShape()[1] == dTensorTy.getShape()[1] &&
         "DotOp's $c operand should pass the same number of values as $d");

  auto loc = op.getLoc();
  auto mmaLayout = op.getResult()
                       .getType()
                       .cast<RankedTensorType>()
                       .getEncoding()
                       .cast<MmaEncodingAttr>();

  DotOpMFMAConversionHelper helper(mmaLayout, rewriter, typeConverter, loc);

  return helper.convertDot(op, adaptor);
}

#endif // ifdef USE_ROCM
