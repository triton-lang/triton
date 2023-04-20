#include "../DotOpToLLVM.h"
#include "../Utility.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

struct DotOpMFMAConversionHelper {
  enum class MatrixCoreType : uint8_t {
    // D = AB + C
    FP32_FP16_FP16_FP32 = 0, // default
    FP32_BF16_BF16_FP32,
    FP32_FP32_FP32_FP32,
    FP64_FP64_FP64_FP64,
    INT32_INT8_INT8_INT32,
    NOT_APPLICABLE,
  };

  MmaEncodingAttr mmaLayout;
  ArrayRef<unsigned int> wpt;

  Value thread, lane, wave;

  ConversionPatternRewriter &rewriter;
  TritonGPUToLLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;
  explicit DotOpMFMAConversionHelper(
      Type dotOperand, MmaEncodingAttr mmaLayout, Value thread,
      ConversionPatternRewriter &rewriter,
      TritonGPUToLLVMTypeConverter *typeConverter, Location loc)
      : mmaLayout(mmaLayout), wpt(mmaLayout.getWarpsPerCTA()), thread(thread),
        rewriter(rewriter), typeConverter(typeConverter), loc(loc),
        ctx(mmaLayout.getContext()) {
    deduceMFMAType(dotOperand);
    Value waveSize = i32_val(64);
    lane = urem(thread, waveSize);
    wave = udiv(thread, waveSize);
  }

  void deduceMFMAType(DotOp op) const { mfmaType = getMFMAType(op); }
  void deduceMFMAType(Type operandTy) const {
    mfmaType = getMatrixCoreTypeFromOperand(operandTy);
  }

  ArrayRef<int> getMFMAInstrShape() const {
    assert(mfmaType != MatrixCoreType::NOT_APPLICABLE &&
           "Unknown MFMA type found.");
    return mfmaInstrShape.at(mfmaType);
  }

  Value generateMFMAOp(Value valA, Value valB, Value valC) const;

  static ArrayRef<int> getMFMAInstrShape(MatrixCoreType matrixCoreType) {
    assert(matrixCoreType != MatrixCoreType::NOT_APPLICABLE &&
           "Unknown MFMA type found.");
    return mfmaInstrShape.at(matrixCoreType);
  }

  // Deduce the MatrixCoreType from either $a or $b's type.
  static MatrixCoreType getMatrixCoreTypeFromOperand(Type operandTy);

  static MatrixCoreType getMFMAType(triton::DotOp op);

  std::tuple<int, int, int> getMFMAInstrShape(Type operand) const {
    deduceMFMAType(operand);
    auto instrShape = getMFMAInstrShape();
    int instrM = instrShape[0];
    int instrN = instrShape[1];
    int instrK = instrShape[2];
    return std::make_tuple(instrM, instrN, instrK);
  }

  // \param operand is either $a or $b's type.
  inline int getNumRepM(Type operand, int M) const {
    return getNumRepM(operand, M, wpt[0]);
  }

  // \param operand is either $a or $b's type.
  inline int getNumRepN(Type operand, int N) const {
    return getNumRepN(operand, N, wpt[1]);
  }

  // \param operand is either $a or $b's type.
  inline int getNumRepK(Type operand, int K) const {
    return getNumRepK_(operand, K);
  }

  static int getNumRepM(Type operand, int M, int wpt) {
    auto matrixCoreType = getMatrixCoreTypeFromOperand(operand);
    int instrM = getMFMAInstrShape(matrixCoreType)[0];
    return std::max<int>(M / (wpt * instrM), 1);
  }

  static int getNumRepN(Type operand, int N, int wpt) {
    auto matrixCoreType = getMatrixCoreTypeFromOperand(operand);
    int instrN = getMFMAInstrShape(matrixCoreType)[1];
    return std::max<int>(N / (wpt * instrN), 1);
  }

  static int getNumRepK_(Type operand, int K) {
    auto matrixCoreType = getMatrixCoreTypeFromOperand(operand);
    int instrK = getMFMAInstrShape(matrixCoreType)[2];
    return std::max<int>(K / instrK, 1);
  }

  // Loading $c to registers, returns a Value.
  Value loadC(Value tensor, Value llTensor) const;

  // Conduct the Dot conversion.
  // \param a, \param b, \param c and \param d are DotOp operands.
  // \param loadedA, \param loadedB, \param loadedC, all of them are result of
  // loading.
  LogicalResult convertDot(Value a, Value b, Value c, Value d, Value loadedA,
                           Value loadedB, Value loadedC, DotOp op,
                           DotOpAdaptor adaptor) const;

  ValueTable getValuesFromDotOperandLayoutStruct(Value value, int n0, int n1,
                                                 Type type) const;
  TritonGPUToLLVMTypeConverter *getTypeConverter() const {
    return typeConverter;
  }

private:
  mutable MatrixCoreType mfmaType{MatrixCoreType::NOT_APPLICABLE};

  // Used on AMDGPU mma layout .version == 3
  inline static const std::map<MatrixCoreType, llvm::SmallVector<int>>
      mfmaInstrShape = { // m, n, k
          {MatrixCoreType::FP32_FP16_FP16_FP32, {32, 32, 8}},
          {MatrixCoreType::FP32_BF16_BF16_FP32, {32, 32, 4}},
          {MatrixCoreType::FP32_FP32_FP32_FP32, {32, 32, 2}},

          {MatrixCoreType::INT32_INT8_INT8_INT32, {32, 32, 8}},
          {MatrixCoreType::FP64_FP64_FP64_FP64, {16, 16, 4}}};
};

Value getThreadId(ConversionPatternRewriter &rewriter, TritonGPUToLLVMTypeConverter *typeConverter, Location loc) {
  auto llvmIndexTy = typeConverter->getIndexType();
  auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
      loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::TruncIOp>(loc, i32_ty, tid);
}

// contents of cpp file

// Get the M and N of MFMA instruction shape.
static std::tuple<int, int> getInstrShapeMN() { return {32, 32}; }

Value DotOpMFMAConversionHelper::generateMFMAOp(Value valA, Value valB,
                                                Value valC) const {
  auto resType = valC.getType();
  Value zeroFlag = i32_val(0);
  switch (mfmaType) {
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

DotOpMFMAConversionHelper::MatrixCoreType
DotOpMFMAConversionHelper::getMatrixCoreTypeFromOperand(Type operandTy) {
  auto tensorTy = operandTy.cast<RankedTensorType>();
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

Value DotOpMFMAConversionHelper::loadC(Value tensor, Value llTensor) const {
  auto tensorTy = tensor.getType().cast<RankedTensorType>();

  auto mmaLayout = tensorTy.getEncoding().cast<MmaEncodingAttr>();
  auto wpt = mmaLayout.getWarpsPerCTA();

  int M = tensorTy.getShape()[0];
  int N = tensorTy.getShape()[1];
  auto [instrM, instrN] = getInstrShapeMN();
  int repM = std::max<int>(M / (wpt[0] * instrM), 1);
  int repN = std::max<int>(N / (wpt[1] * instrN), 1);

  size_t accSize = 16 * repM * repN;

  assert(tensorTy.getEncoding().isa<MmaEncodingAttr>() &&
         "Currently, we only support $c with a mma layout.");
  auto structTy = llTensor.getType().cast<LLVM::LLVMStructType>();
  assert(structTy.getBody().size() == accSize &&
         "DotOp's $c operand should pass the same number of values as $d in "
         "mma layout.");
  return llTensor;
}

DotOpMFMAConversionHelper::ValueTable
DotOpMFMAConversionHelper::getValuesFromDotOperandLayoutStruct(
    Value value, int n0, int n1, Type type) const {
  auto elems = typeConverter->unpackLLElements(loc, value, rewriter, type);
  ValueTable vals;
  for (int i = 0; i < n0; i++) {
    for (int j = 0; j < n1; j++) {
      vals[{i, j}] = elems[n1 * i + j];
    }
  }
  return vals;
}

LogicalResult DotOpMFMAConversionHelper::convertDot(
    Value a, Value b, Value c, Value d, Value loadedA, Value loadedB,
    Value loadedC, DotOp op, DotOpAdaptor adaptor) const {
  auto aTensorTy = a.getType().cast<RankedTensorType>();
  auto dTensorTy = d.getType().cast<RankedTensorType>();

  SmallVector<int64_t> aShape(aTensorTy.getShape().begin(),
                              aTensorTy.getShape().end());

  auto dShape = dTensorTy.getShape();

  int numRepM = getNumRepM(aTensorTy, dShape[0]);
  int numRepN = getNumRepN(aTensorTy, dShape[1]);
  int numRepK = getNumRepK(aTensorTy, aShape[1]);
  ValueTable ha = getValuesFromDotOperandLayoutStruct(
      loadedA, numRepM, numRepK, aTensorTy.getElementType());
  ValueTable hb = getValuesFromDotOperandLayoutStruct(
      loadedB, numRepN, numRepK, aTensorTy.getElementType());
  auto dstElemTy = dTensorTy.getElementType();
  auto fc = typeConverter->unpackLLElements(loc, loadedC, rewriter, dstElemTy);

  auto vecTy = vec_ty(dstElemTy, 16);
  for (int m = 0; m < numRepM; ++m) {
    for (int n = 0; n < numRepN; ++n) {
      Value acc = undef(vecTy);
      for (unsigned v = 0; v < 16; ++v) {
        acc = insert_element(vecTy, acc, fc[m * numRepN * 16 + n * 16 + v],
                             i32_val(v));
      }

      for (size_t k = 0; k < numRepK; k++) {
        acc = generateMFMAOp(ha[{m, k}], hb[{n, k}], acc);
      }
      for (unsigned v = 0; v < 16; ++v) {
        fc[m * numRepN * 16 + n * 16 + v] =
            extract_element(dstElemTy, acc, i32_val(v));
      }
    }
  }
  // Type resElemTy = dTensorTy.getElementType();

  // replace with new packed result
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(fc.size(), dstElemTy));
  Value res = typeConverter->packLLElements(loc, fc, rewriter, structTy);
  rewriter.replaceOp(op, res);

  return success();
}

} // namespace


LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto mmaLayout = op.getResult()
                       .getType()
                       .cast<RankedTensorType>()
                       .getEncoding()
                       .cast<MmaEncodingAttr>();
  Value A = op.getA();
  Value B = op.getB();
  Value C = op.getC();

  DotOpMFMAConversionHelper helper(A.getType(), mmaLayout,
                                   getThreadId(rewriter, typeConverter, loc), rewriter,
                                   typeConverter, loc);
  auto ATensorTy = A.getType().cast<RankedTensorType>();
  auto BTensorTy = B.getType().cast<RankedTensorType>();
  assert(ATensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
         BTensorTy.getEncoding().isa<DotOperandEncodingAttr>() &&
         "Both $a and %b should be DotOperand layout.");

  Value loadedA, loadedB, loadedC;
  loadedA = adaptor.getA();
  loadedB = adaptor.getB();
  loadedC = helper.loadC(op.getC(), adaptor.getC());

  return helper.convertDot(A, B, C, op.getD(), loadedA, loadedB, loadedC, op,
                           adaptor);
}