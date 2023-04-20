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

struct GridDescription {
  Value thread, lane, wave, waveSize;
};

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

struct DotOpMFMAConversionHelper {
  MmaEncodingAttr mmaLayout;

  ConversionPatternRewriter &rewriter;
  TritonGPUToLLVMTypeConverter *typeConverter;
  Location loc;
  MLIRContext *ctx{};

  explicit DotOpMFMAConversionHelper(
      MmaEncodingAttr mmaLayout,
      ConversionPatternRewriter &rewriter,
      TritonGPUToLLVMTypeConverter *typeConverter, Location loc)
      : mmaLayout(mmaLayout),
        rewriter(rewriter), typeConverter(typeConverter), loc(loc),
        ctx(mmaLayout.getContext()) { }

  Value getThreadId() const {
    auto llvmIndexTy = typeConverter->getIndexType();
    auto tid = rewriter.create<::mlir::gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x);
    return rewriter.create<arith::TruncIOp>(loc, i32_ty, tid);
  }

  GridDescription generateGridDescription() const {
    GridDescription gd;
    gd.thread = getThreadId();
    gd.waveSize = i32_val(64);
    gd.lane = urem(gd.thread, gd.waveSize);
    gd.wave = udiv(gd.thread, gd.waveSize);
    return gd;
  }

  Value generateMFMAOp(MatrixCoreType mfmaTy, Value valA, Value valB, Value valC) const {
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

  static ArrayRef<int> getMFMAInstrShape(MatrixCoreType matrixCoreType) {
    assert(matrixCoreType != MatrixCoreType::NOT_APPLICABLE &&
           "Unknown MFMA type found.");
    static const std::map<MatrixCoreType, const llvm::SmallVector<int>>
    mfmaInstrShape = { // m, n, k
        {MatrixCoreType::FP32_FP16_FP16_FP32, {32, 32, 8}},
        {MatrixCoreType::FP32_BF16_BF16_FP32, {32, 32, 4}},
        {MatrixCoreType::FP32_FP32_FP32_FP32, {32, 32, 2}},
        {MatrixCoreType::INT32_INT8_INT8_INT32, {32, 32, 8}},
        {MatrixCoreType::FP64_FP64_FP64_FP64, {16, 16, 4}}};
    return mfmaInstrShape.at(matrixCoreType);
  }

  static MatrixCoreType getMatrixCoreTypeFromDot(DotOp op){
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

  static int getNumRepM(const ArrayRef<int> &instrSize,
                        int M, const ArrayRef<unsigned> &warpsPerCTA) {
    int instrM = instrSize[0];
    return std::max<int>(M / (warpsPerCTA[0] * instrM), 1);
  }

  static int getNumRepN(const ArrayRef<int> &instrSize,
                        int N, const ArrayRef<unsigned> & warpsPerCTA) {
    int instrN = instrSize[1];
    return std::max<int>(N / (warpsPerCTA[1] * instrN), 1);
  }

  static int getNumRepK(const ArrayRef<int> &instrSize, int K) {
    int instrK = instrSize[2];
    return std::max<int>(K / instrK, 1);
  }

  // Conduct the Dot conversion.
  LogicalResult convertDot(DotOp op, DotOpAdaptor adaptor) const {

    GridDescription gd = generateGridDescription();

    auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
    auto mfmaTy = getMatrixCoreTypeFromDot(op);

    auto instrShape = getMFMAInstrShape(mfmaTy);

    Value a = op.getA();
    Value d = op.getD();
    auto aTensorTy = a.getType().cast<RankedTensorType>();
    auto dTensorTy = d.getType().cast<RankedTensorType>();
  
    Value loadedA = adaptor.getA();
    Value loadedB = adaptor.getB();
    Value loadedC = adaptor.getC();
  
    SmallVector<int64_t> aShape(aTensorTy.getShape().begin(),
                                aTensorTy.getShape().end());
  
    auto dShape = dTensorTy.getShape();
  
    int numRepM = getNumRepM(instrShape, dShape[0], warpsPerCTA);
    int numRepN = getNumRepN(instrShape, dShape[1], warpsPerCTA);
    int numRepK = getNumRepK(instrShape, aShape[1]);

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
  auto rankedTType =
      [](Value tensor) { return tensor.getType().cast<RankedTensorType>(); };

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
