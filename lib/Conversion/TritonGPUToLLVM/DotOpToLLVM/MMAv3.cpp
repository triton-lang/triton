#include "../DotOpToLLVM.h"
#include "../Utility.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PTXAsmFormat.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;

namespace {

using ::mlir::triton::gpu::BlockedEncodingAttr;
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

  // Get the M and N of MFMA instruction shape.
  static std::tuple<int, int> getInstrShapeMN() { return {32, 32}; }

  static std::tuple<int, int> getRepMN(const RankedTensorType &tensorTy);

  // Get number of elements per thread for $a operand.
  static size_t getANumElemsPerThread(RankedTensorType operand, int wpt) {
    auto shape = operand.getShape();
    int numOfElem = getNumOfElems(operand);
    int repM = getNumRepM(operand, shape[0], wpt);
    int repK = getNumRepK_(operand, shape[1]);
    return repM * repK;
  }

  // Get number of elements per thread for $b operand.
  static size_t getBNumElemsPerThread(RankedTensorType operand, int wpt) {
    auto shape = operand.getShape();
    int numOfElem = getNumOfElems(operand);
    int repK = getNumRepK_(operand, shape[0]);
    int repN = getNumRepN(operand, shape[1], wpt);
    return repN * repK;
  }

  Type getShemPtrTy() const;

  Type getLoadElemTy();

  Type getMRetType() const;

  llvm::SmallVector<Value> computeOffsetsA(Value waveM, Value laneId, int wptA,
                                           int numOfElems, int numM, int numK,
                                           Value cSwizzleOffset) const;
  llvm::SmallVector<Value> computeOffsetsB(Value waveN, Value laneId, int wptB,
                                           int numOfElems, int numK, int numN,
                                           Value cSwizzleOffset) const;

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

  static int getNumOfElems(Type operand) {
    auto matrixCoreType = getMatrixCoreTypeFromOperand(operand);
    int instrM = getMFMAInstrShape(matrixCoreType)[0];
    int instrK = getMFMAInstrShape(matrixCoreType)[2];
    return std::max<int>(instrM * instrK / 64, 1);
  }

  int getNumOfElems() const {
    int instrM = getMFMAInstrShape(mfmaType)[0];
    int instrK = getMFMAInstrShape(mfmaType)[2];
    return std::max<int>(instrM * instrK / 64, 1);
  }

  // Get a waveId for M axis.
  Value getWaveM(int M) const {
    auto instrShape = getMFMAInstrShape();
    return urem(urem(wave, i32_val(wpt[0])), i32_val(M / instrShape[0]));
  }

  // Get a waveId for N axis.
  Value getWaveN(int N) const {
    auto instrShape = getMFMAInstrShape();
    Value waveMN = udiv(wave, i32_val(wpt[0]));
    return urem(urem(waveMN, i32_val(wpt[1])), i32_val(N / instrShape[1]));
  }

  // Loading $a from lds to registers, returns a LLVM::Struct.
  Value loadA(Value tensor, const SharedMemoryObject &smemObj) const;

  // Loading $b from lds to registers, returns a LLVM::Struct.
  Value loadB(Value tensor, const SharedMemoryObject &smemObj) const;

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

Type DotOpMFMAConversionHelper::getShemPtrTy() const {
  switch (mfmaType) {
  case MatrixCoreType::FP32_FP16_FP16_FP32:
    return ptr_ty(type::f16Ty(ctx), 3);
  case MatrixCoreType::FP32_BF16_BF16_FP32:
    return ptr_ty(type::i16Ty(ctx), 3);
  case MatrixCoreType::FP32_FP32_FP32_FP32:
    return ptr_ty(type::f32Ty(ctx), 3);
  case MatrixCoreType::INT32_INT8_INT8_INT32:
    return ptr_ty(type::i8Ty(ctx), 3);
  case MatrixCoreType::FP64_FP64_FP64_FP64:
    return ptr_ty(type::f64Ty(ctx), 3);
  default:
    llvm::report_fatal_error("MFMA data type not supported");
  }
  return Type{};
}

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

llvm::SmallVector<Value>
DotOpMFMAConversionHelper::computeOffsetsA(Value waveM, Value laneId, int wptA,
                                           int numOfElems, int numM, int numK,
                                           Value cSwizzleOffset) const {
  SmallVector<Value> offsets(numM * numK * numOfElems);
  auto mfmaShape = getMFMAInstrShape();
  int lineSize = mfmaShape[2] * numK;
  int blockSize = mfmaShape[0] * numM * lineSize;
  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value waveHalf = udiv(laneId, _32);

  // Value waveOffset = mul(waveM, i32_val(blockSize));
  Value waveOffset = wptA > 1 ? mul(waveM, i32_val(mfmaShape[0] * lineSize))
                              : mul(waveM, i32_val(blockSize));
  Value colOffset = select(icmp_uge(laneId, _32), i32_val(numOfElems), _0);

  for (int block = 0; block < numM; ++block) {
    // Value blockOffset = i32_val(block * mfmaShape[0] * lineSize);
    Value blockOffset = wptA > 1 ? i32_val(block * blockSize)
                                 : i32_val(block * mfmaShape[0] * lineSize);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * mfmaShape[2]);
      for (int elem = 0; elem < numOfElems; ++elem) {
        Value rowOffset =
            add(mul(urem(laneId, _32), i32_val(lineSize)), i32_val(elem));
        Value elemOffset = add(rowOffset, colOffset);
        Value offset =
            add(add(add(waveOffset, blockOffset), tileOffset), elemOffset);
        offsets[numK * numOfElems * block + numOfElems * tile + elem] = offset;
      }
    }
  }
  return offsets;
}

llvm::SmallVector<Value>
DotOpMFMAConversionHelper::computeOffsetsB(Value waveN, Value laneId, int wptB,
                                           int numOfElems, int numK, int numN,
                                           Value cSwizzleOffset) const {
  SmallVector<Value> offsets(numK * numN * numOfElems);
  auto mfmaShape = getMFMAInstrShape();

  int lineSize = wpt[1] * mfmaShape[1] * numN;
  Value _0 = i32_val(0);
  Value _32 = i32_val(32);
  Value waveOffset = wptB > 1 ? mul(waveN, i32_val(mfmaShape[1]))
                              : mul(waveN, i32_val(mfmaShape[1] * numN));
  // Value waveOffset = mul(waveN, i32_val(mfmaShape[1] * numN));
  Value colOffset = urem(laneId, _32);

  for (int block = 0; block < numN; ++block) {
    // Value blockOffset = i32_val(block * mfmaShape[1]);
    Value blockOffset = wptB > 1 ? i32_val(block * mfmaShape[1] * numN)
                                 : i32_val(block * mfmaShape[1]);
    for (int tile = 0; tile < numK; ++tile) {
      Value tileOffset = i32_val(tile * mfmaShape[2] * lineSize);
      for (int elem = 0; elem < numOfElems; ++elem) {
        Value halfOffset =
            select(icmp_uge(laneId, _32), i32_val(numOfElems * lineSize), _0);
        Value rowOffset = add(i32_val(elem * lineSize), halfOffset);
        Value elemOffset = add(rowOffset, colOffset);
        Value offset =
            add(add(add(waveOffset, blockOffset), tileOffset), elemOffset);
        offsets[numK * numOfElems * block + numOfElems * tile + elem] = offset;
      }
    }
  }
  return offsets;
}

Value DotOpMFMAConversionHelper::loadA(
    Value tensor, const SharedMemoryObject &smemObj) const {
  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape(aTensorTy.getShape().begin(),
                             aTensorTy.getShape().end());
  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  SmallVector<Value> ha;
  auto [mfmaInstrM, mfmaInstrN, mfmaInstrK] = getMFMAInstrShape(aTensorTy);

  int numRepM = getNumRepM(aTensorTy, shape[0]);
  int numRepK = getNumRepK(aTensorTy, shape[1]);

  Value waveM = getWaveM(shape[0]);
  int numOfElems = getNumOfElems(aTensorTy);
  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
  int macroTileM =
      std::max<int>(shape[0] / (mmaLayout.getWarpsPerCTA()[0] * 32), 1);
  int wptM = std::min<int>(mmaLayout.getWarpsPerCTA()[0], macroTileM);
  int macroTileN =
      std::max<int>(shape[1] / (mmaLayout.getWarpsPerCTA()[1] * 32), 1);
  int wptN = std::min<int>(mmaLayout.getWarpsPerCTA()[1], macroTileN);
  int wpt = std::max<int>(wptM, wptN);
  auto offsets = computeOffsetsA(waveM, lane, wpt, numOfElems, numRepM, numRepK,
                                 cSwizzleOffset);

  Value smemBase = smemObj.getBaseBeforeSwizzle(order[0], loc, rewriter);

  Type smemPtrTy = getShemPtrTy();
  Type elemTy = aTensorTy.getElementType();
  for (int m = 0; m < numRepM; ++m) {
    for (int k = 0; k < numRepK; ++k) {
      auto vecTy = vec_ty(elemTy, numOfElems);
      Value valVec = undef(vecTy);
      for (unsigned elem = 0; elem < numOfElems; ++elem) {
        Value elemOffset =
            offsets[m * numOfElems * numRepK + k * numOfElems + elem];
        Value elemValue = load(gep(smemPtrTy, smemBase, elemOffset));
        if (numOfElems > 1)
          valVec = insert_element(vecTy, valVec, elemValue, i32_val(elem));
        else
          valVec = elemValue;
      }
      if (elemTy == i8_ty)
        valVec = bitcast(valVec, i32_ty);
      ha.push_back(valVec);
    }
  }

  elemTy = ha[0].getType();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(ha.size(), elemTy));
  auto result = typeConverter->packLLElements(loc, ha, rewriter, structTy);
  return result;
}

Value DotOpMFMAConversionHelper::loadB(
    Value tensor, const SharedMemoryObject &smemObj) const {
  auto bTensorTy = tensor.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape(bTensorTy.getShape().begin(),
                             bTensorTy.getShape().end());
  auto sharedLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  SmallVector<Value> hb;
  auto [mfmaInstrM, mfmaInstrN, mfmaInstrK] = getMFMAInstrShape(bTensorTy);

  int numRepK = getNumRepK(bTensorTy, shape[0]);
  int numRepN = getNumRepN(bTensorTy, shape[1]);

  Value waveN = getWaveN(shape[1]);
  int numOfElems = getNumOfElems(bTensorTy);
  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);

  int macroTileM =
      std::max<int>(shape[0] / (mmaLayout.getWarpsPerCTA()[0] * 32), 1);
  int wptM = std::min<int>(mmaLayout.getWarpsPerCTA()[0], macroTileM);
  int macroTileN =
      std::max<int>(shape[1] / (mmaLayout.getWarpsPerCTA()[1] * 32), 1);
  int wptN = std::min<int>(mmaLayout.getWarpsPerCTA()[1], macroTileN);
  int wpt = std::max<int>(wptM, wptN);

  auto offsets = computeOffsetsB(waveN, lane, wpt, numOfElems, numRepK, numRepN,
                                 cSwizzleOffset);

  Value smemBase = smemObj.getBaseBeforeSwizzle(order[0], loc, rewriter);

  Type smemPtrTy = getShemPtrTy();

  Type elemTy = bTensorTy.getElementType();
  for (int n = 0; n < numRepN; ++n) {
    for (int k = 0; k < numRepK; ++k) {
      auto vecTy = vec_ty(bTensorTy.getElementType(), numOfElems);
      Value valVec = undef(vecTy);
      for (unsigned elem = 0; elem < numOfElems; ++elem) {
        Value elemOffset =
            offsets[n * numOfElems * numRepK + k * numOfElems + elem];
        Value elemValue = load(gep(smemPtrTy, smemBase, elemOffset));
        if (numOfElems > 1)
          valVec = insert_element(vecTy, valVec, elemValue, i32_val(elem));
        else
          valVec = elemValue;
      }
      if (elemTy == i8_ty)
        valVec = bitcast(valVec, i32_ty);
      hb.push_back(valVec);
    }
  }

  elemTy = hb[0].getType();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(hb.size(), elemTy));
  auto result = typeConverter->packLLElements(loc, hb, rewriter, structTy);
  return result;
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