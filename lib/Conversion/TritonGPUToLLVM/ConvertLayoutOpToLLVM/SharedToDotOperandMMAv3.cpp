#include "../ConvertLayoutOpToLLVM.h"
#include "../Utility.h"

using CoordTy = SmallVector<Value>;
using ValueTable = std::map<std::pair<int, int>, std::pair<Value, Value>>;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

enum class MatrixCoreType : uint8_t {
  // D = AB + C
  FP32_FP16_FP16_FP32 = 0, // default
  FP32_BF16_BF16_FP32,
  FP32_FP32_FP32_FP32,
  FP64_FP64_FP64_FP64,
  INT32_INT8_INT8_INT32,
  NOT_APPLICABLE,
};

MatrixCoreType
getMatrixCoreTypeFromOperand(Type operandTy) {
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

Type getShemPtrTy(MLIRContext *ctx, MatrixCoreType mfmaType) {
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

inline static const std::map<MatrixCoreType, llvm::SmallVector<int>>
    mfmaInstrShape = { // m, n, k
        {MatrixCoreType::FP32_FP16_FP16_FP32, {32, 32, 8}},
        {MatrixCoreType::FP32_BF16_BF16_FP32, {32, 32, 4}},
        {MatrixCoreType::FP32_FP32_FP32_FP32, {32, 32, 2}},
        {MatrixCoreType::INT32_INT8_INT8_INT32, {32, 32, 8}},
        {MatrixCoreType::FP64_FP64_FP64_FP64, {16, 16, 4}}};

ArrayRef<int> getMFMAInstrShape(MatrixCoreType matrixCoreType) {
  assert(matrixCoreType != MatrixCoreType::NOT_APPLICABLE &&
         "Unknown MFMA type found.");
  return mfmaInstrShape.at(matrixCoreType);
}

std::tuple<int, int, int> getMFMAInstrShape(Type operandTy) {
  auto coreType = getMatrixCoreTypeFromOperand(operandTy);
  auto instrShape = getMFMAInstrShape(coreType);
  int instrM = instrShape[0];
  int instrN = instrShape[1];
  int instrK = instrShape[2];
  return std::make_tuple(instrM, instrN, instrK);
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
static int getNumRepK(Type operand, int K) {
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

// Get a waveId for M axis.
Value getWaveM(ConversionPatternRewriter &rewriter, Location loc, Value wave, const ArrayRef<unsigned int> &wpt, int elemPerInstr, int M) {
  return urem(urem(wave, i32_val(wpt[0])), i32_val(M / elemPerInstr));
}
// Get a waveId for N axis.
Value getWaveN(ConversionPatternRewriter &rewriter, Location loc, Value wave, const ArrayRef<unsigned int> &wpt, int elemPerInstr, int N) {
  Value waveMN = udiv(wave, i32_val(wpt[0]));
  return urem(urem(waveMN, i32_val(wpt[1])), i32_val(N / elemPerInstr));
}

} // namespace

namespace SharedToDotOperandMMAv3 {

llvm::SmallVector<Value> computeOffsetsA(ConversionPatternRewriter &rewriter,
                                         Location loc, const ArrayRef<int> &mfmaShape,
                                         Value waveM, Value laneId, int wptA,
                                         int numOfElems, int numM, int numK,
                                         Value cSwizzleOffset) {
  SmallVector<Value> offsets(numM * numK * numOfElems);
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

llvm::SmallVector<Value> computeOffsetsB(ConversionPatternRewriter &rewriter,
                                         Location loc, const ArrayRef<int> &mfmaShape,
                                         int warpsPerGroupN,
                                         Value waveN, Value laneId, int wptB,
                                         int numOfElems, int numK, int numN,
                                         Value cSwizzleOffset) {
  SmallVector<Value> offsets(numK * numN * numOfElems);

  int lineSize = warpsPerGroupN * mfmaShape[1] * numN;
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

Value loadA(ConversionPatternRewriter &rewriter,
    Location loc, Value thread, DotOperandEncodingAttr encoding,
    TritonGPUToLLVMTypeConverter *typeConverter,
    Value tensor, const SharedMemoryObject &smemObj) {
  auto mmaLayout = encoding.getParent().cast<MmaEncodingAttr>();
  MLIRContext *ctx = mmaLayout.getContext();
  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();

  auto aTensorTy = tensor.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape(aTensorTy.getShape().begin(),
                             aTensorTy.getShape().end());
  auto sharedLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  SmallVector<Value> ha;
  auto mfmaInstrShape = getMFMAInstrShape(aTensorTy);
  auto [mfmaInstrM, mfmaInstrN, mfmaInstrK] = mfmaInstrShape;

  int numRepM = getNumRepM(aTensorTy, shape[0], warpsPerCTA[0]);
  int numRepK = getNumRepK(aTensorTy, shape[1]);

  Value waveSize = i32_val(64);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveM = getWaveM(rewriter, loc, wave, warpsPerCTA, mfmaInstrM, shape[0]);
  int numOfElems = getNumOfElems(aTensorTy);
  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
  // TODO make macro tile size granilarity configurable
  int macroTileM =
      std::max<int>(shape[0] / (mmaLayout.getWarpsPerCTA()[0] * 32), 1);
  int wptM = std::min<int>(mmaLayout.getWarpsPerCTA()[0], macroTileM);
  int macroTileN =
      std::max<int>(shape[1] / (mmaLayout.getWarpsPerCTA()[1] * 32), 1);
  int wptN = std::min<int>(mmaLayout.getWarpsPerCTA()[1], macroTileN);
  int wpt = std::max<int>(wptM, wptN);
  auto mfmaShape = getMFMAInstrShape(getMatrixCoreTypeFromOperand(aTensorTy));
  auto offsets = computeOffsetsA(rewriter, loc, mfmaShape, waveM, lane, wpt, numOfElems, numRepM, numRepK,
                                 cSwizzleOffset);

  Value smemBase = smemObj.getBaseBeforeSwizzle(order[0], loc, rewriter);

  Type smemPtrTy = getShemPtrTy(ctx, getMatrixCoreTypeFromOperand(aTensorTy));
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

Value loadB(ConversionPatternRewriter &rewriter,
    Location loc, Value thread, DotOperandEncodingAttr encoding,
    TritonGPUToLLVMTypeConverter *typeConverter,
    Value tensor, const SharedMemoryObject &smemObj) {
  auto mmaLayout = encoding.getParent().cast<MmaEncodingAttr>();
  MLIRContext *ctx = mmaLayout.getContext();
  auto warpsPerCTA = mmaLayout.getWarpsPerCTA();

  auto bTensorTy = tensor.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape(bTensorTy.getShape().begin(),
                             bTensorTy.getShape().end());
  auto sharedLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto order = sharedLayout.getOrder();

  SmallVector<Value> hb;
  auto mfmaInstrShape = getMFMAInstrShape(bTensorTy);
  auto [mfmaInstrM, mfmaInstrN, mfmaInstrK] = mfmaInstrShape;

  int numRepK = getNumRepK(bTensorTy, shape[0]);
  int numRepN = getNumRepN(bTensorTy, shape[1], warpsPerCTA[1]);

  Value waveSize = i32_val(64);
  Value wave = udiv(thread, waveSize);
  Value lane = urem(thread, waveSize);

  Value waveN = getWaveN(rewriter, loc, wave, mmaLayout.getWarpsPerCTA(), mfmaInstrN, shape[1]);
  int numOfElems = getNumOfElems(bTensorTy);
  Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);

  int macroTileM =
      std::max<int>(shape[0] / (mmaLayout.getWarpsPerCTA()[0] * 32), 1);
  int wptM = std::min<int>(mmaLayout.getWarpsPerCTA()[0], macroTileM);
  int macroTileN =
      std::max<int>(shape[1] / (mmaLayout.getWarpsPerCTA()[1] * 32), 1);
  int wptN = std::min<int>(mmaLayout.getWarpsPerCTA()[1], macroTileN);
  int wpt = std::max<int>(wptM, wptN);

  auto mfmaShape = getMFMAInstrShape(getMatrixCoreTypeFromOperand(bTensorTy));
  int warpsPerGroupN = mmaLayout.getWarpsPerCTA()[1];
  auto offsets = computeOffsetsB(rewriter, loc, mfmaShape, warpsPerGroupN, waveN, lane, wpt, numOfElems, numRepK, numRepN,
                                 cSwizzleOffset);

  Value smemBase = smemObj.getBaseBeforeSwizzle(order[0], loc, rewriter);

  Type smemPtrTy = getShemPtrTy(ctx, getMatrixCoreTypeFromOperand(bTensorTy));

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

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    TritonGPUToLLVMTypeConverter *typeConverter, Value thread) {
  switch (opIdx) {
  case 0:
    // operand $a
    return loadA(rewriter, loc, thread, encoding, typeConverter, tensor, smemObj);
  case 1:
    // operand $b
    return loadB(rewriter, loc, thread, encoding, typeConverter, tensor, smemObj);
  default:
    assert(false && "unexpected operand idx");
    return Value();
  }
}

} // namespace SharedToDotOperandMMAv3
