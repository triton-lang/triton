#include "TypeConverter.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

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

// Get number of elements per thread for $a operand.
static size_t getANumElemsPerThread(RankedTensorType operand, int wpt) {
  auto shape = operand.getShape();
  int numOfElem = getNumOfElems(operand);
  int repM = getNumRepM(operand, shape[0], wpt);
  int repK = getNumRepK(operand, shape[1]);
  return repM * repK;
}
// Get number of elements per thread for $b operand.
static size_t getBNumElemsPerThread(RankedTensorType operand, int wpt) {
  auto shape = operand.getShape();
  int numOfElem = getNumOfElems(operand);
  int repK = getNumRepK(operand, shape[0]);
  int repN = getNumRepN(operand, shape[1], wpt);
  return repN * repK;
}

} // namespace


TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([&](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
  // Internally store float8 as int8
  addConversion([&](mlir::Float8E4M3FNType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E5M2Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  // Internally store bfloat16 as int16
  addConversion([&](BFloat16Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 16);
  });
}

Type TritonGPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  // Recursively translate pointee type
  return LLVM::LLVMPointerType::get(convertType(type.getPointeeType()),
                                    type.getAddressSpace());
}

Value TritonGPUToLLVMTypeConverter::packLLElements(
    Location loc, ValueRange resultVals, ConversionPatternRewriter &rewriter,
    Type type) {
  auto structType = this->convertType(type).dyn_cast<LLVM::LLVMStructType>();
  if (!structType) {
    assert(resultVals.size() == 1);
    return *resultVals.begin();
  }

  auto elementTypes = structType.getBody();
  if (elementTypes.size() != resultVals.size()) {
    emitError(loc) << " size mismatch when packing elements for LLVM struct"
                   << " expected " << elementTypes.size() << " but got "
                   << resultVals.size();
  }
  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
  for (const auto &v : llvm::enumerate(resultVals)) {
    if (!v.value()) {
      emitError(loc)
          << "cannot insert null values into struct, but tried to insert"
          << v.value();
    }
    if (v.value().getType() != elementTypes[v.index()]) {
      emitError(loc) << "invalid element type in packLLEElements. Expected "
                     << elementTypes[v.index()] << " but got "
                     << v.value().getType();
    }
    llvmStruct = insert_val(structType, llvmStruct, v.value(), v.index());
  }
  return llvmStruct;
}

SmallVector<Value> TritonGPUToLLVMTypeConverter::unpackLLElements(
    Location loc, Value llvmStruct, ConversionPatternRewriter &rewriter,
    Type type) {
  assert(bool(llvmStruct) && "can not unpack null values");
  if (llvmStruct.getType().isIntOrIndexOrFloat() ||
      llvmStruct.getType().isa<triton::PointerType>() ||
      llvmStruct.getType().isa<LLVM::LLVMPointerType>())
    return {llvmStruct};
  ArrayRef<Type> types =
      llvmStruct.getType().cast<LLVM::LLVMStructType>().getBody();
  SmallVector<Value> results(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    results[i] = extract_val(type, llvmStruct, i);
  }
  return results;
}

Type TritonGPUToLLVMTypeConverter::getElementTypeForStruct(
    RankedTensorType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  Type elemTy = convertType(type.getElementType());
  auto dotOpLayout = layout.dyn_cast<DotOperandEncodingAttr>();
  if (!dotOpLayout)
    return elemTy;
  auto mmaParent = dotOpLayout.getParent().dyn_cast<MmaEncodingAttr>();
  if (!mmaParent)
    return elemTy;
  if (mmaParent.isAmpere()) {
    int bitwidth = elemTy.getIntOrFloatBitWidth();
    // sub-word integer types need to be packed for perf reasons
    if (elemTy.isa<IntegerType>() && bitwidth < 32)
      return IntegerType::get(ctx, 32);
    // TODO: unify everything to use packed integer-types
    // otherwise, vector types are ok
    const llvm::DenseMap<int, Type> elemTyMap = {
        {32, vec_ty(elemTy, 1)},
        {16, vec_ty(elemTy, 2)},
        {8, vec_ty(elemTy, 4)},
    };
    return elemTyMap.lookup(bitwidth);
#ifdef USE_ROCM
  } else if (mmaParent.isMI200()) {
    const llvm::DenseMap<int, Type> targetTyMap = {
        {32, elemTy},
        {16, vec_ty(elemTy, 4)},
        {8, IntegerType::get(ctx, 32)},
    };
    Type targetTy = targetTyMap.lookup(elemTy.getIntOrFloatBitWidth());
    auto wpt = mmaParent.getWarpsPerCTA();
    if (dotOpLayout.getOpIdx() == 0) { // $a
      auto elems =
          getANumElemsPerThread(type, wpt[0]);
      return struct_ty(SmallVector<Type>(elems, targetTy));
    }
    if (dotOpLayout.getOpIdx() == 1) { // $b
      auto elems =
          getBNumElemsPerThread(type, wpt[1]);
      return struct_ty(SmallVector<Type>(elems, targetTy));
    }
    return Type();
  // llvm_unreachable("if (mmaLayout.isMI200()) not implemented");
#endif
  } else {
    assert(mmaParent.isVolta());
    return vec_ty(elemTy, 2);
  }
}

Type TritonGPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
  Type eltType = getElementTypeForStruct(type);

  if (auto shared_layout = layout.dyn_cast<SharedEncodingAttr>()) {
    SmallVector<Type, 4> types;
    // base ptr
    auto ptrType = LLVM::LLVMPointerType::get(eltType, 3);
    types.push_back(ptrType);
    // shape dims
    auto rank = type.getRank();
    // offsets + strides
    for (auto i = 0; i < rank * 2; i++) {
      types.push_back(IntegerType::get(ctx, 32));
    }
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }

  unsigned numElementsPerThread = getTotalElemsPerThread(type);
  SmallVector<Type, 4> types(numElementsPerThread, eltType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}
