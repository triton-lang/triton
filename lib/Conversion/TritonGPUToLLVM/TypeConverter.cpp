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

// TODO merge this function with same function from SharedToDotOperandMMAv3
SmallVector<int64_t> getMFMAInstrShape(Type operandTy) {
  auto tensorTy = operandTy.cast<RankedTensorType>();
  auto elemTy = tensorTy.getElementType();
  if (elemTy.isF16())
    return {32l, 32l, 8l}; // FP32_FP16_FP16_FP32
  if (elemTy.isF32())
    return {32l, 32l, 2l}; // FP32_FP32_FP32_FP32;
  if (elemTy.isBF16())
    return {32l, 32l, 4l}; // FP32_BF16_BF16_FP32;
  if (elemTy.isInteger(8))
    return {32l, 32l, 8l}; // INT32_INT8_INT8_INT32;
  if (elemTy.isF64())
    return {16l, 16l, 4l}; // FP64_FP64_FP64_FP64;
  assert(false && "unsupported operand data type");
  return {};
}

// Get number of elements per thread for $a operand.
static size_t getMFMAOperandNumElemsPerThread(RankedTensorType operand) {
  auto encoding = operand.getEncoding().cast<DotOperandEncodingAttr>();
  auto mfmaInstrShape = getMFMAInstrShape(operand);
  auto reps = encoding.getMMAv3Rep(operand.getShape(), mfmaInstrShape);
  return reps[0] * reps[1];
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
    auto elems = getMFMAOperandNumElemsPerThread(type);
    return struct_ty(SmallVector<Type>(elems, targetTy));
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
