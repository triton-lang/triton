#include "TypeConverter.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> llvm::Optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([&](RankedTensorType type) -> llvm::Optional<Type> {
    return convertTritonTensorType(type);
  });
  // Internally store float8 as int8
  addConversion([&](mlir::Float8E4M3FNType type) -> llvm::Optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E5M2Type type) -> llvm::Optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  // Internally store bfloat16 as int16
  addConversion([&](BFloat16Type type) -> llvm::Optional<Type> {
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
  auto structType = this->convertType(type);
  if (!structType.isa<LLVM::LLVMStructType>()) {
    return *resultVals.begin();
  }

  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
  // llvm::outs() << structType << "\n";
  for (const auto &v : llvm::enumerate(resultVals)) {
    assert(v.value() && "can not insert null values");
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

  unsigned numElementsPerThread = getElemsPerThread(type);
  SmallVector<Type, 4> types(numElementsPerThread, eltType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}
