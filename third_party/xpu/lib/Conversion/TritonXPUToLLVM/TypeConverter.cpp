//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "triton/Conversion/TritonXPUToLLVM/TypeConverter.h" // TritonXPUToLLVMTypeConverter

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::xpu::getTotalElemsPerThread;

TritonXPUToLLVMTypeConverter::TritonXPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  // deal RankedTensorType to calculate elemNum
  addConversion([&](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
}

Type TritonXPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  auto ctx = type.getContext();
  auto pointeeType = type.getPointeeType();
  if (isa<RankedTensorType>(pointeeType)) {
    auto rankedTensorType = cast<RankedTensorType>(pointeeType);
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto eleType = rankedTensorType.getElementType();
    auto shape = rankedTensorType.getShape();
    SmallVector<Type, 4> types;
    // offsets
    for (size_t i = 0; i < shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 32));
    // shapes, strides
    for (size_t i = 0; i < 2 * shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 64));

    types.push_back(LLVM::LLVMPointerType::get(ctx, type.getAddressSpace()));

    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
  return LLVM::LLVMPointerType::get(ctx, type.getAddressSpace());
}

Type TritonXPUToLLVMTypeConverter::getElementTypeForStruct(
    TensorOrMemDesc type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  Type elemTy = convertType(type.getElementType());
  return elemTy;
}

Type TritonXPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
  Type eltType = getElementTypeForStruct(cast<TensorOrMemDesc>(type));

  unsigned numElementsPerThread = getTotalElemsPerThread(type);
  SmallVector<Type, 4> types(numElementsPerThread, eltType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}
