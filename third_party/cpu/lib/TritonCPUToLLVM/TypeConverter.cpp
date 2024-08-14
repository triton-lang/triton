#include "TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;

TritonCPUToLLVMTypeConverter::TritonCPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([this](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
}

Type TritonCPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  auto ctx = type.getContext();
  auto pointeeType = type.getPointeeType();
  if (isa<RankedTensorType>(pointeeType)) {
    // struct {
    //   ptr base_ptr;
    //   array<rank x i64> offsets;
    //   array<rank x i64> shape;
    //   array<rank x i64> strides;
    // }
    auto tensorTy = cast<RankedTensorType>(pointeeType);
    auto rank = tensorTy.getShape().size();
    auto i64Ty = IntegerType::get(ctx, 64);
    SmallVector<Type, 4> types;
    types.push_back(LLVM::LLVMPointerType::get(ctx));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank));
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
  return LLVM::LLVMPointerType::get(ctx);
}

Type TritonCPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type) {
  if (isa<PointerType>(type.getElementType()))
    return VectorType::get(type.getShape(),
                           IntegerType::get(type.getContext(), 64));
  llvm_unreachable("No tensor types are expected in TTCIR");
}
