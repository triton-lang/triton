#include "TypeConverter.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

TritonToTritonCPUTypeConverter::TritonToTritonCPUTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion([this](RankedTensorType tensorTy) -> Type {
    Type elemTy = convertType(tensorTy.getElementType());
    if (isa<triton::PointerType>(elemTy))
      elemTy = IntegerType::get(tensorTy.getContext(), 64);
    return VectorType::get(tensorTy.getShape(), elemTy);
  });

  // Converted ops produce vectors instead of tensors. Provide conversion
  // here for users.
  addSourceMaterialization([&](OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
        .getResult(0);
  });

  // Provide conversion for vector users.
  addTargetMaterialization([&](OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (isa<VectorType>(type))
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    llvm_unreachable("Unexpected target materizalization");
  });
}
