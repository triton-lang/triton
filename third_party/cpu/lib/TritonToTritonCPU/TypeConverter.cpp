#include "TypeConverter.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

TritonToTritonCPUTypeConverter::TritonToTritonCPUTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion([](triton::PointerType ptrTy) -> Type {
    if (triton::isTensorPointerType(ptrTy)) {
      // Tensor pointer is translated into a memref
      auto tensorTy = dyn_cast<RankedTensorType>(ptrTy.getPointeeType());
      auto elemTy = tensorTy.getElementType();
      // TODO: use dynamic strides
      SmallVector<int64_t> shape(tensorTy.getRank(), ShapedType::kDynamic);
      return MemRefType::get(shape, elemTy);
    }
    return IntegerType::get(ptrTy.getContext(), 64);
  });
  addConversion([this](RankedTensorType tensorTy) -> Type {
    Type elemTy = convertType(tensorTy.getElementType());
    return VectorType::get(tensorTy.getShape(), elemTy);
  });

  // Converted ops produce vectors instead of tensors. Provide conversion
  // here for users. Also, convert pointers when required.
  addSourceMaterialization([&](OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (isa<PointerType>(type))
      return builder.create<IntToPtrOp>(loc, type, inputs);
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
        .getResult(0);
  });

  // Converted loads and stores consume memrefs instead of pointers, use extract
  // op to get them. Also, provide conversion for vector users and pointer
  // casts.
  addTargetMaterialization([&](OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (type.isInteger() && isa<PointerType>(inputs.front().getType()))
      return builder.create<PtrToIntOp>(loc, type, inputs);
    if (isa<VectorType>(type))
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    if (isa<MemRefType>(type))
      return builder.create<ExtractMemRefOp>(loc, type, inputs);
    llvm_unreachable("Unexpected target materizalization");
  });
}
