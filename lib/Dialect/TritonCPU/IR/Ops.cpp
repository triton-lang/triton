#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonCPU/IR/Ops.cpp.inc"

// enum attribute definitions
#include "triton/Dialect/TritonCPU/IR/OpsEnums.cpp.inc"

namespace mlir::triton::cpu {

LogicalResult PrintOp::verify() {
  if (getOperands().size() > 1)
    return emitOpError("expects at most one operand");
  return success();
}

void ExternElementwiseOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getPure())
    return;
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult
DotOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = cast<VectorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);
  return success();
}

} // namespace mlir::triton::cpu
