#include "Dialect/Proton/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#define GET_OP_CLASSES
#include "Dialect/Proton/IR/Ops.cpp.inc"
#include "Dialect/Proton/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {
namespace proton {

// -- RecordOp --
void RecordOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

// -- CircularRecordOp --
void CircularRecordOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult CircularRecordOp::verify() {
  // TODO(fywkevin): checks the following:
  // 1. circular buffer size power of 2.
  // 2. function's noinline is false.
  return success();
}

// -- FinalizeOp --
void FinalizeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getDataMutable(),
                       mlir::triton::gpu::SharedMemory::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndexPtrMutable(),
                       mlir::triton::GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrMutable(),
                       mlir::triton::GlobalMemory::get());
}

// -- InitOp --
void InitOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       mlir::triton::GlobalMemory::get());
}

} // namespace proton
} // namespace triton
} // namespace mlir
