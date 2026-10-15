#ifndef TRITON_IR_TYPES_H_
#define TRITON_IR_TYPES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Triton/IR/TypeInterfaces.h.inc"

// clang-format off
// TypesEnums.h.inc must precede Types.h.inc: the generated PointerType
// declarations reference PtrAddrSpace.
#include "triton/Dialect/Triton/IR/TypesEnums.h.inc"
#include "triton/Dialect/Triton/IR/Types.h.inc"
// clang-format on

namespace mlir {

namespace triton {

// Whether a scalar type has a distinct low-precision storage interpretation.
bool isLowPrecisionStorageType(Type type);

// Bit width of a scalar integer, floating-point, or packed storage type.
// A packed E2M1 pair occupies one byte, not one 4-bit element.
unsigned getStorageBitWidth(Type type);

unsigned getPointeeBitWidth(Type type);

Type getPointeeType(Type type);

Type getPointerType(Type type,
                    PtrAddrSpace addressSpace = PtrAddrSpace::Global);

PtrAddrSpace getAddressSpace(Type type);

Type getI1SameShape(Type type);

Type getI32SameShape(Type type);

Type getPointerTypeSameShape(Type type);

bool elementTypeMatchesPointee(Type valueTy, Type ptrTy);

} // namespace triton

} // namespace mlir

#endif // TRITON_IR_TYPES_H_
