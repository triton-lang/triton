#ifndef TRITON_IR_TYPES_H_
#define TRITON_IR_TYPES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Triton/IR/TypeInterfaces.h.inc"

#include "triton/Dialect/Triton/IR/OpsEnums.h.inc" // required by `Types.h.inc`
#include "triton/Dialect/Triton/IR/Types.h.inc"

namespace mlir {

namespace triton {

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
