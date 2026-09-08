#ifndef TRITONGPU_IR_TRAITS_H_
#define TRITONGPU_IR_TRAITS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {
namespace OpTrait {

namespace impl {
LogicalResult verifyEquivalentMemDescType(Type typeA, Type typeB);
LogicalResult verifyMemDescIndexPhase(Operation *op);
LogicalResult verifyMemDescLayouts(Operation *op);
} // namespace impl

// Trait applied to all Triton GPU MLIR ops.  Checks that the layouts of
// MemDescs are valid.
template <class ConcreteType>
class VerifyMemDescLayoutsTrait
    : public TraitBase<ConcreteType, VerifyMemDescLayoutsTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyMemDescLayouts(op);
  }
};

template <typename ConcreteType>
class MemDescViewTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, MemDescViewTrait> {
  // Optional: Add methods or verification logic here
};

// Marks operations that produce, consume, or forward logical-index phases.
// Individual operations may impose additional operand constraints.
template <typename ConcreteType>
class SupportsMemDescIndexPhaseTrait
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      SupportsMemDescIndexPhaseTrait> {};

template <typename ConcreteType>
class LocalLoadTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, LocalLoadTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyMemDescIndexPhase(op);
  }
};

template <typename ConcreteType>
class MemWaitOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, MemWaitOpTrait> {
  // Optional: Add methods or verification logic here
};

// Copies from global memory into local memory. Such copies are asynchronous
template <typename ConcreteType>
class GlobalToLocalCopyTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, GlobalToLocalCopyTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyMemDescIndexPhase(op);
  }
};

} // namespace OpTrait
} // namespace mlir

namespace mlir::triton::gpu {
// Returns whether the operation supports logical-index phase descriptors in
// its current execution context.
bool supportsMemDescIndexPhase(Operation *op);
} // namespace mlir::triton::gpu

#endif
