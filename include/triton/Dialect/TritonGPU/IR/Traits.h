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

// Marks an op that produces a sub-view of a shared memory descriptor
// (e.g., memdesc_subview, memdesc_index).
template <typename ConcreteType>
class MemDescViewTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, MemDescViewTrait> {};

// Marks an op that loads from shared (local) memory into registers.
template <typename ConcreteType>
class LocalLoadTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, LocalLoadTrait> {};

// Marks an op that waits for outstanding asynchronous operations to complete.
template <typename ConcreteType>
class MemWaitOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, MemWaitOpTrait> {};

// Marks an op that initiates an asynchronous write to shared memory.
// Completion is only guaranteed after an explicit wait.
template <typename ConcreteType>
class MemAsyncLocalStoreOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, MemAsyncLocalStoreOpTrait> {
};

} // namespace OpTrait
} // namespace mlir

#endif
