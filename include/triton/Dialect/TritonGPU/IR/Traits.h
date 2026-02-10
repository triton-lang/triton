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

template <typename ConcreteType>
class MemDescViewTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, MemDescViewTrait> {
  // Optional: Add methods or verification logic here
};

template <typename ConcreteType>
class LocalLoadTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, LocalLoadTrait> {
  // Optional: Add methods or verification logic here
};

template <typename ConcreteType>
class MemWaitOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, MemWaitOpTrait> {
  // Optional: Add methods or verification logic here
};

// Async DMA write to shared memory, visible only after an explicit wait.
template <typename ConcreteType>
class MemAsyncWriteOpTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, MemAsyncWriteOpTrait> {};

} // namespace OpTrait
} // namespace mlir

#endif
