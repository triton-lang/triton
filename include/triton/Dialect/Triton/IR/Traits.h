#ifndef TRITON_IR_TRAITS_H_
#define TRITON_IR_TRAITS_H_

#include "mlir/IR/OpDefinition.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include <iostream>

namespace mlir {
namespace OpTrait {
// TODO: should have `namespace triton {}` here

template <class ConcreteType>
class TensorSizeTrait : public TraitBase<ConcreteType, TensorSizeTrait> {
public:
  // TODO: move impl to .cc files
  static LogicalResult verifyTrait(Operation *op) {
    // The rationale for this number is to prevent users from creating programs
    // that would have catastrophic register pressure and cause the compiler to
    // hang.
    // Since H100 has 256KB registers, we should allow users to create tensors
    // of size up to 256K elements. It will spill for datatypes wider than 1B,
    // but we probably should limit number of elements (rather than bytes) to
    // keep specs simple
    int constexpr maxElement = 1048576;
    for (auto opType : op->getOperandTypes()) {
      if (auto tensorType = opType.dyn_cast<RankedTensorType>()) {
        int64_t numElements = 1;
        for (int64_t s : tensorType.getShape())
          numElements *= s;
        if (numElements > maxElement)
          return op->emitError("Maximum allowed number of elements is ")
                 << maxElement << ", but " << *op << " has more than that";
        if ((numElements & (numElements - 1)) != 0)
          return op->emitError("Number of elements must be power-of-two, but ")
                 << *op << " doesn't follow the rule";
      }
    }

    for (auto opType : op->getResultTypes()) {
      if (auto tensorType = opType.dyn_cast<RankedTensorType>()) {
        int64_t numElements = 1;
        for (int64_t s : tensorType.getShape())
          numElements *= s;
        if (numElements > maxElement)
          return op->emitError("Maximum allowed number of elements is ")
                 << maxElement << ", but " << *op << " has more than that";
        if ((numElements & (numElements - 1)) != 0)
          return op->emitError("Number of elements must be power-of-two, but ")
                 << *op << " doesn't follow the rule";
      }
    }

    return success();
  }
};

} // namespace OpTrait
} // namespace mlir

#endif
