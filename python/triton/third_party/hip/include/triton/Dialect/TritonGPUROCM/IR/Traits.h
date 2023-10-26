#ifndef TRITON_DIALECT_TRITONGPUROCM_IR_TRAITS_H_
#define TRITON_DIALECT_TRITONGPUROCM_IR_TRAITS_H_

#include "mlir/IR/OpDefinition.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace OpTrait {

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes.  This avoids them being template
// instantiated/duplicated.
namespace impl {
LogicalResult verifyResultsAreSharedEncodingROCM(Operation *op);
} // namespace impl

template <typename ConcreteType>
class ResultsAreSharedEncodingROCM
    : public TraitBase<ConcreteType, ResultsAreSharedEncodingROCM> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyResultsAreSharedEncodingROCM(op);
  }
};

} // namespace OpTrait
} // namespace mlir

#endif
