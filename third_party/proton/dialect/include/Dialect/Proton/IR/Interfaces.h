#ifndef DIALECT_PROTON_IR_INTERFACES_H_
#define DIALECT_PROTON_IR_INTERFACES_H_

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::triton::proton {

//===----------------------------------------------------------------------===//
// ProtonDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

struct ProtonInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       IRMapping &) const final {
    return true;
  }
};
} // namespace mlir::triton::proton

#endif // DIALECT_PROTON_IR_INTERFACES_H_
