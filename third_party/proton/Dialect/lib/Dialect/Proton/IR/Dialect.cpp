#include "Dialect/Proton/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "Dialect/Proton/IR/Dialect.cpp.inc"

namespace mlir::triton::proton {
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

void ProtonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Proton/IR/Ops.cpp.inc"
      >();
  addInterfaces<ProtonInlinerInterface>();
}
} // namespace mlir::triton::proton
