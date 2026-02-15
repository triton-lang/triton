#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

SmallVector<NamedAttribute>
filterDiscardableAttrs(Operation *op, ArrayRef<StringRef> allowList) {
  SmallVector<NamedAttribute> propagatedAttrs;
  for (auto attrName : allowList) {
    Attribute attr = op->getDiscardableAttr(attrName);
    if (attr)
      propagatedAttrs.emplace_back(attrName, attr);
  }
  return propagatedAttrs;
}

} // namespace mlir::triton
