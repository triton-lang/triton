#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

SmallVector<NamedAttribute> getAllowedDiscardableAttrs(triton::AddPtrOp op) {
  std::array<StringRef, 3> allowList{"tt.divisibility", "tt.contiguity",
                                     "tt.constancy"};
  SmallVector<NamedAttribute> propagatedAttrs;
  for (auto attrName : allowList) {
    Attribute attr = op->getDiscardableAttr(attrName);
    if (attr)
      propagatedAttrs.emplace_back(attrName, attr);
  }
  return propagatedAttrs;
}

} // namespace mlir::triton
