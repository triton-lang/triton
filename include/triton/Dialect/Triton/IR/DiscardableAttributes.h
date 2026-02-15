#ifndef TRITON_DIALECT_TRITON_IR_DISCARDABLE_ATTRIBUTES_H_
#define TRITON_DIALECT_TRITON_IR_DISCARDABLE_ATTRIBUTES_H_

#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

// Filter out attributes from the given operation that are not present in
// the allowList.
[[nodiscard]] SmallVector<NamedAttribute>
filterDiscardableAttrs(Operation *op, ArrayRef<StringRef> allowList);

} // namespace mlir::triton
#endif // TRITON_DIALECT_TRITON_IR_DISCARDABLE_ATTRIBUTES_H_
