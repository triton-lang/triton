#ifndef TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PATTERNS_H
#define TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PATTERNS_H

#include <functional>

namespace mlir {
class ModuleOp;
class RankedTensorType;

namespace triton::gpu {

/// Replaces `blocked -> dot_op` with `blocked -> shared -> dot_op` in the given
/// |module| op because the codegen doesn't handle `blocked -> dot_op` directly.
void decomposeBlockedToDotLayoutConversion(ModuleOp module);

} // namespace triton::gpu

} // namespace mlir

#endif
