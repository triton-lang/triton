#ifndef TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PATTERNS_H
#define TRITONGPU_CONVERSION_TRITONGPUTOLLVM_PATTERNS_H

namespace mlir {

class ModuleOp;

namespace triton::gpu {

/// Replaces `splat -> shared` with `splat -> blocked -> shared` in the given
/// |module| op.
void decomposeSplatOpToSharedLayoutConversion(ModuleOp module);

/// Replaces `blocked -> dot_op` with `blocked -> shared -> dot_op` in the given
/// |module| op because the codegen doesn't handle `blocked -> dot_op` directly.
void decomposeBlockedToDotLayoutConversion(ModuleOp module);

} // namespace triton::gpu

} // namespace mlir

#endif
