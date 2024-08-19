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

/// Replaces `splat -> shared` with `splat -> blocked -> shared` in the given
/// |module| op.
void decomposeSplatOpToSharedLayoutConversion(ModuleOp module);

/// Replaces `mma/mfma -> dot_op` with `mma/mfma -> blocked -> dot_op` in the
/// given |module| op, but bypass the decomposition if |shortcutFn| returns
/// true.
using ShortcutFn = std::function<bool(RankedTensorType, RankedTensorType)>;
void decomposeTensorCoreToDotLayoutConversion(ModuleOp module,
                                              ShortcutFn shortcutFn);

} // namespace triton::gpu

} // namespace mlir

#endif
