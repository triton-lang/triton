#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_LOOP_PEELING_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_LOOP_PEELING_H_

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace triton {

// Peel the single last iteration of the loop.
void peelLoopEpilogue(
    scf::ForOp forOp,
    function_ref<Operation *(RewriterBase &, Operation *, bool)>
        processPeeledOp = nullptr);

} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITON_TRANSFORMS_LOOP_PEELING_H_
