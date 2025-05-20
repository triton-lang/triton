#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_LOOP_PEELING_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_LOOP_PEELING_H_

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace triton {

constexpr StringRef kPeelEpilogueIterationsAttrName =
    "tt.peel_epilogue_iterations";

void annotateLoopForEpiloguePeeling(RewriterBase &rewriter, scf::ForOp forOp,
                                    int numIterations);

// Peel the epilogue of the loop.
void peelLoopEpilogue(
    scf::ForOp forOp, int numIterations,
    function_ref<Operation *(RewriterBase &, Operation *, Value)> predicateOp,
    SmallVector<Operation *> *peeledOps = nullptr);

// Peel the epilogue of the loop based on the `tt.peel_epilogue_iterations`
// attribute.
void peelLoopEpilogue(
    scf::ForOp forOp,
    function_ref<Operation *(RewriterBase &, Operation *, Value)> predicateOp,
    SmallVector<Operation *> *peeledOps = nullptr);

} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITON_TRANSFORMS_LOOP_PEELING_H_
