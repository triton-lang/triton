#ifndef TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_
#define TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include <vector>

namespace mlir {
namespace triton {

static const char *kNumStagesAttrName = "tt.num_stages";

/// Function to mask operations during scheduling.
Operation *predicateOp(RewriterBase &rewriter, Operation *op, Value pred);

/// Collect ssa dependencies of `op` in `deps`. if `includeArg` is true,
/// continue looking through loop block arguments.
void addDep(Operation *op, DenseSet<Operation *> &deps, bool includeArg = true,
            DenseSet<Operation *> *filter = nullptr);

/// Add operations from `forOp` into a pipeline schedule with the the given
/// `stage` when filter is true. This will add operation in the original loop
/// order.
void addOps(scf::ForOp forOp, int stage,
            std::vector<std::pair<Operation *, unsigned>> &schedule,
            std::function<bool(Operation *)> filter);

/// Replace all uses of `oldUse` with `val` and propagate the type if needed.
/// This is useful when we need to change a memory descriptor from immutable to
/// mutable.
void replaceUsesAndPropagateType(OpBuilder &builder, Operation *oldUse,
                                 Value val);

/// Create a map from load ops to their indirection level and the
/// final use of the load op (another load op, or a dot op).
/// Indirection level is "0" for the load op directly used by the dot op,
/// "1" for the load op used by the load op used by the dot op, and so on.
llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
loadOpsToIndirectionLevelAndUse(scf::ForOp forOp);
} // namespace triton
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_
