#ifndef TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_
#define TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <optional>
#include <utility>
#include <vector>

namespace mlir {
namespace triton {

static const char *kNumStagesAttrName = "tt.num_stages";
static const char *kDisallowAccMultiBufferAttrName =
    "tt.disallow_acc_multi_buffer";
static const char *kLoopStageAttrName = "loop.stage";
static const char *kLoopClusterAttrName = "loop.cluster";

bool loopHasDistGreaterThanOne(scf::ForOp forOp);
bool isOuterLoop(scf::ForOp forOp);

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

// Return true if the given ForOp has the attribute
// `tt.disallow_acc_multi_buffer` set to true.
bool getDisallowAccMultiBuffer(scf::ForOp forOp);

/// Visit the operands of `op` and the operands of any nested ops defined
/// outside of `op`.
void visitNestedOperands(Operation *op, function_ref<void(Value)> visitor);
/// Get the operands of `op` and the operands of any nested ops defined outside
/// of `op`.
SetVector<Value> getNestedOperands(Operation *op);

// Return the minClusterId and maxClusterId for the given ForOp.
std::pair<int, int> getMinMaxCluster(scf::ForOp &forOp);

// Return maximum scheduled stage for any of the ops in the loop.
int getMaxStage(scf::ForOp &forOp);

// Return the stage and cluster for the given operation.
std::pair<int, int> getStageCluster(Operation *op);

// Return the stage and cluster for the given operation if it has been
// scheduled.
std::optional<std::pair<int, int>> maybeGetStageCluster(Operation *op);

// Set the stage and cluster for the given operation.
void setStageCluster(Operation *op, int stage, int cluster);

// Return maxumum length of the vectorized copy between registers and shared
// memory for the given tensor type and shared encoding.
int getCopyVecBytes(RankedTensorType registerTy,
                    gpu::SharedEncodingTrait sharedEnc);
} // namespace triton
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_
