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
static const char *kLatencyAttrName = "tt.latency";

bool loopHasDistGreaterThanOne(scf::ForOp forOp);
bool isOuterLoop(scf::ForOp forOp);

/// Function to mask operations during scheduling.
Operation *predicateOp(RewriterBase &rewriter, Operation *op, Value pred);

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
std::optional<int> tryGetMaxStage(scf::ForOp &forOp);

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

// Serialize the latencies of the operations in the loops into the latency
// attribute.
void serializeLatencies(ModuleOp module, DenseMap<Operation *, int> &opLatency);

// Deserialize the latencies of the operations in the loops from the attribute.
DenseMap<Operation *, int> deserializeLatencies(ModuleOp module);

// Given a result of MemDescSubview, or Alloca, create a MemDescSubview with a
// single buffer slice (leading dimension equal to 1), at the given index.
template <typename BuilderT>
Value createSingleBufferView(BuilderT &builder, Value alloc, Value idx) {
  assert(isa<triton::gpu::MemDescType>(alloc.getType()) &&
         "Expected MemDescType");
  auto allocDescType = cast<triton::gpu::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape;
  if (allocDescType.getShape().size() > 1) {
    shape.insert(shape.end(), allocDescType.getShape().begin() + 1,
                 allocDescType.getShape().end());
  } else {
    shape.push_back(1);
  }
  auto viewDescType = triton::gpu::MemDescType::get(
      shape, allocDescType.getElementType(), allocDescType.getEncoding(),
      allocDescType.getMemorySpace(), allocDescType.getMutableMemory(),
      /*allocShape=*/allocDescType.getAllocShape());
  SmallVector<Value> idxs = {idx};
  if (allocDescType.getShape().size() > 1) {
    Value zero =
        builder.template create<arith::ConstantIntOp>(alloc.getLoc(), 0, 32);
    for (unsigned i = 1; i < allocDescType.getShape().size(); i++) {
      idxs.push_back(zero);
    }
  }
  return builder.template create<triton::gpu::MemDescSubviewOp>(
      alloc.getLoc(), viewDescType, alloc, idxs);
}

template <typename BuilderT>
Value createSingleBufferView(BuilderT &builder, Value alloc, int idx) {
  return createSingleBufferView(
      builder, alloc,
      builder.template create<arith::ConstantIntOp>(alloc.getLoc(), idx, 32));
}

// Create an allocation and init the mbarriers.
Value createBarrierAlloc(scf::ForOp forOp, int numBarriers);

} // namespace triton
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_
