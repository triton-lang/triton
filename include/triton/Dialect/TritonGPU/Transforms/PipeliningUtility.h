#ifndef TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_
#define TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <optional>
#include <utility>
#include <vector>

namespace mlir {
class DominanceInfo;
class ImplicitLocOpBuilder;
namespace triton {

static const char *kNumStagesAttrName = "tt.num_stages";
static const char *kDisallowAccMultiBufferAttrName =
    "tt.disallow_acc_multi_buffer";
static const char *kWarpSpecializeAttrName = "tt.warp_specialize";
static const char *kLoopStageAttrName = "loop.stage";
static const char *kLoopClusterAttrName = "loop.cluster";
static const char *kScheduledMaxStageAttrName = "tt.scheduled_max_stage";
class CoarseSchedule;
class ModuleAxisInfoAnalysis;
//===----------------------------------------------------------------------===//
// Hoisting Utilities
//===----------------------------------------------------------------------===//

// By default, an operation can be hoisted if it is pure scalar operation.
bool isPureScalarOp(Operation *op);

// Given a set of values and a reference operation, return true if all of the
// values dominate the reference operation OR a set of "trivial" operations can
// be moved before the reference operation such that the value set dominates the
// reference operation.
//
// Returns false if it is not possible to make the values dominate the reference
// operation. The function determines "trivial"-ness with the given callback.
// By default, it determines that memory-effect-free and scalar operations are
// trivial.
bool getDominatingValueSetOpsToHoist(
    DominanceInfo &domInfo, Operation *refOp, ArrayRef<Value> valueSet,
    llvm::SetVector<Operation *> &toHoist,
    function_ref<bool(Operation *)> canHoist = isPureScalarOp);

// Hoist the given set of operations above the reference operation.
void hoistOpsBefore(Operation *refOp,
                    const llvm::SetVector<Operation *> &toHoist);
// Hoist the given set of operations before the iterator.
void hoistOpsBefore(Block *block, Block::iterator it,
                    const llvm::SetVector<Operation *> &toHoist);

//===----------------------------------------------------------------------===//
// Sinking Utilities
//===----------------------------------------------------------------------===//

// Sink a value redefinition into a block, provided that the block is dominated
// by `in` and postdominated by `out`.
Value sinkValueRedefinition(RewriterBase &rewriter, Value in, Value out,
                            Block *block);

//===----------------------------------------------------------------------===//
// Loop Pipelining Utilities
//===----------------------------------------------------------------------===//

bool loopHasDistGreaterThanOne(scf::ForOp forOp);
bool isOuterLoop(scf::ForOp forOp);

/// Function to mask operations during scheduling.
Operation *predicateOp(RewriterBase &rewriter, Operation *op, Value pred);

// Return true if the given ForOp has the attribute
// `tt.disallow_acc_multi_buffer` set to true.
bool getDisallowAccMultiBuffer(scf::ForOp forOp);

// Return the definition of the given value. If the value is a loop-carried
// dependency, return the definition and the distance to it.
std::pair<OpResult, int64_t> getDefinitionAndDistance(scf::ForOp forOp,
                                                      Value value);
// Return the defining op of the given value, if the Value is an argument of the
// loop return the associated defining op in the loop and its distance to the
// Value.
std::pair<Operation *, int64_t> getDefiningOpAndDistance(scf::ForOp forOp,
                                                         Value value);

// Return maximum length of the vectorized copy between registers and shared
// memory for the given tensor type and shared encoding.
int getCopyVecBytes(RankedTensorType registerTy,
                    gpu::SharedEncodingTrait sharedEnc);

bool canBeConvertedToAsyncLoad(
    triton::LoadOp loadOp, triton::ModuleAxisInfoAnalysis &axisInfoAnalysis);

// Serialize the latencies of the operations in the loops into the latency
// attribute.
void serializeLatencies(ModuleOp module, DenseMap<Operation *, int> &opLatency);

// Serialize the self latencies of the operations in the loops into the
// self_latency attribute.
void serializeSelfLatencies(ModuleOp module,
                            DenseMap<Operation *, int> &opSelfLatency);

// Deserialize the latencies of the operations in the loops from the attribute.
DenseMap<Operation *, int> deserializeLatencies(Operation *op);

// Create an allocation for multibuffered scalars.
Value createScalarAlloc(ImplicitLocOpBuilder &rewriter, Type type,
                        unsigned numBuffers);
// Create an allocation and init the mbarriers.
Value createBarrierAlloc(scf::ForOp forOp, int numBarriers,
                         int arriveCount = 1);
// Create an allocation that can hold distance number of tensor shapes.
Value createAlloc(Operation *insertBefore, RankedTensorType ty, Location loc,
                  gpu::SharedEncodingTrait sharedEnc, unsigned distance);

// Determine if the operation is a TMA load.
bool isTMALoad(Operation *op);

// Determine if the operation can be lowered to an async load.
bool canBeAsyncLoad(Operation *op);

// Look for consecutive wait ops and combine them into a single wait op.
void combineRedundantWaitOps(
    llvm::SmallSetVector<gpu::AsyncWaitOp, 8> &waitOps);

// Get the type of the view of a multi-buffered tensor value.
gpu::MemDescType getBufferViewType(gpu::MemDescType allocTy);
// Get a generic shared encoding for a tensor.
gpu::SharedEncodingTrait getSharedEncoding(RankedTensorType ty);
// Get a shared encoding for a tensor based on its uses.
gpu::SharedEncodingTrait getSharedEncoding(Operation *loadOp);

// Get the number of stages to pipeline the loop with, if it is explicitly
// specified.
int getNumStagesOrDefault(scf::ForOp forOp, int defaultNumStages);

// Given a result of MemDescSubview, or Alloca, create a MemDescSubview with a
// single buffer slice (leading dimension equal to 1), at the given index.
TypedValue<triton::gpu::MemDescType>
createSingleBufferView(OpBuilder &builder, Value alloc, Value idx);
// Given a result of MemDescSubview, or Alloca, create a MemDescSubview with a
// single buffer slice (leading dimension equal to 1), at the given index.
TypedValue<triton::gpu::MemDescType>
createSingleBufferView(OpBuilder &builder, Value alloc, int idx);

Value createIncrementModulo(OpBuilder &builder, Location loc, Value counter,
                            Value modulus, Value zero, Value one,
                            Value *outWrapCond = nullptr);

scf::ForOp lowerTMADescriptors(scf::ForOp forOp, CoarseSchedule &schedule);

} // namespace triton
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_
