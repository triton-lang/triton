#include "Dialect/NVWS/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/hopper/lib/Transforms/WarpSpecialization/CodePartitionUtility.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/Dump.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "nvgpu-warp-specialization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

void doTaskPartition(triton::FuncOp &funcOp, unsigned numWarpGroups);
int doTaskIdPropagate(triton::FuncOp &funcOp);
bool doDataPartition(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers);
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups);

#define GEN_PASS_DEF_NVGPUWARPSPECIALIZATION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUWarpSpecializationPass
    : public impl::NVGPUWarpSpecializationBase<NVGPUWarpSpecializationPass> {
public:
  using impl::NVGPUWarpSpecializationBase<
      NVGPUWarpSpecializationPass>::NVGPUWarpSpecializationBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    SmallVector<scf::ForOp> loops;
    funcOp->walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(mlir::triton::kWarpSpecializeAttrName) &&
          triton::getNumStagesOrDefault(forOp, numStages) > 1)
        loops.push_back(forOp);
    });
    if (loops.empty())
      return;

    int numWarps = mlir::triton::gpu::lookupNumWarps(funcOp);
    if (numWarps != 4)
      return;

    bool hasPreexistingTaskIds = false;
    funcOp.walk([&](Operation *op) {
      hasPreexistingTaskIds |= op->hasAttr("async_task_id");
    });

    // FIXME: skip warpspec if there is else block. Need to improve
    // CodePartitioning to correctly handle channels in else block.
    bool hasElse = false;
    funcOp->walk([&](scf::IfOp ifOp) {
      if (ifOp.elseBlock()) {
        hasElse = true;
      }
    });
    if (hasElse)
      return;

    // Gather is not supported by the data partitioner. Decline warp
    // specialization when a Gather is in, feeds, or depends on a selected loop
    // or partition root, or joins a partitioned value at a consumer whose
    // operands are sliced together.
    auto isInWarpSpecializedLoop = [&](Operation *op) {
      for (scf::ForOp loop : loops)
        if (loop.getOperation() == op || loop->isAncestor(op))
          return true;
      return false;
    };
    auto isPartitionRoot = [](Operation *op) {
      // Data partitioning considers both dot forms after task-ID propagation.
      return isa<triton::nvidia_gpu::WarpGroupDotOp,
                 triton::nvidia_gpu::TCGen5MMAOp>(op);
    };
    auto hasPartitionableDimension = [](auto shape) {
      for (int64_t dim : shape)
        if (dim > 1)
          return true;
      return false;
    };
    // The sibling preflight runs before the partition dimension is known.
    // Keep values that could pass the partitioner's shape boundary on some
    // dimension, without following scalar broadcasts back into the loop.
    auto hasPartitionableShape = [&](auto &&self, Type type) -> bool {
      if (auto ptrType = dyn_cast<triton::PointerType>(type))
        return self(self, ptrType.getPointeeType());
      if (auto memDescType = dyn_cast<triton::gpu::MemDescType>(type))
        return hasPartitionableDimension(memDescType.getShape());
      if (auto tensorType = dyn_cast<RankedTensorType>(type))
        return hasPartitionableDimension(tensorType.getShape());
      if (auto tensorDescType = dyn_cast<triton::TensorDescType>(type))
        return hasPartitionableDimension(tensorDescType.getShape());
      return false;
    };
    auto hasPartitionableValue = [&](Value value) {
      return hasPartitionableShape(hasPartitionableShape, value.getType());
    };
    auto hasPartitionRootInBackwardSlice = [&](Value root, auto shouldFollow) {
      SetVector<Value> worklist;
      SetVector<Value> visited;
      worklist.insert(root);
      while (!worklist.empty()) {
        Value value = worklist.pop_back_val();
        if (!visited.insert(value) || !shouldFollow(value))
          continue;

        if (auto blockArg = dyn_cast<BlockArgument>(value)) {
          if (auto forOp =
                  dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
            if (blockArg.getArgNumber() > 0) {
              unsigned iterArg = blockArg.getArgNumber() - 1;
              // A loop-carried block argument depends on both its initial
              // value and the value yielded by the previous iteration.
              worklist.insert(forOp.getInitArgs()[iterArg]);
              worklist.insert(forOp.getYieldedValues()[iterArg]);
            }
          }
          continue;
        }

        Operation *op = value.getDefiningOp();
        if (!op)
          continue;
        if (isInWarpSpecializedLoop(op) || isPartitionRoot(op))
          return true;

        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
          // The partitioner's forward slice treats all loop results as
          // partitioned when any init arg is partitioned.
          for (Value initArg : forOp.getInitArgs())
            worklist.insert(initArg);
          for (Value yieldedValue : forOp.getYieldedValues())
            worklist.insert(yieldedValue);
          continue;
        }
        for (Value operand : op->getOperands())
          worklist.insert(operand);
      }
      return false;
    };
    auto hasPartitionedUseInForwardSlice = [&](Value root) {
      SetVector<Value> worklist;
      SetVector<Value> visited;
      worklist.insert(root);
      while (!worklist.empty()) {
        Value value = worklist.pop_back_val();
        if (!visited.insert(value))
          continue;

        for (Operation *user : value.getUsers()) {
          if (isInWarpSpecializedLoop(user) || isPartitionRoot(user))
            return true;

          // These consumers are sliced with their operands. A Gather can
          // therefore meet a partitioned value here even when neither value
          // depends on the other.
          if (user->hasTrait<OpTrait::Elementwise>() ||
              isa<triton::StoreOp, triton::DescriptorStoreOp,
                  triton::AtomicRMWOp, triton::JoinOp, triton::LoadOp,
                  triton::ReduceOp>(user)) {
            for (Value operand : user->getOperands()) {
              if (operand != value && hasPartitionRootInBackwardSlice(
                                          operand, hasPartitionableValue))
                return true;
            }
          }

          if (auto forOp = dyn_cast<scf::ForOp>(user)) {
            for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
              if (forOp.getInitArgs()[i] == value) {
                worklist.insert(forOp.getRegionIterArgs()[i]);
                worklist.insert(forOp.getResult(i));
              }
            }
            continue;
          }

          if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
            // Preserve the yielded operand's index when crossing the loop.
            for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
              if (yieldOp->getOperand(i) != value)
                continue;
              if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
                // The yielded value becomes the loop-carried block argument
                // on the next iteration, even if the loop result is unused.
                worklist.insert(forOp.getRegionIterArgs()[i]);
                worklist.insert(forOp.getResult(i));
              }
            }
            continue;
          }

          for (Value result : user->getResults())
            worklist.insert(result);
        }
      }
      return false;
    };
    bool hasUnsupportedGather = false;
    funcOp.walk([&](triton::GatherOp gatherOp) {
      if (hasUnsupportedGather)
        return;
      if (isInWarpSpecializedLoop(gatherOp.getOperation())) {
        hasUnsupportedGather = true;
        return;
      }

      for (Value operand : gatherOp->getOperands()) {
        if (hasPartitionRootInBackwardSlice(operand,
                                            [](Value) { return true; })) {
          hasUnsupportedGather = true;
          return;
        }
      }
      if (hasUnsupportedGather || gatherOp->getResult(0).use_empty())
        return;

      hasUnsupportedGather =
          hasPartitionedUseInForwardSlice(gatherOp->getResult(0));
    });
    if (hasUnsupportedGather) {
      if (hasPreexistingTaskIds) {
        funcOp.emitError()
            << "warp specialization cannot fall back from unsupported gather "
               "in "
               "warp-specialized function with preexisting async_task_id "
               "attributes";
        return signalPassFailure();
      }
      funcOp.walk([](scf::ForOp loop) {
        loop->removeAttr(triton::kWarpSpecializeAttrName);
      });
      return;
    }

    OpBuilder builder(funcOp);
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    unsigned numWarpGroups = 3;
    // FIXME: skip data partitioning with on-host TMA.
    bool success = false;
    for (; numWarpGroups >= 2; numWarpGroups--) {
      // Partition key ops into multiple async tasks.
      doTaskPartition(funcOp, numWarpGroups);
      if (dumpIntermediateSteps) {
        ::mlir::triton::tools::mlirDumpsOrDbgs()
            << "// -----// WarpSpec internal IR Dump After: doTaskPartition\n"
            << moduleOp << "\n\n\n";
      }
      // Propagate taskId.
      int retCode = doTaskIdPropagate(funcOp);
      if (retCode == -1)
        continue;
      if (dumpIntermediateSteps) {
        ::mlir::triton::tools::mlirDumpsOrDbgs()
            << "// -----// WarpSpec internal IR Dump After: doTaskIdPropagate\n"
            << moduleOp << "\n\n\n";
      }

      // Partition ops into parallel sub ops.
      if (doDataPartition(funcOp, numWarpGroups - 1)) {
        if (dumpIntermediateSteps) {
          ::mlir::triton::tools::mlirDumpsOrDbgs()
              << "// -----// WarpSpec internal IR Dump After: doDataPartition\n"
              << moduleOp << "\n\n\n";
        }
        success = true;
        break;
      }
      // Clear async_task.
    }
    if (!success) {
      mlir::emitError(
          getOperation()->getLoc(),
          "failed to partition the function into warp-specialized code");
      return signalPassFailure();
    }

    doCodePartition(funcOp, numStages);
    if (dumpIntermediateSteps) {
      ::mlir::triton::tools::mlirDumpsOrDbgs()
          << "// -----// WarpSpec internal IR Dump After: doCodePartition\n"
          << moduleOp << "\n\n\n";
    }
    doTokenLowering(funcOp, numWarpGroups - 1);
    invalidateWarpSpecializeBarriers(funcOp);
    // Clear num_stages to disable SWP.
    funcOp->walk([&](scf::ForOp forOp) {
      forOp->setAttr(mlir::triton::kNumStagesAttrName,
                     builder.getI32IntegerAttr(0));
    });
  }

  void runOnOperation() override {
    if (numStages <= 1)
      return;

    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
