#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;

static bool willIncreaseRegisterPressure(Operation *op) {
  if (isa<triton::gpu::LocalLoadOp>(op))
    return true;
  if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op))
    return isa<triton::gpu::DotOperandEncodingAttr>(
        cvt.getType().getEncoding());
  return false;
}

static bool isDescendent(Operation *op, Block *block) {
  Block *b = op->getBlock();
  while (b != nullptr) {
    if (b == block)
      return true;
    b = b->getParentOp()->getBlock();
  }
  return false;
}

static bool gatherDFG(Operation *op, Block *block,
                      SmallVector<Operation *> &dfg) {
  // BFS (filo)
  SmallVector<Operation *> oprs;
  bool leadsToLoad = false;
  for (auto operand : op->getOperands()) {
    if (Operation *pop = operand.getDefiningOp()) {
      if (isDescendent(pop, block)) {
        // only move ops that reside in same block
        if (pop->getBlock() == block)
          dfg.push_back(pop);
        oprs.push_back(pop);
        leadsToLoad |= isa<triton::LoadOp>(pop);
      } else {
        // only operands from current block or ancestor
        assert(isDescendent(block->getParentOp(), pop->getBlock()));
      }
    }
  }
  // check sub-regions
  for (auto &subregion : op->getRegions()) {
    for (auto &subblock : subregion) {
      for (auto &sop : subblock) {
        if (gatherDFG(&sop, block, dfg))
          leadsToLoad = true;
      }
    }
  }

  // process next level ops
  for (auto *op : oprs) {
    if (gatherDFG(op, block, dfg))
      leadsToLoad = true;
  }
  return leadsToLoad;
}

class TritonAMDGPUReorderInstructionsPass
    : public TritonAMDGPUReorderInstructionsBase<
          TritonAMDGPUReorderInstructionsPass> {
public:
  TritonAMDGPUReorderInstructionsPass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);
    // Sink conversions into loops when they will increase
    // register pressure
    DenseMap<Operation *, Operation *> opToMove;
    auto moveAfter = [](Operation *lhs, Operation *rhs) {
      lhs->moveAfter(rhs);
    };
    m.walk([&](Operation *op) {
      if (!willIncreaseRegisterPressure(op))
        return;
      if (!op->hasOneUse())
        return;
      Operation *user = op->getUses().begin()->getOwner();
      if (user->getParentOfType<scf::ForOp>() ==
          op->getParentOfType<scf::ForOp>())
        return;
      opToMove.insert({op, user});
    });
    for (auto &kv : opToMove)
      kv.first->moveBefore(kv.second);
    opToMove.clear();
    // Move LocalLoadOp and LocalAllocOp immediately after their operands.
    m.walk([&](Operation *op) {
      if (!isa<triton::gpu::LocalLoadOp, triton::gpu::LocalAllocOp>(op) ||
          op->getNumOperands() < 1) {
        return;
      }
      if (Operation *argOp = op->getOperand(0).getDefiningOp())
        moveAfter(op, argOp);
    });
    // Move transpositions just after their definition
    m.walk([&](triton::TransOp op) {
      Operation *argOp = op.getSrc().getDefiningOp();
      if (!argOp)
        return;
      moveAfter(op, argOp);
    });
    SmallVector<Operation *> moveOps;
    m.walk([&](triton::gpu::LocalStoreOp op) {
      // Move local stores early if it's global load is outside loop
      moveOps.push_back(op);
    });
    m.walk([&](triton::LoadOp op) {
      // Move global loads early (prefetch)
      moveOps.push_back(op);
    });
    for (auto op : moveOps) {
      // 0. gather DFG
      Block *block = op->getBlock();
      SmallVector<Operation *> dfg{op};
      bool leadsToLoad = gatherDFG(op, block, dfg);
      if (!isa<triton::gpu::LocalStoreOp>(op) || !leadsToLoad) {
        // 1. move to beginning of enclosing block
        for (auto *op : dfg)
          op->moveAfter(block, block->begin());
      }
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUReorderInstructionsPass() {
  return std::make_unique<TritonAMDGPUReorderInstructionsPass>();
}
