#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
//
// This file implements loop software pipelining
// The implementation here is inspired by the pipeline pass in Triton 2.0 and
// SCF's LoopPipelining.
//
//===----------------------------------------------------------------------===//


using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {
class LoopPipeliner {
  struct PipelineInfo {
    triton::DotOp dotOp;
    triton::LoadOp aLoadOp;
    triton::LoadOp bLoadOp;
  };

  /// comments on numStages:
  ///   [0, numStages-1) are in the prologue
  ///   numStages-1 is appended after the loop body
  int numStages;
  /// cache forOp we are working on
  scf::ForOp forOp;
  /// dot & loads
  PipelineInfo info;
  /// value (in loop) => value at stage N
  DenseMap<Value, SmallVector<Value>> valueMapping;
  /// stage => loop condition
  DenseMap<int, Value> loopConds;

  DenseSet<BlockArgument> depArgs;
  DenseSet<Operation*> depOps;

  void setValueMapping(Value origin, Value newValue, int stage);

  /// collect values that v depends on and are defined inside the loop
  void collectDeps(Value v);
public:
  LoopPipeliner(scf::ForOp forOp, int numStages) 
      : forOp(forOp), numStages(numStages) {}

  /// Collect loop info. Return success if we can pipeline this loop
  LogicalResult initialize();

  ///
  void emitPrologue();

  friend class PipelinePass;
};

// helpers
void LoopPipeliner::setValueMapping(Value origin, Value newValue, int stage) {
  if (valueMapping.find(origin) == valueMapping.end())
    valueMapping[origin] = SmallVector<Value>(numStages);
  valueMapping[origin][stage] = newValue;
}

void LoopPipeliner::collectDeps(Value v) {
  if (v.getParentRegion() != &forOp.getLoopBody())
    return;
  if (auto arg = v.dyn_cast<BlockArgument>())
    depArgs.insert(arg);
  else { // value
    Operation *defOp = v.getDefiningOp();
    depOps.insert(defOp);
    for (Value op : defOp->getOperands())
      collectDeps(op);
  }
}

/// A load instruction can be pipelined if:
///   - the pointer is a block argument (redefined inside the loop)
///   - the load has only a single use in a dot instruction
LogicalResult LoopPipeliner::initialize() {
  Region &bodyRegion = forOp.getLoopBody();
  assert(bodyRegion.hasOneBlock());
  Block &loop = bodyRegion.front();

  // TODO: can we use forOp.walk(...) here?
  SmallVector<triton::DotOp, 2> dots;
  for (Operation &op : loop) {
    if (auto dotOp = dyn_cast<triton::DotOp>(&op)) {
      dots.push_back(dotOp);
    }
  }

  // Don't know what to do if we have more than 1 dots inside the loop
  if (dots.size() != 1)
    return failure();

  triton::DotOp dotOp = dots[0];
  // dot (cvt (load %ptr0)), (cvt (load %ptr1))
  auto getDefinintLoad = [&](Value v) -> triton::LoadOp {
    auto cvt = v.getDefiningOp<triton::gpu::ConvertLayoutOp>();
    if (cvt) {
      return cvt.src().getDefiningOp<triton::LoadOp>();
    }
    return nullptr;
  };
  auto aLoad = getDefinintLoad(dotOp.a());
  auto bLoad = getDefinintLoad(dotOp.b());

  // ptrs must be block args (phi nodes)
  if (aLoad && bLoad) {
    if (aLoad.ptr().isa<BlockArgument>() && bLoad.ptr().isa<BlockArgument>()) {
      info.dotOp = dotOp; info.aLoadOp = aLoad; info.bLoadOp = bLoad;
      collectDeps(dotOp.a());
      collectDeps(dotOp.b());
      return success();
    }
  }

  return failure();
}

void LoopPipeliner::emitPrologue() {
  // TODO: should we use rewriter here?
  OpBuilder builder(forOp);
  for (BlockArgument &arg : forOp.getRegionIterArgs()) {
    OpOperand &operand = forOp.getOpOperandForRegionIterArg(arg);
    setValueMapping(arg, operand.get(), 0);
  }

  // pro
  Value iv = forOp.getInductionVar();
  for (int stage = 0; stage < numStages - 1; ++stage) {
    // special handling for induction variable as the increment is implicit
    if (stage != 0)
      iv = builder.create<arith::AddIOp>(iv.getLoc(), iv, forOp.getStep());
    setValueMapping(forOp.getInductionVar(), iv, stage);

    // special handling for loop condition as there is no condition in ForOp
    Value loopCond = builder.create<arith::CmpIOp>(
      iv.getLoc(), arith::CmpIPredicate::slt, iv, forOp.getUpperBound());
    loopConds[stage] = loopCond;

    // rematerialize peeled values
    SmallVector<Operation*> orderedDeps;
    for (Operation &op : forOp.getLoopBody().front())
      if (depOps.contains(&op))
        orderedDeps.push_back(&op);
    assert(depOps.size() == orderedDeps.size() && "depOps contains invalid values");
    for (Operation *op : orderedDeps) {
      Operation *newOp = builder.clone(*op);
      for (unsigned opIdx = 0; opIdx < op->getNumOperands(); ++opIdx) {
        auto it = valueMapping.find(op->getOperand(opIdx));
        if (it != valueMapping.end()) {
          Value v = it->second[stage];
          assert(v);
          newOp->setOperand(opIdx, v);
        } // else, op at opIdx is a loop-invariant value
      }

      // update mapping of results
      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults())) {
        setValueMapping(op->getResult(dstIdx), newOp->getResult(dstIdx), stage);
        // TODO: update mapping for loop-carried values (args)
      }
    }
  }
}

// ref: mlir/lib/Dialect/SCF/Transforms/LoopPipelining.cpp
struct PipelinePass : public TritonGPUPipelineBase<PipelinePass> {
  void runOnOperation() override {
    // TODO: collect numStages from ModuleOp
    int numStages = 2;

    if (numStages <= 1)
      return;

    getOperation()->walk([&](scf::ForOp forOp) -> void {
      LoopPipeliner pipeliner(forOp, numStages);

      if (pipeliner.initialize().failed())
        return;

      llvm::errs() << "candidate for pipelining: " << pipeliner.info.dotOp
                   << "\n";

      // pipeliner.emitPrologue();

      // scf::ForOp newForOp = pipeliner.createNewForOp();

      // // replace the original loop
      // if (forOp->getNumResults() > 0)
      //   rewriter.replaceOp(forOp, newForOp->getResults());
      // else
      //   rewriter.eraseOp(forOp);
    });
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUPipelinePass() {
  return std::make_unique<PipelinePass>();
}
