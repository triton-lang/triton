#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

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

  int numStages;
  /// cache forOp we are working on
  scf::ForOp forOp;
  /// dot & loads
  PipelineInfo info;
  /// value (in loop) => value at stage N
  DenseMap<Value, SmallVector<Value>> valueMapping;

  void setStageValueMapping(Value origin, Value prefetched, int idx);
public:
  LoopPipeliner(scf::ForOp forOp, int numStages) 
      : forOp(forOp), numStages(numStages) {}

  /// Collect loop info. Return success if we can pipeline this loop
  LogicalResult initialize();

  /// 
  void emitPrologue();

  friend class PipelinePass;
};

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
      return success();
    }
  }

  return failure();
}

void LoopPipeliner::emitPrologue() {
  OpBuilder builder(forOp);
  // 
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
    });
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUPipelinePass() {
  return std::make_unique<PipelinePass>();
}
