#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/LogicalResult.h"



#define DEBUG_TYPE "tritongpu-wgmma-prefetch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
using namespace mlir;
namespace ttng = mlir::triton::nvidia_gpu;


namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUWGMMAPREFETCH
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"


namespace {

// RSGEMM means the lhs is in register and the rhs is in
// shared memory
LogicalResult isRSGEMM(ttng::WarpGroupDotOp dotOp){
  Value operandA = dotOp.getA();
  Value operandB = dotOp.getB();

  // operandB is in shared memoryassert
  auto encA = dyn_cast<TensorOrMemDesc>(operandA.getType()).getEncoding();
  auto encB = dyn_cast<TensorOrMemDesc>(operandB.getType()).getEncoding();

  if(isa<DotOperandEncodingAttr>(encA) && isa<NVMMASharedEncodingAttr>(encB))
    return llvm::success();

  return llvm::failure();
}


class WGMMAPrefetcher {
/// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  ///
  // For Hopper, we implicitly use the prefetchWidth 16
  unsigned prefetchWidth = 16;

  /// dots to be prefetched
  SetVector<ttng::WarpGroupDotOp> dots;
/// dot => dot operand
  DenseMap<Value, Value> dot2aLoopArg;
  DenseMap<Value, Value> dot2aHeaderDef;
  DenseMap<Value, Value> dot2bLoopArg;
  DenseMap<Value, Value> dot2bHeaderDef;
  DenseMap<Value, Value> dot2aSrcMemDesc;
  DenseMap<Value, Value> dot2bSrcMemDesc;
  DenseMap<Value, Value> dot2aYield;
  DenseMap<Value, Value> dot2bYield;
  DenseMap<Value, SmallVector<Value>> dot2aVals;
  DenseMap<Value, SmallVector<Value>> dot2aValsLocalLoad;
  DenseMap<Value, SmallVector<Value>> dot2aValsElementWise;
  DenseMap<Value, SmallVector<Value>> dot2bVals;
  DenseMap<Value, Value> dot2Wait;

public:
  WGMMAPrefetcher() = delete;

  WGMMAPrefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();


};

LogicalResult WGMMAPrefetcher::initialize() {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<TensorOrMemDesc>(v.getType()).getEncoding();
  };

  SmallVector<nvidia_gpu::WarpGroupDotOp> dotsInFor;

  // Step 1: check the condition if the forloop can be prefetched
  for (Operation &op : *loop){
    if(auto dotOp = dyn_cast<ttng::WarpGroupDotOp>(op)){
      auto dstMmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(dotOp.getResult()));
      dotsInFor.push_back(dotOp);
    }
    else if(auto dotOp = dyn_cast<triton::DotOp>(op)){
      // To be conservative, we only allow WarpGroupDotOp in the foor loop;
      return llvm::failure();
    }
  }

  if (dotsInFor.empty()){
    return llvm::failure();
  }

  if (dotsInFor.size() > 1){
    return llvm::failure();
  }

  for(ttng::WarpGroupDotOp dotOp : dotsInFor){
    if(isRSGEMM(dotOp).failed())
      return llvm::failure();
  }

  LDBG("Pass all the precondition test");

  auto getPrefetchSrc = [](Value v) -> SmallVector<Value> {
    // walk back to conversion
    Operation *op = v.getDefiningOp();
    bool foundConvertFromShared = false;
    SmallVector<Value> rets;
    rets.push_back(op->getResult(0));
    if (dyn_cast<MemDescSubviewOp>(op))
      return rets;
    LDBG("Prefetch src: " << *op);
    while (op) {
      if (op->getNumOperands() != 1)
        break;
      if (!op->getResult(0).hasOneUse())
        break;
      rets.push_back(op->getOperand(0));
      if (auto cvt = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        // NYI for other encodings, for example if we have transpose
        // in the chain
        if (isa<DotOperandEncodingAttr>(cvt.getType().getEncoding()))
          foundConvertFromShared = true;
        break;
      }
      op = op->getOperand(0).getDefiningOp();
      if (op)
        LDBG("op: " << *op);
    }
    std::reverse(rets.begin(), rets.end());

    if (foundConvertFromShared)
      return rets;
    return {};
  };

  // [To Do]: OpenAI is working on refactoring the
  // pipeling logic. Currently, the subviewed matrix is
  // Not from block arg. Instead, it is from the beginning
  // of the loop after some calculation
  auto getIncomingOp = [this](Value v) -> Value {
    if (auto arg = mlir::dyn_cast<BlockArgument>(v))
      if (arg.getOwner()->getParentOp() == forOp.getOperation())
        return forOp.getTiedLoopInit(arg)->get();
    return Value();
  };

  for (ttng::WarpGroupDotOp dotOp : dotsInFor){
    auto aType = dotOp.getA().getType();
    auto bType = dotOp.getB().getType();

    auto aEnc = mlir::cast<DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc = mlir::cast<NVMMASharedEncodingAttr>(bType.getEncoding());

    auto kSize = aType.getShape().back();

    LDBG("kSize is " << kSize);

    if (kSize < prefetchWidth)
      return failure();

    auto aVals = getPrefetchSrc(dotOp.getA());
    for(auto op : aVals){
      LDBG("aVals: " << op);
    }

    auto bVals = getPrefetchSrc(dotOp.getB());
    for(auto op : bVals){
      LDBG("bVals: " << op);
    }
    if (aVals.size()) {
      Value aSmem = aVals.front();
      Value bSmem = bVals.front();
      if(!dyn_cast<MemDescSubviewOp>(aSmem.getDefiningOp()) || !dyn_cast<MemDescSubviewOp>(bSmem.getDefiningOp())){
        return failure();
      }
      auto dotOpResult = dotOp.getResult();
      LDBG("dotOpResult is " << dotOpResult);

      if(!dotOpResult.hasOneUse())
        return failure();

      auto dotOpUser = *(dotOpResult.getUsers().begin());
      auto dotWait = dyn_cast<nvidia_gpu::WarpGroupDotWaitOp>(dotOpUser);

      if(!dotWait)
        return failure();

      LDBG("dotWait is " << dotWait);


      LDBG("Successfully check memory source");
      dots.insert(dotOp);
      dot2aVals[dotOp] = aVals;
      for(auto op : aVals){
        if(isa<MemDescSubviewOp, LocalLoadOp>(op.getDefiningOp())){
          dot2aValsLocalLoad[dotOp].push_back(op);
        }
        else{
          dot2aValsElementWise[dotOp].push_back(op);
        }
      }

      for(auto op : dot2aValsLocalLoad[dotOp])
        LDBG("aVal, mem load op " << op);

      for(auto op : dot2aValsElementWise[dotOp])
        LDBG("aVal, elementwise op " << op);

      dot2bVals[dotOp] = bVals;
      dot2aSrcMemDesc[dotOp] = aSmem;
      dot2bSrcMemDesc[dotOp] = bSmem;
      dot2Wait[dotOp] = dotWait.getResult(0);

    }
  }

  return llvm::success();
}

scf::ForOp WGMMAPrefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for(auto v : forOp.getInitArgs())
    loopArgs.push_back(v);

  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  auto setInsertionPointBeforeYield = [](OpBuilder &builder,
                                         scf::ForOp newForOp) {
    if (newForOp.getBody()->mightHaveTerminator()) {
      builder.setInsertionPoint(newForOp.getBody()->getTerminator());
    } else {
      builder.setInsertionPointToEnd(newForOp.getBody());
    }
  };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (op.getNumRegions() > 0){
      setInsertionPointBeforeYield(builder, newForOp);
    }
    for (auto operand : op.getOperands()){
      if (auto def = operand.getDefiningOp()){
        auto dot = dyn_cast<nvidia_gpu::WarpGroupDotOp>(def);
        if (dot && dots.contains(dot)){
          setInsertionPointBeforeYield(builder, newForOp);
        }
      }

    }
    Operation *newOp = builder.clone(op, mapping);
    auto dot = dyn_cast<nvidia_gpu::WarpGroupDotOp>(&op);
    if (dot && dots.contains(dot)) {



    }


  }

  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));

  builder.setInsertionPointToEnd(newForOp.getBody());
  if (!yieldValues.empty())
    builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);

  return newForOp;
}


}



struct WGMMAPrefetchPass : public impl:: TritonGPUWGMMAPrefetchBase<WGMMAPrefetchPass> {
  void runOnOperation() override {

    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }

    getOperation()->walk([&](scf::ForOp forOp) {
      LDBG("check if the layout convert has been removed");
      LDBG(forOp);

      WGMMAPrefetcher prefetcher(forOp);

      if (prefetcher.initialize().failed())
        return;

      scf::ForOp newForOp = prefetcher.createNewForOp();

      for(unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();

    });


  }

};






} // namespace gpu
} // namespace triton
} // namespace mlir


