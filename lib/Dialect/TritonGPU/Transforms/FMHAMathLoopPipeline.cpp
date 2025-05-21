#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WSUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUFMHAMATHLOOPPIPELINE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonnvidiagpu-fmha-math-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

Operation *getSingleUserOp(Operation *op) {
  if (op->getNumResults() == 1 && op->getResult(0).hasOneUse()) {
    return *op->getResult(0).getUsers().begin();
  }
  return nullptr;
}

scf::IfOp createEmptyLoopCheckIfOp(scf::ForOp forOp, OpBuilderWithGroup builder,
                                   Value normMathIdxVar) {
  builder.setInsertionPoint(forOp);
  auto cond = builder.create<arith::CmpIOp>(
      forOp.getLoc(), arith::CmpIPredicate::slt, forOp.getLowerBound(),
      forOp.getUpperBound());

  SmallVector<Type> ifResultTypes = {forOp.getResultTypes().begin(),
                                     forOp.getResultTypes().end()};
  // add normalized index to the ifOp result
  ifResultTypes.push_back(normMathIdxVar.getType());
  auto ifOp = builder.create<scf::IfOp>(forOp.getLoc(), ifResultTypes, cond,
                                        /*withElseRegion*/ true);
  builder.setInsertionPointToEnd(ifOp.thenBlock());
  builder.create<scf::YieldOp>(forOp.getLoc(), forOp.getResults());
  auto thenYield = ifOp.thenBlock()->getTerminator();
  forOp->moveBefore(thenYield);
  builder.setInsertionPointToEnd(ifOp.thenBlock());
  for (auto const &v : llvm::enumerate(forOp.getResults())) {
    forOp.getResult(v.index()).replaceAllUsesExcept(ifOp.getResult(v.index()),
                                                    thenYield);
  }
  builder.setInsertionPointToEnd(ifOp.elseBlock());
  SmallVector<Value> elseYieldValues = forOp.getInits();
  // if the loop is not executed, keep normalized index as 0
  auto zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  elseYieldValues.push_back(zero);
  builder.create<scf::YieldOp>(forOp.getLoc(), elseYieldValues);
  return ifOp;
}

// collect ops related to second GEMM op. basically we start from the
// second dot op and traverse backward to collect all the ops that are
// only used by the second dot op, but there are some hardcoded special cases.
SmallVector<Operation *> collectSecondGemmOp(scf::ForOp forOp) {
  // iterate fom the end
  Operation *secondDotOp = nullptr;
  SmallVector<Operation *> secondGemmOps;
  for (Operation &op : llvm::reverse(forOp.getBody()->getOperations())) {
    if (isa<triton::nvidia_gpu::WarpGroupDotOp>(op)) {
      secondDotOp = &op;
      break;
    }
  }
  assert(secondDotOp && "no dot op found in math loop\n");
  secondGemmOps.push_back(secondDotOp);
  // after split loop, we already inserted dot wait after the dot op
  Operation *waitOp = getSingleUserOp(secondDotOp);
  assert(waitOp && isa<triton::nvidia_gpu::WarpGroupDotWaitOp>(waitOp));
  secondGemmOps.push_back(waitOp);

  assert(secondDotOp->getNumOperands() == 3);
  // the first operand of the warp_group_dot is the result of softmax
  // don't peel the convert_layout op if there is one.
  Value operandA = secondDotOp->getOperand(0);
  if (isa<mlir::triton::gpu::ConvertLayoutOp>(operandA.getDefiningOp())) {
    secondGemmOps.push_back(operandA.getDefiningOp());
  }
  // the second operand of the warp_group_dot is V which is loaded by tma_get
  Value operandB = secondDotOp->getOperand(1);
  Operation *localAllocOp = nullptr;
  if (isa<triton::gpu::LocalAllocOp>(operandB.getDefiningOp())) {
    localAllocOp = operandB.getDefiningOp();
    secondGemmOps.push_back(localAllocOp);
  } else if (isa<triton::TransOp>(operandB.getDefiningOp())) {
    Operation *transOp = operandB.getDefiningOp();
    secondGemmOps.push_back(transOp);
    localAllocOp = transOp->getOperand(0).getDefiningOp();
    assert(isa<triton::nvidia_gpu::ArefGetEnterOp>(localAllocOp) &&
           "The second operand of GEMM PV should be a ArefWaitGetOp");
    secondGemmOps.push_back(localAllocOp);
  } else if (isa<triton::gpu::MemDescTransOp>(operandB.getDefiningOp())) {
    Operation *transOp = operandB.getDefiningOp();
    secondGemmOps.push_back(transOp);
    localAllocOp = transOp->getOperand(0).getDefiningOp();
    assert(isa<triton::nvidia_gpu::ArefGetEnterOp>(localAllocOp) &&
           "The second operand of GEMM PV should be a ArefGetEnterOp");
    secondGemmOps.push_back(localAllocOp);
  } else if (isa<triton::nvidia_gpu::ArefGetEnterOp>(
                 operandB.getDefiningOp())) {
    localAllocOp = operandB.getDefiningOp();
    secondGemmOps.push_back(localAllocOp);
  } else {
    assert(0 && "Unsupported operand");
  }
  SmallVector<Operation *> ops;
  ops.push_back(localAllocOp);
  secondGemmOps.push_back(localAllocOp);
  while (!ops.empty()) {
    Operation *op = ops.pop_back_val();
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      // if the defining op is inside the loop body
      if (defOp && defOp->getBlock() == forOp.getBody()) {
        ops.push_back(defOp);
      }
      bool usedOnlyBySecondGemm = true;
      for (Operation *userOp : operand.getUsers()) {
        if (std::find(secondGemmOps.begin(), secondGemmOps.end(), userOp) ==
            secondGemmOps.end()) {
          usedOnlyBySecondGemm = false;
          break;
        }
      }
      if (usedOnlyBySecondGemm) {
        secondGemmOps.push_back(defOp);
      }
    }
  }
  // the third operand of the warp_group_dot is acc
  Value operandAcc = secondDotOp->getOperand(2);
  // some pattern match here to try to avoid creating too many new
  // loop carried variables
  assert(isa<arith::MulFOp>(operandAcc.getDefiningOp()));
  Operation *mulOp = operandAcc.getDefiningOp();
  secondGemmOps.push_back(mulOp);
  assert(isa<triton::BroadcastOp>(mulOp->getOperand(1).getDefiningOp()));
  Operation *broadcastOp = mulOp->getOperand(1).getDefiningOp();
  secondGemmOps.push_back(broadcastOp);
  if (isa<triton::gpu::ConvertLayoutOp>(
          broadcastOp->getOperand(0).getDefiningOp())) {
    Operation *convertOp = broadcastOp->getOperand(0).getDefiningOp();
    secondGemmOps.push_back(convertOp);
    assert(isa<triton::ExpandDimsOp>(convertOp->getOperand(0).getDefiningOp()));
    Operation *expandOp = convertOp->getOperand(0).getDefiningOp();
    secondGemmOps.push_back(expandOp);
  } else {
    assert(
        isa<triton::ExpandDimsOp>(broadcastOp->getOperand(0).getDefiningOp()));
    Operation *expandOp = broadcastOp->getOperand(0).getDefiningOp();
    secondGemmOps.push_back(expandOp);
  }
  return secondGemmOps;
}

std::pair<scf::ForOp, Value> peelFirstGemmAndSoftmax(
    scf::ForOp forOp, SmallVector<Operation *> &newSecondGemmOps,
    Operation *&peeledBeforeDotOp, Value normMathIdx, int AREF_SIZE) {
  // collect all ops related to second gemm. everything else in the loop are
  // to be peeled out
  SmallVector<Operation *> secondGemmOps = collectSecondGemmOp(forOp);

  IRMapping peelMapping;
  // initial values of the loop variables
  auto iterOperands = forOp.getInits();
  // loop variables
  auto iterArgs = forOp.getRegionIterArgs();
  // when peeling out the ops, replace all loop variables with the
  // corresponding initial values
  for (auto e : llvm::zip(iterArgs, iterOperands)) {
    peelMapping.map(std::get<0>(e), std::get<1>(e));
  }
  peelMapping.map(forOp.getInductionVar(), forOp.getLowerBound());

  // some initial value will change after peeling
  // mapping from index to new initial value
  DenseMap<int, Value> updateInitValMap;
  // the loop may have new loop variables
  // mapping from the value in the old loop to the value in the new loop. the
  // latter will become the initial value of the loop variable
  DenseMap<Value, Value> newLoopVarMap;

  // set insertpoint to before the old forOp loop
  OpBuilderWithGroup builder(forOp, ATTR_WS_MMA);
  builder.setInsertionPoint(forOp);
  for (Operation &op : forOp.getBody()->without_terminator()) {
    // peel out the ops that are not in the second gemm
    if (std::find(secondGemmOps.begin(), secondGemmOps.end(), &op) ==
        secondGemmOps.end()) {
      Operation *cloneOp = builder.clone(op, peelMapping);
      if (isa<triton::nvidia_gpu::WarpGroupDotOp>(cloneOp)) {
        assert(peeledBeforeDotOp == nullptr);
        peeledBeforeDotOp = cloneOp;
      }
      // if the op defines yield variables, after peeling out we replace the
      // corresponding loop init val with the value defined by the peeled op
      // ======
      // Before
      // ======
      // for arg = init
      //   y = x
      //   yield y
      // =====
      // After
      // =====
      // y' = x
      // for arg = y'
      //   y = x
      //   yield y
      for (const auto &v :
           llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
        if (v.value().getDefiningOp() == &op) {
          for (const auto &o : llvm::enumerate(op.getResults())) {
            if (v.value() == o.value()) {
              // map y -> y'
              updateInitValMap[v.index()] = cloneOp->getResult(o.index());
              break;
            }
          }
        }
      }
      // if a to-be-hoisted op defines a value used by a not hoisted op, this
      // value will eventually be a new loop variable
      // ======
      // Before
      // ======
      // for
      //   y = x // is hoisted out
      //   z = y // isn't hoisted out
      // =====
      // After
      // =====
      // y' = x
      // for arg = y'
      //   y = x'
      //   z = arg
      //   yield y
      for (OpResult opResult : op.getOpResults()) {
        for (auto &use : opResult.getUses()) {
          if (std::find(secondGemmOps.begin(), secondGemmOps.end(),
                        use.getOwner()) != secondGemmOps.end()) {
            if (isa<IndexType>(opResult.getType())) {
              // hack: skip the index type which is used by extract op
              // we'll create a new index for the tma_get for the second gemm
              assert(isa<tensor::ExtractOp>(use.getOwner()));
              continue;
            }
            // map y -> y'
            newLoopVarMap[use.getOwner()->getOperand(use.getOperandNumber())] =
                cloneOp->getResult(opResult.getResultNumber());
          }
        }
      }
    }
  }

  // now create a new forOp for the peeled loop
  IRMapping forMapping;
  SmallVector<Value> newInitVals(forOp.getInitArgs().begin(),
                                 forOp.getInitArgs().end());
  // the last init value is the normMathIdx which was 0. after peeling it
  // should be 1
  Value one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
  int normMathIdxIdx = forOp.getInitArgs().size() - 1;
  updateInitValMap[normMathIdxIdx] = one;
  // update loop initial values for certain values we have collected
  int origNumInitVals = forOp.getInitArgs().size();
  for (auto &e : updateInitValMap) {
    newInitVals[e.first] = e.second;
  }
  // add y' in the above example to new loop init values
  for (auto &e : newLoopVarMap) {
    newInitVals.push_back(e.second);
  }

  // notice the loop lower bound change here
  Value ub = forOp.getUpperBound();
  Value step = forOp.getStep();
  Value lb = builder.create<arith::AddIOp>(forOp.getLoc(),
                                           forOp.getLowerBound(), step);
  auto newForOp =
      builder.create<scf::ForOp>(forOp.getLoc(), lb, ub, step, newInitVals);

  // the normMathIdx within the peeled loop is the corresponding loop variable
  auto normMathIdxVar = newForOp.getRegionIterArgs()[normMathIdxIdx];

  builder.setInsertionPointToStart(newForOp.getBody());
  // map the old loop var to the new loop var
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
    forMapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  }
  forMapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  // this map from y to arg in the above example
  DenseMap<Value, Value> newYieldValueToLoopVar;
  for (const auto &v : llvm::enumerate(newLoopVarMap)) {
    newYieldValueToLoopVar[v.value().first] =
        newForOp.getRegionIterArgs()[v.index() + origNumInitVals];
  }
  for (Operation &op : forOp.getBody()->without_terminator()) {
    Operation *clonedOp = builder.clone(op, forMapping);
    // if an extract op is used by tma_get that is part of second GEMM loading
    // V, we need to update its index to normMathIdx - 1
    if (std::find(secondGemmOps.begin(), secondGemmOps.end(), &op) !=
        secondGemmOps.end()) {
      if (isa<triton::nvidia_gpu::ArefGetEnterOp>(op)) {
        auto one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
        auto minus =
            builder.create<arith::SubIOp>(forOp.getLoc(), normMathIdxVar, one);
        newSecondGemmOps.push_back(one);
        newSecondGemmOps.push_back(minus);
        auto clonedGetEnterOp =
            cast<triton::nvidia_gpu::ArefGetEnterOp>(clonedOp);
        clonedGetEnterOp.setIndex(minus.getResult());
      }
      newSecondGemmOps.push_back(clonedOp);
    }
  }
  // create the yield op for the cloned loop
  // first collect the mapped values of the original yield operands
  SmallVector<Value> newYieldValues;
  for (const auto &v :
       llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
    newYieldValues.push_back(forMapping.lookupOrDefault(v.value()));
  }
  // we may have added a few new yield operands
  for (auto &e : newLoopVarMap) {
    Value newYieldValue = forMapping.lookupOrDefault(e.first);
    // replace the use of new y in newSecondGemmOps with arg
    newYieldValue.replaceUsesWithIf(
        newYieldValueToLoopVar[e.first], [&](OpOperand &operand) {
          Operation *op = operand.getOwner();
          return std::find(newSecondGemmOps.begin(), newSecondGemmOps.end(),
                           op) != newSecondGemmOps.end();
        });
    newYieldValues.push_back(newYieldValue);
  }
  builder.create<scf::YieldOp>(forOp.getLoc(), newYieldValues);
  for (const auto &v :
       llvm::enumerate(forOp.getBody()->getTerminator()->getOperands())) {
    forOp.getResult(v.index()).replaceAllUsesWith(
        newForOp.getResult(v.index()));
  }
  // replace all uses of loop induction variable i in secondGemmOps with
  // (i - step)
  builder.setInsertionPointToStart(newForOp.getBody());
  Value offsetInd = builder.create<arith::SubIOp>(
      newForOp.getLoc(), newForOp.getInductionVar(), newForOp.getStep());
  for (auto &use : newForOp.getInductionVar().getUses()) {
    if (std::find(newSecondGemmOps.begin(), newSecondGemmOps.end(),
                  use.getOwner()) != newSecondGemmOps.end()) {
      use.getOwner()->setOperand(use.getOperandNumber(), offsetInd);
    }
  }

  forOp.erase();
  return {newForOp, normMathIdxVar};
}

Operation *findFirstDotOp(scf::ForOp forOp) {
  Operation *firstDotOp = nullptr;
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (isMMAOp(&op)) {
      firstDotOp = &op;
      break;
    }
  }
  return firstDotOp;
}

void reorderSecondGemm(scf::ForOp forOp,
                       SmallVector<Operation *> &secondGemmOps) {
  OpBuilder builder(forOp);
  // the insertion point is the last op of the first gemm
  Operation *firstDotOp = findFirstDotOp(forOp);
  Operation *waitOp = getSingleUserOp(firstDotOp);
  assert(waitOp);
  Operation *insertPoint = waitOp;
  for (Operation *op : secondGemmOps) {
    op->moveAfter(insertPoint);
    insertPoint = op;
  }
}

void updateWarpGroupDotWaitOpForFMHA(
    scf::ForOp forOp, OpBuilderWithGroup &builder,
    SmallVector<triton::nvidia_gpu::ArefCreateOp> &arefOps,
    Operation *peeledBeforeDotOp, Value normMathIdx, int AREF_SIZE) {
  // QK GEMM
  Operation *firstDot = nullptr;
  // PV GEMM
  Operation *secondDot = nullptr;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<mlir::triton::nvidia_gpu::WarpGroupDotOp>(op)) {
      if (firstDot == nullptr) {
        firstDot = &op;
      } else {
        assert(secondDot == nullptr);
        secondDot = &op;
      }
    }
  }

  // insert a wait 1 after the second gemm in the loop body to make sure the
  // first gemm in this iteration is done
  auto waitOp1 = dyn_cast<triton::nvidia_gpu::WarpGroupDotWaitOp>(
      getSingleUserOp(secondDot));
  waitOp1.setPendings(1);
  builder.setInsertionPointAfter(waitOp1);
// here we can guarantee the k-th iteration of the first GEMM is done, where
// k is normalized loop index
  {
    assert(arefOps.size() == 2);
    builder.create<triton::nvidia_gpu::ArefGetExitOp>(
        forOp.getLoc(), arefOps[0],  normMathIdx,
        ArrayAttr::get(
            builder.getContext(),
            nvidia_gpu::ArefConsumerAttr::get(
                builder.getContext(), nvidia_gpu::ArefConsumer::WGMMA)));
  }

  // move the wait 0 to the end of the loop body so that the second gemm in the
  // loop body can overlap with softmax and we make sure the second gemm is done
  // before end of the iteration
  auto waitOp2 = dyn_cast<triton::nvidia_gpu::WarpGroupDotWaitOp>(
      getSingleUserOp(secondDot));
  waitOp2.setPendings(0);
  waitOp2.getOperation()->moveBefore(forOp.getBody()->getTerminator());
  builder.setInsertionPointAfter(waitOp2);
// here we can guarantee the (k-1)th iteration of the second GEMM is done
// because we want k-1 to start from 0, so we actually don't need to add
// offset here
  {
    auto one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
    auto minus =
        builder.create<arith::SubIOp>(forOp.getLoc(), normMathIdx, one);
    assert(arefOps.size() == 2);
    builder.create<triton::nvidia_gpu::ArefGetExitOp>(
        forOp.getLoc(), arefOps[1], minus.getResult(),
        ArrayAttr::get(
            builder.getContext(),
            nvidia_gpu::ArefConsumerAttr::get(
                builder.getContext(), nvidia_gpu::ArefConsumer::WGMMA)));
  }

  // handle the dot op that peeled to before the loop. the normalized index
  // for this dot is always 0
  auto waitOp3 = dyn_cast<triton::nvidia_gpu::WarpGroupDotWaitOp>(
      getSingleUserOp(peeledBeforeDotOp));
  waitOp3.setPendings(0);
  builder.setInsertionPointAfter(waitOp3);
  {
    auto zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
    assert(arefOps.size() == 2);
    builder.create<triton::nvidia_gpu::ArefGetExitOp>(
        forOp.getLoc(), arefOps[0], zero.getResult(),
        ArrayAttr::get(
            builder.getContext(),
            nvidia_gpu::ArefConsumerAttr::get(
                builder.getContext(), nvidia_gpu::ArefConsumer::WGMMA)));
  }
}

/*
  k in below is normalized

  load Q
  P =
  for k = [1, ktiles-1) {
    K = tma_get(aref_K)
    GEMMQK k
    V = tma_get(aref_V)
    GEMMPV k-1
    wait 1
    consumed(K) k
    SoftMax k
    wait 0
    consumed(V) k-1
  }
  GEMMPV(P)
  wait 0
  consumed(V) k
*/
Value appendSecondGemm(scf::ForOp forOp, OpBuilderWithGroup &builder,
                       SmallVector<triton::nvidia_gpu::ArefCreateOp> &arefOps,
                       SmallVector<Operation *> &secondGemmOps,
                       Value normMathIdx, int AREF_SIZE) {
  // set insertion point to after the loop
  builder.setInsertionPointAfter(forOp);

  IRMapping mapping;
  for (const auto &v : llvm::enumerate(forOp.getRegionIterArgs())) {
    mapping.map(v.value(), forOp.getResult(v.index()));
  }
  // map the induction variable to upper - step
  auto newUpper = builder.create<arith::SubIOp>(
      forOp.getLoc(), forOp.getUpperBound(), forOp.getStep());
  mapping.map(forOp.getInductionVar(), newUpper);

  auto yieldOperands = forOp.getBody()->getTerminator()->getOperands();
  SmallVector<Operation *> clonedOps;
  Operation *dotOp = nullptr;
  Operation *tmaGetOp = nullptr;
  for (Operation *op : secondGemmOps) {
    Operation *clonedOp = builder.clone(*op, mapping);
    clonedOps.push_back(clonedOp);
    if (isa<triton::nvidia_gpu::WarpGroupDotOp>(clonedOp)) {
      assert(dotOp == nullptr);
      dotOp = clonedOp;
    }
    if (isa<triton::nvidia_gpu::ArefGetEnterOp>(clonedOp)) {
      assert(tmaGetOp == nullptr);
      tmaGetOp = clonedOp;
    }
    // if the op defines any loop variables, after we cloned it after the loop
    // we need to replace the use of corresponding forOp result to the result
    // of this cloned op
    for (const auto &result : llvm::enumerate(op->getResults())) {
      for (const auto &yield : llvm::enumerate(yieldOperands)) {
        if (result.value() == yield.value()) {
          // only replace the use if the operation is outside the loop
          forOp.getResult(yield.index())
              .replaceUsesWithIf(
                  clonedOp->getResult(result.index()), [&](OpOperand &use) {
                    return std::find(clonedOps.begin(), clonedOps.end(),
                                     use.getOwner()) == clonedOps.end();
                  });
        }
      }
    }
  }
  assert(dotOp != nullptr && tmaGetOp != nullptr);
  // here the normMathIdx is still set to one of the forOp loop variable, we
  // need to find out which loop variable it is and set normMathIdx to the
  // corresponding forOp result
  int normMathIdxIdx = -1;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (arg.value() == normMathIdx) {
      normMathIdxIdx = arg.index();
    }
  }
  assert(normMathIdxIdx != -1);
  auto postLoopNormIdx = forOp.getResult(normMathIdxIdx);

  // set tma_get index to normMathIdx - 1
  builder.setInsertionPoint(tmaGetOp);
  auto one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
  auto minus =
      builder.create<arith::SubIOp>(forOp.getLoc(), postLoopNormIdx, one);

  auto waitOp = getSingleUserOp(dotOp);
  builder.setInsertionPointAfter(waitOp);
  assert(arefOps.size() == 2);
  builder.create<triton::nvidia_gpu::ArefGetExitOp>(
      forOp.getLoc(), arefOps[1], minus.getResult(),
      ArrayAttr::get(
          builder.getContext(),
          nvidia_gpu::ArefConsumerAttr::get(builder.getContext(),
                                            nvidia_gpu::ArefConsumer::WGMMA)));
  return postLoopNormIdx;
}

} // namespace

// remove ifOp and only keep the then block if we can prove a condition is
// always true, for example, i >= 0 where i is a forOp index which has lower
// bound of 0 and step > 0.
class SimplifyIf : public OpRewritePattern<scf::IfOp> {
  bool isGEZero(Value v) const {
    if (auto cst = v.getDefiningOp<arith::ConstantIntOp>()) {
      return cst.value() >= 0;
    } else if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
      return isGEZero(sub.getOperand(0));
    } else if (auto add = v.getDefiningOp<arith::AddIOp>()) {
      return isGEZero(add.getOperand(0)) && isGEZero(add.getOperand(1));
    } else if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
      return isGEZero(mul.getOperand(0)) && isGEZero(mul.getOperand(1));
    } else if (auto div = v.getDefiningOp<arith::DivSIOp>()) {
      // FIXME: operand 1 should > 0
      return isGEZero(div.getOperand(0)) && isGEZero(div.getOperand(1));
    }

    // loop index case
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      auto *parentOp = blockArg.getOwner()->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        if (blockArg == forOp.getInductionVar()) {
          // Check lower bound >= 0 and step > 0
          if (isGEZero(forOp.getLowerBound())) {
            if (auto stepConst =
                    forOp.getStep().getDefiningOp<arith::ConstantIntOp>()) {
              return stepConst.value() > 0;
            }
          }
        }
      }
    }

    return false;
  }

  bool alwaysTrue(Value cond) const {
    if (auto cst = cond.getDefiningOp<arith::ConstantOp>()) {
      return cast<BoolAttr>(cst.getValue()).getValue();
    } else if (auto cmp = cond.getDefiningOp<arith::CmpIOp>()) {
      if (cmp.getPredicate() == arith::CmpIPredicate::sge) {
        if (auto cst =
                cmp.getOperand(1).getDefiningOp<arith::ConstantIntOp>()) {
          return (cst.value() == 0) && isGEZero(cmp.getOperand(0));
        }
      }
    }
    return false;
  }

public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (alwaysTrue(ifOp.getCondition())) {
      rewriter.replaceOp(ifOp,
                         ifOp.thenBlock()->getTerminator()->getOperands());
      return success();
    }
    return failure();
  }
};

class TritonGPUFMHAMathLoopPipelinePass
    : public impl::TritonGPUFMHAMathLoopPipelineBase<
          TritonGPUFMHAMathLoopPipelinePass> {

  void
  pipelineFMHAMathLoop(scf::ForOp forOp, OpBuilderWithGroup builder,
                       SmallVector<triton::nvidia_gpu::ArefCreateOp> &arefOps) {
    int AREF_SIZE = TritonGPUDialect::getNumStages(getOperation());

    // the contract is that the last loop variable is the normalized index which
    // we create at the split loop pass
    auto normMathIdxVar = forOp.getRegionIterArgs().back();

    // create an ifOp and put first loop in math WG in the then block. this
    // is to handle the special case of causal FMHA where the trip count of
    // the first loop is 0. we pipeline the loop inside the ifOp so that the
    // hoisted code won't be executed if the loop is empty
    auto ifOp = createEmptyLoopCheckIfOp(forOp, builder, normMathIdxVar);

    // peel the first GEMM and Softmax before the loop
    SmallVector<Operation *> secondGemmOps;
    Operation *peeledBeforeDotOp = nullptr;
    auto [peeled, newNormMathIdx] = peelFirstGemmAndSoftmax(
        forOp, secondGemmOps, peeledBeforeDotOp, normMathIdxVar, AREF_SIZE);
    // inside the loop body, move the second GEMM before the softmax so that it
    // can be overlapped with softmax
    reorderSecondGemm(peeled, secondGemmOps);
    updateWarpGroupDotWaitOpForFMHA(peeled, builder, arefOps, peeledBeforeDotOp,
                                    newNormMathIdx, AREF_SIZE);
    // peel the second GEMM after the loop
    auto postLoopNormIdx = appendSecondGemm(
        peeled, builder, arefOps, secondGemmOps, newNormMathIdx, AREF_SIZE);

    // replace the normalized index yield from the then block with the updated
    // one
    auto thenYield = ifOp.thenBlock()->getTerminator();
    SmallVector<Value> thenYieldValues = thenYield->getOperands();
    thenYieldValues.push_back(postLoopNormIdx);
    thenYield->erase();
    builder.setInsertionPointToEnd(ifOp.thenBlock());
    builder.create<scf::YieldOp>(peeled.getLoc(), thenYieldValues);
  }

public:
  void runOnOperation() override {
    // the input IR should already be in the form of warp specialized loops
    ModuleOp m = getOperation();
    // collect all Math loop in the module
    SmallVector<scf::ForOp> fmhaMathForOps;
    m.walk([&](scf::ForOp forOp) {
      if (isFMHAMathLoop(forOp))
        fmhaMathForOps.push_back(forOp);
    });
    if (fmhaMathForOps.empty())
      return;

    mlir::RewritePatternSet patterns(m.getContext());
    patterns.add<SimplifyIf>(m.getContext());
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();

    assert(fmhaMathForOps.size() == 1 || fmhaMathForOps.size() == 2);
    OpBuilderWithGroup builder(m, ATTR_WS_MMA);
    // collect all ArefCreateOp
    SmallVector<triton::nvidia_gpu::ArefCreateOp> arefOps;
    m.walk([&](triton::nvidia_gpu::ArefCreateOp arefOp) {
      arefOps.push_back(arefOp);
    });
    assert(arefOps.size() == 3);
    arefOps.erase(arefOps.end() - 1);
    std::swap(arefOps[0], arefOps[1]);

    SmallVector<triton::nvidia_gpu::ArefGetExitOp> getExitOps;
    m.walk([&](triton::nvidia_gpu::ArefGetExitOp getOp) {
      for (auto arefOp : arefOps) {
        if (arefOp.getResult() == getOp.getAref()) {
          getExitOps.push_back(getOp);
          break;
        }
      }
    });
    for (auto op : getExitOps)
      op.erase();

    pipelineFMHAMathLoop(fmhaMathForOps[0], builder, arefOps);
  }
};
} // namespace gpu
} // namespace triton
} // namespace mlir
