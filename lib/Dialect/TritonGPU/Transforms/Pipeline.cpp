#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file implements software pipelining for loops. The implementation here
// is inspired by the pipeline pass in Triton (version 2.0) and SCF's
// LoopPipelining.
//
// We divide the loop body into the following phases:
// a. Pre-load operations: for instance, index computation.
// b. Load operations: loading from global memory to shared memory.
// c. Compute operations: for instance, Triton dot.
// d. Post-load operations: for instance, index computation.
//
// To pipeline the loop, we need to:
// - Hoist the pipelinable load operations for the first numStages-1 iterations
// to the loop pre-header
// - Find all the dependencies of the load operations.
// - Rematerialize the dependencies for their values at the first numStage-1
// iterations
// - Assemble the loop body (numStage) and prefetch (numStage + 1).
//
// In the prologue, the sequence of operations is the same as the original loop
// body, following the (a) -> (b) -> (c) -> (d) order. In the loop body,
// however, we first execute the compute operations, then pre-load operations,
// post-load operations, and eventually the asynchronous load operations - in
// the (c) -> (a) -> (d) -> (b) order. This is used to better hide the latency
// of the load operations. Because of this, if post-load operations have direct
// dependencies on the load operations, we could repeat the post-load
// operations. More specifically, this occurs when:
// 1. Any load operand has an immediate dependency argument used at numStage-1.
// 2. The argument is first defined at numStage-2.
// To avoid the repeat, we peeled off post-load operations in the prologue that
// satisfy the above two conditions. See the example below for the definition of
// immediate and non-immediate dependencies.
// If we have a load that immediately depends on a block argument in the
// current iteration, it is an immediate dependency. Otherwise, it is a
// non-immediate dependency, which means the load depends on a block argument
// in the previous iterations.
// For example:
// scf.for (%arg0, %arg1, %arg2) {
//   %0 = load %arg0  <--- immediate dep, this address is initialized before
//   numStages-1.
//   %1 = load %arg1
//   %2 = add %1, %arg2
//   %3 = load %2  <--- non-immediate dep, %arg1 must be an
//   update-to-date value.
// }
//
// Our pipelining pass share some common characteristics with SCF's
// LoopPipelining. However, it is also noteworthy that our pipelining pass has
// the following characteristics different from SCF's LoopPipelining:
// 1. It can handle loop-carried dependencies of distance greater than 1.
// 2. It does not have a complicated epilogue but instead uses masking to handle
// boundary conditions.
// 3. Each operation/loop-carried argument cannot provide values to both
// immediate and non-immediate dependencies. Otherwise, we have to rematerialize
// the operation and arguments, which would likely increase register pressure.
// For example:
// scf.for (%arg0, %arg1, %arg2) {
//  %0 = load %arg0
//  %1 = load %arg1, %0  <--- %0 is both a post-load op at numStages-2 and a
//  pre-load op at numStages-1, so that we need two versions of %0.
//  %2 = add %0, %arg2
//  scf.yield %arg0, %2, %arg2
//  }
//
//===----------------------------------------------------------------------===//

using llvm::MapVector;
using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
/// FIXME(Keren): The pipeline pass shouldn't be aware of nvidia_gpu dialect
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define int_attr(num) builder.getI64IntegerAttr(num)

namespace {

// Pass named attrs (e.g., tt.contiguity) from Triton to Triton
void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  NamedAttrList attrs = op->getDiscardableAttrs();
  // Collect the attributes to propagate: the ones in dictAttrs and not yet on
  // the operation.
  SmallVector<NamedAttribute> toPropagate;
  for (const NamedAttribute attr : dictAttrs.getValue()) {
    if (!attrs.get(attr.getName()))
      toPropagate.push_back(attr);
  }
  // If we found any, let's set them here as a single step.
  if (toPropagate.size()) {
    attrs.append(toPropagate);
    op->setDiscardableAttrs(attrs);
  }
}

struct ConsumerReleaseInfo {
  Value iterVar;
  Value stageVar;
  Value phaseVar;
  Value nextIVVar;
  Value stepVar;
  Value upperBoundVar;
  ttg::CTALayoutAttr CTALayout;
  DenseMap</*consumer=*/Operation *, /*stage=*/int> consumerStageMap;
};
typedef DenseMap</*mbarrierTensor=*/Value, ConsumerReleaseInfo>
    ConsumerReleaseMap;

class LoopPipeliner {
  /// Cache of ForOp and YieldOp related to this pipeliner.
  scf::ForOp forOp;
  scf::YieldOp yieldOp;

  /// Loads to be pipelined
  SetVector<Value> validLoads;
  /// The value that each load will be mapped to (after layout conversion)
  DenseMap<Value, Value> loadsMapping;
  /// load => buffer
  DenseMap<Value, Value> loadsBuffer;
  /// load => buffer type (with shared layout after swizzling)
  DenseMap<Value, RankedTensorType> loadsBufferType;
  /// load => buffer at stage N
  DenseMap<Value, SmallVector<Value>> loadStageBuffer;
  /// load => after extract
  DenseMap<Value, Value> loadsExtract;

  /// XXX(Keren): The following are h100 only and disabled
  /// load => full barrier arrive
  DenseMap<Value, Operation *> loadsBarrierArvOp;
  /// load => mbarriers
  DenseMap<Value, Value> loadsFullBarriers;
  DenseMap<Value, Value> loadsEmptyBarriers;
  /// load => null value or previous load which can share barrier with
  DenseMap<Value, Value> loadsCanShareBarriers;
  /// Maintains the information to emit consumer_release mbarrier_arrive
  ConsumerReleaseMap &consumerReleaseMap;
  bool hasHopperDot = false;
  // XXX(Keren): why the variable name is hopper dot and why do we need this
  // check?
  void checkHopperDots(SetVector<Operation *> &ops);
  // XXX(Keren): it looks more like an optimization to be, not sure if it should
  // exist in the base pipeliner
  void checkOpShareBarriers(SetVector<Operation *> &ops);
  int numLoadsRequireAsyncWait = 0;
  int numLoadsRequireMBarrier = 0;
  // Number of buffers to allocate for each input.
  int numSharedMemorySlices = 0;

  /// Iterator values
  Value nextIV;
  Value pipelineIterIdx;
  Value curWaitIdx;

  // Only needed when numLoadsRequireMBarrier > 0
  Value loopIterIdx;
  Value curPhase;
  Value curEmptyPhase;

  /// Yield values
  SmallVector<Value> nextBuffers;
  SmallVector<Value> extractSlices;
  SmallVector<Value> yieldValues;

  /// The number of stages in the pipeline.
  /// Stages in the range of [0, numStages-1) are in the prologue.
  /// numStages-1 is appended after the loop body.
  int numStages;

  /// Arg indicies
  size_t bufferIdx, loadIdx, depArgsBeginIdx, ivIdx;
  DenseMap<BlockArgument, size_t> depArgsIdx;

  /// XXX(Keren): The mode parameter is hacky, should be refactored
  // false: legacy mode as a temporary solution for backward compatibility
  // true: new mode for hopper
  bool mode;
  int numWarps;
  int numCTAs;

  /// value (in loop) => value at stage N
  DenseMap<Value, SmallVector<Value>> valueMapping;
  /// loop iter arg => value
  DenseMap<BlockArgument, Value> depArgsMapping;
  /// forOp value => newForOp value
  IRMapping mapping;
  /// forOp value => prefetch value
  IRMapping nextMapping;

  /// Dependency ops by program order
  SmallVector<Operation *> orderedDeps;

  /// arg => source operand defined stages
  DenseMap<BlockArgument, DenseSet<int>> immediateArgStages;

  /// block arguments that loads depend on
  SetVector<BlockArgument> depArgs;

  /// operation => source operand defined stages
  DenseMap<Operation *, DenseSet<int>> immediateOpStages;

  /// operations that loads depend on
  SetVector<Operation *> depOps;

  /// Collect all pipelinable ops
  LogicalResult collectOps(SetVector<Operation *> &ops);

  /// Collect values that `v` depends on and are defined inside the loop
  void collectValueDep(Value v, int stage, SetVector<Value> &opDeps);

  /// Collect all op dependencies
  void collectDeps(SetVector<Operation *> &ops,
                   MapVector<Operation *, SetVector<Value>> &opDeps);

  /// Check if none of the ops has valid uses
  LogicalResult checkOpUses(SetVector<Operation *> &ops);

  /// Check if ops have dependencies that are not pipelinable
  void checkOpDeps(SetVector<Operation *> &ops);

  void createBufferTypes();

  void createOrderedDeps();

  /// Return the stage at which `v` is defined prior to `stage`
  int getValueDefStage(Value v, int stage);

  /// Map `origin` to `newValue` at `stage`
  void setValueMapping(Value origin, Value newValue, int stage);

  /// Map `origin` to `newValue` at `stage` according to the association between
  /// yieldOp and forOp
  void setValueMappingYield(Value origin, Value newValue, int stage);

  /// Map `origin` to `newValue` at the next stage according to the association
  /// between yieldOp and forOp
  void setValueMappingYield(scf::ForOp newForOp, Value origin, Value newValue);

  /// Return the value mapped to `origin` at `stage`, if it exists.
  Value lookupOrDefault(Value origin, int stage);

  /// Get the load mask for `loadOp`, given the mapped mask `mappedMask` (if
  /// exists) and the current iteration's `loopCond`.
  Value getLoadMask(tt::LoadOp loadOp, Value mappedMask, Value loopCond,
                    OpBuilder &builder);

  /// Return an empty buffer of size <numStages, ...>
  ttg::AllocTensorOp allocateEmptyBuffer(tt::LoadOp loadOp, OpBuilder &builder);

  /// Collect all args of the new loop
  SmallVector<Value> collectNewLoopArgs();

  /// Clone the forOp and return the new forOp
  scf::ForOp cloneForOp(ArrayRef<Value> newLoopArgs, OpBuilder &builder);

  /// Prefetch the next iteration for `newForOp`
  void prefetchNextIteration(scf::ForOp newForOp, OpBuilder &builder);

  /// Check if curIdx is out of bound and wrap value around if necessary
  Value getBoundedIterationValue(OpBuilder &builder, Value curIdx,
                                 Value upperBoundIdx, Value curValue,
                                 Value initValue);

  /// Assemble `newForOp`'s yield op
  void finalizeYield(scf::ForOp newForOp, OpBuilder &builder);

public:
  LoopPipeliner(scf::ForOp forOp, int numStages, int numWarps, int numCTAs,
                bool mode, int numSharedMemorySlices,
                ConsumerReleaseMap &consumerReleaseMap)
      : forOp(forOp), numStages(numStages), numWarps(numWarps),
        numCTAs(numCTAs), mode(mode),
        numSharedMemorySlices(numSharedMemorySlices),
        consumerReleaseMap(consumerReleaseMap) {
    // cache yieldOp
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LoopPipeliner() = delete;

  /// Collect loads to pipeline. Return success if we can pipeline this loop
  LogicalResult initialize();

  /// Emit pipelined loads (before loop body)
  void emitPrologue();

  /// emit pipelined loads (after loop body)
  void emitEpilogue();

  /// create the new ForOp (add new args & insert prefetched ops)
  scf::ForOp createNewForOp();

  friend struct PipelinePass;
};

/// Collect loads to pipeline. Return success if we can pipeline this loop
LogicalResult LoopPipeliner::collectOps(SetVector<Operation *> &ops) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  for (Operation &op : forOp)
    if (auto loadOp = dyn_cast<tt::LoadOp>(&op)) {
      if (isLoadFromTensorPtr(loadOp)) {
        ops.insert(loadOp);
      } else {
        auto ptr = loadOp.getPtr();
        unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
        if (auto mask = loadOp.getMask())
          vec =
              std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

        auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
        if (!tensorTy || tensorTy.getRank() < 2)
          continue;
        auto ty =
            tensorTy.getElementType().cast<tt::PointerType>().getPointeeType();
        unsigned width = vec * ty.getIntOrFloatBitWidth();
        // We do not pipeline all loads for the following reasons:
        // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8 and 16.
        // 2. It's likely that pipling small loads won't offer much performance
        //    improvement and may even hurt performance by increasing register
        //    pressure.
        if (width >= 32)
          ops.insert(loadOp);
      }
    }

  if (ops.empty())
    return failure();
  else
    return success();
}

void LoopPipeliner::collectValueDep(Value v, int stage,
                                    SetVector<Value> &deps) {
  // Loop-invariant value, skip
  if (v.getParentRegion() != &forOp.getRegion())
    return;

  // Since we only need to peel the loop numStages-1 times, don't worry
  // about depends that are too far away
  if (stage < 0)
    return;

  if (auto arg = v.dyn_cast<BlockArgument>()) {
    if (arg.getArgNumber() > 0) {
      deps.insert(v);
      collectValueDep(yieldOp->getOperand(arg.getArgNumber() - 1), stage - 1,
                      deps);
    }
  } else { // value
    deps.insert(v);
    for (Value op : v.getDefiningOp()->getOperands())
      collectValueDep(op, stage, deps);
  }
}

void LoopPipeliner::collectDeps(
    SetVector<Operation *> &ops,
    MapVector<Operation *, SetVector<Value>> &valueDeps) {
  for (auto op : ops) {
    for (Value v : op->getOperands()) {
      SetVector<Value> deps;
      collectValueDep(v, numStages - 1, deps);
      valueDeps[op] = deps;
    }
  }
}

LogicalResult LoopPipeliner::checkOpUses(SetVector<Operation *> &ops) {
  DenseSet<Operation *> invalidOps;
  // Collect all ops' dependencies
  MapVector<Operation *, SetVector<Value>> opDeps;
  collectDeps(ops, opDeps);

  for (Operation *op : ops) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      // Don't pipeline valid loads that depend on other valid loads
      // (Because if a valid load depends on another valid load, this load needs
      // to wait on the other load in the prologue, which is against the point
      // of the pipeline pass)
      bool isCandidate = true;
      for (Operation *other : ops)
        if (isa<tt::LoadOp>(other))
          if (opDeps[op].contains(other->getResult(0))) {
            isCandidate = false;
            break;
          }
      // We only pipeline loads that have one covert_layout (to dot_op) use
      // TODO: lift this constraint in the future
      if (isCandidate && loadOp.getResult().hasOneUse() &&
          !isLoadFromTensorPtr(loadOp)) {
        isCandidate = false;
        Operation *use = *loadOp.getResult().getUsers().begin();
        Operation *preUse = nullptr;

        // Advance to the first conversion as long as the use resides in shared
        // memory and it has a single use itself
        while (use) {
          if (use->getNumResults() != 1 || !use->getResult(0).hasOneUse())
            break;
          auto tensorType =
              use->getResult(0).getType().dyn_cast<RankedTensorType>();
          if (!tensorType.getEncoding().isa<ttg::SharedEncodingAttr>())
            break;
          preUse = use;
          use = *use->getResult(0).getUsers().begin();
        }

        if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(use)) {
          if (auto tensorType = convertLayout.getResult()
                                    .getType()
                                    .dyn_cast<RankedTensorType>())
            if (auto dotOpEnc = tensorType.getEncoding()
                                    .dyn_cast<ttg::DotOperandEncodingAttr>()) {
              isCandidate = true;
              loadsMapping[loadOp] = convertLayout;
            }
        } else if (preUse && isa<tt::DotOp>(use)) {
          isCandidate = false;
          // for MMAv3 whose dot take SharedEncoding as operands directly
          Operation *post = *loadOp.getResult().getUsers().begin();
          auto newOrder = post->getResult(0)
                              .getType()
                              .cast<RankedTensorType>()
                              .getEncoding()
                              .cast<ttg::SharedEncodingAttr>()
                              .getOrder();
          auto ty = loadOp.getType().cast<RankedTensorType>();
          auto oldOrder = ttg::getOrder(ty.getEncoding());
          // The operand of MMAv3 is in SharedEncoding and it's order should not
          // be changed after FuseTranspositions Pass. So we only pipeline the
          // load if the order of the loaded BlockedEncoding is the same as the
          // order of the SharedEncoding it is converted to.
          // TODO: remove this constraint once the LoadOp supports transpose
          // fusion
          if (newOrder[0] == oldOrder[0] || newOrder[1] == oldOrder[1]) {
            isCandidate = true;
            loadsMapping[loadOp] = preUse->getResult(0);
          }
        }
      } else if (isCandidate && mode && isLoadFromTensorPtr(loadOp)) {
        loadsMapping[loadOp] = loadOp.getResult();
      } else
        isCandidate = false;

      if (!isCandidate)
        invalidOps.insert(loadOp);
      else {
        validLoads.insert(loadOp);
        if (!isLoadFromTensorPtr(loadOp))
          numLoadsRequireAsyncWait++;
        else
          numLoadsRequireMBarrier++;
      }
    }
  }

  for (Operation *op : invalidOps)
    ops.remove(op);

  if (ops.empty())
    return failure();
  else
    return success();
}

void LoopPipeliner::checkHopperDots(SetVector<Operation *> &ops) {
  // dots to be pipelined
  SetVector<Value> dots;
  for (Operation &op : forOp) {
    if (auto dotOp = dyn_cast<tt::DotOp>(&op)) {
      auto resTy = dotOp.getResult().getType().dyn_cast<RankedTensorType>();
      if (auto resEnc = resTy.getEncoding().dyn_cast<ttg::MmaEncodingAttr>()) {
        if (resEnc && resEnc.isHopper()) {
          // Don't pipeline valid dots that depend on ops other than scf.yield
          // and scf.for
          auto dot = dotOp.getResult();
          bool valid = true;

          // all users of dot should be scf.yield
          if (!dot.hasOneUse())
            valid = false;
          if (!isa<scf::YieldOp>(*dot.getUsers().begin()))
            valid = false;

          // C should be a block argument
          auto CArg = dotOp.getOperand(2).dyn_cast<BlockArgument>();
          if (!CArg || !CArg.hasOneUse())
            valid = false;

          if (valid)
            dots.insert(dotOp);
        }
      }
    }
  }

  hasHopperDot = true;
}

void LoopPipeliner::checkOpShareBarriers(SetVector<Operation *> &ops) {
  // Check if loads can share barriers
  auto canShare = [&](Value load0, Value load1) -> bool {
    if (!load0.hasOneUse() || !load1.hasOneUse())
      return false;
    auto use0 = *load0.getUsers().begin();
    auto use1 = *load1.getUsers().begin();
    if (!use0->hasOneUse() || !use1->hasOneUse())
      return false;
    if (*use0->getUsers().begin() != *use1->getUsers().begin())
      return false;
    return true;
  };
  // XXX(Keren): the logic here is pretty weird and might be incomplete
  for (Value loadOp : validLoads) {
    Value depLoad;
    for (auto oldPair : loadsCanShareBarriers) {
      Value oldLoad = oldPair.first;
      if (canShare(loadOp, oldLoad)) {
        depLoad = oldLoad;
        break;
      }
    }
    loadsCanShareBarriers[loadOp] = depLoad;
  }
}

void LoopPipeliner::checkOpDeps(SetVector<Operation *> &ops) {
  SetVector<BlockArgument> nonImmediateDepArgs;
  SetVector<Operation *> nonImmediateOps;
  for (Operation *op : ops) {
    for (Value v : op->getOperands()) {
      SetVector<Value> deps;
      collectValueDep(v, numStages - 1, deps);
      int defStage = getValueDefStage(v, numStages - 1);
      assert(defStage >= 0 &&
             "newLoopArgs has null args without a define op. Consider either "
             "rewrite the loop to reduce cross iteration dependencies or "
             "increase the num_stages value.");
      for (auto dep : deps) {
        auto immediate = deps.front().isa<BlockArgument>();
        if (auto arg = dyn_cast<BlockArgument>(dep)) {
          depArgs.insert(arg);
          if (immediate)
            immediateArgStages[arg].insert(defStage);
          else
            nonImmediateDepArgs.insert(arg);
        } else {
          depOps.insert(dep.getDefiningOp());
          if (immediate)
            immediateOpStages[dep.getDefiningOp()].insert(defStage);
          else
            nonImmediateOps.insert(dep.getDefiningOp());
        }
      }
    }
  }

  // We could remove the following constraints if we can rematerialize in the
  // loop. Check if immediateDepArgs and nonImmediateDepArgs are disjoint.
  for (auto &[arg, stages] : immediateArgStages) {
    assert(stages.size() == 1 &&
           "Triton doesn't support an argument provides values for "
           "immediate operands of loads from multiple stages. Consider "
           "removing post load instructions dependency on this argument.");
    assert(!(nonImmediateDepArgs.contains(arg) &&
             stages.contains(numStages - 2)) &&
           "Loop-carried arguments provide values for both immediate and "
           "non-immediate operands of loads. Please consider removing "
           "pre/post load instructions dependency on this argument.");
  }

  // Check if immediateOps and nonImmediateOps are disjoint.
  for (auto &[op, stages] : immediateOpStages) {
    assert(stages.size() == 1 &&
           "Triton doesn't support an operation provides values for "
           "immediate operands of loads from multiple stages. Consider "
           "removing post load instructions dependency on this argument.");
    assert(!(nonImmediateOps.contains(op) && stages.contains(numStages - 2)) &&
           "Operations provide values for both immediate and "
           "non-immediate operands of loads.  Please consider "
           "removing pre/post load instructions dependency on this "
           "operation.");
  }
}

// helpers
void LoopPipeliner::setValueMapping(Value origin, Value newValue, int stage) {
  if (valueMapping.find(origin) == valueMapping.end())
    valueMapping[origin] = SmallVector<Value>(numStages);
  valueMapping[origin][stage] = newValue;
}

void LoopPipeliner::setValueMappingYield(Value origin, Value newValue,
                                         int stage) {
  for (OpOperand &operand : origin.getUses()) {
    if (operand.getOwner() == yieldOp) {
      auto yieldIdx = operand.getOperandNumber();
      auto value = forOp.getRegionIterArgs()[yieldIdx];
      setValueMapping(value, newValue, stage);
    }
  }
}

void LoopPipeliner::setValueMappingYield(scf::ForOp newForOp, Value origin,
                                         Value newValue) {
  for (OpOperand &operand : origin.getUses()) {
    if (operand.getOwner() == yieldOp) {
      auto yieldIdx = operand.getOperandNumber();
      auto depYieldIdx = depArgsIdx[forOp.getRegionIterArgs()[yieldIdx]];
      auto originArg = forOp.getRegionIterArgs()[yieldIdx];
      nextMapping.map(originArg, newValue);
      auto newArg = newForOp.getRegionIterArgs()[depYieldIdx];
      if (!depArgsMapping.contains(newArg))
        depArgsMapping[newArg] = newValue;
    }
  }
}

Value LoopPipeliner::lookupOrDefault(Value origin, int stage) {
  if (valueMapping.find(origin) == valueMapping.end())
    return origin;
  return valueMapping[origin][stage];
}

void LoopPipeliner::createBufferTypes() {
  for (auto loadCvt : loadsMapping) {
    auto loadOp = loadCvt.first;
    Value cvt = loadCvt.second;
    auto ty = loadOp.getType().cast<RankedTensorType>();
    SmallVector<int64_t> bufferShape(ty.getShape().begin(),
                                     ty.getShape().end());
    bufferShape.insert(bufferShape.begin(), numSharedMemorySlices);
    auto CTALayout = ttg::getCTALayout(ty.getEncoding());
    Attribute sharedEnc;
    if (auto dotOpEnc = cvt.getType()
                            .cast<RankedTensorType>()
                            .getEncoding()
                            .dyn_cast<ttg::DotOperandEncodingAttr>()) {
      // MMAv1 and MMAv2
      bool needTrans = dyn_cast_or_null<tt::TransOp>(
          cvt.getDefiningOp()->getOperand(0).getDefiningOp());
      unsigned bitWidth = ty.getElementType().getIntOrFloatBitWidth();
      sharedEnc = ttg::SharedEncodingAttr::get(
          ty.getContext(), dotOpEnc, ty.getShape(),
          ttg::getOrder(ty.getEncoding()), CTALayout, bitWidth, needTrans);
    } else {
      // MMAv3
      sharedEnc = ttg::SharedEncodingAttr::get(ty.getContext(), ty.getShape(),
                                               ttg::getOrder(ty.getEncoding()),
                                               CTALayout, ty.getElementType());
    }
    // FIXME(Keren): block ptr not handled
    loadsBufferType[loadOp] =
        RankedTensorType::get(bufferShape, ty.getElementType(), sharedEnc);
  }
}

void LoopPipeliner::createOrderedDeps() {
  for (Operation &op : *forOp.getBody()) {
    if (depOps.contains(&op))
      orderedDeps.push_back(&op);
    else if (op.getNumResults() > 0 && validLoads.contains(op.getResult(0)))
      orderedDeps.push_back(&op);
  }
  assert(depOps.size() + validLoads.size() == orderedDeps.size() &&
         "depOps contains invalid values");
}

int LoopPipeliner::getValueDefStage(Value v, int stage) {
  if (stage < 0)
    return -1;
  if (auto arg = v.dyn_cast<BlockArgument>()) {
    if (arg.getArgNumber() > 0)
      return getValueDefStage(yieldOp->getOperand(arg.getArgNumber() - 1),
                              stage - 1);
    llvm_unreachable("Loop induction variable should not be a dependency");
  } else
    return stage;
}

ttg::AllocTensorOp LoopPipeliner::allocateEmptyBuffer(tt::LoadOp loadOp,
                                                      OpBuilder &builder) {
  // Allocate a buffer for each pipelined tensor
  // shape: e.g. (numStages==4), <32x64xbf16> -> <4x32x64xbf16>
  Value convertLayout = loadsMapping[loadOp];
  if (auto tensorType = convertLayout.getType().dyn_cast<RankedTensorType>())
    return builder.create<ttg::AllocTensorOp>(convertLayout.getLoc(),
                                              loadsBufferType[loadOp]);
  llvm_unreachable("Async copy's return should be of RankedTensorType");
}

LogicalResult LoopPipeliner::initialize() {
  // All ops that maybe pipelined
  SetVector<Operation *> ops;

  if (collectOps(ops).failed())
    return failure();

  if (checkOpUses(ops).failed())
    return failure();

  // XXX(Keren): hopper specific, should be cleaned up
  checkHopperDots(ops);

  checkOpShareBarriers(ops);

  checkOpDeps(ops);

  createBufferTypes();

  createOrderedDeps();

  return success();
}

Value LoopPipeliner::getLoadMask(tt::LoadOp loadOp, Value mappedMask,
                                 Value loopCond, OpBuilder &builder) {
  Type maskType = tt::getI1SameShape(loadOp.getType());
  Value mask = loadOp.getMask();
  Value newMask;
  if (mask) {
    Value cond = loopCond;
    if (isa<RankedTensorType>(maskType)) {
      cond = builder.create<tt::SplatOp>(mask.getLoc(), maskType, loopCond);
    }
    newMask = builder.create<arith::AndIOp>(mask.getLoc(), mappedMask, cond);
  } else {
    if (isa<RankedTensorType>(maskType)) {
      newMask =
          builder.create<tt::SplatOp>(loopCond.getLoc(), maskType, loopCond);
    } else {
      newMask = loopCond;
    }
  }
  return newMask;
}

void LoopPipeliner::emitPrologue() {
  OpBuilder builder(forOp);
  // Get init operands for loop carried values
  for (BlockArgument &arg : forOp.getRegionIterArgs()) {
    OpOperand &operand = forOp.getOpOperandForRegionIterArg(arg);
    setValueMapping(arg, operand.get(), 0);
  }

  // Alloc a vector of MBarriers in size numStages for each load to be pipelined
  bool isMcast = false;
  for (Value loadOp : validLoads) {
    auto load = cast<tt::LoadOp>(loadOp.getDefiningOp());
    if (isLoadFromTensorPtr(load)) {
      auto loadTy = loadOp.getType().cast<RankedTensorType>();
      auto CTALayout = ttg::CTALayoutAttr::get(
          load.getContext(),
          /*CTAsPerCGA*/ {static_cast<unsigned>(numCTAs)},
          /*CTASplitNum*/ {1},
          /*CTAOrder*/ {0});
      auto sharedEncoding = ttg::SharedEncodingAttr::get(
          load.getContext(), 1, 1, 1, {0}, CTALayout, false);
      auto mBarriersTy = RankedTensorType::get(
          {numStages}, builder.getIntegerType(64), sharedEncoding);

      if (!loadsCanShareBarriers[loadOp]) {
        Value fullBarriers = builder.create<ttng::AllocMBarrierOp>(
            load.getLoc(), mBarriersTy, 1);
        loadsFullBarriers[loadOp] = fullBarriers;
      }
      auto layout = loadTy.getEncoding();
      auto CTASplitNum = ttg::getCTASplitNum(layout);
      auto CTAsPerCGA = ttg::getCTAsPerCGA(layout);
      if (CTASplitNum != CTAsPerCGA) {
        isMcast = true;
        // FIXME: numConsumerThreads could be 32 as well instead of 128
        // incase the consumer is not GMMA
        unsigned arriveCnt = ttg::getNumWarpsPerCTA(layout);
        if (hasHopperDot)
          arriveCnt /= 4;
        arriveCnt *=
            product<unsigned>(CTAsPerCGA) / product<unsigned>(CTASplitNum);

        Value emptyBarriers = builder.create<ttng::AllocMBarrierOp>(
            load.getLoc(), mBarriersTy, arriveCnt);
        loadsEmptyBarriers[loadOp] = emptyBarriers;
      }
    }
  }

  if (isMcast) {
    builder.create<ttng::ClusterArriveOp>(forOp.getLoc(), /*relaxed*/ 1);
    builder.create<ttng::ClusterWaitOp>(forOp.getLoc());
  }

  // prologue from [0, numStage-1)
  Value iv = forOp.getLowerBound();
  pipelineIterIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
  for (int stage = 0; stage < numStages - 1; ++stage) {
    // Special handling for induction variable as the increment is implicit
    if (stage != 0)
      iv = builder.create<arith::AddIOp>(iv.getLoc(), iv, forOp.getStep());
    setValueMapping(forOp.getInductionVar(), iv, stage);

    // Special handling for loop condition as there is no condition in ForOp
    Value loopCond = builder.create<arith::CmpIOp>(
        iv.getLoc(), arith::CmpIPredicate::slt, iv, forOp.getUpperBound());
    for (Operation *op : orderedDeps) {
      Operation *newOp = nullptr;
      if (validLoads.contains(op->getResult(0))) {
        auto load = cast<tt::LoadOp>(op);
        // Allocate empty buffer
        if (stage == 0) {
          loadsBuffer[load] = allocateEmptyBuffer(load, builder);
          loadStageBuffer[load] = {loadsBuffer[load]};
        }
        // load => copy async
        if (auto loadOp = llvm::dyn_cast<tt::LoadOp>(op)) {
          Value newMask =
              getLoadMask(loadOp, lookupOrDefault(loadOp.getMask(), stage),
                          loopCond, builder);

          if (mode && isLoadFromTensorPtr(loadOp)) {
            auto loc = op->getLoc();
            auto mBarTy = tt::PointerType::get(builder.getIntegerType(64), 3);
            Value stageVal =
                builder.create<arith::ConstantIntOp>(loc, stage, 32);
            // producer_acquire
            if (loadsEmptyBarriers.count(loadOp)) {
              Value emptyBarrier = builder.create<ttng::ExtractMBarrierOp>(
                  loc, mBarTy, loadsEmptyBarriers[loadOp], stageVal);
              auto trueVal =
                  builder.create<arith::ConstantIntOp>(loc, 1, /*bitWidth*/ 1);
              builder.create<ttng::MBarrierWaitOp>(loc, emptyBarrier, trueVal);
            }

            // producer_commit
            Value fullBarrier;
            if (!loadsCanShareBarriers[loadOp]) {
              fullBarrier = builder.create<ttng::ExtractMBarrierOp>(
                  loc, mBarTy, loadsFullBarriers[loadOp], stageVal);
              loadsExtract[loadOp] = fullBarrier;
            } else {
              // Reuse the barrier from previouse load.
              fullBarrier = loadsExtract[loadsCanShareBarriers[loadOp]];
            }

            auto loadTy = loadOp.getType().dyn_cast<RankedTensorType>();
            assert(loadTy);
            auto CTASplitNum = ttg::getCTASplitNum(loadTy.getEncoding());
            auto shapePerSlice =
                ttg::getShapePerCTA(CTASplitNum, loadTy.getShape());
            unsigned elems =
                std::accumulate(shapePerSlice.begin(), shapePerSlice.end(), 1,
                                std::multiplies{});
            elems *= (loadTy.getElementType().getIntOrFloatBitWidth() / 8);

            if (!loadsCanShareBarriers[loadOp]) {
              Value _0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
              Value threadId = builder.create<ttng::GetThreadIdOp>(loc);
              Value pred = builder.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::eq, threadId, _0);
              pred = builder.create<arith::AndIOp>(loc, pred, loopCond);
              Operation *barrierArvOp = builder.create<ttng::MBarrierArriveOp>(
                  loc, fullBarrier, pred,
                  /*remoteCtaId*/ nullptr, /*trackAsyncOp*/ false, elems);
              loadsBarrierArvOp[loadOp] = barrierArvOp;
            } else {
              // Increase the transcnt for barrier of previouse load by the
              // bytes of current load.
              Operation *barrierArvOp =
                  loadsBarrierArvOp[loadsCanShareBarriers[loadOp]];
              unsigned base_elems =
                  barrierArvOp->getAttr("txCount").cast<IntegerAttr>().getInt();
              barrierArvOp->setAttr("txCount",
                                    IntegerAttr::get(builder.getIntegerType(32),
                                                     base_elems + elems));
            }
            newOp = builder.create<ttng::InsertSliceAsyncV2Op>(
                loc, loadsBuffer[loadOp].getType(),
                lookupOrDefault(loadOp.getPtr(), stage),
                loadStageBuffer[loadOp][stage], pipelineIterIdx, fullBarrier,
                newMask, lookupOrDefault(loadOp.getOther(), stage),
                loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile(),
                /*axis*/ 0);
          } else {
            newOp = builder.create<ttg::InsertSliceAsyncOp>(
                op->getLoc(), loadsBuffer[loadOp].getType(),
                lookupOrDefault(loadOp.getPtr(), stage),
                loadStageBuffer[loadOp][stage], pipelineIterIdx, newMask,
                lookupOrDefault(loadOp.getOther(), stage), loadOp.getCache(),
                loadOp.getEvict(), loadOp.getIsVolatile(), /*axis*/ 0);
            builder.create<ttg::AsyncCommitGroupOp>(op->getLoc());
          }
          loadStageBuffer[loadOp].push_back(newOp->getResult(0));
        } else
          llvm_unreachable("This should be LoadOp");
      } else {
        if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
          Value newMask =
              getLoadMask(loadOp, lookupOrDefault(loadOp.getMask(), stage),
                          loopCond, builder);
          newOp = builder.create<tt::LoadOp>(
              loadOp.getLoc(), loadOp.getResult().getType(),
              lookupOrDefault(loadOp.getPtr(), stage), newMask,
              lookupOrDefault(loadOp.getOther(), stage),
              loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
              loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
          addNamedAttrs(newOp, op->getDiscardableAttrDictionary());
        } else
          newOp = builder.clone(*op);
        // Update loop-carried uses
        for (unsigned opIdx = 0; opIdx < op->getNumOperands(); ++opIdx) {
          auto it = valueMapping.find(op->getOperand(opIdx));
          if (it != valueMapping.end()) {
            Value v = it->second[stage];
            assert(v && "Value not found in valueMapping");
            newOp->setOperand(opIdx, v);
          } // else, op at opIdx is a loop-invariant value
        }
      }

      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults())) {
        Value originResult = op->getResult(dstIdx);
        if (validLoads.contains(originResult))
          break;
        setValueMapping(originResult, newOp->getResult(dstIdx), stage);
        // Update mapping for loop-carried values (args)
        setValueMappingYield(op->getResult(dstIdx), newOp->getResult(dstIdx),
                             stage + 1);
      }
    } // for (Operation *op : orderedDeps)

    // Update pipeline index
    pipelineIterIdx = builder.create<arith::AddIOp>(
        iv.getLoc(), pipelineIterIdx,
        builder.create<arith::ConstantIntOp>(iv.getLoc(), 1, 32));
    Value numSlices = builder.create<arith::ConstantIntOp>(
        iv.getLoc(), numSharedMemorySlices, 32);
    Value _0 = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
    pipelineIterIdx = getBoundedIterationValue(builder, pipelineIterIdx,
                                               numSlices, pipelineIterIdx, _0);
    // Some values have not been used by any ops in the loop body
    for (BlockArgument arg : forOp.getRegionIterArgs())
      setValueMappingYield(arg, valueMapping[arg][stage], stage + 1);
  } // for (int stage = 0; stage < numStages - 1; ++stage)

  // async.wait & extract_slice
  if (numLoadsRequireAsyncWait > 0)
    builder.create<ttg::AsyncWaitOp>(validLoads.front().getLoc(),
                                     validLoads.size() * (numStages - 2));
  for (Value loadOp : validLoads) {
    auto bufferType = loadStageBuffer[loadOp][numStages - 1]
                          .getType()
                          .cast<RankedTensorType>();
    auto bufferShape = bufferType.getShape();
    auto sliceType = loadsMapping[loadOp].getType().cast<RankedTensorType>();
    sliceType = RankedTensorType::get({bufferShape[1], bufferShape[2]},
                                      sliceType.getElementType(),
                                      loadsBufferType[loadOp].getEncoding());
    Value extractSlice = builder.create<ttg::ExtractSliceOp>(
        loadOp.getLoc(), sliceType, loadStageBuffer[loadOp][numStages - 1],
        SmallVector<OpFoldResult>{int_attr(0), int_attr(0), int_attr(0)},
        SmallVector<OpFoldResult>{int_attr(1),
                                  int_attr(sliceType.getShape()[0]),
                                  int_attr(sliceType.getShape()[1])},
        SmallVector<OpFoldResult>{int_attr(1), int_attr(1), int_attr(1)});
    loadsExtract[loadOp] = extractSlice;
  }
  curWaitIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
  loopIterIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
  curPhase = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 1);
  curEmptyPhase = builder.create<arith::ConstantIntOp>(iv.getLoc(), 1, 1);
}

void LoopPipeliner::emitEpilogue() {
  // If there's any outstanding async copies, we need to wait for them.
  if (numLoadsRequireAsyncWait > 0) {
    OpBuilder builder(forOp);
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointAfter(forOp);
    builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), 0);
  }
}

SmallVector<Value> LoopPipeliner::collectNewLoopArgs() {
  // Order of new args:
  //   (original args)
  //   (insertSliceAsync buffer at stage numStages - 1) for each load
  //   (extracted tensor) for each load
  //   (depArgs at stage numStages - 1)
  //   (depArgs at stage numStages - 2)
  //   ...
  //   (iv at stage numStages - 2)
  //   (pipeline iteration index)
  //   (loop iteration index)
  //   (wait index)
  //   (phase index)
  //   (empty phase index)

  // We need this to update operands for yield
  // original block arg => new arg's idx
  SmallVector<Value> newLoopArgs;
  for (auto v : forOp.getInitArgs())
    newLoopArgs.push_back(v);

  bufferIdx = newLoopArgs.size();
  for (auto loadOp : validLoads)
    newLoopArgs.push_back(loadStageBuffer[loadOp].back());

  loadIdx = newLoopArgs.size();
  for (auto loadOp : validLoads)
    newLoopArgs.push_back(loadsExtract[loadOp]);

  depArgsBeginIdx = newLoopArgs.size();
  for (auto depArg : depArgs) {
    depArgsIdx[depArg] = newLoopArgs.size();
    if (immediateArgStages[depArg].contains(numStages - 2))
      // Peel off post load ops in numStage-1
      newLoopArgs.push_back(valueMapping[depArg][numStages - 2]);
    else
      newLoopArgs.push_back(valueMapping[depArg][numStages - 1]);
  }

  ivIdx = newLoopArgs.size();
  newLoopArgs.push_back(valueMapping[forOp.getInductionVar()][numStages - 2]);
  newLoopArgs.push_back(pipelineIterIdx);
  newLoopArgs.push_back(curWaitIdx);
  if (numLoadsRequireMBarrier > 0) {
    newLoopArgs.push_back(loopIterIdx);
    newLoopArgs.push_back(curPhase);
    newLoopArgs.push_back(curEmptyPhase);
  }

  return newLoopArgs;
}

scf::ForOp LoopPipeliner::cloneForOp(ArrayRef<Value> newLoopArgs,
                                     OpBuilder &builder) {
  // Clone the original ForOp
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newLoopArgs);

  // Set mapping on body of the new ForOp
  builder.setInsertionPointToStart(newForOp.getBody());
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  // Loop iteration args
  Value upperBound = newForOp.getUpperBound();
  Value step = newForOp.getStep();
  Value curIV = newForOp.getRegionIterArgs()[ivIdx];
  pipelineIterIdx = newForOp.getRegionIterArgs()[ivIdx + 1];
  curWaitIdx = newForOp.getRegionIterArgs()[ivIdx + 2];
  if (numLoadsRequireMBarrier > 0) {
    loopIterIdx = newForOp.getRegionIterArgs()[ivIdx + 3];
    curPhase = newForOp.getRegionIterArgs()[ivIdx + 4];
    curEmptyPhase = newForOp.getRegionIterArgs()[ivIdx + 5];
  }

  // Clone the loop body, replace original args with args of the new ForOp.
  SmallVector<Value> loadsFromTensorPtr;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto cvtOp = dyn_cast<ttg::ConvertLayoutOp>(op)) {
      auto result = op.getResult(0);
      auto cvtDstTy = result.getType().cast<RankedTensorType>();
      auto it =
          std::find(validLoads.begin(), validLoads.end(), op.getOperand(0));
      if (it != validLoads.end()) {
        auto loadArgIdx = std::distance(validLoads.begin(), it);
        if (cvtDstTy.getEncoding().isa<ttg::DotOperandEncodingAttr>()) {
          // We want to find cvt ops that match the following pattern:
          // %0 = load %ptr
          // %1 (dotOperand) = cvt %0
          // We replace the use new load use with a convert layout
          auto cvt = builder.create<ttg::ConvertLayoutOp>(
              result.getLoc(), cvtDstTy,
              newForOp.getRegionIterArgs()[loadIdx + loadArgIdx]);
          mapping.map(result, cvt.getResult());
          continue;
        } else if (cvtDstTy.getEncoding().isa<ttg::SharedEncodingAttr>()) {
          // We want to find cvt ops that match the following pattern:
          // %0 = load %ptr
          // %1 (sharedEncoding) = cvt %0
          // We replace the use new load use with insert_slice_async's result
          mapping.map(result,
                      newForOp.getRegionIterArgs()[loadIdx + loadArgIdx]);
          continue;
        }
      }
    } else if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      if (isLoadFromTensorPtr(loadOp)) {
        // XXX(Keren): The comparison operator using std::find on tensor ptr
        // doesn't work as expected
        auto operand = loadOp.getPtr();
        auto tensorTy =
            operand.getType().cast<tt::PointerType>().getPointeeType();
        auto loadArgIdx = 0;
        for (auto validLoad : validLoads) {
          auto defOp = cast<tt::LoadOp>(validLoad.getDefiningOp());
          if (isLoadFromTensorPtr(defOp)) {
            auto validOperand = defOp.getOperand(0);
            auto validTensorTy =
                validOperand.getType().cast<tt::PointerType>().getPointeeType();
            if (tensorTy == validTensorTy)
              break;
          }
          loadArgIdx++;
        }
        // consumer_wait, emitted before the first consumer
        auto firstConsumer = getFirstUser(loadOp);
        mapping.map(loadOp, newForOp.getRegionIterArgs()[loadIdx + loadArgIdx]);

        // If current load can reuse barriers shared by previous load, then we
        // do nothing.
        if (!loadsCanShareBarriers[loadOp]) {
          // emit mbarrier wait before the first consumer of the loaD
          OpBuilder mBarBuilder(firstConsumer);
          auto mBarTy = tt::PointerType::get(builder.getIntegerType(64), 3);
          Value fullBarrier = mBarBuilder.create<ttng::ExtractMBarrierOp>(
              loadOp.getLoc(), mBarTy, loadsFullBarriers[loadOp], curWaitIdx);
          mBarBuilder.create<ttng::MBarrierWaitOp>(loadOp.getLoc(), fullBarrier,
                                                   curPhase);
        }

        loadsFromTensorPtr.push_back(loadOp);
        continue;
      }
    }
    cloneWithInferType(builder, &op, mapping);
  }

  for (Value load : loadsFromTensorPtr) {
    // consumer_relase, emitted after the last consumer
    // 'the last consumer' might be updated in the following Phase_1 since
    // some of the consumers might be pipelined. Thus we maintain this
    // information in 'consumerReleaseMap' and move the position of
    // consumer_release barrier in a seperate Phase_2 in case necessary.
    if (loadsEmptyBarriers.count(load)) {
      auto users = mapping.lookup(load).getUsers();
      DenseMap</*consumer=*/Operation *, /*stage=*/int> consumerStageMap;
      for (Operation *user : users) {
        // All the stage is initialized to zero before Phase_1,
        // since no consumers has been pipelined yet.
        consumerStageMap[user] = 0;
      }
      auto CTALayout = ttg::getCTALayout(
          load.getType().cast<RankedTensorType>().getEncoding());
      ConsumerReleaseInfo info{
          loopIterIdx, pipelineIterIdx, curEmptyPhase, curIV,
          step,        upperBound,      CTALayout,     consumerStageMap};
      consumerReleaseMap[loadsEmptyBarriers[load]] = info;
    }
  }

  // Remove redundant conversions
  // e.g., %145 = triton_gpu.convert_layout %arg15 : (tensor<128x64xf16,
  // #shared1>) -> tensor<128x64xf16, #shared1>
  for (Operation &op : newForOp.getBody()->without_terminator()) {
    if (auto convert_layout = dyn_cast<ttg::ConvertLayoutOp>(op)) {
      auto result = op.getResult(0);
      auto cvtDstTy = result.getType();
      auto operand = convert_layout.getOperand();
      auto tensorTy = operand.getType();
      if (cvtDstTy == tensorTy)
        result.replaceAllUsesWith(operand);
    }
  }

  return newForOp;
}

Value LoopPipeliner::getBoundedIterationValue(OpBuilder &builder, Value curIdx,
                                              Value upperBoundIdx,
                                              Value curValue, Value initValue) {
  Value cond = builder.create<arith::CmpIOp>(
      curIdx.getLoc(), arith::CmpIPredicate::uge, curIdx, upperBoundIdx);
  Value selectValue = builder.create<mlir::arith::SelectOp>(
      curIdx.getLoc(), cond, initValue, curValue);
  return selectValue;
}

void LoopPipeliner::prefetchNextIteration(scf::ForOp newForOp,
                                          OpBuilder &builder) {
  // Map the dep args of the next iteration to the dep args of the current
  size_t argIdx = 0;
  for (auto depArg : depArgs) {
    BlockArgument nextArg =
        newForOp.getRegionIterArgs()[argIdx + depArgsBeginIdx];
    nextMapping.map(depArg, nextArg);
    ++argIdx;
  }

  // Update loop iteration args
  Value curIV = newForOp.getRegionIterArgs()[ivIdx];
  pipelineIterIdx = newForOp.getRegionIterArgs()[ivIdx + 1];
  curWaitIdx = newForOp.getRegionIterArgs()[ivIdx + 2];
  if (numLoadsRequireMBarrier > 0) {
    loopIterIdx = newForOp.getRegionIterArgs()[ivIdx + 3];
    curPhase = newForOp.getRegionIterArgs()[ivIdx + 4];
    curEmptyPhase = newForOp.getRegionIterArgs()[ivIdx + 5];
  }

  // Special handling for iv & loop condition
  auto idxLoc = curIV.getLoc();
  nextIV = builder.create<arith::AddIOp>(idxLoc, curIV, newForOp.getStep());
  Value nextLoopCond = builder.create<arith::CmpIOp>(
      idxLoc, arith::CmpIPredicate::slt, nextIV, newForOp.getUpperBound());

  // Constants
  Value _0 = builder.create<arith::ConstantIntOp>(idxLoc, 0, 32);
  Value _1 = builder.create<arith::ConstantIntOp>(idxLoc, 1, 32);
  Value numStagesVal =
      builder.create<arith::ConstantIntOp>(idxLoc, numStages, 32);
  Value numSlices =
      builder.create<arith::ConstantIntOp>(idxLoc, numSharedMemorySlices, 32);

  // nextWaitIdx
  Value waitIdxPlusOne = builder.create<arith::AddIOp>(idxLoc, curWaitIdx, _1);
  Value nextWaitIdx = getBoundedIterationValue(builder, waitIdxPlusOne,
                                               numSlices, waitIdxPlusOne, _0);

  // Indices of InsertSliceAsyncOp and ExtractSliceOp
  Value insertSliceIndex = pipelineIterIdx;
  Value extractSliceIndex = nextWaitIdx;

  // Prefetch load deps
  // If a load-dependent instruction that uses a block argument, we
  // shouldn't update the new mapping of the block argument in the current
  // iteration.
  // For example.
  // %a = add %arg0, %c
  // %b = add %arg0, %d
  //
  // Update %arg0 will cause the value of %b to be incorrect.
  // We do need to use the next iteration value of %arg0 because it could be a
  // immediate arg of a load op.
  // load %arg0
  // %a = add %arg0, %c
  // yield %a
  //
  // We reroder instructions so %a and yield are actually before load. load
  // %arg0 should use the updated %arg0.
  IRMapping curMapping = nextMapping;
  for (Operation *op : orderedDeps)
    if (!validLoads.contains(op->getResult(0))) {
      if (immediateOpStages[op].contains(numStages - 2))
        // A post load op that provides values for numStage - 2
        curMapping.map(forOp.getInductionVar(), curIV);
      else
        curMapping.map(forOp.getInductionVar(), nextIV);
      Operation *nextOp;
      if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
        auto newMask =
            getLoadMask(loadOp, curMapping.lookupOrDefault(loadOp.getMask()),
                        nextLoopCond, builder);
        nextOp = builder.create<tt::LoadOp>(
            loadOp.getLoc(), loadOp.getResult().getType(),
            curMapping.lookupOrDefault(loadOp.getPtr()), newMask,
            curMapping.lookupOrDefault(loadOp.getOther()),
            loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
            loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
        addNamedAttrs(nextOp, op->getDiscardableAttrDictionary());
        curMapping.map(loadOp.getResult(), nextOp->getResult(0));
        nextMapping.map(loadOp.getResult(), nextOp->getResult(0));
      } else {
        nextOp = builder.clone(*op, curMapping);
        for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
          nextMapping.map(op->getResult(dstIdx), nextOp->getResult(dstIdx));
      }

      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
        setValueMappingYield(newForOp, op->getResult(dstIdx),
                             nextOp->getResult(dstIdx));
    }

  // loads -> async loads
  for (Operation *op : orderedDeps) {
    Operation *nextOp = nullptr;
    // Update loading mask
    if (validLoads.contains(op->getResult(0))) {
      auto loadOp = llvm::cast<tt::LoadOp>(op);
      auto mask = loadOp.getMask();
      auto newMask =
          getLoadMask(loadOp, nextMapping.lookupOrDefault(loadOp.getMask()),
                      nextLoopCond, builder);
      if (mask) {
        // If mask is defined outside the loop, don't update the map more than
        // once
        if (!(forOp.isDefinedOutsideOfLoop(mask) && nextMapping.contains(mask)))
          nextMapping.map(mask, newMask);
        newMask = nextMapping.lookupOrDefault(loadOp.getMask());
      }
      Value insertedVal;
      if (mode && isLoadFromTensorPtr(loadOp)) {
        auto loc = op->getLoc();
        auto mBarTy = tt::PointerType::get(builder.getIntegerType(64), 3);

        // producer_acquire
        if (loadsEmptyBarriers.count(loadOp)) {
          auto ifOp = builder.create<scf::IfOp>(loc, ArrayRef<Type>{},
                                                nextLoopCond, false);
          builder.setInsertionPointToStart(ifOp.thenBlock());
          Value emptyBarrier = builder.create<ttng::ExtractMBarrierOp>(
              loc, mBarTy, loadsEmptyBarriers[loadOp], insertSliceIndex);
          builder.create<ttng::MBarrierWaitOp>(loc, emptyBarrier,
                                               curEmptyPhase);
          builder.setInsertionPointAfter(ifOp);
        }

        // producer_commit
        Value fullBarrier;
        if (!loadsCanShareBarriers[loadOp]) {
          fullBarrier = builder.create<ttng::ExtractMBarrierOp>(
              loc, mBarTy, loadsFullBarriers[loadOp], insertSliceIndex);
          loadsExtract[loadOp] = fullBarrier;
        } else {
          // Reuse the barrier from previouse load.
          fullBarrier = loadsExtract[loadsCanShareBarriers[loadOp]];
        }

        auto loadTy = loadOp.getType().dyn_cast<RankedTensorType>();
        assert(loadTy);
        auto CTASplitNum = ttg::getCTASplitNum(loadTy.getEncoding());
        auto shapePerSlice =
            ttg::getShapePerCTA(CTASplitNum, loadTy.getShape());
        unsigned elems = std::accumulate(
            shapePerSlice.begin(), shapePerSlice.end(), 1, std::multiplies{});
        elems *= (loadTy.getElementType().getIntOrFloatBitWidth() / 8);
        if (!loadsCanShareBarriers[loadOp]) {
          Value _0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
          Value threadId = builder.create<ttng::GetThreadIdOp>(loc);
          Value pred = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, threadId, _0);
          pred = builder.create<arith::AndIOp>(loc, pred, nextLoopCond);
          Operation *barrierArvOp = builder.create<ttng::MBarrierArriveOp>(
              loc, fullBarrier, pred,
              /*remoteCtaId*/ nullptr,
              /*trackAsyncOp*/ false, elems);
          loadsBarrierArvOp[loadOp] = barrierArvOp;
        } else {
          // Increase the transcnt for barrier of previouse load by the bytes of
          // current load.
          Operation *barrierArvOp =
              loadsBarrierArvOp[loadsCanShareBarriers[loadOp]];
          unsigned base_elems =
              barrierArvOp->getAttr("txCount").cast<IntegerAttr>().getInt();
          barrierArvOp->setAttr(
              "txCount",
              IntegerAttr::get(builder.getIntegerType(32), base_elems + elems));
        }
        insertedVal = builder.create<tt::nvidia_gpu::InsertSliceAsyncV2Op>(
            loc, loadsBuffer[loadOp].getType(),
            nextMapping.lookupOrDefault(loadOp.getPtr()),
            newForOp.getRegionIterArgs()[bufferIdx + nextBuffers.size()],
            insertSliceIndex, fullBarrier, newMask,
            nextMapping.lookupOrDefault(loadOp.getOther()), loadOp.getCache(),
            loadOp.getEvict(), loadOp.getIsVolatile(), /*axis*/ 0);
      } else {
        insertedVal = builder.create<ttg::InsertSliceAsyncOp>(
            op->getLoc(), loadsBuffer[loadOp].getType(),
            nextMapping.lookupOrDefault(loadOp.getPtr()),
            newForOp.getRegionIterArgs()[bufferIdx + nextBuffers.size()],
            insertSliceIndex, newMask,
            nextMapping.lookupOrDefault(loadOp.getOther()), loadOp.getCache(),
            loadOp.getEvict(), loadOp.getIsVolatile(), /*axis*/ 0);
        builder.create<ttg::AsyncCommitGroupOp>(op->getLoc());
      }
      nextBuffers.push_back(insertedVal);
      // Extract slice
      auto bufferType = insertedVal.getType().cast<RankedTensorType>();
      auto bufferShape = bufferType.getShape();
      auto sliceType = loadsMapping[loadOp].getType().cast<RankedTensorType>();
      sliceType = RankedTensorType::get({bufferShape[1], bufferShape[2]},
                                        sliceType.getElementType(),
                                        loadsBufferType[loadOp].getEncoding());

      nextOp = builder.create<ttg::ExtractSliceOp>(
          op->getLoc(), sliceType, insertedVal,
          SmallVector<OpFoldResult>{extractSliceIndex, int_attr(0),
                                    int_attr(0)},
          SmallVector<OpFoldResult>{int_attr(1),
                                    int_attr(sliceType.getShape()[0]),
                                    int_attr(sliceType.getShape()[1])},
          SmallVector<OpFoldResult>{int_attr(1), int_attr(1), int_attr(1)});
      extractSlices.push_back(nextOp->getResult(0));

      // Update mapping of results
      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
        // If this is a loop-carried value, update the mapping for yield
        setValueMappingYield(newForOp, op->getResult(dstIdx),
                             nextOp->getResult(dstIdx));
    }
  }

  // Some values have not been used by any ops in the loop body
  for (BlockArgument arg : forOp.getRegionIterArgs())
    setValueMappingYield(newForOp, arg,
                         newForOp.getRegionIterArgs()[depArgsIdx[arg]]);

  // async.wait & extract_slice
  if (numLoadsRequireAsyncWait > 0) {
    Operation *asyncWait = builder.create<ttg::AsyncWaitOp>(
        validLoads[0].getLoc(), validLoads.size() * (numStages - 2));
    for (auto it = extractSlices.rbegin(); it != extractSlices.rend(); ++it) {
      // move extract_slice after asyncWait
      it->getDefiningOp()->moveAfter(asyncWait);
    }
  }

  // Bump pipelineIterIdx
  Value pipelineIterIdxPlusOne =
      builder.create<arith::AddIOp>(idxLoc, pipelineIterIdx, _1);
  pipelineIterIdx = getBoundedIterationValue(
      builder, pipelineIterIdxPlusOne, numSlices, pipelineIterIdxPlusOne, _0);

  // Bump curWaitIdx
  curWaitIdx = nextWaitIdx;

  if (numLoadsRequireMBarrier > 0) {
    // Bump loopIterIdx
    loopIterIdx = builder.create<arith::AddIOp>(idxLoc, loopIterIdx, _1);

    Value _1_1b = builder.create<arith::ConstantIntOp>(idxLoc, 1, 1);

    // Flip curPhase
    Value nextPhase = builder.create<arith::XOrIOp>(idxLoc, curPhase, _1_1b);
    curPhase = getBoundedIterationValue(builder, waitIdxPlusOne, numStagesVal,
                                        curPhase, nextPhase);

    // Flip curEmptyPhase
    Value nextEmptyPhase =
        builder.create<arith::XOrIOp>(idxLoc, curEmptyPhase, _1_1b);
    curEmptyPhase =
        getBoundedIterationValue(builder, pipelineIterIdxPlusOne, numStagesVal,
                                 curEmptyPhase, nextEmptyPhase);
  }
}

void LoopPipeliner::finalizeYield(scf::ForOp newForOp, OpBuilder &builder) {
  SmallVector<Value> yieldValues;
  for (Value v : yieldOp->getOperands())
    yieldValues.push_back(mapping.lookup(v));
  for (Value nextBuffer : nextBuffers)
    yieldValues.push_back(nextBuffer);
  for (Value nextSlice : extractSlices)
    yieldValues.push_back(nextSlice);

  for (size_t i = depArgsBeginIdx; i < ivIdx; ++i) {
    auto arg = newForOp.getRegionIterArgs()[i];
    assert(depArgsMapping.count(arg) && "Missing loop-carried value");
    yieldValues.push_back(depArgsMapping[arg]);
  }

  // Loop iteration args
  yieldValues.push_back(nextIV);
  yieldValues.push_back(pipelineIterIdx);
  yieldValues.push_back(curWaitIdx);
  if (numLoadsRequireMBarrier > 0) {
    yieldValues.push_back(loopIterIdx);
    yieldValues.push_back(curPhase);
    yieldValues.push_back(curEmptyPhase);
  }

  builder.setInsertionPointToEnd(newForOp.getBody());
  builder.create<scf::YieldOp>(yieldOp->getLoc(), yieldValues);
}

scf::ForOp LoopPipeliner::createNewForOp() {
  OpBuilder builder(forOp);
  auto newLoopArgs = collectNewLoopArgs();
  auto newForOp = cloneForOp(newLoopArgs, builder);
  prefetchNextIteration(newForOp, builder);
  finalizeYield(newForOp, builder);
  return newForOp;
}

// ref: mlir/lib/Dialect/SCF/Transforms/LoopPipelining.cpp
struct PipelinePass : public TritonGPUPipelineBase<PipelinePass> {
  PipelinePass() = default;
  PipelinePass(int numStages, int numWarps, int numCTAs,
               int computeCapability) {
    this->numStages = numStages;
    this->numWarps = numWarps;
    this->numCTAs = numCTAs;
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    // TODO[goostavz]: mode = 0 is temporary for backward compatible, will be
    // deprecated after the refactor of pipeline fully gets done
    // TODO[goostavz]: When mode = 1, the mask of prefetch insert_slice in the
    // prologue is currently not properly provided. Need some second thought on
    // the mask definition of InsertSliceOp when the src is ptr<tensor>
    bool mode =
        computeCapability >= 90 && ::triton::tools::getBoolEnv("ENABLE_TMA");
    if (this->numStages <= 1)
      return;

    // phase 0: pipeline loads in loops
    // Pre-processing
    // we make sure element-wise ops are done *after* the conversion
    // to dot operands
    // we can achieve this with simple recursive pattern matching
    // MLIRContext *context = &getContext();
    // mlir::RewritePatternSet patterns(context);
    // patterns.add<MoveOpAfterLayoutConversion>(context);
    // auto didPreprocess =
    //     applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    llvm::SmallVector<scf::ForOp> newForOps;

    // Currently we schedule stage 0 after stage `numStages - 1` during
    // pipelining therefore we only need `numStages - 1` slice of memory.
    // On Hopper we have a separate post-processing that pipelines wgmma so we
    // need an extra buffer for each input.
    // Note that an alternative would be to keep allocating `numStages` buffers
    // and remove the barrier between the loads from shared memory and the
    // copies from global to shared. This would require improving existing
    // membar analysis.
    int numSharedMemorySlices =
        computeCapability < 90 ? numStages - 1 : numStages;

    // Do the pipelining
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      LoopPipeliner pipeliner(forOp, this->numStages, this->numWarps,
                              this->numCTAs, mode, numSharedMemorySlices,
                              consumerReleaseMap);
      if (pipeliner.initialize().failed())
        return;

      pipeliner.emitPrologue();
      scf::ForOp newForOp = pipeliner.createNewForOp();
      pipeliner.emitEpilogue();
      newForOps.push_back(newForOp);

      // Replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
    });

    // phase 1: pipeline dots in loops
    // A tt.dot suitable for GMMA will be converted to ttg.dot_async. And a
    // ttg.DotWaitOp will synchronize it lagging just one iteration, which is
    // a hueristic rule.
    for (auto forOp : newForOps)
      asyncLaunchDots(forOp);

    // phase 2: emit consumer_release (empty barrier arrive) logics in case of
    //          TMA multicast.
    // For each load ops, it is emitted after its last consumer, if the consumer
    // is another async op, find its associated sync op. Each async load will be
    // emitted with a consumer_release action. The merge of redundant mbarriers
    // will be processed in the consequent OptimizeBarriers pass.
    for (const auto &item : consumerReleaseMap)
      emitConsumerRelease(item.first, item.second, numStages);
  }

private:
  Value getRemoteCTAId(OpBuilder &b, Location loc, ttg::CTALayoutAttr CTALayout,
                       Value remoteCTAIdIdx) const;
  void updateConsumerReleaseInfo(Operation *oldOp, Operation *newOp, int stage);
  void asyncLaunchDots(scf::ForOp forOp);
  void emitConsumerRelease(Value mbarTensor, const ConsumerReleaseInfo &info,
                           int numStages);

  ConsumerReleaseMap consumerReleaseMap;
};

void PipelinePass::updateConsumerReleaseInfo(Operation *oldOp, Operation *newOp,
                                             int stage) {
  for (auto &item : consumerReleaseMap) {
    auto &m = item.second.consumerStageMap;
    if (m.count(oldOp)) {
      m.erase(oldOp);
      m[newOp] = stage;
    }

    for (Value operand : oldOp->getOperands()) {
      Operation *op = operand.getDefiningOp();
      if (op && isa<ttg::ConvertLayoutOp>(op)) {
        auto cvt = cast<ttg::ConvertLayoutOp>(op);
        auto src = cvt.getSrc();
        auto srcEncoding = src.getType().cast<RankedTensorType>().getEncoding();
        auto dstEncoding =
            cvt.getResult().getType().cast<RankedTensorType>().getEncoding();
        if (srcEncoding == dstEncoding && m.count(op)) {
          m.erase(op);
          m[newOp] = stage;
        }
      }
    }
  }
}

void PipelinePass::asyncLaunchDots(scf::ForOp forOp) {
  Block *loop = forOp.getBody();

  /// XXX(Keren): Clean up the following duplicate code with checkDotOp
  /// dots to be pipelined
  SmallVector<tt::DotOp> dots;
  SmallVector<unsigned> resultNeedSync;
  for (Operation &op : *loop) {
    if (auto dotOp = dyn_cast<tt::DotOp>(&op)) {
      auto resTy = dotOp.getResult().getType().dyn_cast<RankedTensorType>();
      if (auto resEnc = resTy.getEncoding().dyn_cast<ttg::MmaEncodingAttr>()) {
        if (resEnc && resEnc.isHopper()) {
          // Don't pipeline valid dots that depend on ops other than scf.yield
          // and scf.for
          auto dot = dotOp.getResult();
          bool valid = true;

          // all users of dot should be scf.yield
          if (!dot.hasOneUse())
            valid = false;
          if (!isa<scf::YieldOp>(*dot.getUsers().begin()))
            valid = false;

          // C should be a block argument
          auto CArg = dotOp.getOperand(2).dyn_cast<BlockArgument>();
          if (!CArg || !CArg.hasOneUse())
            valid = false;

          if (valid) {
            dots.push_back(dotOp);
            resultNeedSync.push_back(
                dotOp->getUses().begin()->getOperandNumber());
          }
        }
      }
    }
  }

  // Early stop: no need to continue if there is no valid dot in the loop.
  if (dots.empty())
    return;

  OpBuilder builder(forOp);
  // 0. insert dot_wait after the last dot in the loop as we implicitly pipeline
  // wgmma ops by one stage.
  // This is needed to prevent shared memory inputs to be overriden before the
  // operation is completed.
  // TODO: merge this with the rest of the pipelining transformation and look at
  // a better representation for async dots.
  tt::DotOp lastDot = dots.back();
  builder.setInsertionPointAfter(lastDot);
  auto dotWait = builder.create<tt::nvidia_gpu::DotWaitOp>(
      lastDot.getLoc(), lastDot.getResult(), dots.size());

  // 1. replace Dot with DotAsync
  for (size_t idx = 0; idx < dots.size(); ++idx) {
    tt::DotOp dotOp = dots[idx];
    builder.setInsertionPoint(dotOp);
    auto dotAsync = builder.create<tt::nvidia_gpu::DotAsyncOp>(
        dotOp.getLoc(), dotOp.getA(), dotOp.getB(), dotOp.getC(),
        dotOp.getAllowTF32(), dotOp.getMaxNumImpreciseAcc());
    dotOp.replaceAllUsesWith(dotAsync.getResult());
    updateConsumerReleaseInfo(dotOp, dotWait, /*stage=*/1);
    dotOp->erase();
  }

  // 2. If there's any outstanding DotAsyncOps, we need to wait for them.
  builder.setInsertionPointAfter(forOp);
  for (unsigned resultIndex : resultNeedSync) {
    Value result = forOp->getResult(resultIndex);
    if (result.use_empty())
      continue;
    auto dotWait =
        builder.create<tt::nvidia_gpu::DotWaitOp>(forOp.getLoc(), result, 0);
    result.replaceAllUsesExcept(dotWait.getResult(), dotWait);
  }
}

Value PipelinePass::getRemoteCTAId(OpBuilder &b, Location loc,
                                   ttg::CTALayoutAttr CTALayout,
                                   Value remoteCTAIdIdx) const {
  auto CTAsPerCGA = CTALayout.getCTAsPerCGA();
  auto CTAOrder = CTALayout.getCTAOrder();
  auto CTASplitNum = CTALayout.getCTASplitNum();

  // Short path when bcastMask is a constant
  bool isConstMcastMask = true;
  for (unsigned s : CTASplitNum) {
    if (s > 1) {
      isConstMcastMask = false;
      break;
    }
  }
  if (isConstMcastMask)
    return remoteCTAIdIdx;

  Value linearCTAId = b.create<ttng::GetClusterCTAIdOp>(loc);
  SmallVector<Value> multiDimCTAId =
      delinearize(b, loc, linearCTAId, CTAsPerCGA, CTAOrder);
  auto rank = CTAOrder.size();
  int bcastDim = -1;
  for (size_t i = 0; i < rank; ++i) {
    if (CTAsPerCGA[i] != CTASplitNum[i]) {
      assert(bcastDim < 0 && "bcast in multiple dims is not expected");
      bcastDim = i;
    }
  }
  multiDimCTAId[bcastDim] = remoteCTAIdIdx;
  return linearize(b, loc, multiDimCTAId, CTAsPerCGA, CTAOrder);
}

void PipelinePass::emitConsumerRelease(Value mbarTensor,
                                       const ConsumerReleaseInfo &info,
                                       int numStages) {
  Value iterVar = info.iterVar;
  Value stage = info.stageVar;
  Value phase = info.phaseVar;
  Value nextIV = info.nextIVVar;
  Value step = info.stepVar;
  Value upperBound = info.upperBoundVar;

  const auto &consumerStageMap = info.consumerStageMap;
  // find the the last consumer among all the consumers with the largest stage.
  SmallVector<Operation *> consumersWithLargestStage;
  int maxStage = 0;
  for (const auto &it : consumerStageMap) {
    if (it.second > maxStage) {
      consumersWithLargestStage.clear();
      consumersWithLargestStage.push_back(it.first);
      maxStage = it.second;
    } else if (it.second == maxStage) {
      consumersWithLargestStage.push_back(it.first);
    }
  }
  assert(consumersWithLargestStage.size() > 0);
  DenseMap<Operation *, size_t> operationId;
  consumersWithLargestStage[0]->getBlock()->walk<WalkOrder::PostOrder>(
      [&](Operation *op) { operationId[op] = operationId.size(); });
  size_t maxId = 0;
  Operation *lastUserWithLargestStage;
  for (Operation *op : consumersWithLargestStage) {
    assert(operationId.find(op) != operationId.end());
    size_t userId = operationId[op];
    if (userId > maxId) {
      maxId = userId;
      lastUserWithLargestStage = op;
    }
  }

  OpBuilder b(&getContext());
  b.setInsertionPointAfter(lastUserWithLargestStage);
  auto loc = lastUserWithLargestStage->getLoc();
  auto maxStageVal = b.create<arith::ConstantIntOp>(loc, maxStage, 32);

  // pred = (iterVar >= maxStage) &&
  //        (threadId % (numConsumerThreads / numRemoteCTAs) == 0);

  // [benzh] maybe we can simplify the logics here
  auto cmpOp = arith::CmpIPredicate::sge;
  if (maxStage == 0)
    cmpOp = arith::CmpIPredicate::sgt;
  Value pred = b.create<arith::CmpIOp>(loc, cmpOp, iterVar, maxStageVal);

  Value threadId = b.create<ttng::GetThreadIdOp>(loc);
  auto CTAsPerCGA = info.CTALayout.getCTAsPerCGA();
  auto CTASplitNum = info.CTALayout.getCTASplitNum();
  auto numRemoteCTAs = std::accumulate(CTAsPerCGA.begin(), CTAsPerCGA.end(), 1,
                                       std::multiplies{}) /
                       std::accumulate(CTASplitNum.begin(), CTASplitNum.end(),
                                       1, std::multiplies{});
  auto numConsumerThreads =
      isa<ttng::DotWaitOp>(lastUserWithLargestStage) ? 128 : 32;
  Value _0 = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value numArrives = b.create<arith::ConstantIntOp>(
      loc, numConsumerThreads / numRemoteCTAs, 32);
  pred = b.create<arith::AndIOp>(
      loc, pred,
      b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          b.create<arith::RemUIOp>(loc, threadId, numArrives), _0));
  // remoteCtaIdIdx = (threadId % numConsumerThreads) / (numConsumerThreads /
  // numRemoteCTAs);
  Value remoteCTAIdIdx = b.create<arith::DivUIOp>(
      loc,
      b.create<arith::RemUIOp>(
          loc, threadId,
          b.create<arith::ConstantIntOp>(loc, numConsumerThreads, 32)),
      numArrives);
  Value remoteCTAId = getRemoteCTAId(b, loc, info.CTALayout, remoteCTAIdIdx);
  Value emptyBarrier = b.create<ttng::ExtractMBarrierOp>(
      loc, tt::PointerType::get(b.getIntegerType(64), 3), mbarTensor, stage);

  Value newNextIV = b.create<arith::AddIOp>(loc, nextIV, step);
  Value nextLoopCond = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               newNextIV, upperBound);
  auto ifOp = b.create<scf::IfOp>(loc, ArrayRef<Type>{}, nextLoopCond,
                                  /*hasElse*/ false);
  b.setInsertionPointToStart(ifOp.thenBlock());

  b.create<ttng::MBarrierArriveOp>(loc, emptyBarrier, pred, remoteCTAId,
                                   /*trackAsyncOp*/ false);
}

} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUPipelinePass(int numStages,
                                                        int numWarps,
                                                        int numCTAs,
                                                        int computeCapability) {
  return std::make_unique<PipelinePass>(numStages, numWarps, numCTAs,
                                        computeCapability);
}
