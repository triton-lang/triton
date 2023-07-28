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
#include "llvm/ADT/MapVector.h"

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
namespace ttg = triton::gpu;

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

  /// Iterator values
  Value pipelineIterIdx;
  Value loopIterIdx;
  Value nextIV;

  /// Yield values
  SmallVector<Value> nextBuffers;
  SmallVector<Value> extractSlices;
  SmallVector<Value> yieldValues;

  /// The number of stages in the pipeline.
  /// Stages in the range of [0, numStages-1) are in the prologue.
  /// numStages-1 is appended after the loop body.
  int numStages;

  /// Arg indicies
  size_t bufferIdx, loadIdx, depArgsBeginIdx, ivIndex;
  DenseMap<BlockArgument, size_t> depArgsIdx;

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
  Value getLoadMask(triton::LoadOp loadOp, Value mappedMask, Value loopCond,
                    OpBuilder &builder);

  /// Return an empty buffer of size <numStages, ...>
  ttg::AllocTensorOp allocateEmptyBuffer(triton::LoadOp loadOp,
                                         OpBuilder &builder);

  /// Collect all args of the new loop
  SmallVector<Value> collectNewLoopArgs();

  /// Clone the forOp and return the new forOp
  scf::ForOp cloneForOp(ArrayRef<Value> newLoopArgs, OpBuilder &builder);

  /// Prefetch the next iteration for `newForOp`
  void prefetchNextIteration(scf::ForOp newForOp, OpBuilder &builder);

  /// Assemble `newForOp`'s yield op
  void finalizeYield(scf::ForOp newForOp, OpBuilder &builder);

public:
  LoopPipeliner(scf::ForOp forOp, int numStages)
      : forOp(forOp), numStages(numStages) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

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
    if (auto loadOp = dyn_cast<triton::LoadOp>(&op)) {
      auto ptr = loadOp.getPtr();
      unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);

      if (auto mask = loadOp.getMask())
        vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

      auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
      if (!tensorTy || tensorTy.getRank() < 2)
        continue;
      auto ty = tensorTy.getElementType()
                    .cast<triton::PointerType>()
                    .getPointeeType();
      unsigned width = vec * ty.getIntOrFloatBitWidth();
      // We do not pipeline all loads for the following reasons:
      // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8 and 16.
      // 2. It's likely that pipling small loads won't offer much performance
      //    improvement and may even hurt performance by increasing register
      //    pressure.
      if (width >= 32)
        ops.insert(loadOp);
    }

  if (ops.empty())
    return failure();
  else
    return success();
}

void LoopPipeliner::collectValueDep(Value v, int stage,
                                    SetVector<Value> &deps) {
  // Loop-invariant value, skip
  if (v.getParentRegion() != &forOp.getLoopBody())
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
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      // Don't pipeline valid loads that depend on other valid loads
      // (Because if a valid load depends on another valid load, this load needs
      // to wait on the other load in the prologue, which is against the point
      // of the pipeline pass)
      bool isCandidate = true;
      for (Operation *other : ops)
        if (isa<triton::LoadOp>(other))
          if (opDeps[op].contains(other->getResult(0))) {
            isCandidate = false;
            break;
          }
      // We only pipeline loads that have one covert_layout (to dot_op) use
      // TODO: lift this constraint in the future
      if (isCandidate && loadOp.getResult().hasOneUse()) {
        isCandidate = false;
        Operation *use = *loadOp.getResult().getUsers().begin();

        // Advance to the first conversion as long as the use resides in shared
        // memory and it has a single use itself
        while (use) {
          if (use->getNumResults() != 1 || !use->getResult(0).hasOneUse())
            break;
          auto tensorType =
              use->getResult(0).getType().dyn_cast<RankedTensorType>();
          if (!tensorType.getEncoding().isa<ttg::SharedEncodingAttr>())
            break;
          use = *use->getResult(0).getUsers().begin();
        }

        if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(use))
          if (auto tensorType = convertLayout.getResult()
                                    .getType()
                                    .dyn_cast<RankedTensorType>())
            if (auto dotOpEnc = tensorType.getEncoding()
                                    .dyn_cast<ttg::DotOperandEncodingAttr>()) {
              isCandidate = true;
              loadsMapping[loadOp] = convertLayout;
            }
      } else
        isCandidate = false;

      if (!isCandidate)
        invalidOps.insert(loadOp);
      else
        validLoads.insert(loadOp);
    }
  }

  for (Operation *op : invalidOps)
    ops.remove(op);

  if (ops.empty())
    return failure();
  else
    return success();
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

  // XXX: We could remove the following constraints if we can rematerialize in
  // the loop.
  // Check if immediateDepArgs and nonImmediateDepArgs are disjoint.
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
    auto dotOpEnc = cvt.getType()
                        .cast<RankedTensorType>()
                        .getEncoding()
                        .cast<ttg::DotOperandEncodingAttr>();
    auto ty = loadOp.getType().cast<RankedTensorType>();
    SmallVector<int64_t> bufferShape(ty.getShape().begin(),
                                     ty.getShape().end());
    bufferShape.insert(bufferShape.begin(), numStages);
    unsigned bitWidth = ty.getElementType().getIntOrFloatBitWidth();
    auto sharedEnc =
        ttg::SharedEncodingAttr::get(ty.getContext(), dotOpEnc, ty.getShape(),
                                     ttg::getOrder(ty.getEncoding()), bitWidth);
    loadsBufferType[loadOp] =
        RankedTensorType::get(bufferShape, ty.getElementType(), sharedEnc);
  }
}

void LoopPipeliner::createOrderedDeps() {
  for (Operation &op : forOp.getLoopBody().front()) {
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

ttg::AllocTensorOp LoopPipeliner::allocateEmptyBuffer(triton::LoadOp loadOp,
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

  checkOpDeps(ops);

  createBufferTypes();

  createOrderedDeps();

  return success();
}

Value LoopPipeliner::getLoadMask(triton::LoadOp loadOp, Value mappedMask,
                                 Value loopCond, OpBuilder &builder) {
  Type maskType = triton::getI1SameShape(loadOp.getType());
  Value mask = loadOp.getMask();
  Value newMask;
  if (mask) {
    Value cond = loopCond;
    if (isa<RankedTensorType>(maskType)) {
      cond = builder.create<triton::SplatOp>(mask.getLoc(), maskType, loopCond);
    }
    newMask = builder.create<arith::AndIOp>(mask.getLoc(), mappedMask, cond);
  } else {
    if (isa<RankedTensorType>(maskType)) {
      newMask = builder.create<triton::SplatOp>(loopCond.getLoc(), maskType,
                                                loopCond);
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

  // Emit prologue from [0, numStage-1)
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
        auto load = cast<triton::LoadOp>(op);
        // Allocate empty buffer
        if (stage == 0) {
          loadsBuffer[load] = allocateEmptyBuffer(load, builder);
          loadStageBuffer[load] = {loadsBuffer[load]};
        }
        // load => copy async
        if (auto loadOp = llvm::dyn_cast<triton::LoadOp>(op)) {
          Value newMask =
              getLoadMask(loadOp, lookupOrDefault(loadOp.getMask(), stage),
                          loopCond, builder);
          newOp = builder.create<ttg::InsertSliceAsyncOp>(
              op->getLoc(), loadsBuffer[loadOp].getType(),
              lookupOrDefault(loadOp.getPtr(), stage),
              loadStageBuffer[loadOp][stage], pipelineIterIdx, newMask,
              lookupOrDefault(loadOp.getOther(), stage), loadOp.getCache(),
              loadOp.getEvict(), loadOp.getIsVolatile(), /*axis*/ 0);
          builder.create<ttg::AsyncCommitGroupOp>(op->getLoc());
          loadStageBuffer[loadOp].push_back(newOp->getResult(0));
        } else
          llvm_unreachable("This should be LoadOp");
      } else {
        if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
          Value newMask =
              getLoadMask(loadOp, lookupOrDefault(loadOp.getMask(), stage),
                          loopCond, builder);
          newOp = builder.create<triton::LoadOp>(
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
    // Some values have not been used by any ops in the loop body
    for (BlockArgument arg : forOp.getRegionIterArgs())
      setValueMappingYield(arg, valueMapping[arg][stage], stage + 1);
  } // for (int stage = 0; stage < numStages - 1; ++stage)

  // async.wait & extract_slice
  builder.create<ttg::AsyncWaitOp>(validLoads.front().getLoc(),
                                   validLoads.size() * (numStages - 2));
  loopIterIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
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
  // Bump up loopIterIdx, this is used for getting the correct slice for the
  // `next` iteration
  loopIterIdx = builder.create<arith::AddIOp>(
      loopIterIdx.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(loopIterIdx.getLoc(), 1, 32));
}

void LoopPipeliner::emitEpilogue() {
  // If there's any outstanding async copies, we need to wait for them.
  OpBuilder builder(forOp);
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointAfter(forOp);
  builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), 0);
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

  // We need this to update operands for yield
  // original block arg => new arg's idx
  SmallVector<Value> newLoopArgs;
  for (auto v : forOp.getIterOperands())
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

  ivIndex = newLoopArgs.size();
  newLoopArgs.push_back(valueMapping[forOp.getInductionVar()][numStages - 2]);
  newLoopArgs.push_back(pipelineIterIdx);
  newLoopArgs.push_back(loopIterIdx);
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

  // Clone the loop body, replace original args with args of the new ForOp.
  // We want to find cvt ops that match the following pattern:
  // %0 = load %ptr
  // %1 (dotOperand) = cvt %0
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto result = op.getResult(0);
      auto cvtDstTy = result.getType().cast<RankedTensorType>();
      if (cvtDstTy.getEncoding().isa<ttg::DotOperandEncodingAttr>()) {
        auto it =
            std::find(validLoads.begin(), validLoads.end(), op.getOperand(0));
        if (it != validLoads.end()) {
          // We replace the use new load use with a convert layout
          auto loadArgIdx = std::distance(validLoads.begin(), it);
          auto cvt = builder.create<ttg::ConvertLayoutOp>(
              result.getLoc(), cvtDstTy,
              newForOp.getRegionIterArgs()[loadIdx + loadArgIdx]);
          mapping.map(result, cvt.getResult());
          continue;
        }
      }
    }
    cloneWithInferType(builder, &op, mapping);
  }

  return newForOp;
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

  // Special handling for iv & loop condition
  Value curIV = newForOp.getRegionIterArgs()[ivIndex];
  nextIV = builder.create<arith::AddIOp>(newForOp.getInductionVar().getLoc(),
                                         curIV, newForOp.getStep());
  Value nextLoopCond =
      builder.create<arith::CmpIOp>(nextIV.getLoc(), arith::CmpIPredicate::slt,
                                    nextIV, newForOp.getUpperBound());

  pipelineIterIdx = newForOp.getRegionIterArgs()[ivIndex + 1];
  Value insertSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), pipelineIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages, 32));
  loopIterIdx = newForOp.getRegionIterArgs()[ivIndex + 2];
  Value extractSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages, 32));

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
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        auto newMask =
            getLoadMask(loadOp, curMapping.lookupOrDefault(loadOp.getMask()),
                        nextLoopCond, builder);
        nextOp = builder.create<triton::LoadOp>(
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
      auto loadOp = llvm::cast<triton::LoadOp>(op);
      auto mask = loadOp.getMask();
      auto newMask =
          getLoadMask(loadOp, nextMapping.lookupOrDefault(loadOp.getMask()),
                      nextLoopCond, builder);
      if (mask) {
        // If mask is defined outside the loop, don't update the map more than
        // once
        if (!(forOp.isDefinedOutsideOfLoop(mask) && nextMapping.contains(mask)))
          nextMapping.map(loadOp.getMask(), newMask);
        newMask = nextMapping.lookupOrDefault(mask);
      }
      Value insertAsyncOp = builder.create<ttg::InsertSliceAsyncOp>(
          op->getLoc(), loadsBuffer[loadOp].getType(),
          nextMapping.lookupOrDefault(loadOp.getPtr()),
          newForOp.getRegionIterArgs()[bufferIdx + nextBuffers.size()],
          insertSliceIndex, newMask,
          nextMapping.lookupOrDefault(loadOp.getOther()), loadOp.getCache(),
          loadOp.getEvict(), loadOp.getIsVolatile(), /*axis*/ 0);
      builder.create<ttg::AsyncCommitGroupOp>(op->getLoc());
      nextBuffers.push_back(insertAsyncOp);
      // Extract slice
      auto bufferType = insertAsyncOp.getType().cast<RankedTensorType>();
      auto bufferShape = bufferType.getShape();
      auto sliceType = loadsMapping[loadOp].getType().cast<RankedTensorType>();
      sliceType = RankedTensorType::get({bufferShape[1], bufferShape[2]},
                                        sliceType.getElementType(),
                                        loadsBufferType[loadOp].getEncoding());

      nextOp = builder.create<ttg::ExtractSliceOp>(
          op->getLoc(), sliceType, insertAsyncOp,
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
  Operation *asyncWait = builder.create<ttg::AsyncWaitOp>(
      validLoads[0].getLoc(), validLoads.size() * (numStages - 2));
  for (auto it = extractSlices.rbegin(); it != extractSlices.rend(); ++it) {
    // move extract_slice after asyncWait
    it->getDefiningOp()->moveAfter(asyncWait);
  }

  // Bump iteration count
  pipelineIterIdx = builder.create<arith::AddIOp>(
      nextIV.getLoc(), pipelineIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), 1, 32));
  loopIterIdx = builder.create<arith::AddIOp>(
      nextIV.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), 1, 32));
}

void LoopPipeliner::finalizeYield(scf::ForOp newForOp, OpBuilder &builder) {
  SmallVector<Value> yieldValues;
  for (Value v : yieldOp->getOperands())
    yieldValues.push_back(mapping.lookup(v));
  for (Value nextBuffer : nextBuffers)
    yieldValues.push_back(nextBuffer);
  for (Value nextSlice : extractSlices)
    yieldValues.push_back(nextSlice);

  for (size_t i = depArgsBeginIdx; i < ivIndex; ++i) {
    auto arg = newForOp.getRegionIterArgs()[i];
    assert(depArgsMapping.count(arg) && "Missing loop-carried value");
    yieldValues.push_back(depArgsMapping[arg]);
  }
  yieldValues.push_back(nextIV);
  yieldValues.push_back(pipelineIterIdx);
  yieldValues.push_back(loopIterIdx);

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
  PipelinePass(int numStages) { this->numStages = numStages; }

  void runOnOperation() override {
    int numStages = this->numStages;

    if (numStages <= 1)
      return;

    // Pre-processing
    // we make sure element-wise ops are done *after* the conversion
    // to dot operands
    // we can achieve this with simple recursive pattern matching
    // MLIRContext *context = &getContext();
    // mlir::RewritePatternSet patterns(context);
    // patterns.add<MoveOpAfterLayoutConversion>(context);
    // auto didPreprocess =
    //     applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    // Do the pipelining
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      LoopPipeliner pipeliner(forOp, numStages);

      if (pipeliner.initialize().failed())
        return;

      pipeliner.emitPrologue();
      scf::ForOp newForOp = pipeliner.createNewForOp();
      pipeliner.emitEpilogue();

      // Replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
    });
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUPipelinePass(int numStages) {
  return std::make_unique<PipelinePass>(numStages);
}
