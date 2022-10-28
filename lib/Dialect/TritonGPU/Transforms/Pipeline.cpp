#include "mlir/IR/BlockAndValueMapping.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
//
// This file implements loop software pipelining
// The implementation here is inspired by the pipeline pass in Triton (-v2.0)
// and SCF's LoopPipelining.
//
//===----------------------------------------------------------------------===//

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {
class LoopPipeliner {
  /// cache forOp we are working on
  scf::ForOp forOp;

  /// cache YieldOp for this forOp
  scf::YieldOp yieldOp;

  /// loads to be pipelined
  SetVector<Value> loads;
  /// the value that each load will be mapped to (after layout conversion)
  DenseMap<Value, Value> loadsMapping;
  /// load => buffer
  DenseMap<Value, Value> loadsBuffer;
  /// load => buffer at stage N
  DenseMap<Value, SmallVector<Value>> loadStageBuffer;
  /// load => after extract
  DenseMap<Value, Value> loadsExtract;
  ///
  Value pipelineIterIdx;
  ///
  Value loopIterIdx;

  /// comments on numStages:
  ///   [0, numStages-1) are in the prologue
  ///   numStages-1 is appended after the loop body
  int numStages;

  /// value (in loop) => value at stage N
  DenseMap<Value, SmallVector<Value>> valueMapping;

  /// Block arguments that loads depend on
  DenseSet<BlockArgument> depArgs;
  /// Operations (inside the loop body) that loads depend on
  DenseSet<Operation *> depOps;

  /// collect values that v depends on and are defined inside the loop
  void collectDeps(Value v, int stages, DenseSet<Value> &deps);

  void setValueMapping(Value origin, Value newValue, int stage);

  Value lookupOrDefault(Value origin, int stage);

  /// returns a empty buffer of size <numStages, ...>
  triton::gpu::AllocTensorOp allocateEmptyBuffer(Operation *op,
                                                 OpBuilder &builder);

public:
  LoopPipeliner(scf::ForOp forOp, int numStages)
      : forOp(forOp), numStages(numStages) {
    // cache yieldOp
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  /// Collect loads to pipeline. Return success if we can pipeline this loop
  LogicalResult initialize();

  /// emit pipelined loads (before loop body)
  void emitPrologue();

  /// emit pipelined loads (after loop body)
  void emitEpilogue();

  /// create the new ForOp (add new args & insert prefetched ops)
  scf::ForOp createNewForOp();

  friend struct PipelinePass;
};

// helpers
void LoopPipeliner::setValueMapping(Value origin, Value newValue, int stage) {
  if (valueMapping.find(origin) == valueMapping.end())
    valueMapping[origin] = SmallVector<Value>(numStages);
  valueMapping[origin][stage] = newValue;
}

Value LoopPipeliner::lookupOrDefault(Value origin, int stage) {
  if (valueMapping.find(origin) == valueMapping.end())
    return origin;
  return valueMapping[origin][stage];
}

void LoopPipeliner::collectDeps(Value v, int stages, DenseSet<Value> &deps) {
  // Loop-invarant value. skip
  if (v.getParentRegion() != &forOp.getLoopBody())
    return;

  // Since we only need to peel the loop numStages-1 times, don't worry about
  // depends that are too far away
  if (stages < 0)
    return;

  if (auto arg = v.dyn_cast<BlockArgument>()) {
    deps.insert(v);
    // Note: we have iv as the first arg, so the op idx is arg.getArgNumber()-1
    collectDeps(yieldOp->getOperand(arg.getArgNumber() - 1), stages - 1, deps);
  } else { // value
    // v might be in deps, but we still need to visit v.
    // This is because v might depends on value in previous iterations
    deps.insert(v);
    for (Value op : v.getDefiningOp()->getOperands())
      collectDeps(op, stages, deps);
  }
}

triton::gpu::AllocTensorOp
LoopPipeliner::allocateEmptyBuffer(Operation *op, OpBuilder &builder) {
  // allocate a buffer for each pipelined tensor
  // shape: e.g. (numStages==4), <32x64xbf16> -> <4x32x64xbf16>
  Value convertLayout = loadsMapping[op->getResult(0)];
  if (auto tensorType = convertLayout.getType().dyn_cast<RankedTensorType>()) {
    SmallVector<int64_t> shape(tensorType.getShape().begin(),
                               tensorType.getShape().end());
    shape.insert(shape.begin(), numStages);
    Type elementType = tensorType.getElementType();
    // The encoding of the buffer is similar to the original tensor
    Attribute encoding = tensorType.getEncoding();
    auto bufferType = RankedTensorType::get(shape, elementType, encoding);
    return builder.create<triton::gpu::AllocTensorOp>(convertLayout.getLoc(),
                                                      bufferType);
  }
  llvm_unreachable("Async copy's return should be of RankedTensorType");
}

/// A load instruction can be pipelined if:
///   - the load doesn't depend on any other loads (after loop peeling)
///   - (?) this load is not a loop-invariant value (we should run LICM before
///                                                  this pass?)
LogicalResult LoopPipeliner::initialize() {
  Block *loop = forOp.getBody();

  // can we use forOp.walk(...) here?
  SmallVector<triton::LoadOp, 2> allLoads;
  for (Operation &op : *loop)
    if (auto loadOp = dyn_cast<triton::LoadOp>(&op))
      allLoads.push_back(loadOp);

  // Early stop: no need to continue if there is no load in the loop.
  if (allLoads.empty())
    return failure();

  // load => values that it depends on
  DenseMap<Value, DenseSet<Value>> loadDeps;
  for (triton::LoadOp loadOp : allLoads) {
    DenseSet<Value> deps;
    for (Value op : loadOp->getOperands())
      collectDeps(op, numStages - 1, deps);
    loadDeps[loadOp] = deps;
  }

  // Don't pipeline loads that depend on other loads
  // (Because if a load depends on another load, this load needs to wait on the
  //  other load in the prologue, which is against the point of the pipeline
  //  pass)
  for (triton::LoadOp loadOp : allLoads) {
    bool isCandiate = true;
    for (triton::LoadOp other : allLoads) {
      if (loadDeps[loadOp].contains(other)) {
        isCandiate = false;
        break;
      }
    }

    // For now, we only pipeline loads that have one covert_layout (to smem) use
    // TODO: lift this constraint in the future
    if (isCandiate && loadOp.getResult().hasOneUse()) {
      isCandiate = false;
      Operation *use = *loadOp.getResult().getUsers().begin();
      if (auto convertLayout =
              llvm::dyn_cast<triton::gpu::ConvertLayoutOp>(use)) {
        if (auto tensorType = convertLayout.getResult()
                                  .getType()
                                  .dyn_cast<RankedTensorType>()) {
          if (tensorType.getEncoding().isa<triton::gpu::SharedEncodingAttr>()) {
            isCandiate = true;
            loadsMapping[loadOp] = convertLayout;
          }
        }
      }
    } else
      isCandiate = false;

    if (isCandiate)
      loads.insert(loadOp);
  }

  // we have some loads to pipeline
  if (!loads.empty()) {
    // update depArgs & depOps
    for (Value loadOp : loads) {
      for (Value dep : loadDeps[loadOp]) {
        // TODO: we should record the stage that the value is depended on
        if (auto arg = dep.dyn_cast<BlockArgument>())
          depArgs.insert(arg);
        else
          depOps.insert(dep.getDefiningOp());
      }
    }
    return success();
  }

  return failure();
}

void LoopPipeliner::emitPrologue() {
  // llvm::errs() << "loads to pipeline...:\n";
  // for (Value load : loads)
  //   llvm::errs() << load << "\n";

  OpBuilder builder(forOp);
  for (BlockArgument &arg : forOp.getRegionIterArgs()) {
    OpOperand &operand = forOp.getOpOperandForRegionIterArg(arg);
    setValueMapping(arg, operand.get(), 0);
  }

  // prologue from [0, numStage-1)
  Value iv = forOp.getLowerBound();
  pipelineIterIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
  for (int stage = 0; stage < numStages - 1; ++stage) {
    // special handling for induction variable as the increment is implicit
    if (stage != 0)
      iv = builder.create<arith::AddIOp>(iv.getLoc(), iv, forOp.getStep());
    setValueMapping(forOp.getInductionVar(), iv, stage);

    // special handling for loop condition as there is no condition in ForOp
    Value loopCond = builder.create<arith::CmpIOp>(
        iv.getLoc(), arith::CmpIPredicate::slt, iv, forOp.getUpperBound());

    // rematerialize peeled values
    SmallVector<Operation *> orderedDeps;
    for (Operation &op : forOp.getLoopBody().front()) {
      if (depOps.contains(&op))
        orderedDeps.push_back(&op);
      else if (loads.contains(op.getResult(0)))
        orderedDeps.push_back(&op);
    }
    assert(depOps.size() + loads.size() == orderedDeps.size() &&
           "depOps contains invalid values");
    for (Operation *op : orderedDeps) {
      Operation *newOp = nullptr;
      if (loads.contains(op->getResult(0))) {
        // Allocate empty buffer
        if (stage == 0) {
          loadsBuffer[op->getResult(0)] = allocateEmptyBuffer(op, builder);
          loadStageBuffer[op->getResult(0)] = {loadsBuffer[op->getResult(0)]};
        }
        // load => copy async
        // TODO: check if the hardware supports async copy
        if (auto loadOp = llvm::dyn_cast<triton::LoadOp>(op)) {
          newOp = builder.create<triton::gpu::InsertSliceAsyncOp>(
              op->getLoc(), loadsBuffer[loadOp].getType(),
              lookupOrDefault(loadOp.ptr(), stage),
              loadStageBuffer[loadOp][stage], pipelineIterIdx,
              lookupOrDefault(loadOp.mask(), stage),
              lookupOrDefault(loadOp.other(), stage), loadOp.cache(),
              loadOp.evict(), loadOp.isVolatile(), /*axis*/ 0);
          loadStageBuffer[loadOp].push_back(newOp->getResult(0));
        } else
          llvm_unreachable("This should be LoadOp");
      } else {
        newOp = builder.clone(*op);
        // Update loop-carried uses
        for (unsigned opIdx = 0; opIdx < op->getNumOperands(); ++opIdx) {
          auto it = valueMapping.find(op->getOperand(opIdx));
          if (it != valueMapping.end()) {
            Value v = it->second[stage];
            assert(v);
            newOp->setOperand(opIdx, v);
          } // else, op at opIdx is a loop-invariant value
        }
      }

      // If this is a load/async_copy, we need to update the mask
      if (Value mask = [&]() {
            if (auto loadOp = llvm::dyn_cast<triton::LoadOp>(newOp)) {
              return loadOp.mask();
            } else if (auto insertSliceAsyncOp =
                           llvm::dyn_cast<triton::gpu::InsertSliceAsyncOp>(
                               newOp)) {
              return insertSliceAsyncOp.mask();
            } else {
              return mlir::Value();
            }
          }()) {
        // assert(I1 or TensorOf<[I1]>);
        OpBuilder::InsertionGuard g(builder);
        // TODO: move this out of the loop
        builder.setInsertionPoint(newOp);
        Value splatCond = builder.create<triton::SplatOp>(
            mask.getLoc(), mask.getType(), loopCond);
        Value newMask =
            builder.create<arith::AndIOp>(mask.getLoc(), mask, splatCond);
        // TODO: better way to do this?
        if (llvm::isa<triton::LoadOp>(newOp))
          newOp->setOperand(1, newMask);
        else // InsertSliceAsyncOp
          newOp->setOperand(3, newMask);
      }

      // update mapping of results
      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults())) {
        Value originalResult = op->getResult(dstIdx);
        // copy_async will update the value of its only use
        // TODO: load should no be used in the preheader?
        if (loads.contains(originalResult)) {
          break;
          // originalResult = loadsMapping[originalResult];
        }
        setValueMapping(originalResult, newOp->getResult(dstIdx), stage);
        // update mapping for loop-carried values (args)
        for (OpOperand &operand : yieldOp->getOpOperands()) {
          if (operand.get() == op->getResult(dstIdx))
            setValueMapping(
                forOp.getRegionIterArgs()[operand.getOperandNumber()],
                newOp->getResult(dstIdx), stage + 1);
        }
      }
    }

    pipelineIterIdx = builder.create<arith::AddIOp>(
        iv.getLoc(), pipelineIterIdx,
        builder.create<arith::ConstantIntOp>(iv.getLoc(), 1, 32));
  } // for (int stage = 0; stage < numStages - 1; ++stage)

  // async.wait & extract_slice
  builder.create<triton::gpu::AsyncWaitOp>(loads[0].getLoc(),
                                           loads.size() * (numStages - 2));
  loopIterIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
  for (Value loadOp : loads) {
    Value extractSlice = builder.create<triton::gpu::ExtractSliceOp>(
        loadOp.getLoc(), loadsMapping[loadOp].getType(),
        loadStageBuffer[loadOp][numStages - 1], loopIterIdx, /*axis*/ 0);
    loadsExtract[loadOp] = extractSlice;
  }
  // bump up loopIterIdx, this is used for getting the correct slice for the
  // *next* iteration
  loopIterIdx = builder.create<arith::AddIOp>(
      loopIterIdx.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(loopIterIdx.getLoc(), 1, 32));
}

void LoopPipeliner::emitEpilogue() {
  // If there's any outstanding async copies, we need to wait for them.
  // TODO(Keren): We may want to completely avoid the async copies in the last
  // few iterations by setting is_masked attribute to true. We don't want to use
  // the mask operand because it's a tensor but not a scalar.
  OpBuilder builder(forOp);
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointAfter(forOp);
  builder.create<triton::gpu::AsyncWaitOp>(forOp.getLoc(), 0);
}

scf::ForOp LoopPipeliner::createNewForOp() {
  OpBuilder builder(forOp);

  // order of new args:
  //   (original args),
  //   (insertSliceAsync buffer at stage numStages - 1)  for each load
  //   (extracted tensor)  for each load
  //   (depArgs at stage numStages-1)
  //   (iv at stage numStages-1)
  //   (pipeline iteration index)
  //   (loop iteration index)
  SmallVector<Value> newLoopArgs;
  // We need this to update operands for yield
  // original block arg => new arg's idx
  DenseMap<BlockArgument, size_t> depArgsIdx;
  for (auto v : forOp.getIterOperands())
    newLoopArgs.push_back(v);

  size_t bufferIdx = newLoopArgs.size();
  for (Value loadOp : loads)
    newLoopArgs.push_back(loadStageBuffer[loadOp].back());
  size_t loadIdx = newLoopArgs.size();
  for (Value loadOp : loads)
    newLoopArgs.push_back(loadsExtract[loadOp]);

  size_t depArgsBeginIdx = newLoopArgs.size();
  for (BlockArgument depArg : depArgs) {
    depArgsIdx[depArg] = newLoopArgs.size();
    newLoopArgs.push_back(valueMapping[depArg][numStages - 1]);
  }

  size_t nextIVIdx = newLoopArgs.size();
  newLoopArgs.push_back(valueMapping[forOp.getInductionVar()][numStages - 2]);
  newLoopArgs.push_back(pipelineIterIdx);
  newLoopArgs.push_back(loopIterIdx);

  for (size_t i = 0; i < newLoopArgs.size(); ++i)
    assert(newLoopArgs[i]);

  // 1. signature of the new ForOp
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newLoopArgs);

  // 2. body of the new ForOp
  builder.setInsertionPointToStart(newForOp.getBody());
  BlockAndValueMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);

  // 2.1 clone the loop body, replace original args with args of the new ForOp
  // Insert async wait if necessary.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    Operation *newOp = builder.clone(op, mapping);
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // 3. replace loads with block args (from prologue)
  for (size_t idx = 0; idx < loads.size(); ++idx) {
    Value load = loads[idx];
    assert(load.hasOneUse() &&
           "we assume that this load has one use (ConvertLayout)");
    Value loadUse = load.getUsers().begin()->getResult(0);
    mapping.lookup(loadUse).replaceAllUsesWith(
        newForOp.getRegionIterArgs()[loadIdx + idx]);
    // delete old load and layout conversion
    mapping.lookup(loadUse).getDefiningOp()->erase();
    mapping.lookup(load).getDefiningOp()->erase();
  }

  // 4. prefetch the next iteration
  SmallVector<Operation *> orderedDeps;
  for (Operation &op : forOp.getLoopBody().front()) {
    if (depOps.contains(&op))
      orderedDeps.push_back(&op);
    else if (loads.contains(op.getResult(0)))
      orderedDeps.push_back(&op);
  }
  assert(depOps.size() + loads.size() == orderedDeps.size() &&
         "depOps contains invalid values");
  BlockAndValueMapping nextMapping;
  DenseMap<BlockArgument, Value> depArgsMapping;
  size_t argIdx = 0;
  for (BlockArgument arg : depArgs) {
    nextMapping.map(arg,
                    newForOp.getRegionIterArgs()[argIdx + depArgsBeginIdx]);
    ++argIdx;
  }
  // special handling for iv & loop condition
  Value nextIV = builder.create<arith::AddIOp>(
      newForOp.getInductionVar().getLoc(),
      newForOp.getRegionIterArgs()[nextIVIdx], newForOp.getStep());
  Value nextLoopCond =
      builder.create<arith::CmpIOp>(nextIV.getLoc(), arith::CmpIPredicate::slt,
                                    nextIV, newForOp.getUpperBound());

  // slice index
  SmallVector<Value> nextBuffers;
  SmallVector<Value> extractSlices;

  pipelineIterIdx = newForOp.getRegionIterArgs()[nextIVIdx + 1];
  Value insertSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), pipelineIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages, 32));
  loopIterIdx = newForOp.getRegionIterArgs()[nextIVIdx + 2];
  Value extractSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages, 32));

  for (Operation *op : orderedDeps) {
    Operation *nextOp = nullptr;
    // TODO(da): does this work if loadOp has no mask?
    // update loading mask
    if (loads.contains(op->getResult(0))) {
      auto loadOp = llvm::cast<triton::LoadOp>(op);
      Value mask = loadOp.mask();
      if (mask) {
        Value splatCond = builder.create<triton::SplatOp>(
            mask.getLoc(), mask.getType(), nextLoopCond);
        Value newMask = builder.create<arith::AndIOp>(
            mask.getLoc(), splatCond, nextMapping.lookupOrDefault(mask));
        // if mask is defined outside the loop, don't update the map more than
        // once
        if (!(forOp.isDefinedOutsideOfLoop(mask) && nextMapping.contains(mask)))
          nextMapping.map(mask, newMask);
      }
      Value insertAsyncOp = builder.create<triton::gpu::InsertSliceAsyncOp>(
          op->getLoc(), loadsBuffer[loadOp].getType(),
          nextMapping.lookupOrDefault(loadOp.ptr()),
          newForOp.getRegionIterArgs()[bufferIdx + nextBuffers.size()],
          insertSliceIndex, nextMapping.lookupOrDefault(loadOp.mask()),
          nextMapping.lookupOrDefault(loadOp.other()), loadOp.cache(),
          loadOp.evict(), loadOp.isVolatile(), /*axis*/ 0);
      nextBuffers.push_back(insertAsyncOp);
      nextOp = builder.create<triton::gpu::ExtractSliceOp>(
          op->getLoc(), loadsMapping[loadOp].getType(), insertAsyncOp,
          extractSliceIndex, /*axis*/ 0);
      extractSlices.push_back(nextOp->getResult(0));
    } else
      nextOp = builder.clone(*op, nextMapping);
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults())) {
      nextMapping.map(op->getResult(dstIdx), nextOp->getResult(dstIdx));
      // if this is a loop-carried value, update the mapping for yield
      auto originYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (OpOperand &operand : originYield->getOpOperands()) {
        if (operand.get() == op->getResult(dstIdx)) {
          size_t originIdx = operand.getOperandNumber();
          size_t newArgIdx = depArgsIdx[forOp.getRegionIterArgs()[originIdx]];
          BlockArgument newArg = newForOp.getRegionIterArgs()[newArgIdx];
          depArgsMapping[newArg] = nextOp->getResult(dstIdx);
        }
      }
    }
  }

  // async.wait & extract_slice
  Operation *asyncWait = builder.create<triton::gpu::AsyncWaitOp>(
      loads[0].getLoc(), loads.size() * (numStages - 2));
  for (auto it = extractSlices.rbegin(); it != extractSlices.rend(); ++it) {
    // move extract_slice after asyncWait
    it->getDefiningOp()->moveAfter(asyncWait);
  }

  // bump iteration count
  pipelineIterIdx = builder.create<arith::AddIOp>(
      nextIV.getLoc(), pipelineIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), 1, 32));
  loopIterIdx = builder.create<arith::AddIOp>(
      nextIV.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), 1, 32));

  // Finally, the YieldOp, need to sync with the order of newLoopArgs
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookup(v));
  for (Value nextBuffer : nextBuffers)
    yieldValues.push_back(nextBuffer);
  for (Value nextSlice : extractSlices)
    yieldValues.push_back(nextSlice);

  for (size_t i = depArgsBeginIdx; i < nextIVIdx; ++i)
    yieldValues.push_back(
        depArgsMapping.lookup(newForOp.getRegionIterArgs()[i]));
  yieldValues.push_back(nextIV);
  yieldValues.push_back(pipelineIterIdx);
  yieldValues.push_back(loopIterIdx);

  builder.setInsertionPointToEnd(newForOp.getBody());
  builder.create<scf::YieldOp>(forOp.getBody()->getTerminator()->getLoc(),
                               yieldValues);
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

    getOperation()->walk([&](scf::ForOp forOp) -> void {
      LoopPipeliner pipeliner(forOp, numStages);

      if (pipeliner.initialize().failed())
        return;

      pipeliner.emitPrologue();

      scf::ForOp newForOp = pipeliner.createNewForOp();

      pipeliner.emitEpilogue();

      // replace the original loop
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
