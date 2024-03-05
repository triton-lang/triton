#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"

//===----------------------------------------------------------------------===//
// This file implements stream software pipelining for loops. The implementation
// here is inspired by the pipeline pass in Triton and the rocMLIR pipeliner.
//
// We divide the loop body into the following phases:
// a. Pre-load operations: for instance, index computation.
// b. Load operations: loading from global memory to shared memory.
// c. Compute operations: for instance, Triton dot.
// d. Post-load operations: for instance, index computation.
//
// To pipeline the loop, we need to:
// - Find all the dependencies of the load operations.
// - Prologue: Hoist the pipelinable load operations and shared memory store
// for the ramp up stage
// - Pipelined Loop: Assemble the loop body minus last iteration
//   - Prefetch next tile from global into regs (while computing from previous)
//   - Non-load loop body
//   - Store next tile into shared mem
// - Epilogue: Peeled non-load loop body for last iteration
//
//===----------------------------------------------------------------------===//

using llvm::MapVector;
using namespace mlir;
namespace ttg = triton::gpu;

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

class LoopPipeliner {
  /// Cache of ForOp and YieldOp related to this pipeliner.
  scf::ForOp forOp;
  scf::YieldOp yieldOp;

  bool peelLastIter = true;

  /// The new pipelined ForOp.
  scf::ForOp pplForOp;
  
  /// Loads to be pipelined
  SetVector<Operation *> validLoads;
  /// The value that each load will be mapped to (after layout conversion)
  DenseMap<Value, Value> convertMapping;
  /// load => buffer
  DenseMap<Value, Value> loadsBuffer;
  /// load => buffer type (with shared layout after swizzling)
  DenseMap<Value, triton::MemDescType> loadsBufferType;

  /// Iterator values
  Value nextLoopCond;

  /// Yield values
  SmallVector<Value> nextBuffers;
  SmallVector<Value> yieldValues;

  /// The number of stages in the pipeline is fixed to '2' for
  /// analysis since there will be a current buffer stored in
  /// shared mem and a next buffer stored in regs.
  int numStages = 2;

  /// Arg indicies
  size_t bufferIdx, depArgsBeginIdx;
  DenseMap<BlockArgument, size_t> depArgsIdx;

  /// value (in loop) => value at stage N
  DenseMap<Value, SmallVector<Value>> valueMapping;
  /// loop iter arg => value
  DenseMap<BlockArgument, Value> depArgsMapping;

  /// forOp value => pplForOp value
  IRMapping curMapping;
  /// forOp value => prefetch value
  IRMapping nextMapping;
  
  /// Dependency ops by program order
  SmallVector<Operation *> orderedDeps;

  SetVector<Operation*> currentDeps;
  
  /// block arguments that loads depend on
  SetVector<BlockArgument> depArgs;

  /// operation => source operand defined stages
  DenseMap<Operation *, DenseSet<int>> immediateOpStages;

  /// operations that loads depend on
  SetVector<Operation *> depOps;

  /// Collect values that `v` depends on and are defined inside the loop
  void collectValueDep(Value v, int stage, SetVector<Operation*> &deps,
                       SetVector<BlockArgument> &args);

  /// Collect all op dependencies
  void collectDeps(SetVector<Operation *> &ops,
                   MapVector<Operation *, SetVector<Operation*>> &opDeps);

  void collectDepChain(Operation *op, SetVector<Operation*> &ops);
  
  /// Check if none of the for-ops has valid uses
  LogicalResult checkOpUses();

  /// Check if ops have dependencies that are not pipelinable
  LogicalResult checkOpDeps();

  void createBufferTypes();

  void createOrderedDeps();

  void createCurrentDeps();
  
  /// Return the stage at which `v` is defined prior to `stage`
  int getValueDefStage(Value v, int stage);

  /// Map `origin` to `newValue` at `stage`
  void setValueMapping(Value origin, Value newValue, int stage);

  /// Map `origin` to `newValue` at `stage` according to the association between
  /// yieldOp and forOp
  void setValueMappingYield(Value origin, Value newValue, int stage);

  /// Map `origin` to `newValue` at the next stage according to the association
  /// between yieldOp and forOp
  void setValueMappingYield(Value origin, Value newValue);

  /// Return the value mapped to `origin` at `stage`, if it exists.
  Value lookupOrDefault(Value origin, int stage);

  Value getLoadMask(triton::LoadOp loadOp, Value mappedMask,
                    Value loopCond, OpBuilder &builder);
  /// Collect all args of the new loop
  SmallVector<Value> collectNewLoopArgs();

  /// Clone the forOp and return the new forOp
  scf::ForOp cloneForOp(ArrayRef<Value> newLoopArgs, OpBuilder &builder);

  void updateLoadMask(triton::LoadOp loadOp, Value newMask);
  /// Prefetch the next iteration for `pplForOp`
  void prefetchNextBuffer(OpBuilder &builder);
  void cloneCurrentBody(OpBuilder &builder);
  void storeNextBuffer(OpBuilder &builder);

  bool isLoadChain(Operation *op) const;
  
  /// Assemble `pplForOp`'s yield op
  void finalizeYield(OpBuilder &builder);

public:
  LoopPipeliner(scf::ForOp forOp)
      : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  /// Collect loads to pipeline. Return success if we can pipeline this loop
  LogicalResult initialize();

  /// Emit pipelined loads (before loop body)
  void emitPrologue();

  /// emit pipelined loads (after loop body)
  void emitEpilogue(DenseMap<Value, Value> &newResults);

  /// create the new ForOp (add new args & insert prefetched ops)
  scf::ForOp createNewForOp();

  friend struct PipelinePass;
};

void LoopPipeliner::collectValueDep(Value v, int stage,
                                    SetVector<Operation*> &deps,
                                    SetVector<BlockArgument> &args) {
  // Since we only need to peel the loop numStages-1 times, don't worry
  // about depends that are too far away
  if (stage < 0)
    return;

  // Loop-invariant value, skip
  if (v.getParentRegion() != &forOp.getRegion())
    return;

  if (Operation *op = v.getDefiningOp()) {
    if (!deps.contains(op)) {
      deps.insert(op);
      for (Value opr : op->getOperands())
        collectValueDep(opr, stage, deps, args);
    }
  } else if (auto arg = v.dyn_cast<BlockArgument>()) {
    if (arg.getArgNumber() > 0) {
      args.insert(arg);
      collectValueDep(yieldOp->getOperand(arg.getArgNumber() - 1), stage - 1,
                      deps, args);
    }
  }
}

void LoopPipeliner::collectDeps(
    SetVector<Operation *> &ops,
    MapVector<Operation *, SetVector<Operation*>> &valueDeps) {
  for (auto op : ops) {
    for (Value v : op->getOperands()) {
      SetVector<Operation*> deps;
      SetVector<BlockArgument> args;
      collectValueDep(v, numStages - 1, deps, args);
      valueDeps[op] = deps;
    }
  }
}

LogicalResult LoopPipeliner::checkOpUses() {
  SetVector<Operation *> ops;
  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  for (Operation &op : forOp) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(&op))
      ops.insert(&op);
  }

  // Collect all ops' dependencies
  MapVector<Operation *, SetVector<Operation*>> opDeps;
  collectDeps(ops, opDeps);

  for (Operation *op : ops) {
    auto loadOp = dyn_cast<triton::LoadOp>(op);
    // Don't pipeline valid loads that depend on other valid loads
    // (Because if a valid load depends on another valid load, this load needs
    // to wait on the other load in the prologue, which is against the point
    // of the pipeline pass)
    bool isCandidate = true;
    for (Operation *other : ops)
      if (isa<triton::LoadOp>(other))
        if (opDeps[op].contains(other)) {
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
        if (!tensorType || !tensorType.getEncoding().isa<ttg::SharedEncodingAttr>())
          break;
        use = *use->getResult(0).getUsers().begin();
      }

      // TODO: handle fp_to_fp conversions in between
      if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(use))
        if (auto tensorType = convertLayout.getResult()
            .getType()
            .dyn_cast<RankedTensorType>())
          if (auto dotOpEnc = tensorType.getEncoding()
              .dyn_cast<ttg::DotOperandEncodingAttr>()) {
            isCandidate = true;
            convertMapping[loadOp] = convertLayout;
          }
    } else
      isCandidate = false;

    if (isCandidate)
      validLoads.insert(op);
  }

  return validLoads.empty() ? failure() : success();
}

LogicalResult LoopPipeliner::checkOpDeps() {
  /// arg => source operand defined stages
  DenseMap<BlockArgument, DenseSet<int>> immediateArgStages;
  SetVector<BlockArgument> nonImmediateDepArgs;
  SetVector<Operation *> nonImmediateOps;
  for (Operation *op : validLoads) {
    for (Value v : op->getOperands()) {
      SetVector<Operation*> deps;
      SetVector<BlockArgument> args;
      collectValueDep(v, numStages - 1, deps, args);
      int defStage = getValueDefStage(v, numStages - 1);
      if (defStage < 0) {
        // assert(defStage >= 0 &&
        //        "newLoopArgs has null args without a define op. Consider either "
        //        "rewrite the loop to reduce cross iteration dependencies or "
        //        "increase the num_stages value.");
        return failure();
      }
      bool immediate = args.size() > 0;
      for (auto *dep : deps) {
        depOps.insert(dep);
        if (immediate)
          immediateOpStages[dep].insert(defStage);
        else
          nonImmediateOps.insert(dep);
      }
      for (auto arg : args) {
        depArgs.insert(arg);
        if (immediate)
          immediateArgStages[arg].insert(defStage);
        else
          nonImmediateDepArgs.insert(arg);
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
  return success();
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

void LoopPipeliner::setValueMappingYield(Value origin,
                                         Value newValue) {
  for (OpOperand &operand : origin.getUses()) {
    if (operand.getOwner() == yieldOp) {
      auto yieldIdx = operand.getOperandNumber();
      auto depYieldIdx = depArgsIdx[forOp.getRegionIterArgs()[yieldIdx]];
      auto originArg = forOp.getRegionIterArgs()[yieldIdx];
      nextMapping.map(originArg, newValue);
      auto newArg = pplForOp.getRegionIterArgs()[depYieldIdx];
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
  for (auto loadCvt : convertMapping) {
    auto loadOp = loadCvt.first;
    Value cvt = loadCvt.second;
    auto dotOpEnc = cvt.getType()
                        .cast<RankedTensorType>()
                        .getEncoding()
                        .cast<ttg::DotOperandEncodingAttr>();
    auto ty = loadOp.getType().cast<RankedTensorType>();
    SmallVector<int64_t> bufferShape(ty.getShape().begin(),
                                     ty.getShape().end());
    Type eType = ty.getElementType();
    auto blockedEnc = ty.getEncoding().cast<ttg::BlockedEncodingAttr>();
    auto CTALayout = ttg::getCTALayout(ty.getEncoding());
    // unsigned bitWidth = dotOpEnc.getMMAv2kWidth()
    //                         ? 32 / dotOpEnc.getMMAv2kWidth()
    //                         : ty.getElementType().getIntOrFloatBitWidth();
    auto sharedEnc =
        ttg::SharedEncodingAttr::get(ty.getContext(), dotOpEnc, ty.getShape(),
                                     ttg::getOrder(ty.getEncoding()), CTALayout, eType);
    loadsBufferType[loadOp] =
        triton::MemDescType::get(bufferShape, eType, sharedEnc);
  }
}

void LoopPipeliner::createOrderedDeps() {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (depOps.contains(&op))
      orderedDeps.push_back(&op);
    else if (op.getNumResults() > 0 && validLoads.contains(&op))
      orderedDeps.push_back(&op);
  }
  assert(depOps.size() + validLoads.size() == orderedDeps.size() &&
         "depOps contains invalid values");
}

void LoopPipeliner::collectDepChain(Operation *op, SetVector<Operation*> &ops) {
  if (op->getNumResults() == 1 && validLoads.contains(op))
    return;
  if (!ops.contains(op)) {
    ops.insert(op);
    for (Value opr : op->getOperands())
      if (Operation *oprOp = opr.getDefiningOp())
        collectDepChain(oprOp, ops);
  }
}

void LoopPipeliner::createCurrentDeps() {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!llvm::is_contained(orderedDeps, &op))
      collectDepChain(&op, currentDeps);
  }
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

LogicalResult LoopPipeliner::initialize() {
  if (checkOpUses().failed())
    return failure();

  if (checkOpDeps().failed())
    return failure();

  createBufferTypes();

  createOrderedDeps();

  createCurrentDeps();

  return success();
}

Value LoopPipeliner::getLoadMask(triton::LoadOp loadOp, Value mappedMask,
                                 Value loopCond, OpBuilder &builder) {
  if (!peelLastIter) {
    // add mask for last iteration when not peeled to epilogue
    Value mask = loadOp.getMask();
    Type maskType = triton::getI1SameShape(loadOp.getType());
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
  // use original mask when peeling last iteration bc the loop will not do
  // extra loads for the tail of the pipeline
  return mappedMask;
}

bool LoopPipeliner::isLoadChain(Operation *op) const {
  if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
    Value loadVal = cvtOp.getSrc();
    if (auto f2fOp = dyn_cast<triton::FpToFpOp>(op))
      loadVal = f2fOp.getSrc();
    if (validLoads.contains(loadVal.getDefiningOp())) {
      if (cvtOp.getType().getEncoding().isa<ttg::DotOperandEncodingAttr>())
        return true;
    }
  }
  return false;
}

void LoopPipeliner::emitPrologue() {
  /// forOp block args => forOp operands
  /// forOp iterator => lower bound
  IRMapping prologueMap;
  OpBuilder builder(forOp);
  // Get init operands for loop carried values
  for (BlockArgument &arg : forOp.getRegionIterArgs()) {
    OpOperand &operand = *forOp.getTiedLoopInit(arg);
    prologueMap.map(arg, operand.get());
  }

  // Emit prologue
  // Map IV to lower bound
  prologueMap.map(forOp.getInductionVar(), forOp.getLowerBound());

  // Emit Iteration 0 loads, etc
  for (Operation *op : orderedDeps) {
    Operation *newOp = nullptr;
    if (validLoads.contains(op)) {
      auto loadOp = cast<triton::LoadOp>(op);
      // Load from global -> regs
      auto newLoadOp = cloneWithInferType(builder, op, prologueMap);
      Value loadVal = newLoadOp->getResult(0);
      // Convert from regs to shared mem
      newOp = builder.create<ttg::LocalAllocOp>(loadOp.getLoc(),
                                                   loadsBufferType[loadOp],
                                                   loadVal);
      Value cvtVal = newOp->getResult(0);
      prologueMap.map(loadOp->getResult(0), cvtVal);
      loadsBuffer[loadOp] = cvtVal;
    } else {
      newOp = cloneWithInferType(builder, op, prologueMap);
    }
    // Capture loop carried results for pipelined for input
    for (unsigned idx : llvm::seq(unsigned(0), op->getNumResults()))
      setValueMappingYield(op->getResult(idx), newOp->getResult(idx), 1);
  } // for (Operation *op : orderedDeps)
}

void LoopPipeliner::emitEpilogue(DenseMap<Value, Value> &newResults) {
  if (!peelLastIter)
    return;
  OpBuilder builder(pplForOp);
  builder.setInsertionPointAfter(pplForOp);

  IRMapping epilogueMap;
  // Map 'for' iteration args to pipelined-for results
  auto args = forOp.getRegionIterArgs();
  for (uint32_t i = 0; i < args.size(); ++i)
    epilogueMap.map(args[i], pplForOp.getResult(i));
  for (auto load : llvm::enumerate(validLoads))
    epilogueMap.map(load.value()->getResult(0), pplForOp.getResult(bufferIdx + load.index()));
  // Map IV to original upper bound (ie. last iteration)
  epilogueMap.map(forOp.getInductionVar(), forOp.getUpperBound());

  const auto &yieldOprs = yieldOp.getOperands();
  // Clone the loop body after the new ForOp
  // , replace original args with results of the new ForOp.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (currentDeps.contains(&op)) {
      Operation *newOp = nullptr;
      if (isLoadChain(&op)) {
        if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(&op)) {
          Value mappedValue = epilogueMap.lookup(cvt.getSrc());
          if (isa<triton::MemDescType>(mappedValue.getType())) {
            auto newCvt = builder.create<triton::gpu::LocalLoadOp>(
                cvt.getLoc(), cvt.getType(), mappedValue);
            epilogueMap.map(cvt.getResult(), newCvt);
            newOp = newCvt;
          }
        }
        if (!newOp)
          newOp = builder.clone(op, epilogueMap);
      } else {
        newOp = cloneWithInferType(builder, &op, epilogueMap);
      }
      // substitute for these results for the results of the new for loop
      for (const auto &pair : llvm::zip(op.getResults(), newOp->getResults())) {
        auto val = std::get<0>(pair);
        auto it = llvm::find(yieldOprs, val);
        if (it != yieldOprs.end()) {
          uint32_t idx = std::distance(yieldOprs.begin(), it);
          newResults[forOp->getResult(idx)] = std::get<1>(pair);
        }
      }
    }
  }
}

SmallVector<Value> LoopPipeliner::collectNewLoopArgs() {
  // Order of new args:
  //   (original args)
  //   (shared mem buffers for each load)
  //   (depArgs at stage numStages - 1)

  // We need this to update operands for yield
  // original block arg => new arg's idx
  SmallVector<Value> newLoopArgs;
  for (auto v : forOp.getInitArgs()) {
    newLoopArgs.push_back(lookupOrDefault(v, numStages - 1));/*1*/
  }

  // Shared mem locations from iteration 0
  bufferIdx = newLoopArgs.size();
  for (auto *loadOp : validLoads)
    newLoopArgs.push_back(loadsBuffer[loadOp->getResult(0)]);

  // Loop carried vals
  depArgsBeginIdx = newLoopArgs.size();
  for (auto depArg : depArgs) {
    depArgsIdx[depArg] = newLoopArgs.size();
    newLoopArgs.push_back(valueMapping[depArg][numStages - 1]);/*1*/
  }

  return newLoopArgs;
}

scf::ForOp LoopPipeliner::cloneForOp(ArrayRef<Value> newLoopArgs,
                                     OpBuilder &builder) {
  auto loc = forOp.getLoc();
  // Peel off the last iteration
  auto pplUpperBound = forOp.getUpperBound();
  if (peelLastIter)
    pplUpperBound = builder.create<arith::SubIOp>(loc, pplUpperBound,
                                                  forOp.getStep());

  // Clone the original ForOp
  pplForOp = builder.create<scf::ForOp>(
      loc, forOp.getLowerBound(), pplUpperBound,
      forOp.getStep(), newLoopArgs);

  // Set mapping on body of the new ForOp
  builder.setInsertionPointToStart(pplForOp.getBody());
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    curMapping.map(arg.value(), pplForOp.getRegionIterArgs()[arg.index()]);
  uint32_t bufIdx = bufferIdx;
  for (auto *loadOp : validLoads)
    curMapping.map(loadOp->getResult(0), pplForOp.getRegionIterArgs()[bufIdx++]);
  curMapping.map(forOp.getInductionVar(), pplForOp.getInductionVar());

  nextMapping = curMapping;
  // Map the dep args of the next iteration to the dep args of the current
  auto iterArgs = pplForOp.getRegionIterArgs();
  size_t argIdx = 0;
  for (auto depArg : depArgs) {
    BlockArgument nextArg = iterArgs[argIdx + depArgsBeginIdx];
    nextMapping.map(depArg, nextArg);
    ++argIdx;
  }

  // Compute next IV for pre-loads
  Value iv = pplForOp.getInductionVar();
  curMapping.map(forOp.getInductionVar(), iv);
  Value nextIV = builder.create<arith::AddIOp>(iv.getLoc(), iv, pplForOp.getStep());
  nextMapping.map(forOp.getInductionVar(), nextIV);
  nextLoopCond =
      builder.create<arith::CmpIOp>(nextIV.getLoc(), arith::CmpIPredicate::slt,
                                    nextIV, pplForOp.getUpperBound());
  
  return pplForOp;
}

void LoopPipeliner::updateLoadMask(triton::LoadOp loadOp, Value newMask) {
  if (newMask) {
    if (loadOp->getNumOperands() > 1)
      loadOp->setOperand(1, newMask);
    else {
      auto mask = loadOp.getMaskMutable();
      mask.assign(newMask);
    }
  }
}

void LoopPipeliner::prefetchNextBuffer(OpBuilder &builder) {
  // Emit prefetch loads of next buffer before compute of current buffer
  for (Operation *op : orderedDeps) {
    Operation *nextOp = nullptr;
    if (validLoads.contains(op)) {
      // Update loading mask
      auto loadOp = llvm::cast<triton::LoadOp>(op);
      auto mask = loadOp.getMask();
      // pre-load global -> regs
      Value newMask = getLoadMask(loadOp, nextMapping.lookupOrDefault(mask),
                                  nextLoopCond, builder);
      if (mask) {
        // If mask is defined outside the loop, don't update the map more than
        // once
        if (!(forOp.isDefinedOutsideOfLoop(mask) && nextMapping.contains(mask)))
          nextMapping.map(loadOp.getMask(), newMask);
        newMask = nextMapping.lookupOrDefault(mask);
      }
      auto newOp = builder.clone(*op, nextMapping);
      updateLoadMask(cast<triton::LoadOp>(newOp), newMask);
    } else if (!immediateOpStages[op].contains(numStages - 2)) {
      Operation *nextOp = builder.clone(*op, nextMapping);
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        if (auto newMask = getLoadMask(loadOp,
                                       nextMapping.lookupOrDefault(loadOp.getMask()),
                                       nextLoopCond, builder)) {
          updateLoadMask(cast<triton::LoadOp>(nextOp), newMask);
        }
      }

      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
        nextMapping.map(op->getResult(dstIdx), nextOp->getResult(dstIdx));
      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
        setValueMappingYield(op->getResult(dstIdx),
                             nextOp->getResult(dstIdx));
    }
  }
}

void LoopPipeliner::cloneCurrentBody(OpBuilder &builder) {
  auto loc = forOp.getLoc();
  // only add instructions that are not part of the restructuring
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (currentDeps.contains(&op)) {
      Operation *newOp = nullptr;
      if (isLoadChain(&op)) {
        if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(&op)) {
          Value mappedValue = curMapping.lookup(cvt.getSrc());
          if (isa<triton::MemDescType>(mappedValue.getType())) {
            auto newCvt = builder.create<triton::gpu::LocalLoadOp>(
                cvt.getLoc(), cvt.getType(), mappedValue);
            curMapping.map(cvt.getResult(), newCvt);
            newOp = newCvt;
          }
        }
        if (!newOp)
          newOp = builder.clone(op, curMapping);
      } else {
        newOp = cloneWithInferType(builder, &op, curMapping);
      }
    }
  }
}
  
void LoopPipeliner::storeNextBuffer(OpBuilder &builder) {
  // Store the next buffer at the end of the loop body for the next iteration
  for (Operation *op : orderedDeps) {
    if (!validLoads.contains(op)) {
      if (immediateOpStages[op].contains(numStages - 2)) {
        Operation *nextOp = builder.clone(*op, nextMapping);
        if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
          auto newMask = getLoadMask(loadOp,
                                     nextMapping.lookupOrDefault(loadOp.getMask()),
                                     nextLoopCond, builder);
          updateLoadMask(cast<triton::LoadOp>(nextOp), newMask);
        }

        for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
          setValueMappingYield(op->getResult(dstIdx),
                               nextOp->getResult(dstIdx));
      }
    }
  }
  
  // PL loads -> store next to shared
  for (auto *loadOp : validLoads) {
    Value loadVal = nextMapping.lookup(loadOp->getResult(0));
    // then store regs -> shared
    Value storeBuf = pplForOp.getRegionIterArgs()[bufferIdx + nextBuffers.size()];
    auto alloc = builder.create<ttg::LocalAllocOp>(
          loadOp->getLoc(), storeBuf.getType(), loadVal);
    nextBuffers.push_back(alloc);
  }

  // Some values have not been used by any ops in the loop body
  for (BlockArgument arg : forOp.getRegionIterArgs())
    setValueMappingYield(arg,
                         pplForOp.getRegionIterArgs()[depArgsIdx[arg]]);

}

void LoopPipeliner::finalizeYield(OpBuilder &builder) {
  SmallVector<Value> yieldValues;
  for (const auto &opr : llvm::enumerate(yieldOp->getOperands())) {
    if (curMapping.contains(opr.value()))
      yieldValues.push_back(curMapping.lookup(opr.value()));
    else
      yieldValues.push_back(pplForOp.getRegionIterArgs()[opr.index()]);
  }
  for (Value nextBuffer : nextBuffers)
    yieldValues.push_back(nextBuffer);

  for (size_t i = 0; i < depArgsMapping.size(); ++i) {
    auto arg = pplForOp.getRegionIterArgs()[depArgsBeginIdx + i];
    assert(depArgsMapping.count(arg) && "Missing loop-carried value");
    yieldValues.push_back(depArgsMapping[arg]);
  }

  builder.setInsertionPointToEnd(pplForOp.getBody());
  builder.create<scf::YieldOp>(yieldOp->getLoc(), yieldValues);
}

scf::ForOp LoopPipeliner::createNewForOp() {
  OpBuilder builder(forOp);
  auto newLoopArgs = collectNewLoopArgs();
  cloneForOp(newLoopArgs, builder);
  prefetchNextBuffer(builder);
  cloneCurrentBody(builder);
  storeNextBuffer(builder);
  finalizeYield(builder);
  return pplForOp;
}

// Stream Pipeline
struct PipelinePass : public TritonAMDGPUStreamPipelineBase<PipelinePass> {
  PipelinePass() = default;

  void runOnOperation() override {
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
      LoopPipeliner pipeliner(forOp);

      if (pipeliner.initialize().failed())
        return;

      pipeliner.emitPrologue();
      scf::ForOp pplForOp = pipeliner.createNewForOp();
      DenseMap<Value, Value> newResults;
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        newResults[forOp->getResult(i)] = pplForOp->getResult(i);
      pipeliner.emitEpilogue(newResults);

      // Replace the original loop
      for (auto &pair : newResults)
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
      forOp->erase();
    });
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUStreamPipelinePass() {
  return std::make_unique<PipelinePass>();
}
