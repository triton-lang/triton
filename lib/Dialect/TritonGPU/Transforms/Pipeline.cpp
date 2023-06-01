#include "Utility.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"

//===----------------------------------------------------------------------===//
//
// This file implements loop software pipelining
// The implementation here is inspired by the pipeline pass in Triton (-v2.0)
// and SCF's LoopPipelining.
//
//===----------------------------------------------------------------------===//

using llvm::MapVector;
using namespace mlir;
namespace ttg = triton::gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
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

#define int_attr(num) builder.getI64IntegerAttr(num)

namespace {

class LoopPipeliner {
  /// Cache forOp we are working on
  scf::ForOp forOp;

  /// Cache YieldOp for this forOp
  scf::YieldOp yieldOp;

  /// Loads to be pipelined
  SetVector<Value> loads;
  /// Smallest data-type for each load (used to optimize swizzle and
  /// (create DotOpEncoding layout)
  DenseMap<Value, Type> loadsSmallestType;
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
  ///
  Value pipelineIterIdx;
  ///
  Value loopIterIdx;

  /// Comments on numStages:
  ///   [0, numStages-1) are in the prologue
  ///   numStages-1 is appended after the loop body
  int numStages;

  /// value (in loop) => value at stage N
  DenseMap<Value, SmallVector<Value>> valueMapping;

  /// For each argument, we need to record at which stage it is defined.
  /// If we have a load that immediately depends on a block argument in the
  /// current iteration, it is an immediate dependency. Otherwise, it is a
  /// non-immediate dependency, which means the load depends on a block argument
  /// in the previous iterations.
  /// For example:
  /// scf.for (%arg0, %arg1, %arg2) {
  ///   %0 = load %arg0  <--- immediate dep, this address is initialized before
  ///   numStages-1
  ///   %1 = load %arg1
  ///   %2 = add %1, %arg2
  ///   %3 = load %2  <--- non-immediate dep, %arg1 must be an update-to-date
  ///   value
  /// }
  /// Collect values that v depends on and are defined inside the loop
  LogicalResult collectDeps(Value v, int stage,
                            MapVector<Value, int> &depStage);

  /// Associate each variable with a unique stage. If a variable is defined
  /// at multiple stages, we don't pipeline it.
  LogicalResult addDep(Value v, int stage, MapVector<Value, int> &depStage);

  int getArgDefStage(Value v, int stage);

  /// Block arguments that loads depend on
  MapVector<BlockArgument, int> depArgUseStage;

  /// Block arguments that loads depend on (defined in the loop body)
  MapVector<BlockArgument, int> depArgDefStage;

  /// Operations (inside the loop body) that loads depend on
  MapVector<Operation *, int> depOpDefStage;

  /// Operations (inside the loop body) that loads depend on (defined in the
  /// loop body)
  SetVector<BlockArgument> immediateDepArgs;

  /// Operations (inside the loop body) that loads depend on (defined in the
  /// previous iterations)
  SetVector<BlockArgument> nonImmediateDepArgs;

  void setValueMapping(Value origin, Value newValue, int stage);

  Value lookupOrDefault(Value origin, int stage);

  Value getLoadMask(triton::LoadOp loadOp, Value mappedMask, Value loopCond,
                    OpBuilder &builder);

  /// Returns a empty buffer of size <numStages, ...>
  ttg::AllocTensorOp allocateEmptyBuffer(Operation *op, OpBuilder &builder);

public:
  LoopPipeliner(scf::ForOp forOp, int numStages)
      : forOp(forOp), numStages(numStages) {
    // cache yieldOp
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

LogicalResult LoopPipeliner::addDep(Value v, int stage,
                                    MapVector<Value, int> &depStage) {
  if (!depStage.contains(v)) {
    depStage.insert(std::make_pair(v, stage));
  } else if (depStage[v] != stage)
    return failure();
  return success();
}

LogicalResult LoopPipeliner::collectDeps(Value v, int stage,
                                         MapVector<Value, int> &depStage) {
  // Loop-invariant value, skip
  if (v.getParentRegion() != &forOp.getLoopBody())
    return success();

  // Since we only need to peel the loop numStages-1 times, don't worry about
  // depends that are too far away
  if (stage < 0)
    return success();

  if (auto arg = v.dyn_cast<BlockArgument>()) {
    // Skip the first arg (loop induction variable)
    // Otherwise the op idx is arg.getArgNumber()-1
    if (arg.getArgNumber() > 0) {
      // If we've found the first definition of this arg, we're done, don't
      // recurse
      if (addDep(v, stage, depStage).succeeded())
        if (collectDeps(yieldOp->getOperand(arg.getArgNumber() - 1), stage - 1,
                        depStage)
                .failed())
          return failure();
    }
  } else { // value
    // An operation cannot be dependent on different stages
    if (addDep(v, stage, depStage).failed())
      return failure();
    for (Value op : v.getDefiningOp()->getOperands())
      if (collectDeps(op, stage, depStage).failed())
        return failure();
  }
  return success();
}

int LoopPipeliner::getArgDefStage(Value v, int stage) {
  if (stage < 0)
    return -1;
  if (auto arg = v.dyn_cast<BlockArgument>()) {
    if (arg.getArgNumber() > 0) {
      return getArgDefStage(yieldOp->getOperand(arg.getArgNumber() - 1),
                            stage - 1);
    }
    llvm_unreachable("Loop induction variable should not be a dependency");
  } else
    return stage;
}

ttg::AllocTensorOp LoopPipeliner::allocateEmptyBuffer(Operation *op,
                                                      OpBuilder &builder) {
  // Allocate a buffer for each pipelined tensor
  // shape: e.g. (numStages==4), <32x64xbf16> -> <4x32x64xbf16>
  Value convertLayout = loadsMapping[op->getResult(0)];
  if (auto tensorType = convertLayout.getType().dyn_cast<RankedTensorType>()) {
    return builder.create<ttg::AllocTensorOp>(
        convertLayout.getLoc(), loadsBufferType[op->getResult(0)]);
  }
  llvm_unreachable("Async copy's return should be of RankedTensorType");
}

/// A load instruction can be pipelined if:
///   - the load doesn't depend on any other loads (after loop peeling)
///   - (?) this load is not a loop-invariant value (we should run LICM before
///                                                  this pass?)
LogicalResult LoopPipeliner::initialize() {
  Block *loop = forOp.getBody();
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  SmallVector<triton::LoadOp, 2> validLoads;
  for (Operation &op : *loop)
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
      // 2. It's likely that pipling small load won't offer much performance
      //    improvement and may even hurt performance by increasing register
      //    pressure.
      if (width >= 32)
        validLoads.push_back(loadOp);
    }

  // Early stop: no need to continue if there is no load in the loop.
  if (validLoads.empty())
    return failure();

  // load => values that it depends on
  // Don't pipeline if any load's operands
  DenseMap<Value, SetVector<Value>> loadDeps;
  MapVector<Value, int> depStage;
  for (triton::LoadOp loadOp : validLoads) {
    for (Value op : loadOp->getOperands()) {
      MapVector<Value, int> operandDepStage;
      if (collectDeps(op, numStages - 1, operandDepStage).failed())
        return failure();
      for (auto [v, stage] : operandDepStage) {
        auto immedidate = operandDepStage.front().first.isa<BlockArgument>();
        if (v.isa<BlockArgument>()) {
          auto arg = v.cast<BlockArgument>();
          if (immedidate)
            immediateDepArgs.insert(arg);
          else
            nonImmediateDepArgs.insert(arg);
        }
        loadDeps[loadOp].insert(v);
        if (addDep(v, stage, depStage).failed())
          return failure();
      }
    }
  }

  // Don't pipeline valid loads that depend on other valid loads
  // (Because if a valid load depends on another valid load, this load needs to
  // wait on the other load in the prologue, which is against the point of the
  // pipeline pass)
  for (triton::LoadOp loadOp : validLoads) {
    bool isCandidate = true;
    for (triton::LoadOp other : validLoads) {
      if (loadDeps[loadOp].contains(other)) {
        isCandidate = false;
        break;
      }
    }

    // We only pipeline loads that have one covert_layout (to dot_op) use
    // TODO: lift this constraint in the future
    if (isCandidate && loadOp.getResult().hasOneUse()) {
      isCandidate = false;
      Operation *use = *loadOp.getResult().getUsers().begin();

      // advance to the first conversion as long
      // as the use resides in shared memory and it has
      // a single use itself
      while (use) {
        if (use->getNumResults() != 1 || !use->getResult(0).hasOneUse())
          break;
        auto tensorType =
            use->getResult(0).getType().dyn_cast<RankedTensorType>();
        if (!tensorType.getEncoding().isa<ttg::SharedEncodingAttr>())
          break;
        use = *use->getResult(0).getUsers().begin();
      }

      auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(use);
      if (!convertLayout)
        continue;
      auto tensorType =
          convertLayout.getResult().getType().dyn_cast<RankedTensorType>();
      if (!tensorType)
        continue;
      auto dotOpEnc =
          tensorType.getEncoding().dyn_cast<ttg::DotOperandEncodingAttr>();
      if (!dotOpEnc)
        continue;
      isCandidate = true;
      loadsMapping[loadOp] = convertLayout;
    } else
      isCandidate = false;

    if (isCandidate)
      loads.insert(loadOp);
  }

  // we need to find the smallest ocmmon dtype
  // since this determines the layout of `mma.sync` operands
  // in mixed-precision mode
  Type smallestType;
  for (auto loadCvt : loadsMapping) {
    auto loadOp = loadCvt.first;
    auto ty = loadOp.getType().cast<RankedTensorType>();
    Type eltTy = ty.getElementType();
    if (!smallestType ||
        (eltTy.getIntOrFloatBitWidth() < smallestType.getIntOrFloatBitWidth()))
      smallestType = eltTy;
  }

  for (auto loadCvt : loadsMapping)
    loadsSmallestType[loadCvt.first] = smallestType;

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
    auto sharedEnc = ttg::SharedEncodingAttr::get(
        ty.getContext(), dotOpEnc, ty.getShape(),
        ttg::getOrder(ty.getEncoding()), loadsSmallestType[loadOp]);
    loadsBufferType[loadOp] =
        RankedTensorType::get(bufferShape, ty.getElementType(), sharedEnc);
  }

  // We have some loads to pipeline
  if (!loads.empty()) {
    // Update depArgs & depOps
    for (auto [dep, stage] : depStage) {
      if (auto arg = dep.dyn_cast<BlockArgument>())
        depArgUseStage.insert({arg, stage});
      else
        depOpDefStage.insert({dep.getDefiningOp(), stage});
    }
    return success();
  }

  // Check if immedidateDepArgs and nonImmedidateDepArgs are disjoint
  // If yes, we cannot pipeline the loop for now
  for (BlockArgument arg : immediateDepArgs)
    if (nonImmediateDepArgs.contains(arg)) {
      return failure();
    }

  return failure();
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

    // Rematerialize peeled values
    SmallVector<Operation *> orderedDeps;
    for (Operation &op : forOp.getLoopBody().front()) {
      if (depOpDefStage.contains(&op))
        orderedDeps.push_back(&op);
      else if (op.getNumResults() > 0 && loads.contains(op.getResult(0)))
        orderedDeps.push_back(&op);
    }
    assert(depOpDefStage.size() + loads.size() == orderedDeps.size() &&
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
            assert(v);
            newOp->setOperand(opIdx, v);
          } // else, op at opIdx is a loop-invariant value
        }
      }

      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults())) {
        Value originResult = op->getResult(dstIdx);
        // copy_async will update the value of its only use
        if (loads.contains(originResult))
          break;
        setValueMapping(originResult, newOp->getResult(dstIdx), stage);
        // update mapping for loop-carried values (args)
        for (OpOperand &operand : yieldOp->getOpOperands()) {
          if (operand.get() == op->getResult(dstIdx)) {
            auto yieldIdx = operand.getOperandNumber();
            auto value = forOp.getRegionIterArgs()[yieldIdx];
            setValueMapping(value, newOp->getResult(dstIdx), stage + 1);
          }
        }
      }
    } // for (Operation *op : orderedDeps)

    // Update pipeline index
    pipelineIterIdx = builder.create<arith::AddIOp>(
        iv.getLoc(), pipelineIterIdx,
        builder.create<arith::ConstantIntOp>(iv.getLoc(), 1, 32));

    // Some values have not been used by any ops in the loop body
    for (BlockArgument arg : forOp.getRegionIterArgs()) {
      // Check if arg has a yieldOp use
      for (OpOperand &operand : arg.getUses()) {
        if (operand.getOwner() == yieldOp) {
          auto yieldIdx = operand.getOperandNumber();
          auto value = forOp.getRegionIterArgs()[yieldIdx];
          if (!valueMapping[value][stage + 1])
            setValueMapping(value, valueMapping[arg][stage], stage + 1);
        }
      }
    }
  } // for (int stage = 0; stage < numStages - 1; ++stage)

  // async.wait & extract_slice
  builder.create<ttg::AsyncWaitOp>(loads[0].getLoc(),
                                   loads.size() * (numStages - 2));
  loopIterIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
  for (Value loadOp : loads) {
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
  // *next* iteration
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

scf::ForOp LoopPipeliner::createNewForOp() {
  OpBuilder builder(forOp);

  // Order of new args:
  //   (original args)
  //   (insertSliceAsync buffer at stage numStages - 1) for each load
  //   (extracted tensor) for each load
  //   (depArgs at stage numStages - 1):
  //   for each dep arg that is not an immediate block argument
  //   (depArgs at stage numStages - 2):
  //   for each dep arg that is an immediate block argument
  //   (iv at stage numStages - 2)
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
  for (auto [depArg, useStage] : depArgUseStage) {
    depArgsIdx[depArg] = newLoopArgs.size();
    auto defStage = getArgDefStage(depArg, useStage);
    assert(defStage >= 0 &&
           "newLoopArgs has null args without a define op. Consider either "
           "rewrite the loop to reduce cross iteration dependencies or "
           "increase the num_stages value.");
    if (immediateDepArgs.contains(depArg) && defStage == numStages - 2) {
      newLoopArgs.push_back(valueMapping[depArg][numStages - 2]);
    } else
      newLoopArgs.push_back(valueMapping[depArg][numStages - 1]);
  }

  size_t ivIndex = newLoopArgs.size();
  newLoopArgs.push_back(valueMapping[forOp.getInductionVar()][numStages - 2]);
  newLoopArgs.push_back(pipelineIterIdx);
  newLoopArgs.push_back(loopIterIdx);

  // 1. signature of the new ForOp
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newLoopArgs);

  // 2. body of the new ForOp
  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  // 3. clone the loop body, replace original args with args of the new ForOp
  // Insert async wait if necessary.
  DenseSet<Value> isModified;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    // is modified
    auto it = std::find(loads.begin(), loads.end(), op.getOperand(0));
    if (it == loads.end()) {
      Operation *newOp = cloneWithInferType(builder, &op, mapping);
      continue;
    }

    // we replace the use new load use with a convert layout
    size_t i = std::distance(loads.begin(), it);
    auto cvtDstTy = op.getResult(0).getType().cast<RankedTensorType>();
    auto cvtDstEnc =
        cvtDstTy.getEncoding().dyn_cast<ttg::DotOperandEncodingAttr>();
    if (!cvtDstEnc) {
      builder.clone(op, mapping);
      continue;
    }
    auto newDstTy = RankedTensorType::get(
        cvtDstTy.getShape(), cvtDstTy.getElementType(),
        ttg::DotOperandEncodingAttr::get(
            cvtDstEnc.getContext(), cvtDstEnc.getOpIdx(), cvtDstEnc.getParent(),
            loadsSmallestType[op.getOperand(0)]));
    auto cvt = builder.create<ttg::ConvertLayoutOp>(
        op.getResult(0).getLoc(), newDstTy,
        newForOp.getRegionIterArgs()[loadIdx + i]);
    mapping.map(op.getResult(0), cvt.getResult());
    isModified.insert(op.getResult(0));
  }

  // 4. prefetch the next iteration
  SmallVector<Operation *> orderedDeps;
  for (Operation &op : forOp.getLoopBody().front()) {
    if (depOpDefStage.contains(&op))
      orderedDeps.push_back(&op);
    else if (op.getNumResults() > 0 && loads.contains(op.getResult(0)))
      orderedDeps.push_back(&op);
  }
  assert(depOpDefStage.size() + loads.size() == orderedDeps.size() &&
         "depOps contains invalid values");
  IRMapping nextMapping;
  DenseMap<BlockArgument, Value> depArgsMapping;
  size_t argIdx = 0;
  for (auto [depArg, useStage] : depArgUseStage) {
    BlockArgument nextArg =
        newForOp.getRegionIterArgs()[argIdx + depArgsBeginIdx];
    nextMapping.map(depArg, nextArg);
    ++argIdx;
  }

  // Special handling for iv & loop condition
  Value curIV = newForOp.getRegionIterArgs()[ivIndex];
  Value nextIV = builder.create<arith::AddIOp>(
      newForOp.getInductionVar().getLoc(), curIV, newForOp.getStep());
  Value nextLoopCond =
      builder.create<arith::CmpIOp>(nextIV.getLoc(), arith::CmpIPredicate::slt,
                                    nextIV, newForOp.getUpperBound());

  // Slice index
  SmallVector<Value> nextBuffers;
  SmallVector<Value> extractSlices;

  pipelineIterIdx = newForOp.getRegionIterArgs()[ivIndex + 1];
  Value insertSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), pipelineIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages, 32));
  loopIterIdx = newForOp.getRegionIterArgs()[ivIndex + 2];
  Value extractSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages, 32));

  // Prefetch load deps
  for (Operation *op : orderedDeps)
    if (!loads.contains(op->getResult(0))) {
      if (depOpDefStage[op] == numStages - 2)
        nextMapping.map(forOp.getInductionVar(), curIV);
      else
        nextMapping.map(forOp.getInductionVar(), nextIV);
      Operation *nextOp;
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        auto newMask =
            getLoadMask(loadOp, nextMapping.lookupOrDefault(loadOp.getMask()),
                        nextLoopCond, builder);
        nextOp = builder.create<triton::LoadOp>(
            loadOp.getLoc(), loadOp.getResult().getType(),
            nextMapping.lookupOrDefault(loadOp.getPtr()), newMask,
            nextMapping.lookupOrDefault(loadOp.getOther()),
            loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
            loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
        addNamedAttrs(nextOp, op->getDiscardableAttrDictionary());
        nextMapping.map(loadOp.getResult(), nextOp->getResult(0));
      } else {
        nextOp = builder.clone(*op, nextMapping);
      }

      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults())) {
        for (OpOperand &operand : yieldOp->getOpOperands()) {
          if (operand.get() == op->getResult(dstIdx)) {
            size_t yieldIdx = operand.getOperandNumber();
            size_t depYieldIdx =
                depArgsIdx[forOp.getRegionIterArgs()[yieldIdx]];
            BlockArgument newArg = newForOp.getRegionIterArgs()[depYieldIdx];
            nextMapping.map(forOp.getRegionIterArgs()[yieldIdx],
                            nextOp->getResult(dstIdx));
            depArgsMapping[newArg] = nextOp->getResult(dstIdx);
          }
        }
      }
    }

  // loads -> async loads
  for (Operation *op : orderedDeps) {
    Operation *nextOp = nullptr;
    // Update loading mask
    if (loads.contains(op->getResult(0))) {
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
      // ExtractSlice
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
      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults())) {
        nextMapping.map(op->getResult(dstIdx), nextOp->getResult(dstIdx));
        // If this is a loop-carried value, update the mapping for yield
        for (OpOperand &operand : yieldOp->getOpOperands()) {
          if (operand.get() == op->getResult(dstIdx)) {
            auto yieldIdx = operand.getOperandNumber();
            auto depYieldIdx = depArgsIdx[forOp.getRegionIterArgs()[yieldIdx]];
            auto newArg = newForOp.getRegionIterArgs()[depYieldIdx];
            depArgsMapping[newArg] = nextOp->getResult(dstIdx);
          }
        }
      }
    }
  }

  // Some values have not been used by any ops in the loop body
  for (BlockArgument arg : forOp.getRegionIterArgs()) {
    // Check if arg has a yieldOp use
    for (OpOperand &operand : arg.getUses()) {
      if (operand.getOwner() == yieldOp) {
        auto yieldIdx = operand.getOperandNumber();
        auto depYieldIdx = depArgsIdx[forOp.getRegionIterArgs()[yieldIdx]];
        auto newArg = newForOp.getRegionIterArgs()[depYieldIdx];
        if (!depArgsMapping.contains(newArg)) {
          auto argIdx = depArgsIdx[arg];
          depArgsMapping[newArg] = newForOp.getRegionIterArgs()[argIdx];
        }
      }
    }
  }

  // async.wait & extract_slice
  Operation *asyncWait = builder.create<ttg::AsyncWaitOp>(
      loads[0].getLoc(), loads.size() * (numStages - 2));
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

  // Finally, the YieldOp, need to sync with the order of newLoopArgs
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
