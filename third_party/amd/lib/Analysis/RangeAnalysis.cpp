#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <numeric>
#include <optional>

// Some notes:
//
// 1. Framework
//  1.1) This pass is based on MLIR's dataflow framework. In hindsight, maybe it
//    is ill-fit for what we need.
//  1.2) If I understand correctly, the MLIR's dataflow framework is a
//     combination of traditional iterative dataflow analysis and a mighty
//     Sparse Conditional Constant propagation (SCCP).
//  1.3) Iterative dataflow analysis requires transfer function to be monotone.
//    However, not all value-ranges keep increasing when the analysis progress.
//    Consider the expression x - y, while x and y's value-range may keep
//    increasing, the difference between them does not necessarily keep
//    increasing as well.
//  1.4) The 1st C in SCCP, i.e. "conditional" part in SCCP part is unnecessary
//    for this pass, because we don't expect many dead code at the moment when
//    this analysis is invoked. Price for being "conditional" is less about
//    compile time but complexity (in terms of debugging and understanding).
//  1.5 Maybe just walking the code top-dowm is sufficient for range-analysis:
//    For loops, figuring out IVs' value-ranges before loops are entered, and
//    progress to loop-body, without visiting back-edge for non-SCF loops.
//
// 2: tl.assume statements
//  2.1) A value may have multiple assume-operations (assume-ops for short)
//    associated with it. At point p, we only take into account those assume-ops
//    whose enclosing basic blocks dominate the basic-block where p belongs to.
//  2.2) See some examples in the comment to maybeGetAssumedRangeHelper().
//  2.3) The assumed value-range for source and result operands are inferred
//  right before an operation is visited.
//  2.4) For now, if a value has a assumed value-range, we use assumed
//    value-range and ignore its inferred value range. It would be nice to
//    use the intersection of assumed-value-range and inferred-value-range.
//    However, it is not always possible: iterative dataflow analysis
//    requires that the transfer function must be monotone; in general it's
//    dangerous to use both meet() and join() operations. In this pass,
//    intersecting inferred value-range with assumed-value-range still guarantee
//    its monotonicity. However, the underlying lattice's meet() operation is
//    a silent no-op.
//

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-range-analysis"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace tt = mlir::triton;

namespace {

constexpr int64_t kDefaultMaxTripCount = 1024;
constexpr uint64_t kDefaultMaxPrograms = 1L << 31; // 2147483648

void getEnclosingLoops(Operation &op, SmallVector<LoopLikeOpInterface> &ops) {
  Operation *currOp = op.getParentOp();
  while (currOp) {
    if (isa<LoopLikeOpInterface>(currOp))
      ops.push_back(llvm::cast<LoopLikeOpInterface>(currOp));
    currOp = currOp->getParentOp();
  }
}

tt::FuncOp getEnclosingFunction(Value v) {
  tt::FuncOp funcOp = nullptr;

  auto definingOp = v.getDefiningOp();
  if (!definingOp)
    if (auto blk = v.getParentBlock())
      definingOp = blk->getParentOp();

  if (definingOp) {
    if (auto selfIsFunc = dyn_cast<tt::FuncOp>(definingOp))
      funcOp = selfIsFunc;
    else
      funcOp = definingOp->getParentOfType<tt::FuncOp>();
  }

  assert(funcOp && "No enclosing tt::FuncOp");
  return funcOp;
}

Block *getFuncEntryBlock(tt::FuncOp func) { return &func.getRegion().front(); }

void inferResultRangesPID(Operation *op, uint64_t max,
                          SetIntRangeFn setResultRange) {
  assert(op->getNumResults() == 1 && "expected op to have one result");
  auto result = op->getResult(0);
  assert(llvm::isa<IntegerType>(result.getType()) &&
         "expected result type to be int");
  IntegerType resTy = llvm::cast<IntegerType>(result.getType());
  auto bitWidth = mlir::ConstantIntRanges::getStorageBitwidth(resTy);
  setResultRange(result, ConstantIntRanges::range(
                             /*min*/ {/*numBits*/ bitWidth, /*val*/ 0,
                                      /*isSigned*/ resTy.isSigned()},
                             /*max*/
                             {/*numBits*/ bitWidth, /*val*/ max,
                              /*isSigned*/ resTy.isSigned()},
                             /*isSigned*/ resTy.isSigned()));
}

void inferResultRanges(tt::MakeRangeOp *op, SetIntRangeFn setResultRange) {
  auto result = op->getResult();
  RankedTensorType resTy = result.getType();
  assert(llvm::isa<IntegerType>(resTy.getElementType()) && "expected int type");
  IntegerType elTy = llvm::cast<IntegerType>(resTy.getElementType());
  auto bitWidth = mlir::ConstantIntRanges::getStorageBitwidth(elTy);
  // NOTE: make_range(begin, end) yields a half open interval, [begin, end).
  setResultRange(result,
                 ConstantIntRanges::range(
                     /*min*/ {/*numBits*/ bitWidth, /*val*/ op->getStart(),
                              /*isSigned*/ elTy.isSigned()},
                     /*max*/
                     {/*numBits*/ bitWidth, /*val*/ op->getEnd() - 1,
                      /*isSigned*/ elTy.isSigned()},
                     /*isSigned*/ elTy.isSigned()));
}

void inferResultRanges(tt::GatherOp *op, ArrayRef<ConstantIntRanges> argRanges,
                       SetIntRangeFn setResultRange) {
  assert(argRanges.size() == 2 && "expected two arg ranges");
  setResultRange(op->getResult(), argRanges[0]);
}

void inferResultRangesUnaryOpForwardArgRange(
    Operation *op, ArrayRef<ConstantIntRanges> argRanges,
    SetIntRangeFn setResultRange) {
  for (const auto &result : op->getResults())
    setResultRange(result, argRanges[0]);
}

void inferResultRangesBinaryOpUnionArgRanges(
    Operation *op, ArrayRef<ConstantIntRanges> argRanges,
    SetIntRangeFn setResultRange) {
  assert(op->getNumOperands() == 2 && "expected op to have two operands");
  assert(argRanges.size() == 2 && "expected two arg ranges");
  for (const auto &result : op->getResults())
    setResultRange(result, argRanges[0].rangeUnion(argRanges[1]));
}

void inferResultRangesMaxNonNegSigned(Operation *op,
                                      SetIntRangeFn setResultRange) {
  for (auto result : op->getResults()) {
    auto bitWidth =
        mlir::ConstantIntRanges::getStorageBitwidth(result.getType());
    setResultRange(result, ConstantIntRanges::fromSigned(
                               APInt::getZero(bitWidth).sext(bitWidth),
                               APInt::getMaxValue(bitWidth).sext(bitWidth)));
  }
}

// Given an assumption operation, try to derive the value range of the value
// <anchor>'s value range at the somewhere in the block "useBlock".
// Note that
//  - The value "anchor" is defined or referenced in the "useBlock"
//  - The location of the reference of "anchor" in the "useBlock" does not
//    matter because the IR is in SSA form, the value-range of a quantity
//    does not change through out the entire block.
//  - The assumption should be ignored if it does not dominate the "useBlock".
//
// Consider following cases:
//
// case 1: both s2 and s3 are applicable to s1 because they dominate s1
//   s2: assume y > 5
//   ...
//   if cond
//     s3: assume z < 3
//     s1: x = y + z
//
// case 2: s2 is applicable to s1 even if s2 stay after s1.
//   blk:
//     s1: x = y + z
//     s2: assume y > 5
//
// case 3: s2 is not applicable to s1 because the block of else-caluse does not
//   domoinate the then-clause block.
//   if cond
//      s1: x = y + z
//   else
//      s2: assume y > 5
//
std::optional<ConstantIntRanges>
maybeGetAssumedRangeHelper(Operation *assumption, Value anchor, Block *useBlock,
                           DominanceInfo *domInfo) {

  arith::CmpIOp cmpOp = llvm::dyn_cast<arith::CmpIOp>(assumption);
  if (!cmpOp) {
    emitRemark(assumption->getLoc(), "unsupported assumption operation");
    return {};
  }

  // The block where tl.assume resides must dominate the block where the value
  // is referenced!
  if (!useBlock || !domInfo->dominates(cmpOp->getBlock(), useBlock))
    return {};

  bool isSigned = true;
  switch (cmpOp.getPredicate()) {
  case arith::CmpIPredicate::uge:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ult:
    isSigned = false;
  default:
    break;
  }

  bool anchorIsLhs = cmpOp.getLhs() == anchor;
  auto maybeConstantIntValue = getConstantIntValue(
      getAsOpFoldResult(anchorIsLhs ? cmpOp.getRhs() : cmpOp.getLhs()));
  if (auto constValue = maybeConstantIntValue) {
    unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(anchor.getType());
    assert(bitWidth > 0 && "expected non-zero bitwdith");
    APInt apVal = {bitWidth, static_cast<uint64_t>(*constValue), isSigned};
    APInt min, max;
    if (isSigned) {
      min = APInt::getSignedMinValue(bitWidth);
      if (llvm::isa_and_nonnull<mlir::triton::GetProgramIdOp,
                                mlir::triton::GetNumProgramsOp>(
              anchor.getDefiningOp())) {
        min = APInt::getZero(bitWidth);
      } else
        min = APInt::getSignedMinValue(bitWidth);
      max = APInt::getSignedMaxValue(bitWidth);
    } else {
      min = APInt::getMinValue(bitWidth);
      max = APInt::getMaxValue(bitWidth);
    }

    switch (cmpOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return mlir::ConstantIntRanges::constant(apVal);
    case arith::CmpIPredicate::uge:
    case arith::CmpIPredicate::sge: {
      // K >= apVal implies K ∈ [apVal, max]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(apVal, max, isSigned);
      // apVal >= K implies K ∈ [min, apVal]
      return mlir::ConstantIntRanges::range(min, apVal, isSigned);
    }
    case arith::CmpIPredicate::ugt:
    case arith::CmpIPredicate::sgt: {
      // K > apVal implies K >= apVal + 1 implies K ∈ [apVal + 1, max]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(apVal + 1, max, isSigned);
      // apVal > K implies apVal - 1 >= K implies K ∈ [min, apVal - 1]
      return mlir::ConstantIntRanges::range(min, apVal - 1, isSigned);
    }
    case arith::CmpIPredicate::ule:
    case arith::CmpIPredicate::sle: {
      // K <= apVal implies K ∈ [min, apVal]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(min, apVal, isSigned);
      // apVal <= K implies K ∈ [apVal, max]
      return mlir::ConstantIntRanges::range(apVal, max, isSigned);
    }
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::slt: {
      // K < apVal implies K <= apVal -1 implies K ∈ [min, apVal - 1]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(min, apVal - 1, isSigned);
      // apVal < K implies apVal + 1 <= K implies K ∈ [apVal + 1, max]
      return mlir::ConstantIntRanges::range(apVal + 1, max, isSigned);
    }
    default:
      emitRemark(cmpOp.getLoc(), "unsupported cmp predicate for assumption");
      return {};
    }
  }
  return {};
}

std::optional<ConstantIntRanges>
maybeGetAssumedRange(const SetVector<Operation *> &allAssumptions, Value anchor,
                     Block *useBlock, DominanceInfo *domInfo) {

  std::optional<ConstantIntRanges> result;
  for (auto assumption : allAssumptions) {
    auto tmpResult =
        maybeGetAssumedRangeHelper(assumption, anchor, useBlock, domInfo);
    if (!tmpResult.has_value())
      continue;

    if (result.has_value())
      result = (*result).intersection(*tmpResult);
    else
      result = *tmpResult;
  }

  if (result) {
    const auto &val = *result;
    if (val.smin().isNonNegative()) {
      // Consider 0 <= x && x <= 1024.
      // When processing x > 0, the value range of x is
      //  vr1={umin=0, umax=0xf...f, smin=0, smax=0x7...f}
      // When processing x < 1024, the value range of x is:
      //  vr2={umin=0, umax=0xf...f, smin=..., smax=1024}
      // and
      //  vr1 ∩ vr2 = {umin=0, umax=0xf...f, smin=0, smax=1024}
      // note that the umax=0xf...f is annoying, need to change to 1024.
      return ConstantIntRanges::range(val.smin(), val.smax(), true);
    }
  }
  return result;
}

} // namespace

namespace mlir::triton::AMD {

std::optional<int64_t>
TritonIntegerRangeAnalysis::maybeGetTripCount(LoopLikeOpInterface loop) {
  std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
  std::optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
  std::optional<OpFoldResult> step = loop.getSingleStep();
  std::optional<Value> iv = loop.getSingleInductionVar();
  if (!iv)
    return {};

  unsigned int width = ConstantIntRanges::getStorageBitwidth(iv->getType());

  auto getLoopRangeInfo = [&](std::optional<OpFoldResult> loopBound,
                              Block *block,
                              std::optional<bool> getUpper = std::nullopt,
                              std::optional<APInt> defaultVal = std::nullopt) {
    if (loopBound.has_value()) {
      if (auto attr = dyn_cast<Attribute>(*loopBound)) {
        if (auto bound = dyn_cast_or_null<IntegerAttr>(attr))
          return bound.getValue();
      } else if (auto value = llvm::dyn_cast_if_present<Value>(*loopBound)) {
        const dataflow::IntegerValueRangeLattice *lattice =
            getLatticeElementFor(getProgramPointBefore(block), value);
        if (lattice != nullptr && !lattice->getValue().isUninitialized())
          return getUpper ? lattice->getValue().getValue().smax()
                          : lattice->getValue().getValue().smin();
      }
    }
    if (defaultVal)
      return *defaultVal;
    return getUpper ? APInt::getSignedMaxValue(width)
                    : APInt::getSignedMinValue(width);
  };

  Block *block = iv->getParentBlock();
  APInt min = getLoopRangeInfo(lowerBound, block,
                               /*getUpper=*/false);
  APInt max = getLoopRangeInfo(upperBound, block,
                               /*getUpper=*/true);
  // We can assume step is 1 if no range information as that gives us the upper
  // bound of the number of iterations.
  APInt stepValDefault = {width, 1, /*isSigned=*/true};
  APInt stepVal =
      getLoopRangeInfo(step, block, /*getUpper=*/{}, stepValDefault);

  if (stepVal.isNegative())
    std::swap(min, max);
  // This is necessary to catch a case like this:
  //  # range = [0 1024]
  //  K = ....
  //  # range = [1, 64]
  //  k = ...
  //  # range = [0, 16] -> stepVal = range.smin() = 0
  //  step = ceildiv(K, k)
  if (stepVal.isZero())
    stepVal = stepValDefault;
  if (max.sge(min))
    return llvm::divideCeilSigned(max.getSExtValue() - min.getSExtValue(),
                                  stepVal.getSExtValue());
  return {};
}

bool isEmptyInitializedRange(ConstantIntRanges rv) {
  if (!rv.umin().getBitWidth() || !rv.umax().getBitWidth() ||
      !rv.smin().getBitWidth() || !rv.smax().getBitWidth())
    return true;
  return false;
}

std::optional<SmallVector<std::optional<ConstantIntRanges>>>
collectRanges(const DataFlowSolver &solver, ValueRange values) {
  SmallVector<std::optional<ConstantIntRanges>> ranges;
  for (Value val : values) {
    auto *maybeInferredRange =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(val);
    if (!maybeInferredRange ||
        maybeInferredRange->getValue().isUninitialized()) {
      ranges.push_back(std::nullopt);
      continue;
    }
    const ConstantIntRanges &inferredRange =
        maybeInferredRange->getValue().getValue();
    if (isEmptyInitializedRange(inferredRange)) {
      ranges.push_back(std::nullopt);
      continue;
    }
    ranges.push_back(inferredRange);
  }
  return ranges;
}

bool cmpIIsStaticallyTrue(const DataFlowSolver &solver, arith::CmpIOp cmpOp) {
  if (auto inputRanges =
          collectRanges(solver, ValueRange{cmpOp.getOperands()})) {
    intrange::CmpPredicate pred =
        static_cast<intrange::CmpPredicate>(cmpOp.getPredicate());
    if (!(*inputRanges)[0] || !(*inputRanges)[1])
      return false;
    return intrange::evaluatePred(pred, *(*inputRanges)[0], *(*inputRanges)[1])
        .value_or(false);
  }
  return false;
}

LogicalResult TritonIntegerRangeAnalysis::initialize(Operation *top) {
  signedIntValues.clear();
  return Base::initialize(top);
}

std::optional<ConstantIntRanges>
TritonIntegerRangeAnalysis::maybeGetAssumedRange(Value anchor,
                                                 Block *useBlock) const {
  const auto &matchingAssumptions = this->assumptions.lookup(anchor);
  if (matchingAssumptions.empty())
    return {};

  return ::maybeGetAssumedRange(matchingAssumptions, anchor, useBlock, domInfo);
}

int64_t
TritonIntegerRangeAnalysis::getTotalLoopTripCount(LoopLikeOpInterface loop) {
  SmallVector<LoopLikeOpInterface> loops{loop};
  getEnclosingLoops(*loop, loops);
  return std::accumulate(loops.begin(), loops.end(), (int64_t)1,
                         [this](int64_t accum, LoopLikeOpInterface loop) {
                           return accum * maybeGetTripCount(loop).value_or(
                                              kDefaultMaxTripCount + 1);
                         });
}

void TritonIntegerRangeAnalysis::setToEntryState(
    dataflow::IntegerValueRangeLattice *lattice) {
  auto anchor = lattice->getAnchor();
  if (!llvm::isa<IndexType>(getElementTypeOrSelf(anchor)) &&
      !llvm::isa<IntegerType>(getElementTypeOrSelf(anchor)))
    return;

  Block *entryBlock = getFuncEntryBlock(getEnclosingFunction(anchor));
  IntegerValueRange range = IntegerValueRange::getMaxRange(anchor);
  if (auto maybeRange = maybeGetAssumedRange(anchor, entryBlock))
    range = *maybeRange;
  auto changed = lattice->join(range);
  LLVM_DEBUG({
    if (changed == ChangeResult::Change) {
      DBGS() << "Set range of ";
      anchor.printAsOperand(llvm::dbgs(), {});
      llvm::dbgs() << " to " << range << "\n";
    }
  });
  propagateIfChanged(lattice, changed);
}

void TritonIntegerRangeAnalysis::defaultTransferFunc(
    Operation *op, Value resultVal,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> srcLattices,
    ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices,
    const IntegerValueRange &incomingRange) {

  // step 1: Preparation
  //  - Get the lattice associated with given particular result value.
  //  - Make a copy of value-range just inferred, as we need to do some
  //   change to it before it's joined to the existing lattice.
  auto result = dyn_cast<OpResult>(resultVal);
  if (!result)
    return;
  assert(llvm::is_contained(op->getResults(), result));

  dataflow::IntegerValueRangeLattice *lattice =
      resultsLattices[result.getResultNumber()];
  IntegerValueRange incomingRange_ = incomingRange;

  // step 2: If there is assumed value range, the assumed one take precedence.
  // TODO: I think this is bit conservative, the better way is:
  //  final_range = (old_range ∪ incomingRange) ∩ assume_range
  if (auto iter = opResultAssumption.find(resultVal);
      iter != opResultAssumption.end()) {
    const auto &range = iter->second;
    if (auto maybeRange = maybeGetAssumedRange(resultVal, op->getBlock())) {
      incomingRange_ =
          IntegerValueRange(incomingRange_.getValue().intersection(range));
    }
  }

  // step 3: Update the value range. Note that we are using `join` operation
  //  which means `union`. Transfer function must be monotone! The resolver
  //  would otherwise fall into infinite loop.
  ChangeResult changed = lattice->join(incomingRange_);
  LLVM_DEBUG({
    OpPrintingFlags flags;
    flags.skipRegions(true);
    DBGS() << ((changed == ChangeResult::Change) ? ">Inferred range for: "
                                                 : ">Remain unchanged: ");
    resultVal.printAsOperand(llvm::dbgs(), flags);
    llvm::dbgs() << ", resulting state:" << lattice->getValue()
                 << ", in value-range: " << incomingRange_ << "\n";
  });

  // step 4: Add those ops that depends on this op to the worklist. The resolver
  // will iterate all items in the worklist until it become empty.
  propagateIfChanged(lattice, changed);
}

LogicalResult TritonIntegerRangeAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
    ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices) {

  // step 1: Figure out the implied value-range of result and source operands
  opResultAssumption.clear();
  for (mlir::OpResult result : op->getResults()) {
    auto assumedRange = maybeGetAssumedRange(result, op->getBlock());
    if (assumedRange.has_value())
      opResultAssumption.insert(std::pair(result, *assumedRange));
  }

  llvm::SmallVector<const dataflow::IntegerValueRangeLattice *, 4>
      opndValueRanges;

  llvm::SmallVector<std::unique_ptr<dataflow::IntegerValueRangeLattice>, 4>
      newSrcLattices;

  for (auto [index, opnd] : llvm::enumerate(op->getOperands())) {
    auto assumedRange = maybeGetAssumedRange(opnd, op->getBlock());
    if (!assumedRange.has_value()) {
      opndValueRanges.push_back(operands[index]);
      continue;
    }

    auto newLattice =
        std::make_unique<dataflow::IntegerValueRangeLattice>(opnd);
    (void)newLattice->join(IntegerValueRange(*assumedRange));
    opndValueRanges.push_back(newLattice.get());
    newSrcLattices.push_back(std::move(newLattice));
  }
  assert(opndValueRanges.size() == operands.size() && "size disagree");

  // step 2: call helper function inferring the value range. If assumed value-
  // range is present, the transfer-function will intersect the assumed value-
  // value with the inferred value range.
  LogicalResult visitResult =
      visitOperationHelper(op, opndValueRanges, resultsLattices);

  // step 3: If previous step failed to infer value-range, apply assumed
  //  value-range is present.
  for (auto [index, lattice] : llvm::enumerate(resultsLattices)) {
    Value result = op->getResult(index);
    const auto assumedIter = opResultAssumption.find(result);
    if (assumedIter == opResultAssumption.end())
      continue;

    const mlir::IntegerValueRange &vr = lattice->getValue();
    if (!vr.isUninitialized() && !AMD::isEmptyInitializedRange(vr.getValue()))
      continue;

    const ConstantIntRanges &assumedVr = assumedIter->second;
    IntegerValueRange range(assumedVr);
    auto changed = lattice->join(range);

    LLVM_DEBUG({
      if (changed == ChangeResult::Change) {
        DBGS() << ">Force apply assumed value range. value:";
        result.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << ", range:" << range << "\n";
      }
    });
    propagateIfChanged(lattice, changed);
  }

  return visitResult;
}

LogicalResult TritonIntegerRangeAnalysis::visitOperationHelper(
    Operation *op,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
    ArrayRef<dataflow::IntegerValueRangeLattice *> resultsLattices) {
  LDBG("Inferring ranges for " << *op);

  // This callback is almost exactly like the callback in
  // IntegerRangeAnalysis::visitOperation except we do not "short-cicruit" the
  // analysis by inferring a maximum range for loop results (instead we
  // perform a check based on visit counts in visitRegionSuccessors).
  auto joinCallback = [&op, &operands, &resultsLattices,
                       this](Value v, const IntegerValueRange &incomingRange) {
    this->defaultTransferFunc(op, v, operands, resultsLattices, incomingRange);
  };

  // Ops with fixed/constant ranges.
  if (llvm::isa<GetProgramIdOp, MakeRangeOp, HistogramOp, GetNumProgramsOp>(
          op)) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<GetProgramIdOp>([&](auto getPIDOp) {
          inferResultRangesPID(getPIDOp, kDefaultMaxPrograms - 1, joinCallback);
        })
        .Case<GetNumProgramsOp>([&](auto getPIDOp) {
          inferResultRangesPID(getPIDOp, kDefaultMaxPrograms, joinCallback);
        })
        .Case<MakeRangeOp>([&](MakeRangeOp makeROp) {
          inferResultRanges(&makeROp, joinCallback);
        })
        .Case<HistogramOp>([&](HistogramOp histOp) {
          return inferResultRangesMaxNonNegSigned(histOp, joinCallback);
        })
        .Default([&](auto) { llvm::report_fatal_error("unsupported op"); });
    return success();
  }

  SmallVector<IntegerValueRange> argIntValueRanges = llvm::map_to_vector(
      operands, [](const dataflow::IntegerValueRangeLattice *lattice) {
        return lattice->getValue();
      });

  if (auto sliceOp = dyn_cast<triton::amdgpu::ExtractSliceOp>(op)) {
    joinCallback(sliceOp->getResult(0), argIntValueRanges[0]);
    return success();
  }

  // Ops with actually changing/variable input/output ranges.
  if (llvm::isa<TransOp, SplitOp, BroadcastOp, ReshapeOp, gpu::ConvertLayoutOp,
                SplatOp, ExpandDimsOp, JoinOp, CatOp, GatherOp>(op)) {
    SmallVector<ConstantIntRanges> argConstIntRanges;
    for (const auto &r : argIntValueRanges) {
      if (r.isUninitialized()) {
        setAllToEntryStates(resultsLattices);
        return success();
      }
      argConstIntRanges.push_back(r.getValue());
    }
    llvm::TypeSwitch<Operation *>(op)
        .Case<TransOp, SplitOp, BroadcastOp, ExpandDimsOp, SplatOp, ReshapeOp,
              gpu::ConvertLayoutOp>([&](auto) {
          return inferResultRangesUnaryOpForwardArgRange(op, argConstIntRanges,
                                                         joinCallback);
        })
        .Case<JoinOp, CatOp>([&](auto joinOp) {
          return inferResultRangesBinaryOpUnionArgRanges(
              joinOp, argConstIntRanges, joinCallback);
        })
        .Case<GatherOp>([&](GatherOp gatherOp) {
          return inferResultRanges(&gatherOp, argConstIntRanges, joinCallback);
        })
        .Default([&](auto) { llvm::report_fatal_error("unsupported op"); });
    return success();
  }

  // TODO: It looks like inferResultRangesFromOptional does not handle bunch
  //  of operations very well:
  //   - arith.shrui, e.g. arith.shrui %arg3, %c5_i32
  //
  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    inferrable.inferResultRangesFromOptional(argIntValueRanges, joinCallback);
    return success();
  }

  setAllToEntryStates(resultsLattices);
  return success();
}

void TritonIntegerRangeAnalysis::initializeFuncOp(tt::FuncOp op) {
  Block *entryBlock = getFuncEntryBlock(op);
  for (BlockArgument argument : op.getArguments()) {
    if (!this->assumptions.count(argument))
      continue;

    dataflow::IntegerValueRangeLattice *argLattice =
        getLatticeElement(argument);

    IntegerValueRange range = IntegerValueRange::getMaxRange(argument);
    if (auto maybeRange = maybeGetAssumedRange(argument, entryBlock))
      range = *maybeRange;

    // The lattice must in "bottom" state, The join() operation is to set the
    // state to the given "range".
    assert(argLattice->getValue().isUninitialized() &&
           "lattice must be in bottom state");
    (void)argLattice->join(range);
  }
}

void TritonIntegerRangeAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionBranchPoint successor,
    ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) {
  LLVM_DEBUG({
    DBGS() << "Visit Region Succesors of ";
    OpPrintingFlags flags;
    flags.skipRegions(true);
    branch.print(llvm::dbgs(), flags);
    llvm::dbgs() << "\n";
  });
  SmallVector<dataflow::IntegerValueRangeLattice *> lattices;
  for (auto abstractLat : abstractLattices) {
    lattices.push_back(
        static_cast<dataflow::IntegerValueRangeLattice *>(abstractLat));
  }
  // Initialize loop trip counts
  LoopLikeOpInterface loop =
      llvm::dyn_cast<LoopLikeOpInterface>(branch.getOperation());
  if (loop) {
    if (!loopTripCounts.contains(loop)) {
      loopTripCounts[loop] = std::numeric_limits<int64_t>::max();
      for (auto argLat : lattices)
        loopVisits[{loop, argLat}] = 0;
    }

    int64_t loopTripCount = getTotalLoopTripCount(loop);
    LLVM_DEBUG({
      DBGS() << "Trip count for ";
      OpPrintingFlags flags;
      flags.skipRegions(true);
      loop->print(llvm::dbgs(), flags);
      llvm::dbgs() << "\n";
      DBGS() << " --> " << loopTripCount << '\n';
    });
    if (loopTripCount < loopTripCounts[loop]) {
      loopTripCounts[loop] = loopTripCount;
    }
  }

  const auto *predecessors =
      getOrCreateFor<dataflow::PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");

  // Note: It does not seems to be quite obvious; this loop could update SCF
  // operations' LHS. e.g. If the given "branch" argument is scf.if, and the
  // scf.if construct looks like following:
  //   x = scf.if cond
  //    m = ... // op_m
  //    yield m
  //   else
  //    n = ... // op_n
  //    yield n
  //
  // This loop tries to update lattice(x) = join(lattice(m), lattice(n),
  // provided lattice(m) and lattice(n) are initialized.
  //
  // Note that the state of lattice(m) and lattice(n) was updated in the
  // "previous" round. In this "round", the scf.if is visitied right now, and
  // it takes this moment to update its LHS.
  //
  // Alternatively, when we visit, say op_m, we notice its result is used by
  // a yieldOp, get the yieldOp's corresponding receiver, in this case x, and
  // update its state accordingly.
  //
  for (Operation *op : predecessors->getKnownPredecessors()) {
    std::optional<OperandRange> operands;
    if (op == branch) {
      operands = branch.getEntrySuccessorOperands(successor);
    } else if (auto regionTerminator =
                   dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
      operands = regionTerminator.getSuccessorOperands(successor);
    }
    if (!operands)
      return setAllToEntryStates(lattices);

    ValueRange inputs = predecessors->getSuccessorInputs(op);
    assert(inputs.size() == operands->size() &&
           "expected the same number of successor inputs as operands");

    unsigned firstIndex = 0;
    if (inputs.size() != lattices.size()) {
      if (!point->isBlockStart()) {
        if (!inputs.empty()) {
          firstIndex = cast<OpResult>(inputs.front()).getResultNumber();
        }
        visitNonControlFlowArguments(branch,
                                     RegionSuccessor(branch->getResults().slice(
                                         firstIndex, inputs.size())),
                                     lattices, firstIndex);
      } else {
        if (!inputs.empty()) {
          firstIndex = cast<BlockArgument>(inputs.front()).getArgNumber();
        }
        Region *region = point->getBlock()->getParent();
        visitNonControlFlowArguments(
            branch,
            RegionSuccessor(region, region->getArguments().slice(
                                        firstIndex, inputs.size())),
            lattices, firstIndex);
      }
    }

    for (auto [oper, argLat] :
         llvm::zip(*operands, ArrayRef(lattices).drop_front(firstIndex))) {
      std::pair loopArgLat = {loop, argLat};
      // If we've "run the loop" #tripcount times, stop propagating.
      if (loop && loopVisits[loopArgLat] >= loopTripCounts[loop])
        continue;

      ChangeResult changed;
      if (loop && loopTripCounts[loop] > kDefaultMaxTripCount) {
        // If the loop's tripcount is too large, infer the maximum range for
        // the arg lattices. This will have the effect that all users will
        // also be inferred to have maximum range and end the analysis will
        // end (the maximum range is the "top" of the lattice and thus no
        // further changes/updates are possible).
        changed = argLat->join(IntegerValueRange::getMaxRange(oper));
      } else {
        // Else, propagate pred operands.
        auto operLat = *getLatticeElementFor(point, oper);
        changed = argLat->join(operLat);
        LLVM_DEBUG({
          if (changed == ChangeResult::Change) {
            DBGS() << "Operand lattice ";
            oper.printAsOperand(llvm::dbgs(), {});
            llvm::dbgs() << " --> " << operLat.getValue() << "\n";
          }
        });
      }
      propagateIfChanged(argLat, changed);
      // Only increase the loop visitation count if have actually update the
      // lattice because otherwise we will over count the number of visits
      // (since not all iter_arg lattices are updated/propagated on each
      // visit).
      if (loop && changed == ChangeResult::Change)
        ++loopVisits[loopArgLat];
    }
  }
}

DenseMap<Value, SetVector<Operation *>>
TritonIntegerRangeAnalysis::collectAssumptions(Operation *rootOp,
                                               bool filterConstants) {
  DenseMap<Value, SetVector<Operation *>> assumptions;
  rootOp->walk([&](LLVM::AssumeOp op) {
    auto assump = op.getCond().getDefiningOp();
    for (auto operand : assump->getOperands()) {
      if (filterConstants && getConstantIntValue(operand))
        continue;
      assumptions[operand].insert(assump);
    }
  });
  return assumptions;
}

struct FoldTrueCmpIOp : OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  FoldTrueCmpIOp(MLIRContext *context, DataFlowSolver *solver)
      : OpRewritePattern(context), solver(solver) {};

  LogicalResult matchAndRewrite(arith::CmpIOp cmpOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::isa<IntegerType, IndexType>(cmpOp.getType()) &&
        cmpIIsStaticallyTrue(*solver, cmpOp)) {
      if (failed(mlir::dataflow::maybeReplaceWithConstant(*solver, rewriter,
                                                          cmpOp.getResult()))) {
        LDBG("failed to replace with constant op: " << cmpOp);
        return failure();
      }
    } else {
      return failure();
    }
    return success();
  }

  DataFlowSolver *solver;
};

void populateFoldTrueCmpIOpPatterns(RewritePatternSet &patterns,
                                    DataFlowSolver *solver) {
  patterns.add<FoldTrueCmpIOp>(patterns.getContext(), solver);
}

void initializeFuncOps(Operation *op,
                       AMD::TritonIntegerRangeAnalysis *rangeAnalysis) {
  op->walk<WalkOrder::PreOrder>([&rangeAnalysis](FuncOp funcOp) {
    rangeAnalysis->initializeFuncOp(funcOp);
  });
}

} // namespace mlir::triton::AMD
