#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-convert-buffer-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using ::mlir::LLVM::AMD::getVectorSize;
using mlir::triton::AMD::ISAFamily;

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {

constexpr int64_t kDefaultMaxTripCount = 1024;
const std::string kConvertBufferOpsPrefix = "__amdgpuconvertbufferops.";
const std::string kOutputRange = kConvertBufferOpsPrefix + "output_range";

std::optional<int64_t> maybeGetTripCount(LoopLikeOpInterface loop) {
  std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
  std::optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
  std::optional<OpFoldResult> step = loop.getSingleStep();
  assert(lowerBound && upperBound && step);
  return constantTripCount(*lowerBound, *upperBound, *step);
}

void getEnclosingLoops(Operation &op, SmallVector<LoopLikeOpInterface> &ops) {
  Operation *currOp = op.getParentOp();
  while (currOp) {
    if (isa<LoopLikeOpInterface>(currOp))
      ops.push_back(llvm::cast<LoopLikeOpInterface>(currOp));
    currOp = currOp->getParentOp();
  }
}

void inferResultRanges(GetProgramIdOp *op, SetIntRangeFn setResultRange) {
  constexpr int64_t kDefaultMaxProgramID = 2048;
  setResultRange(
      op->getResult(),
      ConstantIntRanges::range(
          /*min*/ {/*numBits*/ 32, /*val*/ 0, /*isSigned*/ true},
          /*max*/
          {/*numBits*/ 32, /*val*/ kDefaultMaxProgramID, /*isSigned*/ true},
          /*isSigned*/ true));
}

void inferResultRanges(MakeRangeOp *op, SetIntRangeFn setResultRange) {
  setResultRange(
      op->getResult(),
      ConstantIntRanges::range(
          /*min*/ {/*numBits*/ 32, /*val*/ op->getStart(), /*isSigned*/ true},
          /*max*/ {/*numBits*/ 32, /*val*/ op->getEnd(), /*isSigned*/ true},
          /*isSigned*/ true));
}

void inferResultRanges(SplatOp *op, ArrayRef<ConstantIntRanges> argRanges,
                       SetIntRangeFn setResultRange) {
  setResultRange(op->getResult(), argRanges[0]);
}

void inferResultRanges(ExpandDimsOp *op, ArrayRef<ConstantIntRanges> argRanges,
                       SetIntRangeFn setResultRange) {
  setResultRange(op->getResult(), argRanges[0]);
}

/// This struct (analysis) adapt's upstream's IntegerRangeAnalysis (inferring
/// lower/upperbounds on integer constants) to our needs.
/// Specifically there are 2 points of extension:
///
/// 1. Support for GetProgramIdOp, MakeRangeOp, SplatOp, ExpandDimsOp. *Note*,
/// upstream already supports range inference for shaped types such as tensors
/// (here we just implement effectively implement the interfaces for our ops).
///    * Upstream's semantics for "range of shape type" is union over ranges of
///    elements.
///    * We do not use tablegen to implement
///    DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>
///    in order to keep the entire implementation contained/encapsulated.
///
/// 2. Support for inference "through loops". Upstream's analysis conservatively
/// inferences [min_int, max_int] for loop carried values (and therefore loop
/// body values). Here we attempt to do better by analysis the loop bounds and
/// "abstractly interpreting" the loop when loop bounds are statically known.
/// See visitRegionSuccessors.
struct TritonIntegerRangeAnalysis : dataflow::IntegerRangeAnalysis {
  using dataflow::IntegerRangeAnalysis::IntegerRangeAnalysis;

  llvm::SmallDenseMap<LoopLikeOpInterface, int64_t> loopTripCounts;
  llvm::SmallDenseMap<
      std::pair<LoopLikeOpInterface, dataflow::IntegerValueRangeLattice *>,
      int64_t>
      loopVisits;

  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> results) override {
    LDBG("  Inferring ranges for " << *op << "\n");
    // This callback is almost exactly like the callback in
    // IntegerRangeAnalysis::visitOperation except we do not "short-cicruit" the
    // analysis by inferring a maximum range for loop results (instead we
    // perform a check based on visit counts in visitRegionSuccessors).
    auto joinCallback =
        [&op, &results, this](Value v, const IntegerValueRange &incomingRange) {
          auto result = dyn_cast<OpResult>(v);
          if (!result)
            return;
          assert(llvm::is_contained(op->getResults(), result));

          LDBG("  Inferred range " << incomingRange << "\n");
          dataflow::IntegerValueRangeLattice *lattice =
              results[result.getResultNumber()];
          IntegerValueRange oldRange = lattice->getValue();
          ChangeResult changed = lattice->join(incomingRange);
          propagateIfChanged(lattice, changed);
        };

    if (llvm::isa<GetProgramIdOp, MakeRangeOp, SplatOp, ExpandDimsOp>(op)) {
      SmallVector<ConstantIntRanges> argRanges;
      for (auto lattice : operands) {
        if (lattice->getValue().isUninitialized()) {
          setAllToEntryStates(results);
          return success();
        }
        argRanges.push_back(lattice->getValue().getValue());
      }
      if (auto op_ = llvm::dyn_cast<GetProgramIdOp>(op))
        inferResultRanges(&op_, joinCallback);
      else if (auto op_ = llvm::dyn_cast<SplatOp>(op))
        inferResultRanges(&op_, argRanges, joinCallback);
      else if (auto op_ = llvm::dyn_cast<ExpandDimsOp>(op))
        inferResultRanges(&op_, argRanges, joinCallback);
      else if (auto op_ = llvm::dyn_cast<MakeRangeOp>(op))
        inferResultRanges(&op_, joinCallback);
      else {
        llvm_unreachable("Unsupported operation");
      }
      return success();
    }

    auto inferrable = dyn_cast<InferIntRangeInterface>(op);
    if (!inferrable) {
      setAllToEntryStates(results);
      return success();
    }
    SmallVector<IntegerValueRange> argRanges = llvm::map_to_vector(
        operands, [](const dataflow::IntegerValueRangeLattice *lattice) {
          return lattice->getValue();
        });

    inferrable.inferResultRangesFromOptional(argRanges, joinCallback);
    return success();
  }

  /// This method (which overloads
  /// AbstractSparseForwardDataFlowAnalysis::visitRegionSuccessors) implements
  /// "abstract interpretation" of loops with statically known bounds in order
  /// to infer tight ranges for loop carried values (and therefore loop body
  /// values). By "abstract interpretation" we mean lattice states are
  /// propagated to all region successors N times, where N is the total trip
  /// count of the loop. Recall for scf.for, both the loop itself and the users
  /// of the loop successors. Thus, after N propagations both loop body values
  /// and users of loop results will have accurate ranges (assuming we have
  /// implemented support for range analysis on the ops).
  /// *Note*, this implementation is majority similar to
  /// AbstractSparseForwardDataFlowAnalysis::visitRegionSuccessors (so check
  /// there for more explanation/insight) and basically only does two things
  /// differently:
  ///
  /// 1. If the branch op is a loop (LoopLikeOpInterface) then we attempt to
  /// compute its total trip count (nested loop trip counts multiply) and
  /// initialize a visit count to 0. Note, due to how Dataflow analysis works we
  /// have to actually visit the loop N times for each iter_arg (each argument
  /// lattice) so we actually track visit count for (loop, arg) not just (loop).
  ///
  /// 2. Before propagating, we check if we have propagated for (loop, arg) >= N
  /// times. If so, we do not propagate (and thus the traversal converges/ends).
  ///
  /// Note, for loops where the trip count cannot be inferred *and* loops with a
  /// total trip count larger than `kDefaultMaxTripCount`, fallback to
  /// upstream's conservative inference (i.e., we infer [min_int, max_int]) for
  /// the loop operands and all users and all users of the results of the loop.
  void visitRegionSuccessors(
      ProgramPoint *point, RegionBranchOpInterface branch,
      RegionBranchPoint successor,
      ArrayRef<dataflow::AbstractSparseLattice *> abstractLattices) override {
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
        SmallVector loops{loop};
        getEnclosingLoops(*loop, loops);
        int loopTripCount =
            std::accumulate(loops.begin(), loops.end(), 1,
                            [](int accum, LoopLikeOpInterface loop) {
                              return accum * maybeGetTripCount(loop).value_or(
                                                 kDefaultMaxTripCount);
                            });
        loopTripCounts[loop] = loopTripCount;
      }
      for (auto argLat : lattices) {
        if (!loopVisits.contains({loop, argLat})) {
          loopVisits[{loop, argLat}] = 0;
        }
      }
    }

    const auto *predecessors =
        getOrCreateFor<dataflow::PredecessorState>(point, point);
    assert(predecessors->allPredecessorsKnown() &&
           "unexpected unresolved region successors");
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
          visitNonControlFlowArguments(
              branch,
              RegionSuccessor(
                  branch->getResults().slice(firstIndex, inputs.size())),
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
          changed = argLat->join(*getLatticeElementFor(point, oper));
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
};

void collectRanges(DataFlowSolver &solver, ValueRange values,
                   SmallVectorImpl<ConstantIntRanges> &ranges) {
  for (Value val : values) {
    auto *maybeInferredRange =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(val);
    if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
      return;
    const ConstantIntRanges &inferredRange =
        maybeInferredRange->getValue().getValue();
    ranges.push_back(inferredRange);
  }
}

void collectRanges(std::shared_ptr<DataFlowSolver> solver, ModuleOp mod) {
  mod->walk<WalkOrder::PreOrder>([&solver, &mod](Operation *op) {
    SmallVector<ConstantIntRanges> outputRange;
    collectRanges(*solver, op->getResults(), outputRange);
    if (!outputRange.empty()) {
      APInt min = outputRange[0].smin();
      APInt max = outputRange[0].smax();
      IntegerType i64ty =
          IntegerType::get(mod.getContext(), 64, IntegerType::Signless);
      SmallVector<Attribute> range = {
          IntegerAttr::get(i64ty, min.getSExtValue()),
          IntegerAttr::get(i64ty, max.getSExtValue()),
      };
      op->setAttr(kOutputRange, ArrayAttr::get(mod.getContext(), range));
    }
  });
}

bool verifyNonSmallerByAssumption(
    Value expr, const DenseSet<Value> &assumptions,
    const std::function<bool(Value)> &matchesOther) {
  for (Value assume : assumptions) {
    if (auto cmpOp = assume.getDefiningOp<arith::CmpIOp>()) {
      switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::eq:
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::sgt: {
        if (cmpOp.getLhs() == expr && matchesOther(cmpOp.getRhs())) {
          LDBG("  " << expr << " non-neg by assumption " << cmpOp);
          return true;
        }
        break;
      }
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::slt: {
        if (cmpOp.getRhs() == expr && matchesOther(cmpOp.getLhs())) {
          LDBG("  " << expr << " non-neg by assumption " << cmpOp);
          return true;
        }
        break;
      }
      default:
        break;
      }
    }
  }
  return false;
}

bool verifyNonNegativeByAssumption(Value expr,
                                   const DenseSet<Value> &assumptions) {
  return verifyNonSmallerByAssumption(expr, assumptions, [](auto otherExpr) {
    APInt cst;
    return matchPattern(otherExpr, m_ConstantInt(&cst)) && cst.isNonNegative();
  });
}

bool verifyNonSmallerByAssumption(Value expr,
                                  const DenseSet<Value> &assumptions,
                                  Value other) {
  return verifyNonSmallerByAssumption(
      expr, assumptions, [&](auto otherAssum) { return otherAssum == other; });
}

bool verifyNonNegativeExpr(Value expr, const DenseSet<Value> &assumptions) {
  LDBG("Determing if non-negative: " << expr);

  if (!llvm::isa<mlir::BlockArgument>(expr)) {
    if (auto outputR = llvm::dyn_cast_if_present<mlir::ArrayAttr>(
            expr.getDefiningOp()->getAttr(kOutputRange))) {
      assert(outputR.size() == 2 && llvm::isa<mlir::IntegerAttr>(outputR[0]) &&
             "expected output_range to have 2 integerAttr entries");
      auto lb = llvm::cast<mlir::IntegerAttr>(outputR[0]);
      if (lb.getValue().sge(0))
        return true;
    }
  }

  // Check if the expression is contained in any assumption
  if (verifyNonNegativeByAssumption(expr, assumptions)) {
    return true;
  }

  // Recurse if the operation is defined
  Operation *op = expr.getDefiningOp();
  if (!op) {
    LDBG("  No defining op, assuming possibly negative");
    return false;
  }

  bool nonNegative =
      llvm::TypeSwitch<Operation *, bool>(expr.getDefiningOp())
          // Various unary triton ops that don't change the sign of the operand
          .Case<triton::TransOp, triton::SplitOp, triton::BroadcastOp,
                triton::ExpandDimsOp, triton::SplatOp, triton::ReshapeOp,
                triton::gpu::ConvertLayoutOp>([&](auto unaryOp) {
            return verifyNonNegativeExpr(unaryOp.getOperand(), assumptions);
          })
          .Case<triton::GatherOp>([&](auto gatherOp) {
            return verifyNonNegativeExpr(gatherOp.getSrc(), assumptions);
          })
          // Joining two non-negative tensors is still non-negative
          .Case<triton::JoinOp, triton::CatOp>([&](auto joinOp) {
            return verifyNonNegativeExpr(joinOp.getLhs(), assumptions) &&
                   verifyNonNegativeExpr(joinOp.getRhs(), assumptions);
          })
          // Returns a tensor representing histogram: historgrams only contain
          // buckets of non-negative values.
          .Case<triton::HistogramOp>([&](auto) { return true; })
          .Case<triton::MakeRangeOp>([&](auto makeRangeOp) {
            // See the warning in TritonOps.td: getStart/getEnd return unsigned,
            // so we need to look through get*Attr.
            return makeRangeOp.getStartAttr().getInt() >= 0 &&
                   makeRangeOp.getEndAttr().getInt() >= 0;
          })
          .Case<arith::ConstantIntOp>(
              [&](auto constIntOp) { return constIntOp.value() >= 0; })
          .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
            Value val = constOp.getResult();
            DenseIntElementsAttr constVal;
            if (matchPattern(val, m_Constant(&constVal)) && constVal.isSplat())
              return constVal.getSplatValue<APInt>().isNonNegative();
            return false;
          })
          .Case<triton::GetNumProgramsOp, triton::GetProgramIdOp>([&](auto) {
            // These are defined as signless, but are actually unsigned
            return true;
          })
          .Case<arith::MaxSIOp>([&](auto maxOp) {
            // max(a,b) >= 0 iff a>=0 || b>=0
            return verifyNonNegativeExpr(maxOp.getLhs(), assumptions) ||
                   verifyNonNegativeExpr(maxOp.getRhs(), assumptions);
          })
          .Case<arith::RemSIOp>([&](auto remsiOp) {
            // a % b >= 0 iff a>=0
            return verifyNonNegativeExpr(remsiOp.getLhs(), assumptions);
          })
          .Case<arith::TruncIOp, arith::ExtSIOp>([&](Operation *unaryOp) {
            // a = OP b >= 0 iff b >= 0
            return verifyNonNegativeExpr(unaryOp->getOperand(0), assumptions);
          })
          // Casting from arbitrary data does *not* guarantee the offset is in
          // range (even if pointer, or the data is non-negative when
          // interpreted as the src's type).
          .Case<triton::PtrToIntOp, triton::BitcastOp>(
              [&](auto) { return false; })
          .Case<arith::CeilDivUIOp, arith::DivUIOp, arith::ExtUIOp,
                arith::FPToUIOp, arith::MaxUIOp, arith::MinUIOp, arith::RemUIOp,
                arith::ShRUIOp>(
              // These OPs also return unsigned values.
              // TODO: We can also sniff whether a Value is unsigned by looking
              //       for whether or not it's used as an argument to one of
              //       these OPs.
              [&](auto uOp) { return true; })
          .Case<arith::AddIOp, arith::MinSIOp, arith::MulIOp, arith::DivSIOp>(
              // Generally speaking, a OP b >= 0  iff  a >= 0 && b >= 0 when
              // OP != sub
              [&](Operation *binOp) {
                return verifyNonNegativeExpr(binOp->getOperand(0),
                                             assumptions) &&
                       verifyNonNegativeExpr(binOp->getOperand(1), assumptions);
              })
          // TODO: more scf
          .Case<scf::IfOp>([&](auto ifOp) {
            auto results = ifOp.getResults();
            auto it = std::find(results.begin(), results.end(), expr);
            assert(it != results.end() && "expr should be the result of ifOp");
            auto resultIdx = it - results.begin();

            // If we're here then we must have both then/else regions
            // (each with 1 block) and each region must terminate with an
            // `scf.yield` expression.
            auto thenYield = cast<scf::YieldOp>(ifOp.thenYield());
            auto elseYield = cast<scf::YieldOp>(ifOp.elseYield());
            return verifyNonNegativeExpr(thenYield->getOperand(resultIdx),
                                         assumptions) &&
                   verifyNonNegativeExpr(elseYield->getOperand(resultIdx),
                                         assumptions);
          })
          .Case<arith::SubIOp>([&](auto op) {
            // If a user annotates tl.assume(a >= b) then we know a - b >= 0
            return verifyNonSmallerByAssumption(op.getLhs(), assumptions,
                                                op.getRhs());
          })
          .Default([&](Operation *op) {
            // Conservatively assume that the expression is negative
            LDBG("  Unhandled op, cannot assume non-negative");
            return false;
          });
  return nonNegative;
}

// Quick analysis on the Triton IR to decide if we can safely use
// buffer operations
bool canUseBufferOps(Value ptr, const DenseSet<Value> &assumptions) {
  // 1. Check if the pointer is uniform: i.e., if it comes from a uniform
  // pointer(splatted) and non-uniform offset addition

  LDBG("Buffer op checks for: " << ptr);
  auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp)
    return false;

  auto maybeSplatOp = addPtrOp.getPtr().getDefiningOp<triton::SplatOp>();
  if (!maybeSplatOp)
    return false;
  LDBG("Pattern matched");

  // 2. Check if the offset is a 32-bit tensor
  Value offset = addPtrOp.getOffset();
  if (cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth() != 32)
    return false;
  LDBG("32 bit offset");

  return verifyNonNegativeExpr(offset, assumptions);
}

// Extract stride of the blocked offset of LD/ST ops.
Value getBlockStride(Location loc, Value offset, PatternRewriter &rewriter) {
  // canonicalize pointer pass sets block stride via
  // `offset:add-broadcast-muli-splat`, backtrace that pattern to reach the
  // stride.
  if (auto maybeAdd = offset.getDefiningOp<arith::AddIOp>()) {
    for (auto addOpr : maybeAdd.getOperands()) {
      if (auto maybeBC = addOpr.getDefiningOp<tt::BroadcastOp>()) {
        auto bcSrc = maybeBC.getSrc();
        if (auto maybeMul = bcSrc.getDefiningOp<arith::MulIOp>()) {
          for (auto mulOpr : maybeMul.getOperands()) {
            if (auto maybeSplat = mulOpr.getDefiningOp<tt::SplatOp>()) {
              return maybeSplat.getSrc();
            }
          }
        }
      }
    }
  }
  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  ;
}

} // namespace

struct ConvertTritonAtomicRMWOpToBufferAtomicRMW
    : public mlir::OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicRMWOpToBufferAtomicRMW(
      mlir::MLIRContext *context, DenseSet<Value> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass)
      : mlir::OpRewritePattern<triton::AtomicRMWOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass) {}

  mlir::LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();
    auto atomicRmwOp = op.getAtomicRmwOp();
    auto sem = op.getSem();
    auto scope = op.getScope();

    // In addition to the `canUserBufferOps` check, we should ensure that
    // 1. Perform the canUserBufferOps check
    if (!canUseBufferOps(ptr, assumptions)) {
      return rewriter.notifyMatchFailure(op, "canUseBufferOps check failed");
    }

    // 2. Check the scope. We support GPU and CTA for now (SYSTEM scope is not
    // supported yet)
    switch (scope) {
    case MemSyncScope::GPU:
    case MemSyncScope::CTA:
      break;
    default:
      return rewriter.notifyMatchFailure(op, "RMW with unsupported scope");
    }
    LDBG("RMW supported scope");

    // 3. Check the memory ordering.
    //    TODO: support monotonic
    switch (sem) {
    case MemSemantic::RELAXED:
    case MemSemantic::RELEASE:
    case MemSemantic::ACQUIRE:
    case MemSemantic::ACQUIRE_RELEASE:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "RMW with unsupported memory ordering");
    }

    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    Value tensorPtr = addPtrOp.getPtr();
    Value tensorOffset = addPtrOp.getOffset();
    auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
    Value basePtr = splatOp.getSrc();

    // 4. Buffer atomic RMW does not support FP8 ops
    //    easier to just check what we support
    auto checkType = getElementTypeOrSelf(op.getVal());
    bool isSupportedType = checkType.isF16() || checkType.isBF16() ||
                           checkType.isF32() || checkType.isF64() ||
                           checkType.isInteger(32) || checkType.isInteger(64);
    if (!isSupportedType) {
      return rewriter.notifyMatchFailure(op, "RMW with unsupported type");
    }
    LDBG("RMW supported type");

    // 5. Check if the RMWOp is supported
    switch (atomicRmwOp) {
    case RMWOp::AND:
    case RMWOp::OR:
    case RMWOp::XOR:
    case RMWOp::ADD:
    case RMWOp::FADD:
    case RMWOp::MAX:
    case RMWOp::MIN:
    case RMWOp::UMAX:
    case RMWOp::UMIN:
    case RMWOp::XCHG:
      break;
    default:
      auto rmwOpStr = stringifyRMWOp(atomicRmwOp).str();
      return rewriter.notifyMatchFailure(op, "RMW with unsupported op: " +
                                                 rmwOpStr);
    }
    LDBG("RMW supported Op");

    // 6. Buffer atomics support 32 and 64-bit operations, so inputs must be at
    //    least 32-bits. Otherwise, fall back to the existing path for atomics
    auto opValueType = op.getVal().getType();
    auto opBitWidth = 0;
    if (auto tensorType = dyn_cast<RankedTensorType>(opValueType)) {
      // We can't just compute the opBitWidth using the numElements *
      // elemBitWidth here. In cases such as tensor<2xf16...>, if the elements
      // are contiguous we can emit the buffer op. Otherwise, the buffer ops
      // lowering will try to emit individual (unsupported) f16/bf16 ops.
      auto elemBitWidth = tensorType.getElementTypeBitWidth();
      opBitWidth =
          getVectorSize(basePtr, tensorOffset, axisAnalysisPass) * elemBitWidth;
    } else {
      opBitWidth = opValueType.getIntOrFloatBitWidth();
    }

    if (opBitWidth < 32) {
      return rewriter.notifyMatchFailure(op, "RMW requires opBitWidth >= 32");
    }

    Value maybeMask{};
    if (op.getMask() && !isZeroConst(op.getMask()))
      maybeMask = op.getMask();

    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferAtomicRMWOp>(
        op, op.getVal().getType(), atomicRmwOp, basePtr, tensorOffset,
        op.getVal(), sem, scope, maybeMask);

    return success();
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct ConvertTritonLoadToBufferLoad
    : public mlir::OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonLoadToBufferLoad(mlir::MLIRContext *context,
                                DenseSet<Value> &assumptions)
      : mlir::OpRewritePattern<triton::LoadOp>(context),
        assumptions(assumptions) {}

  mlir::LogicalResult
  matchAndRewrite(triton::LoadOp op, PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (canUseBufferOps(ptr, assumptions)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeOther{};
      if (op.getOther() && !isZeroConst(op.getOther()))
        maybeOther = op.getOther();
      Value maybeMask{};
      if (op.getMask() && !isZeroConst(op.getMask()))
        maybeMask = op.getMask();
      Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);
      auto bufferLoadOp = rewriter.create<triton::amdgpu::BufferLoadOp>(
          op->getLoc(), op.getType(), basePtr, tensorOffset, blockStride,
          op.getCache(), maybeMask, maybeOther);

      // Propagate `OpIdxAttr` if the currently processed `tt.LoadOp` was
      // labeled it. The attribute needs to be preserved for custom instruction
      // scheduling.
      if (auto opIdxAttr = op->getAttrOfType<triton::amdgpu::OpIdxAttr>(
              triton::amdgpu::OpIdxAttr::getMnemonic())) {
        bufferLoadOp->setAttr(triton::amdgpu::OpIdxAttr::getMnemonic(),
                              opIdxAttr);
      }
      rewriter.replaceOp(op, bufferLoadOp);
      return success();
    }

    LDBG("Failed to convert: " << op);
    return rewriter.notifyMatchFailure(op, "Failed to convert LoadOp");
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
};

struct ConvertTritonStoreToBufferStore
    : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonStoreToBufferStore(mlir::MLIRContext *context,
                                  DenseSet<Value> &assumptions)
      : mlir::OpRewritePattern<triton::StoreOp>(context),
        assumptions(assumptions) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (canUseBufferOps(ptr, assumptions)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeMask{};
      if (op.getMask() && !isZeroConst(op.getMask()))
        maybeMask = op.getMask();
      Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);
      rewriter.replaceOpWithNewOp<triton::amdgpu::BufferStoreOp>(
          op, op.getValue(), basePtr, tensorOffset, blockStride, op.getCache(),
          maybeMask);
      return success();
    }
    LDBG("Failed to convert: " << op);
    return rewriter.notifyMatchFailure(op, "Failed to convert StoreOp");
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
};

class TritonAMDGPUConvertToBufferOpsPass
    : public TritonAMDGPUConvertToBufferOpsBase<
          TritonAMDGPUConvertToBufferOpsPass> {

public:
  TritonAMDGPUConvertToBufferOpsPass() = default;
  TritonAMDGPUConvertToBufferOpsPass(StringRef archGen) {
    this->archGenerationName = archGen.data();
  };
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();

    // Collect assumptions in the function
    DenseSet<Value> assumptions;
    mod.walk([&](LLVM::AssumeOp op) {
      if (op->getOperand(0).getDefiningOp<arith::CmpIOp>())
        assumptions.insert(op->getOperand(0));
    });
    LLVM_DEBUG({
      DBGS() << "Number of assumptions found: " << assumptions.size() << "\n";
      for (Value assume : assumptions) {
        DBGS() << "Assumption:" << assume << "\n";
      }
    });

    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<TritonIntegerRangeAnalysis>();
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();
    collectRanges(solver, mod);

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    patterns.add<ConvertTritonLoadToBufferLoad>(context, assumptions);
    patterns.add<ConvertTritonStoreToBufferStore>(context, assumptions);

    // Gate buffer atomics behind CDNA3 (i.e., MI300 series) for now
    // GFX942-specific assumptions regarding cache coherence are made when
    // lowering to LLVM
    if (ISAFamily::CDNA3 == triton::AMD::deduceISAFamily(archGenerationName))
      patterns.add<ConvertTritonAtomicRMWOpToBufferAtomicRMW>(
          context, assumptions, axisInfoAnalysis);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUConvertToBufferOpsPass(std::string archGen) {
  return std::make_unique<TritonAMDGPUConvertToBufferOpsPass>(archGen);
}
