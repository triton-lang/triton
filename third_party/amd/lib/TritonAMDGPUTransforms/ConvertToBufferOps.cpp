#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
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
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#include <deque>
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

/// Gather ranges for all the values in `values`. Appends to the existing
/// vector.
static LogicalResult collectRanges(DataFlowSolver &solver, ValueRange values,
                                   SmallVectorImpl<ConstantIntRanges> &ranges) {
  for (Value val : values) {
    auto *maybeInferredRange =
        solver.lookupState<dataflow::IntegerValueRangeLattice>(val);
    if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
      return failure();

    const ConstantIntRanges &inferredRange =
        maybeInferredRange->getValue().getValue();
    ranges.push_back(inferredRange);
  }
  return success();
}

namespace {

#undef smax
#undef smin

struct MyIntegerRangeAnalysis : DataFlowAnalysis {
  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint *point) override;
  explicit MyIntegerRangeAnalysis(DataFlowSolver &solver);

  void join(dataflow::IntegerValueRangeLattice *lhs,
            const dataflow::IntegerValueRangeLattice &rhs) {
    propagateIfChanged(lhs, lhs->join(rhs));
  }

  LogicalResult initializeRecursively(Operation *op);

  void visitBlock(Block *block);

  void visitRegionSuccessors(
      ProgramPoint *point, RegionBranchOpInterface branch,
      RegionBranchPoint successor,
      ArrayRef<dataflow::IntegerValueRangeLattice *> lattices);

  using StateT = dataflow::IntegerValueRangeLattice;

  virtual void visitExternalCall(CallOpInterface call,
                                 ArrayRef<const StateT *> argumentLattices,
                                 ArrayRef<StateT *> resultLattices) {
    setAllToEntryStates(resultLattices);
  }

  void visitNonControlFlowArguments(Operation *op,
                                    const RegionSuccessor &successor,
                                    ArrayRef<StateT *> argLattices,
                                    unsigned firstIndex);

  StateT *getLatticeElement(Value value) { return getOrCreate<StateT>(value); }

  const StateT *getLatticeElementFor(ProgramPoint *point, Value value) {
    dataflow::IntegerValueRangeLattice *state = getLatticeElement(value);
    addDependency(state, point);
    return state;
  }

  void setAllToEntryStates(ArrayRef<StateT *> lattices) {
    for (dataflow::IntegerValueRangeLattice *lattice : lattices)
      setToEntryState(lattice);
  }

  LogicalResult visitOperation(Operation *op);
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
                 ArrayRef<dataflow::IntegerValueRangeLattice *> results);

  void visitExternalCallImpl(
      CallOpInterface call,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> argumentLattices,
      ArrayRef<dataflow::IntegerValueRangeLattice *> resultLattices) {
    visitExternalCall(
        call,
        {reinterpret_cast<const StateT *const *>(argumentLattices.begin()),
         argumentLattices.size()},
        {reinterpret_cast<StateT *const *>(resultLattices.begin()),
         resultLattices.size()});
  }

  void setToEntryState(dataflow::IntegerValueRangeLattice *lattice) {
    propagateIfChanged(lattice, lattice->join(IntegerValueRange::getMaxRange(
                                    lattice->getAnchor())));
  }
};

MyIntegerRangeAnalysis::MyIntegerRangeAnalysis(DataFlowSolver &solver)
    : DataFlowAnalysis(solver) {
  registerAnchorKind<dataflow::CFGEdge>();
}

LogicalResult MyIntegerRangeAnalysis::initialize(Operation *top) {
  // Mark the entry block arguments as having reached their pessimistic
  // fixpoints.
  for (Region &region : top->getRegions()) {
    if (region.empty())
      continue;
    for (Value argument : region.front().getArguments())
      setToEntryState(getLatticeElement(argument));
  }

  return initializeRecursively(top);
}

LogicalResult MyIntegerRangeAnalysis::initializeRecursively(Operation *op) {
  // Initialize the analysis by visiting every owner of an SSA value (all
  // operations and blocks).
  if (failed(visitOperation(op)))
    return failure();

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      getOrCreate<dataflow::Executable>(getProgramPointBefore(&block))
          ->blockContentSubscribe(this);
      visitBlock(&block);
      for (Operation &op : block)
        if (failed(initializeRecursively(&op)))
          return failure();
    }
  }

  return success();
}

LogicalResult MyIntegerRangeAnalysis::visit(ProgramPoint *point) {
  if (!point->isBlockStart())
    return visitOperation(point->getPrevOp());
  visitBlock(point->getBlock());
  return success();
}

static int visits = 0;

int64_t maybeGetTripCount(Operation *op) {
  int64_t tripCount = 2 << 20;
  if (LoopLikeOpInterface loop = dyn_cast<LoopLikeOpInterface>(op)) {
    std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
    std::optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
    std::optional<OpFoldResult> step = loop.getSingleStep();
    assert(lowerBound && upperBound && step);
    auto maybeTripCout = constantTripCount(*lowerBound, *upperBound, *step);
    assert(maybeTripCout);
    tripCount = *maybeTripCout;
  }
  return tripCount;
}

LogicalResult MyIntegerRangeAnalysis::visitOperation(Operation *op) {
  // Exit early on operations with no results.
  if (op->getNumResults() == 0)
    return success();

  // If the containing block is not executable, bail out.
  if (op->getBlock() != nullptr &&
      !getOrCreate<dataflow::Executable>(getProgramPointBefore(op->getBlock()))
           ->isLive())
    return success();

  // Get the result lattices.
  SmallVector<dataflow::IntegerValueRangeLattice *> resultLattices;
  resultLattices.reserve(op->getNumResults());
  for (Value result : op->getResults()) {
    dataflow::IntegerValueRangeLattice *resultLattice =
        getLatticeElement(result);
    resultLattices.push_back(resultLattice);
  }

  // The results of a region branch operation are determined by control-flow.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "visits : " << visits << "\n");
    if (visits < maybeGetTripCount(op)) {
      visitRegionSuccessors(getProgramPointAfter(branch), branch,
                            /*successor=*/RegionBranchPoint::parent(),
                            resultLattices);
      ++visits;
    }
    return success();
  }

  // Grab the lattice elements of the operands.
  SmallVector<const dataflow::IntegerValueRangeLattice *> operandLattices;
  operandLattices.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    dataflow::IntegerValueRangeLattice *operandLattice =
        getLatticeElement(operand);
    operandLattice->useDefSubscribe(this);
    operandLattices.push_back(operandLattice);
  }

  if (auto call = dyn_cast<CallOpInterface>(op)) {
    // If the call operation is to an external function, attempt to infer the
    // results from the call arguments.
    auto callable =
        dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
    if (!getSolverConfig().isInterprocedural() ||
        (callable && !callable.getCallableRegion())) {
      visitExternalCallImpl(call, operandLattices, resultLattices);
      return success();
    }

    // Otherwise, the results of a call operation are determined by the
    // callgraph.
    const auto *predecessors = getOrCreateFor<dataflow::PredecessorState>(
        getProgramPointAfter(op), getProgramPointAfter(call));
    // If not all return sites are known, then conservatively assume we can't
    // reason about the data-flow.
    if (!predecessors->allPredecessorsKnown()) {
      setAllToEntryStates(resultLattices);
      return success();
    }
    for (Operation *predecessor : predecessors->getKnownPredecessors())
      for (auto &&[operand, resLattice] :
           llvm::zip(predecessor->getOperands(), resultLattices))
        join(resLattice,
             *getLatticeElementFor(getProgramPointAfter(op), operand));
    return success();
  }

  // Invoke the operation transfer function.
  return visitOperation(op, operandLattices, resultLattices);
}

void MyIntegerRangeAnalysis::visitBlock(Block *block) {
  // Exit early on blocks with no arguments.
  if (block->getNumArguments() == 0)
    return;

  // If the block is not executable, bail out.
  // executable == not DCE
  if (!getOrCreate<dataflow::Executable>(getProgramPointBefore(block))
           ->isLive())
    return;

  // Get the argument lattices.
  SmallVector<dataflow::IntegerValueRangeLattice *> argLattices;
  argLattices.reserve(block->getNumArguments());
  for (BlockArgument argument : block->getArguments()) {
    dataflow::IntegerValueRangeLattice *argLattice =
        getLatticeElement(argument);
    argLattices.push_back(argLattice);
  }

  // The argument lattices of entry blocks are set by region control-flow or the
  // callgraph.
  if (block->isEntryBlock()) {
    // Check if this block is the entry block of a callable region.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      const auto *callsites = getOrCreateFor<dataflow::PredecessorState>(
          getProgramPointBefore(block), getProgramPointAfter(callable));
      // If not all callsites are known, conservatively mark all lattices as
      // having reached their pessimistic fixpoints.
      if (!callsites->allPredecessorsKnown() ||
          !getSolverConfig().isInterprocedural()) {
        return setAllToEntryStates(argLattices);
      }
      for (Operation *callsite : callsites->getKnownPredecessors()) {
        auto call = cast<CallOpInterface>(callsite);
        for (auto it : llvm::zip(call.getArgOperands(), argLattices))
          join(std::get<1>(it),
               *getLatticeElementFor(getProgramPointBefore(block),
                                     std::get<0>(it)));
      }
      return;
    }

    // Check if the lattices can be determined from region control flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
      LLVM_DEBUG(llvm::dbgs() << "visits : " << visits << "\n");
      if (visits < maybeGetTripCount(branch.getOperation())) {
        visitRegionSuccessors(getProgramPointBefore(block), branch,
                              /*successor*/ block->getParent(), argLattices);
      }
    }

    // Otherwise, we can't reason about the data-flow.
    return visitNonControlFlowArguments(block->getParentOp(),
                                        RegionSuccessor(block->getParent()),
                                        argLattices, /*firstIndex=*/0);
  }

  // Iterate over the predecessors of the non-entry block.
  for (Block::pred_iterator it = block->pred_begin(), e = block->pred_end();
       it != e; ++it) {
    Block *predecessor = *it;

    // If the edge from the predecessor block to the current block is not live,
    // bail out.
    auto *edgeExecutable = getOrCreate<dataflow::Executable>(
        getLatticeAnchor<dataflow::CFGEdge>(predecessor, block));
    edgeExecutable->blockContentSubscribe(this);
    if (!edgeExecutable->isLive())
      continue;

    // Check if we can reason about the data-flow from the predecessor.
    if (auto branch =
            dyn_cast<BranchOpInterface>(predecessor->getTerminator())) {
      SuccessorOperands operands =
          branch.getSuccessorOperands(it.getSuccessorIndex());
      for (auto [idx, lattice] : llvm::enumerate(argLattices)) {
        if (Value operand = operands[idx]) {
          join(lattice,
               *getLatticeElementFor(getProgramPointBefore(block), operand));
        } else {
          // Conservatively consider internally produced arguments as entry
          // points.
          setAllToEntryStates(lattice);
        }
      }
    } else {
      return setAllToEntryStates(argLattices);
    }
  }
}

void MyIntegerRangeAnalysis::visitRegionSuccessors(
    ProgramPoint *point, RegionBranchOpInterface branch,
    RegionBranchPoint successor,
    ArrayRef<dataflow::IntegerValueRangeLattice *> lattices) {
  const auto *predecessors =
      getOrCreateFor<dataflow::PredecessorState>(point, point);
  LLVM_DEBUG({
    llvm::dbgs() << "*************entering visitRegionSuccessors************\n";
    llvm::dbgs() << "point: " << *point << "\n";
    if (auto succ = successor.getRegionOrNull()) {
      llvm::dbgs() << "successor: ";
      succ->getParentOp()->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    }
  });

  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");
  LLVM_DEBUG(llvm::dbgs() << "num preds: "
                          << predecessors->getKnownPredecessors().size()
                          << "\n\n");

  for (Operation *op : predecessors->getKnownPredecessors()) {
    // Get the incoming successor operands.
    std::optional<OperandRange> operands;

    // Check if the predecessor is the parent op.
    if (op == branch) {
      operands = branch.getEntrySuccessorOperands(successor);
      // Otherwise, try to deduce the operands from a region return-like op.
    } else if (auto regionTerminator =
                   dyn_cast<RegionBranchTerminatorOpInterface>(op)) {
      operands = regionTerminator.getSuccessorOperands(successor);
    }

    if (!operands) {
      // We can't reason about the data-flow.
      return setAllToEntryStates(lattices);
    }

    ValueRange inputs = predecessors->getSuccessorInputs(op);
    assert(inputs.size() == operands->size() &&
           "expected the same number of successor inputs as operands");

    unsigned firstIndex = 0;
    if (inputs.size() != lattices.size()) {
      if (!point->isBlockStart()) {
        if (!inputs.empty())
          firstIndex = cast<OpResult>(inputs.front()).getResultNumber();
        visitNonControlFlowArguments(branch,
                                     RegionSuccessor(branch->getResults().slice(
                                         firstIndex, inputs.size())),
                                     lattices, firstIndex);
      } else {
        if (!inputs.empty())
          firstIndex = cast<BlockArgument>(inputs.front()).getArgNumber();
        Region *region = point->getBlock()->getParent();
        visitNonControlFlowArguments(
            branch,
            RegionSuccessor(region, region->getArguments().slice(
                                        firstIndex, inputs.size())),
            lattices, firstIndex);
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "pred: ";
      op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    });
    for (auto [oper, argLat] :
         llvm::zip(*operands, lattices.drop_front(firstIndex))) {
      LLVM_DEBUG({
        llvm::dbgs() << "oper: " << oper << "\n";
        llvm::dbgs() << "argLat anchor: ";
        if (auto res = llvm::dyn_cast<OpResult>(argLat->getAnchor())) {
          res.printAsOperand(llvm::dbgs(), OpPrintingFlags().skipRegions());
        } else {
          argLat->getAnchor().print(llvm::dbgs(),
                                    OpPrintingFlags().skipRegions());
        }
        llvm::dbgs() << "\n";
      });
      join(argLat, *getLatticeElementFor(point, oper));
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  LLVM_DEBUG(llvm::dbgs()
             << "************leaving visitRegionSuccessors*************\n");
}

LogicalResult MyIntegerRangeAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
    ArrayRef<dataflow::IntegerValueRangeLattice *> results) {
  auto inferrable = dyn_cast<InferIntRangeInterface>(op);
  if (!inferrable) {
    setAllToEntryStates(results);
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
  auto argRanges = llvm::map_to_vector(
      operands, [](const dataflow::IntegerValueRangeLattice *lattice) {
        return lattice->getValue();
      });

  auto joinCallback = [&](Value v, const IntegerValueRange &attrs) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
    dataflow::IntegerValueRangeLattice *lattice =
        results[result.getResultNumber()];
    IntegerValueRange oldRange = lattice->getValue();

    ChangeResult changed = lattice->join(attrs);

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldRange.isUninitialized() &&
        !(lattice->getValue() == oldRange)) {
      LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
      // changed |= lattice->join(IntegerValueRange::getMaxRange(v));
    }
    propagateIfChanged(lattice, changed);
  };

  inferrable.inferResultRangesFromOptional(argRanges, joinCallback);
  return success();
}

void MyIntegerRangeAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<dataflow::IntegerValueRangeLattice *> argLattices,
    unsigned firstIndex) {
  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");

    auto argRanges = llvm::map_to_vector(op->getOperands(), [&](Value value) {
      return getLatticeElementFor(getProgramPointAfter(op), value)->getValue();
    });

    auto joinCallback = [&](Value v, const IntegerValueRange &attrs) {
      auto arg = dyn_cast<BlockArgument>(v);
      if (!arg)
        return;
      if (!llvm::is_contained(successor.getSuccessor()->getArguments(), arg))
        return;

      LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
      dataflow::IntegerValueRangeLattice *lattice =
          argLattices[arg.getArgNumber()];
      IntegerValueRange oldRange = lattice->getValue();

      ChangeResult changed = lattice->join(attrs);

      // Catch loop results with loop variant bounds and conservatively make
      // them [-inf, inf] so we don't circle around infinitely often (because
      // the dataflow analysis in MLIR doesn't attempt to work out trip counts
      // and often can't).
      bool isYieldedValue = llvm::any_of(v.getUsers(), [](Operation *op) {
        return op->hasTrait<OpTrait::IsTerminator>();
      });
      if (isYieldedValue && !oldRange.isUninitialized() &&
          !(lattice->getValue() == oldRange)) {
        LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
        changed |= lattice->join(IntegerValueRange::getMaxRange(v));
      }
      propagateIfChanged(lattice, changed);
    };

    inferrable.inferResultRangesFromOptional(argRanges, joinCallback);
    return;
  }

  /// Given the results of getConstant{Lower,Upper}Bound() or getConstantStep()
  /// on a LoopLikeInterface return the lower/upper bound for that result if
  /// possible.
  auto getLoopBoundFromFold = [&](std::optional<OpFoldResult> loopBound,
                                  Type boundType, bool getUpper) {
    unsigned int width = ConstantIntRanges::getStorageBitwidth(boundType);
    if (loopBound.has_value()) {
      if (auto attr = dyn_cast<Attribute>(*loopBound)) {
        if (auto bound = dyn_cast_or_null<IntegerAttr>(attr))
          return bound.getValue();
      } else if (auto value = llvm::dyn_cast_if_present<Value>(*loopBound)) {
        const dataflow::IntegerValueRangeLattice *lattice =
            getLatticeElementFor(getProgramPointAfter(op), value);
        if (lattice != nullptr && !lattice->getValue().isUninitialized())
          return getUpper ? lattice->getValue().getValue().smax()
                          : lattice->getValue().getValue().smin();
      }
    }
    // Given the results of getConstant{Lower,Upper}Bound()
    // or getConstantStep() on a LoopLikeInterface return the lower/upper
    // bound
    return getUpper ? APInt::getSignedMaxValue(width)
                    : APInt::getSignedMinValue(width);
  };

  // Infer bounds for loop arguments that have static bounds
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    std::optional<Value> iv = loop.getSingleInductionVar();
    assert(iv);
    std::optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
    std::optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
    std::optional<OpFoldResult> step = loop.getSingleStep();
    APInt min = getLoopBoundFromFold(lowerBound, iv->getType(),
                                     /*getUpper=*/false);
    APInt max = getLoopBoundFromFold(upperBound, iv->getType(),
                                     /*getUpper=*/true);
    // Assume positivity for uniscoverable steps by way of getUpper = true.
    APInt stepVal =
        getLoopBoundFromFold(step, iv->getType(), /*getUpper=*/true);

    if (stepVal.isNegative()) {
      std::swap(min, max);
    } else {
      // Correct the upper bound by subtracting 1 so that it becomes a <=
      // bound, because loops do not generally include their upper bound.
      max -= 1;
    }

    // If we infer the lower bound to be larger than the upper bound, the
    // resulting range is meaningless and should not be used in further
    // inferences.
    if (max.sge(min)) {
      dataflow::IntegerValueRangeLattice *ivEntry = getLatticeElement(*iv);
      auto ivRange = ConstantIntRanges::fromSigned(min, max);
      propagateIfChanged(ivEntry, ivEntry->join(IntegerValueRange{ivRange}));
    }
    return;
  }

  setAllToEntryStates(argLattices.take_front(firstIndex));
  setAllToEntryStates(argLattices.drop_front(
      firstIndex + successor.getSuccessorInputs().size()));
}

} // end anonymous namespace

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

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<MyIntegerRangeAnalysis>();
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    mod->walk<WalkOrder::PreOrder>([this, &solver](Operation *op) {
      SmallVector<ConstantIntRanges> inputRange;
      (void)collectRanges(*solver, op->getOperands(), inputRange);
      SmallVector<ConstantIntRanges> outputRange;
      (void)collectRanges(*solver, op->getResults(), outputRange);

      if (outputRange.size()) {
        auto min = outputRange[0].smin();
        auto max = outputRange[0].smax();
        auto i64ty = IntegerType::get(&getContext(), 64, IntegerType::Signless);
        SmallVector<Attribute> range = {
            IntegerAttr::get(i64ty, min.getSExtValue()),
            IntegerAttr::get(i64ty, max.getSExtValue()),
        };
        op->setAttr("output_range", ArrayAttr::get(&getContext(), range));
      }
    });

    llvm::outs() << "\n\n";
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUConvertToBufferOpsPass(std::string archGen) {
  return std::make_unique<TritonAMDGPUConvertToBufferOpsPass>(archGen);
}
