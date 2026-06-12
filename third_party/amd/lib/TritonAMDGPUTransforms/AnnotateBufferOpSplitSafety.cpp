#include "Analysis/RangeAnalysis.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUANNOTATEBUFFEROPSPLITSAFETY
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

namespace AMD = mlir::triton::AMD;
namespace tta = mlir::triton::amdgpu;
namespace ttg = mlir::triton::gpu;

constexpr llvm::StringLiteral kSplitSafeAttrName = "amdgpu.split_soffset_safe";

// Shape/layout ops that forward their operand's values unchanged, and with them
// the operand's sign and its integer-range lattice state.
static bool isTransparentWrapper(Operation *op) {
  bool isWrapper =
      isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
          triton::ReshapeOp, ttg::ConvertLayoutOp>(op);
  assert((!isWrapper || op->getNumOperands() == 1) &&
         "transparent wrapper must have a single SSA operand.");
  return isWrapper;
}

// Peel transparent wrappers to expose the real defining op underneath.
static Value peelTransparentWrappers(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (!isTransparentWrapper(def))
      break;
    v = def->getOperand(0);
  }
  return v;
}

// Conservatively accept an offset only when every leaf in its
// additive/shape expression proves non-negative. This may miss safe splits,
// but never annotates an offset with a possibly-negative voffset.
static bool isLeafNonNegative(Value v, DataFlowSolver &solver) {
  // An `add` is never a leaf to the soffset splitter. It peels the summands
  // apart and lifts the uniform ones into the unsigned soffset. So a sum whose
  // range is non-negative can still hide a negative summand.
  if (peelTransparentWrappers(v).getDefiningOp<arith::AddIOp>())
    return false;

  const auto *range = solver.lookupState<dataflow::IntegerValueRangeLattice>(v);
  if (!range || range->getValue().isUninitialized())
    return false;
  if (AMD::isEmptyInitializedRange(range->getValue().getValue()))
    return false;
  return succeeded(dataflow::staticallyNonNegative(solver, v));
}

static bool isNonNegative(Value v, DataFlowSolver &solver) {
  if (!v)
    return false;

  // Prefer the lattice result. The structural cases below enforce the
  // stronger leaf-wise proof needed before splitting soffset from voffset.
  if (isLeafNonNegative(v, solver))
    return true;

  Operation *def = v.getDefiningOp();
  if (!def)
    return false;

  // Recurse through ops where "all operands non-negative -> result
  // non-negative" (with the same < 2GB wrap caveat the rest of the
  // buffer-op path already accepts on add/mul).
  if (isa<arith::AddIOp, arith::MulIOp, arith::OrIOp, arith::XOrIOp,
          arith::DivSIOp, arith::DivUIOp, arith::MinSIOp, arith::MinUIOp,
          arith::MaxSIOp, arith::MaxUIOp, arith::ExtSIOp>(def)) {
    for (Value operand : def->getOperands())
      if (!isNonNegative(operand, solver))
        return false;
    return true;
  }

  // First operand only (sign carries from operand 0).
  if (isa<arith::ShLIOp, arith::ShRSIOp, arith::RemSIOp, arith::RemUIOp>(def))
    return isNonNegative(def->getOperand(0), solver);

  // Always non-negative regardless of operands.
  if (isa<arith::ShRUIOp, arith::ExtUIOp>(def))
    return true;

  // Triton shape/control ops that are non-negative or preserve sign.
  if (auto mr = dyn_cast<triton::MakeRangeOp>(def))
    return mr.getStartAttr().getInt() >= 0;
  if (isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(def))
    return true;
  if (isTransparentWrapper(def))
    return isNonNegative(def->getOperand(0), solver);

  return false;
}

struct AnnotateBufferOpSplitSafetyPass
    : impl::TritonAMDGPUAnnotateBufferOpSplitSafetyBase<
          AnnotateBufferOpSplitSafetyPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // Reuse AMD integer range analysis so `tl.assume` / `gl.assume` and
    // argument attributes can prove buffer offsets non-negative.
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(mod);
    auto solver = createDataFlowSolver();
    auto *rangeAnalysis = solver->load<AMD::TritonIntegerRangeAnalysis>(
        assumptions, &getAnalysis<DominanceInfo>());
    AMD::initializeFuncOps(mod, rangeAnalysis);
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();

    UnitAttr unit = UnitAttr::get(&getContext());
    auto annotateIfSafe = [&](Operation *op, Value offsets) {
      if (isNonNegative(offsets, *solver))
        op->setAttr(kSplitSafeAttrName, unit);
    };

    mod.walk([&](Operation *op) {
      if (auto load = dyn_cast<tta::BufferLoadOp>(op))
        annotateIfSafe(op, load.getOffsets());
      else if (auto store = dyn_cast<tta::BufferStoreOp>(op))
        annotateIfSafe(op, store.getOffsets());
      else if (auto loadLds = dyn_cast<tta::BufferLoadToLocalOp>(op))
        annotateIfSafe(op, loadLds.getOffsets());
    });
  }
};

} // namespace
} // namespace mlir
