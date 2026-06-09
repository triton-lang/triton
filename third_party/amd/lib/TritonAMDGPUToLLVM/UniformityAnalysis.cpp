#include "TritonAMDGPUToLLVM/UniformityAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using mlir::dataflow::Lattice;
using mlir::dataflow::SparseForwardDataFlowAnalysis;

namespace mlir::triton::AMD {

// Walk back through an insertvalue chain to find the value at `position`.
// Returns `original` if the chain can't be resolved.
static constexpr int kMaxInsertValueChainDepth = 4096;
static Value resolveExtractChain(Value container, ArrayRef<int64_t> position,
                                 Value original) {
  Value cur = container;
  for (int steps = 0; steps < kMaxInsertValueChainDepth && cur; ++steps) {
    auto insert = cur.getDefiningOp<LLVM::InsertValueOp>();
    if (!insert)
      return original;
    auto insertPos = insert.getPosition();
    // Exact match.
    if (insertPos.size() == position.size() &&
        std::equal(insertPos.begin(), insertPos.end(), position.begin())) {
      return insert.getValue();
    }
    // insertPos is a strict prefix of position -> recurse into the value
    // with the remaining suffix.
    if (insertPos.size() < position.size() &&
        std::equal(insertPos.begin(), insertPos.end(), position.begin())) {
      ArrayRef<int64_t> suffix = position.drop_front(insertPos.size());
      return resolveExtractChain(insert.getValue(), suffix, original);
    }
    // Disjoint or position is prefix of insertPos: keep walking up.
    cur = insert.getContainer();
  }
  return original;
}

namespace {

// Integer arith/cast ops: uniformity = join of operand uniformity.
// Float ops excluded (buffer offsets are always i32). Anything not
// listed here and not seeded falls to Divergent.
bool isPureArithOrCast(Operation *op) {
  return isa<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::ShlOp, LLVM::LShrOp,
             LLVM::AShrOp, LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp, LLVM::SExtOp,
             LLVM::ZExtOp, LLVM::TruncOp, LLVM::SelectOp, LLVM::ICmpOp,
             LLVM::URemOp, LLVM::SRemOp, LLVM::UDivOp, LLVM::SDivOp,
             LLVM::BitcastOp, LLVM::SMinOp, LLVM::SMaxOp, LLVM::UMinOp,
             LLVM::UMaxOp, LLVM::AbsOp, LLVM::PtrToIntOp, LLVM::IntToPtrOp,
             LLVM::GEPOp, LLVM::FreezeOp, LLVM::AddrSpaceCastOp, arith::AddIOp,
             arith::SubIOp, arith::MulIOp, arith::ShLIOp, arith::ShRSIOp,
             arith::ShRUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
             arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::SelectOp,
             arith::CmpIOp, arith::IndexCastOp, arith::IndexCastUIOp,
             arith::BitcastOp, arith::DivSIOp, arith::DivUIOp, arith::RemSIOp,
             arith::RemUIOp, arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp,
             arith::MaxUIOp>(op);
}

bool isSeedUniform(Operation *op) {
  return isa<LLVM::ConstantOp, arith::ConstantOp, ROCDL::BlockIdXOp,
             ROCDL::BlockIdYOp, ROCDL::BlockIdZOp, ROCDL::WaveId,
             ROCDL::ReadfirstlaneOp, ROCDL::ReadlaneOp>(op);
}

bool isSeedDivergent(Operation *op) {
  return isa<ROCDL::ThreadIdXOp, ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp,
             gpu::ThreadIdOp, gpu::LaneIdOp>(op);
}

// Kernel entry = public-linkage llvm.func. AMD `FuncOpConversion`
// (FuncOpToLLVM.cpp) sets `Linkage::External` for tt.func kernels and
// `Linkage::Internal` for device helpers; no AMDGPU-kernel cconv or
// rocdl.kernel/amdgpu-flat-work-group-size attrs are attached at MLIR
// level (those appear later during LLVM-IR translation).
bool isKernelEntry(LLVM::LLVMFuncOp func) { return func.isPublic(); }

class UniformityAnalysis
    : public SparseForwardDataFlowAnalysis<UniformityLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UniformityAnalysis)

  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const UniformityLattice *> operands,
                               ArrayRef<UniformityLattice *> results) override {
    // Seed-uniform: always uniform.
    if (isSeedUniform(op)) {
      for (auto *r : results)
        propagateIfChanged(r, r->join(UniformityValue::uniform()));
      return success();
    }
    // Seed-divergent: always divergent.
    if (isSeedDivergent(op)) {
      for (auto *r : results)
        propagateIfChanged(r, r->join(UniformityValue::divergent()));
      return success();
    }

    // ExtractValueOp: chase the insertvalue chain to find the actual value.
    if (auto extract = dyn_cast<LLVM::ExtractValueOp>(op)) {
      Value underlying = lookThroughExtractValue(extract.getResult());
      // Peeled successfully; propagate that value's lattice.
      if (underlying != extract.getResult()) {
        const UniformityLattice *underLat =
            getLatticeElementFor(getProgramPointAfter(op), underlying);
        if (!underLat) {
          for (auto *r : results)
            propagateIfChanged(r, r->join(UniformityValue::divergent()));
          return success();
        }
        for (auto *r : results)
          propagateIfChanged(r, r->join(underLat->getValue()));
        return success();
      }
      // Can't peel -> conservative.
      for (auto *r : results)
        propagateIfChanged(r, r->join(UniformityValue::divergent()));
      return success();
    }

    // InsertValueOp: can't merge per-slot info into one lattice, so mark
    // divergent. ExtractValueOp above sidesteps this by chasing the value.
    if (isa<LLVM::InsertValueOp>(op)) {
      for (auto *r : results)
        propagateIfChanged(r, r->join(UniformityValue::divergent()));
      return success();
    }

    if (isPureArithOrCast(op)) {
      // Join operand uniformity.
      UniformityValue joined = UniformityValue::uniform();
      for (const auto *o : operands)
        joined = UniformityValue::join(joined, o->getValue());
      for (auto *r : results)
        propagateIfChanged(r, r->join(joined));
      return success();
    }

    // Unknown op -> divergent.
    for (auto *r : results)
      propagateIfChanged(r, r->join(UniformityValue::divergent()));
    return success();
  }

  // The framework's per-edge join doesn't catch phis that merge different
  // values under a divergent branch (lanes take different edges). We
  // handle that in phiIsDivergentUnderControlFlow, called from
  // isUniformValue.
  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange nonSuccessorInputs,
      ArrayRef<UniformityLattice *> nonSuccessorInputLattices) override {
    // We don't model region-branch op induction variables specially.
    setAllToEntryStates(nonSuccessorInputLattices);
  }

  // Kernel discrimination lives in isKernelEntry above (public linkage).

  void visitCallableOperation(
      CallableOpInterface callable,
      ArrayRef<dataflow::AbstractSparseLattice *> argLattices) override {
    if (auto func = dyn_cast<LLVM::LLVMFuncOp>(callable.getOperation())) {
      if (isKernelEntry(func)) {
        for (auto *al : argLattices) {
          auto *typed = static_cast<UniformityLattice *>(al);
          propagateIfChanged(typed, typed->join(UniformityValue::uniform()));
        }
        return;
      }
    }
    // Non-kernel: leave args at Bottom -> isUniformValue treats as Divergent.
    for (auto *al : argLattices) {
      auto *typed = static_cast<UniformityLattice *>(al);
      propagateIfChanged(typed, typed->join(UniformityValue::bottom()));
    }
  }

  void setToEntryState(UniformityLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(UniformityValue::bottom()));
  }

  // Seed kernel args as Uniform here because visitCallableOperation
  // doesn't fire reliably for post-lowering LLVMFuncOps.
  LogicalResult initialize(Operation *top) override {
    top->walk([&](LLVM::LLVMFuncOp func) {
      if (!isKernelEntry(func) || func.isExternal())
        return;
      for (BlockArgument ba : func.getBody().front().getArguments()) {
        auto *lat = getLatticeElement(ba);
        propagateIfChanged(lat, lat->join(UniformityValue::uniform()));
      }
    });
    return SparseForwardDataFlowAnalysis<UniformityLattice>::initialize(top);
  }
};

// Look up a value's uniformity, peeling extractvalue chains first.
static UniformityValue lookupUniformity(Value v, const DataFlowSolver &s) {
  Value peeled = lookThroughExtractValue(v);
  if (const auto *lat = s.lookupState<UniformityLattice>(peeled))
    return lat->getValue();
  return UniformityValue::bottom();
}

// A phi can be data-uniform but control-flow-divergent: if a divergent
// branch picks which incoming value each lane gets, the phi is divergent
// even when all incomings are individually uniform.
//
// Example: cond_br %threadIdx, ^join(%u0), ^join(%u1)
// Both %u0 and %u1 are uniform, but lanes get different values.
//
// Returns true when a divergent branch feeds this phi AND the incoming
// values differ (or any incoming is itself divergent). Loop IVs stay
// uniform because their back-edge condition is typically uniform.
bool phiIsDivergentUnderControlFlow(BlockArgument arg,
                                    const DataFlowSolver &solver) {
  Block *block = arg.getOwner();
  if (block->isEntryBlock())
    return false; // function args, not phis

  unsigned idx = arg.getArgNumber();
  bool anyDivergentControl = false;
  llvm::SmallVector<Value, 4> incomings;

  // Collect incoming values and check branch conditions.
  for (Block *pred : block->getPredecessors()) {
    Operation *term = pred->getTerminator();
    if (auto br = dyn_cast<LLVM::BrOp>(term)) {
      // Unconditional branch — always uniform control.
      auto ops = br.getDestOperands();
      if (idx < ops.size())
        incomings.push_back(ops[idx]);
    } else if (auto cb = dyn_cast<LLVM::CondBrOp>(term)) {
      // Check if the branch condition is divergent.
      const auto *condLat =
          solver.lookupState<UniformityLattice>(cb.getCondition());
      bool condDiv = condLat && condLat->getValue().isDivergent();
      if (condDiv)
        anyDivergentControl = true;
      // Collect incoming values from whichever edges reach this block.
      if (cb.getTrueDest() == block) {
        auto ops = cb.getTrueDestOperands();
        if (idx < ops.size())
          incomings.push_back(ops[idx]);
      }
      if (cb.getFalseDest() == block) {
        auto ops = cb.getFalseDestOperands();
        if (idx < ops.size())
          incomings.push_back(ops[idx]);
      }
    } else {
      return true; // unknown terminator — conservatively divergent
    }
  }

  // All branches uniform -> all lanes take the same path. Loop IVs
  // stay uniform here: back-edge cond (%k < %K) is uniform.
  if (!anyDivergentControl)
    return false;

  // Only one incoming edge means no merge; all lanes get the same value.
  if (incomings.size() < 2)
    return false;

  // Divergent branch: lanes may arrive from different preds. Divergent
  // if the incoming values differ or any incoming is itself divergent.
  Value first = incomings.front();
  for (Value v : incomings) {
    if (v != first)
      return true;
    if (lookupUniformity(v, solver).isDivergent())
      return true;
  }
  return false;
}

} // namespace

Value lookThroughExtractValue(Value v) {
  while (auto extract = v.getDefiningOp<LLVM::ExtractValueOp>()) {
    Value resolved =
        resolveExtractChain(extract.getContainer(), extract.getPosition(), v);
    if (resolved == v)
      return v;
    v = resolved;
  }
  return v;
}

void loadUniformityAnalysis(DataFlowSolver &solver) {
  solver.load<UniformityAnalysis>();
}

// Fallback for values created during pattern conversion (not in the
// solver state). Mirrors the analysis: joins operands for pure arith,
// checks seeds, treats unknowns as divergent. Bounded to `depth` levels.
static bool isUniformRecursive(Value v, const DataFlowSolver &solver,
                               llvm::SmallPtrSetImpl<Value> &visited,
                               int depth) {
  if (!v || depth <= 0)
    return false;
  if (!visited.insert(v).second)
    return false;
  Value peeled = lookThroughExtractValue(v);
  // First try the solver snapshot.
  if (const auto *lat = solver.lookupState<UniformityLattice>(peeled)) {
    if (lat->getValue().isUniform()) {
      if (auto ba = dyn_cast<BlockArgument>(peeled))
        if (phiIsDivergentUnderControlFlow(ba, solver))
          return false;
      return true;
    }
    if (lat->getValue().isDivergent())
      return false;
    // Bottom: fall through to defining-op recursion.
  }
  Operation *def = peeled.getDefiningOp();
  if (!def) {
    // BlockArgument not in solver: treat as kernel-arg uniform iff its
    // owning func is a kernel entry.
    if (auto ba = dyn_cast<BlockArgument>(peeled)) {
      Block *blk = ba.getOwner();
      if (blk && blk->isEntryBlock()) {
        if (auto func = dyn_cast_or_null<LLVM::LLVMFuncOp>(blk->getParentOp()))
          return isKernelEntry(func);
      }
    }
    return false;
  }
  if (isSeedUniform(def))
    return true;
  if (isSeedDivergent(def))
    return false;
  if (!isPureArithOrCast(def))
    return false;
  for (Value op : def->getOperands())
    if (!isUniformRecursive(op, solver, visited, depth - 1))
      return false;
  return true;
}

bool isUniformValue(Value v, const DataFlowSolver &solver) {
  llvm::SmallPtrSet<Value, 16> visited;
  return isUniformRecursive(v, solver, visited, /*depth=*/64);
}

} // namespace mlir::triton::AMD
