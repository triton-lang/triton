#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-convert-buffer-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using ::mlir::LLVM::AMD::getVectorSize;
using mlir::triton::AMD::ISAFamily;

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCONVERTTOBUFFEROPS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Return true iff the given value v is a tensor splatting from 1 (int).
// The usefulness of this func stems from the fact than if a buffer-op's mask
// operand is a all-1-tensor, it does not need to take this operand.
bool isSplatOneConstTensor(const Value v) {
  auto constantOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return false;

  if (auto denseAttr =
          dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APInt>().isOne();

  return false;
}

bool verifyNonSmallerByAssumption(
    Value expr, const DenseMap<Value, SetVector<Operation *>> &assumptions,
    const std::function<bool(Value)> &matchesOther) {
  if (!assumptions.contains(expr))
    return false;
  for (Operation *assume : assumptions.at(expr)) {
    auto cmpOp = llvm::dyn_cast<arith::CmpIOp>(assume);
    if (!cmpOp)
      continue;
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
  return false;
}

bool verifyNonSmallerByAssumption(
    Value expr, const DenseMap<Value, SetVector<Operation *>> &assumptions,
    Value other) {
  return verifyNonSmallerByAssumption(
      expr, assumptions, [&](auto otherAssum) { return otherAssum == other; });
}

bool verifyNonNegativeExpr(
    Value expr, const DenseMap<Value, SetVector<Operation *>> &assumptions,
    std::shared_ptr<DataFlowSolver> solver) {
  LDBG("Determing if non-negative: " << expr);

  auto nonNegativePred = [&solver](Value v) -> bool {
    if (const auto *r =
            solver->lookupState<dataflow::IntegerValueRangeLattice>(v)) {
      if (r->getValue().isUninitialized())
        return false;
      if (AMD::isEmptyInitializedRange(r->getValue().getValue()))
        return false;
    }
    return succeeded(dataflow::staticallyNonNegative(*solver, v));
  };

  if (nonNegativePred(expr))
    return true;

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
            return verifyNonNegativeExpr(unaryOp.getOperand(), assumptions,
                                         solver);
          })
          .Case<triton::GatherOp>([&](auto gatherOp) {
            return verifyNonNegativeExpr(gatherOp.getSrc(), assumptions,
                                         solver);
          })
          // Joining two non-negative tensors is still non-negative
          .Case<triton::JoinOp, triton::CatOp>([&](auto joinOp) {
            return verifyNonNegativeExpr(joinOp.getLhs(), assumptions,
                                         solver) &&
                   verifyNonNegativeExpr(joinOp.getRhs(), assumptions, solver);
          })
          // Returns a tensor representing histogram: histograms only contain
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
            return verifyNonNegativeExpr(maxOp.getLhs(), assumptions, solver) ||
                   verifyNonNegativeExpr(maxOp.getRhs(), assumptions, solver);
          })
          .Case<arith::RemSIOp>([&](auto remsiOp) {
            // a % b >= 0 iff a>=0
            return verifyNonNegativeExpr(remsiOp.getLhs(), assumptions, solver);
          })
          .Case<arith::TruncIOp, arith::ExtSIOp>([&](Operation *unaryOp) {
            // a = OP b >= 0 iff b >= 0
            return verifyNonNegativeExpr(unaryOp->getOperand(0), assumptions,
                                         solver);
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
                return verifyNonNegativeExpr(binOp->getOperand(0), assumptions,
                                             solver) &&
                       verifyNonNegativeExpr(binOp->getOperand(1), assumptions,
                                             solver);
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
                                         assumptions, solver) &&
                   verifyNonNegativeExpr(elseYield->getOperand(resultIdx),
                                         assumptions, solver);
          })
          .Case<arith::SubIOp>([&](auto op) {
            // If a user annotates tl.assume(a >= b) then we know a - b >= 0
            return verifyNonSmallerByAssumption(op.getLhs(), assumptions,
                                                op.getRhs());
          })
          .Case<triton::amdgpu::ExtractSliceOp>([&](auto op) {
            return verifyNonNegativeExpr(op->getOperand(0), assumptions,
                                         solver);
          })
          .Default([&](Operation *) {
            // Conservatively assume that the expression is negative
            LDBG("  Unhandled op, cannot assume non-negative");
            return false;
          });
  return nonNegative;
}

bool isFuncArgWith32bitPtrRange(mlir::Value value) {
  if (value.getDefiningOp())
    return false;

  mlir::BlockArgument blockArg = mlir::cast<mlir::BlockArgument>(value);
  auto blk = blockArg.getOwner();
  auto funcOp = dyn_cast_or_null<tt::FuncOp>(blk->getParentOp());

  if (funcOp && blk == &funcOp->getRegion(0).front()) {
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (arg != value)
        continue;
      auto attr = funcOp.getArgAttrOfType<IntegerAttr>(idx, "tt.pointer_range");
      return attr && attr.getInt() <= 32;
    }
  }

  return false;
}

// Quick analysis on the Triton IR to decide if we can safely use
// buffer operations
bool canUseBufferOps(Value ptr,
                     const DenseMap<Value, SetVector<Operation *>> &assumptions,
                     std::shared_ptr<DataFlowSolver> solver,
                     bool analyzeSmallTensorOfst) {
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

  // 2. check if the offset is either 32 or 64-bit.
  Value offset = addPtrOp.getOffset();
  auto ofstBit =
      cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth();
  LLVM_DEBUG(llvm::dbgs() << "offset bits:" << ofstBit << "\n");

  // TODO: step 3 and 4 can be reversed to further optimize for performance.
  // When the base-ptr is func argument and has tt.pointer_range=32 attribute,
  // it's safe to promote the mem-op into buffer-op even if offset is a 64-bit
  // value. If this is the case, offset need to be cast down to 32-bit.

  // 3. Bail out if ofst cannot fit in 32-bit.
  if (ofstBit != 32)
    return false;

  // 4. If the base is function formal argument which has attribute
  //  tt.point_range=32, then it's safe to promote this memory op into
  //  bufferOp. In this case, if offset is 64-bit, we should cast it down to
  //  32-bit.
  if (!analyzeSmallTensorOfst &&
      isFuncArgWith32bitPtrRange(maybeSplatOp.getSrc())) {
    LDBG("base-ptr as tt.pointer_range=32 attribute");
    return true;
  }
  return verifyNonNegativeExpr(offset, assumptions, std::move(solver));
}

// Extract stride of the blocked offset of LD/ST ops.
Value getBlockStride(Location loc, Value offset, PatternRewriter &rewriter) {
  // canonicalize pointer pass sets block stride via
  // `offset:add-broadcast-muli-splat`, backtrace that pattern to reach the
  // stride.
  if (auto maybeAdd = offset.getDefiningOp<arith::AddIOp>())
    for (auto addOpr : maybeAdd.getOperands())
      if (auto maybeBC = addOpr.getDefiningOp<tt::BroadcastOp>()) {
        auto bcSrc = maybeBC.getSrc();
        if (auto maybeMul = bcSrc.getDefiningOp<arith::MulIOp>())
          for (auto mulOpr : maybeMul.getOperands())
            if (auto maybeSplat = mulOpr.getDefiningOp<tt::SplatOp>())
              return maybeSplat.getSrc();
      }
  return nullptr;
}

// /*-----------------AtomicCAS-------------------*/

struct ConvertTritonAtomicCASOpToBufferAtomicCAS
    : public mlir::OpRewritePattern<triton::AtomicCASOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicCASOpToBufferAtomicCAS(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<triton::AtomicCASOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}

  mlir::LogicalResult
  matchAndRewrite(triton::AtomicCASOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();
    auto sem = op.getSem();
    auto scope = op.getScope();

    if (!canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
      return rewriter.notifyMatchFailure(op, "canUseBufferOps check failed");
    }

    switch (scope) {
    case MemSyncScope::GPU:
    case MemSyncScope::CTA:
      break;
    default:
      return rewriter.notifyMatchFailure(op, "CAS with unsupported scope");
    }
    LDBG("CAS supported scope");

    switch (sem) {
    case MemSemantic::RELAXED:
    case MemSemantic::RELEASE:
    case MemSemantic::ACQUIRE:
    case MemSemantic::ACQUIRE_RELEASE:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "CAS with unsupported memory ordering");
    }

    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    Value tensorPtr = addPtrOp.getPtr();
    Value tensorOffset = addPtrOp.getOffset();
    auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
    Value basePtr = splatOp.getSrc();

    // Buffer atomic CAS only supports i32/i64
    auto checkType = getElementTypeOrSelf(op.getVal());
    bool isSupportedType = checkType.isInteger(32) || checkType.isInteger(64);
    if (!isSupportedType) {
      return rewriter.notifyMatchFailure(op, "AtomicCAS with unsupported type");
    }
    LDBG("AtomicCAS supported type");

    // Buffer atomics support 32 and 64-bit operations, so inputs must be at
    // least 32-bits. Otherwise, fall back to the existing path for atomics
    auto opValueType = op.getVal().getType();
    auto opBitWidth = 0;
    if (auto tensorType = dyn_cast<RankedTensorType>(opValueType)) {
      auto elemBitWidth = tensorType.getElementTypeBitWidth();
      opBitWidth =
          getVectorSize(basePtr, tensorOffset, axisAnalysisPass) * elemBitWidth;
    } else {
      opBitWidth = opValueType.getIntOrFloatBitWidth();
    }

    if (opBitWidth < 32) {
      return rewriter.notifyMatchFailure(
          op, "BufferAtomicCAS requires opBitWidth >= 32");
    }
    Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);
    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferAtomicCASOp>(
        op, op.getVal().getType(), basePtr, tensorOffset, op.getCmp(),
        op.getVal(), blockStride, sem, scope);
    return success();
  }

private:
  // Assumptions collected through the function
  const DenseMap<Value, SetVector<Operation *>> &assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
  bool analyzeSmallTensorOfst;
};

struct ConvertTritonAtomicRMWOpToBufferAtomicRMW
    : public mlir::OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicRMWOpToBufferAtomicRMW(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver, ISAFamily isaFamily,
      bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<triton::AtomicRMWOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)), isaFamily(isaFamily),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}

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
    if (!canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
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

    // float16 is the only 16-bit dtype supported by buffer atomic fadd on
    // gfx942
    if (isaFamily == ISAFamily::CDNA3 && checkType.isBF16() &&
        atomicRmwOp == RMWOp::FADD) {
      return rewriter.notifyMatchFailure(op, "RMW FADD does not support bf16");
    }
    LDBG("RMW FADD supported 16-bit type");

    auto vecSize = getVectorSize(ptr, axisAnalysisPass);
    // f16/bf16 dtypes could only be efficiently calculated using instructions
    // that pack 2 elements (e.g. @llvm.amdgcn.raw.buffer.atomic.fadd.v2f16)
    if (vecSize % 2 != 0 && (checkType.isF16() || checkType.isBF16())) {
      return rewriter.notifyMatchFailure(
          op, "RMW float 16 dtypes must be aligned by 2");
    }
    LDBG("RMW passed alignment check");

    // 5. Check if the RMWOp is supported
    switch (atomicRmwOp) {
    case RMWOp::AND:
    case RMWOp::OR:
    case RMWOp::XOR:
    case RMWOp::ADD:
    case RMWOp::FADD:
    case RMWOp::UMAX:
    case RMWOp::UMIN:
    case RMWOp::XCHG:
      break;
    case RMWOp::MAX:
    case RMWOp::MIN:
      // TODO: It likely means smax/smin, for now intrinsic
      // llvm.amdgcn.raw.ptr.buffer.atomic.{min|max} is emitted, and llvm get
      // confused as how to deal with {f|s|u}{min|max}.
      if (!checkType.isInteger())
        break;
      // else fall through
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
      opBitWidth = vecSize * elemBitWidth;
    } else {
      opBitWidth = opValueType.getIntOrFloatBitWidth();
    }

    if (opBitWidth < 32) {
      return rewriter.notifyMatchFailure(op, "RMW requires opBitWidth >= 32");
    }

    Value maybeMask{};
    if (op.getMask() && !isSplatOneConstTensor(op.getMask()))
      maybeMask = op.getMask();
    Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);
    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferAtomicRMWOp>(
        op, op.getVal().getType(), atomicRmwOp, basePtr, tensorOffset,
        op.getVal(), blockStride, sem, scope, maybeMask);

    return success();
  }

private:
  // Assumptions collected through the function
  DenseMap<Value, SetVector<Operation *>> assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
  ISAFamily isaFamily;
  bool analyzeSmallTensorOfst;
};

// Workaround to allow static_assert(false) on older compilers as it was
// ill-formed before defect report CWG2518
// (https://cplusplus.github.io/CWG/issues/2518.html)
template <typename T> struct always_false : std::false_type {};

template <typename SourceOp>
struct ConvertTritonLoadToBufferLoad : public mlir::OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  ConvertTritonLoadToBufferLoad(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<SourceOp>(context), assumptions(assumptions),
        solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getOperand(0);

    if (toDodgeBug(op)) {
      LDBG("To dodge a llc bug arising from f32 load fed to tt.dot " << op);
      return rewriter.notifyMatchFailure(op, "Failed to convert LoadOp");
    }

    if (canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeOther{};
      if (op.getOther() && !isZeroConst(op.getOther()))
        maybeOther = op.getOther();
      Value maybeMask{};
      if (op.getMask() && !isSplatOneConstTensor(op.getMask()))
        maybeMask = op.getMask();
      Value blockStride = getBlockStride(op->getLoc(), tensorOffset, rewriter);

      auto bufferLoadOp = [&]() {
        if constexpr (std::is_same_v<SourceOp, triton::LoadOp>) {
          return rewriter.create<triton::amdgpu::BufferLoadOp>(
              op->getLoc(), op.getType(), basePtr, tensorOffset, blockStride,
              op.getCache(), maybeMask, maybeOther);
        } else if constexpr (std::is_same_v<
                                 SourceOp,
                                 triton::gpu::AsyncCopyGlobalToLocalOp>) {
          return rewriter.create<triton::amdgpu::BufferLoadToLocalOp>(
              op->getLoc(), op.getType(), op.getResult(), basePtr, tensorOffset,
              maybeMask, maybeOther, blockStride, op.getCache());
        } else {
          static_assert(always_false<SourceOp>::value,
                        "Unsupported type in ConvertTritonLoadToBufferLoad");
        }
      }();

      assert(bufferLoadOp);

      rewriter.replaceOp(op, bufferLoadOp);
      return success();
    }

    LDBG("Failed to convert: " << op);
    return rewriter.notifyMatchFailure(op, "Failed to convert LoadOp");
  }

private:
  // Currently, we need to dodge a LLC bug arising from f32 load fed to
  // tt.dot.
  mutable llvm::SmallMapVector<tt::FuncOp, std::optional<bool>, 2> hasDotOpMap;
  bool toDodgeBug(SourceOp ld) const {
    auto ty = getElementTypeOrSelf(ld.getResult());
    if (!ty.isF32())
      return false;

    auto func = ld->template getParentOfType<tt::FuncOp>();
    if (!func)
      return true;

    bool mayHaveDot = false;
    if (auto iter = hasDotOpMap.find(func); iter != hasDotOpMap.end()) {
      mayHaveDot = iter->second.value();
    } else {
      mayHaveDot = false;
      func.walk([&](tt::DotOp dot) { mayHaveDot = true; });
      hasDotOpMap.insert(std::make_pair(func, std::optional<bool>(mayHaveDot)));
    }
    return mayHaveDot;
  }

  // Assumptions collected through the function
  DenseMap<Value, SetVector<Operation *>> assumptions;
  std::shared_ptr<DataFlowSolver> solver;
  bool analyzeSmallTensorOfst;
};

struct ConvertTritonStoreToBufferStore
    : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonStoreToBufferStore(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<triton::StoreOp>(context),
        assumptions(assumptions), solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeMask{};
      if (op.getMask() && !isSplatOneConstTensor(op.getMask()))
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
  DenseMap<Value, SetVector<Operation *>> assumptions;
  std::shared_ptr<DataFlowSolver> solver;
  bool analyzeSmallTensorOfst;
};

} // anonymous namespace

struct TritonAMDGPUConvertToBufferOpsPass
    : impl::TritonAMDGPUConvertToBufferOpsBase<
          TritonAMDGPUConvertToBufferOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();

    // Collect assumptions in the function
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(getOperation());
    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    AMD::TritonIntegerRangeAnalysis *rangeAnalysis =
        solver->load<AMD::TritonIntegerRangeAnalysis>(assumptions);
    AMD::initializeFuncOps(mod, rangeAnalysis);
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    patterns.add<ConvertTritonLoadToBufferLoad<tt::LoadOp>,
                 ConvertTritonLoadToBufferLoad<ttg::AsyncCopyGlobalToLocalOp>,
                 ConvertTritonStoreToBufferStore>(context, assumptions, solver,
                                                  this->analyzeSmallTensorOfst);

    // Gate buffer atomics behind CDNA3 for now
    // GFX942-specific assumptions regarding cache coherence are made when
    // lowering to LLVM
    triton::AMD::ISAFamily isaFamily =
        triton::AMD::deduceISAFamily(archGenerationName);
    if (this->allowBufferAtomics &&
        (ISAFamily::CDNA3 == isaFamily || ISAFamily::CDNA4 == isaFamily))
      patterns.add<ConvertTritonAtomicRMWOpToBufferAtomicRMW>(
          context, assumptions, axisInfoAnalysis, solver, isaFamily,
          this->analyzeSmallTensorOfst);
    patterns.add<ConvertTritonAtomicCASOpToBufferAtomicCAS>(
        context, assumptions, axisInfoAnalysis, solver,
        this->analyzeSmallTensorOfst);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
