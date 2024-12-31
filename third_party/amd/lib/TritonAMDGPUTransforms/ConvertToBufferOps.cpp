#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#include "triton/Analysis/AxisInfo.h"
#include "../TritonAMDGPUToLLVM/Utility.h"
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
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {
template <typename F>
bool verifyNonSmallerByAssumption(Value expr,
                                  const DenseSet<Value> &assumptions,
                                  F matchesOther) {
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

  // 3. Check if the offset is non-negative
  if (!verifyNonNegativeExpr(offset, assumptions))
    return false;

  LDBG("Non-negative");
  return true;
}
} // namespace


struct ConvertTritonAtomicRMWOpToBufferAtomicRMW
    : public mlir::OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicRMWOpToBufferAtomicRMW(mlir::MLIRContext *context,
                                            DenseSet<Value> &assumptions,
                                            ModuleAxisInfoAnalysis axisAnalysisPass)
      : mlir::OpRewritePattern<triton::AtomicRMWOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass) {}

  mlir::LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();
    auto atomicRmwOp= op.getAtomicRmwOp();
    auto sem = op.getSem();
    auto scope = op.getScope();

    // In addition to the `canUserBufferOps` check, we should ensure that
    // 1. Perform the canUserBufferOps check
    if (!canUseBufferOps(ptr, assumptions)) {
      LDBG("Failed to convert: " << op);
      return failure();
    }

    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    Value tensorPtr = addPtrOp.getPtr();
    Value tensorOffset = addPtrOp.getOffset();
    auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
    Value basePtr = splatOp.getSrc();

    // 2. FP8 atomics are not supported with buffer atomics
    auto checkFP8Value = op.getVal().getType();
    if (auto vecType = dyn_cast<RankedTensorType>(checkFP8Value)) {
      checkFP8Value = vecType.getElementType();
    }
    bool isFP8 = checkFP8Value.isFloat8E5M2() || checkFP8Value.isFloat8E4M3FN() ||
                 checkFP8Value.isFloat8E5M2FNUZ() || checkFP8Value.isFloat8E4M3FNUZ();
    if (isFP8) {
      LDBG("Failed to convert: " << op);
      return failure();
    }

    // 3. Check the hardware---only MI-* series GPUs are supported
    //    (i.e., CDNA 1, 2, 3)

    // TODO

    // 4. Check if the RMWOp is supported
    //    see: https://github.com/ytsaurus/ytsaurus/blob/fa3b61994db90ee211d2944e5b385e55a4d6be42/contrib/libs/llvm18/include/llvm/IR/IntrinsicsAMDGPU.h#L765
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
      case RMWOp::SMAX:
      case RMWOp::SMIN:
      case RMWOp::XCHG:
        break
      default:
        LDBG("Failed to convert: " << op);
        return failure();
    }

    // 5. Buffer atomics support 32 and 64-bit operations, so inputs must be at least 32-bits
    //    Otherwise, fall back to the existing path for atomics
    auto opValueType = op.getVal().getType();
    auto opBitWidth = 0;
    if (auto vecType = dyn_cast<RankedTensorType>(opValueType)) {
      // We can't just get the numElements * elemBitWidth here
      // In cases such as tensor<2xf16...>, if the elements are contiguous we can emit
      // the buffer op. Otherwise, the buffer ops lowering will try to emit individual (unsupported) f16/bf16 ops.
      auto elemBitWidth = vecType.getElementType().getIntOrFloatBitWidth();
      opBitWidth = getVectorSize(basePtr, tensorOffset, axisAnalysisPass) * elemBitWidth;
    } else {
      opBitWidth = opValueType.getIntOrFloatBitWidth();
    }

    if (opBitWidth >= 32) {
      Value maybeMask{};
      if (op.getMask() && !isZeroConst(op.getMask()))
        maybeMask = op.getMask();

      auto atomicRMWOp = rewriter.create<triton::amdgpu::BufferAtomicRMWOp>(
        op->getLoc(), op.getVal().getType(), atomicRmwOp, basePtr, tensorOffset, op.getVal(), sem, scope, maybeMask);

      rewriter.replaceOp(op, atomicRMWOp);

      return success();
    }
    LDBG("Failed to convert: " << op);
    return failure();
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
  ModuleAxisInfoAnalysis axisAnalysisPass;
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

      auto bufferLoadOp = rewriter.create<triton::amdgpu::BufferLoadOp>(
          op->getLoc(), op.getType(), basePtr, tensorOffset, op.getCache(),
          maybeMask, maybeOther);

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
    return failure();
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
      rewriter.replaceOpWithNewOp<triton::amdgpu::BufferStoreOp>(
          op, op.getValue(), basePtr, tensorOffset, op.getCache(), maybeMask);
      return success();
    }
    LDBG("Failed to convert: " << op);
    return failure();
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
    LDBG("Number of assumptions found: " << assumptions.size());
    for (Value assume : assumptions) {
      LDBG("Assumption:" << assume);
    }

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    patterns.add<ConvertTritonLoadToBufferLoad>(context, assumptions);
    patterns.add<ConvertTritonStoreToBufferStore>(context, assumptions);
    patterns.add<ConvertTritonAtomicRMWOpToBufferAtomicRMW>(context, assumptions, axisInfoAnalysis);
    if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUConvertToBufferOpsPass() {
  return std::make_unique<TritonAMDGPUConvertToBufferOpsPass>();
}
