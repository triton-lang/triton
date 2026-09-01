#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_CANONICALIZELLVMIR
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
class SelectConstantConditionPattern : public OpRewritePattern<LLVM::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SelectOp op,
                                PatternRewriter &b) const override {
    BoolAttr cond;
    if (!matchPattern(op.getCondition(), m_Constant(&cond)))
      return failure();
    Value val = cond.getValue() ? op.getTrueValue() : op.getFalseValue();
    b.replaceOp(op, ValueRange{val});
    return success();
  }
};

class ElideFullClusterRankMaskPattern : public OpRewritePattern<LLVM::AndOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AndOp op,
                                PatternRewriter &rewriter) const override {
    APInt mask;
    Value rank = op.getLhs();
    if (!matchPattern(op.getRhs(), m_ConstantInt(&mask))) {
      if (!matchPattern(op.getLhs(), m_ConstantInt(&mask)))
        return failure();
      rank = op.getRhs();
    }

    if (!rank.getDefiningOp<triton::nvgpu::ClusterCTAIdOp>())
      return failure();

    unsigned numCTAs = triton::gpu::lookupNumCTAs(op);
    if (mask.countr_one() < llvm::Log2_32_Ceil(numCTAs))
      return failure();

    rewriter.replaceOp(op, rank);
    return success();
  }
};

// Find each of the five lane-id bits in the source-lane expression. Layout
// lowering can assemble this field from several packed coordinates. Memoize
// the search and known-zero bits, and coalesce residuals with the same shift
// rather than expanding the expression DAG or emitting each bit separately.
class LaneXorMatcher {
public:
  bool match(Value value) {
    for (laneBit = 0; laneBit < 5; ++laneBit) {
      visited.clear();
      if (!match(value, laneBit))
        return false;
    }
    return true;
  }

  Value getResidual(Location loc, PatternRewriter &rewriter) {
    Value result;
    for (auto [projection, bits] : residuals) {
      if (!bits)
        continue;
      auto [value, shift] = projection;
      unsigned width = cast<IntegerType>(value.getType()).getWidth();
      if (width < 32)
        value = LLVM::ZExtOp::create(rewriter, loc, rewriter.getI32Type(), value);
      if (shift) {
        Value amount = LLVM::ConstantOp::create(
            rewriter, loc, value.getType(),
            rewriter.getIntegerAttr(value.getType(), shift > 0 ? shift : -shift));
        if (shift > 0)
          value = LLVM::LShrOp::create(rewriter, loc, value, amount);
        else
          value = LLVM::ShlOp::create(rewriter, loc, value, amount);
      }
      if (width > 32)
        value = LLVM::TruncOp::create(rewriter, loc, rewriter.getI32Type(), value);
      if (bits != 31) {
        Value mask = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                             rewriter.getI32IntegerAttr(bits));
        value = LLVM::AndOp::create(rewriter, loc, value, mask);
      }
      if (result)
        result = LLVM::XOrOp::create(rewriter, loc, result, value);
      else
        result = value;
    }
    if (!result)
      result = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                       rewriter.getI32IntegerAttr(0));
    return result;
  }

private:
  APInt getKnownZeroBits(Value value) {
    auto it = knownZeroBits.find(value);
    if (it != knownZeroBits.end())
      return it->second;
    unsigned width = cast<IntegerType>(value.getType()).getWidth();
    APInt zeros(width, 0), constant;
    Operation *op = value.getDefiningOp();
    if (matchPattern(value, m_ConstantInt(&constant))) {
      zeros = ~constant.zextOrTrunc(width);
    } else if (op && isa<NVVM::LaneIdOp>(op)) {
      zeros = APInt::getHighBitsSet(width, width - 5);
    } else if (op && isa<LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp>(op)) {
      APInt lhs = getKnownZeroBits(op->getOperand(0));
      APInt rhs = getKnownZeroBits(op->getOperand(1));
      zeros = isa<LLVM::AndOp>(op) ? lhs | rhs : lhs & rhs;
    } else if (op && isa<LLVM::ShlOp, LLVM::LShrOp, LLVM::AShrOp>(op) &&
               matchPattern(op->getOperand(1), m_ConstantInt(&constant)) &&
               constant.ult(width)) {
      unsigned shift = constant.getZExtValue();
      zeros = getKnownZeroBits(op->getOperand(0));
      if (isa<LLVM::ShlOp>(op))
        zeros = (zeros << shift) | APInt::getLowBitsSet(width, shift);
      else if (isa<LLVM::LShrOp>(op))
        zeros = zeros.lshr(shift) | APInt::getHighBitsSet(width, shift);
      else
        zeros = zeros.ashr(shift);
    } else if (op && isa<LLVM::TruncOp, LLVM::ZExtOp, LLVM::SExtOp>(op)) {
      zeros = getKnownZeroBits(op->getOperand(0));
      if (isa<LLVM::ZExtOp>(op))
        zeros = zeros.zext(width) |
                APInt::getHighBitsSet(width, width - zeros.getBitWidth());
      else
        zeros = zeros.sextOrTrunc(width);
    } else if (op && isa<LLVM::URemOp>(op) &&
               matchPattern(op->getOperand(1), m_ConstantInt(&constant)) &&
               constant.isPowerOf2()) {
      zeros = getKnownZeroBits(op->getOperand(0)) |
              ~(constant.zextOrTrunc(width) - 1);
    }
    knownZeroBits.try_emplace(value, zeros);
    return zeros;
  }

  bool match(Value value, unsigned bit) {
    unsigned width = cast<IntegerType>(value.getType()).getWidth();
    Operation *op = value.getDefiningOp();
    if (!op || bit >= width || !visited.insert({value, bit}).second)
      return false;

    // Triton uses one-dimensional thread blocks, so tid.x has the same low bits.
    if (isa<NVVM::LaneIdOp, NVVM::ThreadIdXOp>(op))
      return bit == laneBit;

    if (auto xorOp = dyn_cast<LLVM::XOrOp>(op)) {
      if (match(xorOp.getLhs(), bit)) {
        residuals[{xorOp.getRhs(), int(bit) - int(laneBit)}] ^= 1u << laneBit;
        return true;
      }
      if (match(xorOp.getRhs(), bit)) {
        residuals[{xorOp.getLhs(), int(bit) - int(laneBit)}] ^= 1u << laneBit;
        return true;
      }
      return false;
    }

    APInt bits = APInt::getOneBitSet(width, bit);
    if (auto orOp = dyn_cast<LLVM::OrOp>(op)) {
      if ((getKnownZeroBits(orOp.getRhs()) & bits) == bits)
        return match(orOp.getLhs(), bit);
      if ((getKnownZeroBits(orOp.getLhs()) & bits) == bits)
        return match(orOp.getRhs(), bit);
      return false;
    }

    if (isa<LLVM::TruncOp, LLVM::ZExtOp, LLVM::SExtOp>(op))
      return match(op->getOperand(0), bit);

    APInt constant;
    if (isa<LLVM::ShlOp, LLVM::LShrOp, LLVM::AShrOp>(op)) {
      if (!matchPattern(op->getOperand(1), m_ConstantInt(&constant)) ||
          constant.uge(width))
        return false;
      unsigned amount = constant.getZExtValue();
      if (isa<LLVM::ShlOp>(op))
        return bit >= amount && match(op->getOperand(0), bit - amount);
      return match(op->getOperand(0), bit + amount);
    }

    if (!isa<LLVM::AndOp, LLVM::AddOp, LLVM::SubOp, LLVM::URemOp>(op))
      return false;
    Value input = op->getOperand(0);
    Value other = op->getOperand(1);
    if (!matchPattern(other, m_ConstantInt(&constant))) {
      if (!isa<LLVM::AndOp, LLVM::AddOp>(op) ||
          !matchPattern(input, m_ConstantInt(&constant)))
        return false;
      input = other;
    }
    constant = constant.zextOrTrunc(width);
    bool preservesBits =
        (isa<LLVM::AndOp>(op) && (constant & bits) == bits) ||
        (isa<LLVM::AddOp, LLVM::SubOp>(op) &&
         constant.countr_zero() > bit) ||
        (isa<LLVM::URemOp>(op) && !constant.isZero() &&
         constant.countr_zero() > bit);
    return preservesBits && match(input, bit);
  }

  DenseMap<Value, APInt> knownZeroBits;
  DenseSet<std::pair<Value, unsigned>> visited;
  llvm::MapVector<std::pair<Value, int>, unsigned> residuals;
  unsigned laneBit;
};

class ShuffleXorPattern : public OpRewritePattern<NVVM::ShflOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NVVM::ShflOp op,
                                PatternRewriter &rewriter) const override {
    APInt clamp;
    if (op.getKind() != NVVM::ShflKind::idx ||
        !matchPattern(op.getMaskAndClamp(), m_ConstantInt(&clamp)) ||
        clamp != 31)
      return failure();

    LaneXorMatcher matcher;
    if (!matcher.match(op.getOffset()))
      return failure();
    Value offset = matcher.getResidual(op.getLoc(), rewriter);

    // With no segmentation and clamp 31, both modes select the same lane and
    // return a true validity predicate. Keep the member mask and convergence.
    rewriter.modifyOpInPlace(op, [&] {
      op.setKind(NVVM::ShflKind::bfly);
      op.getOffsetMutable().assign(offset);
    });
    return success();
  }
};
} // namespace

namespace {
struct CanonicalizeLLVMIR
    : public mlir::triton::gpu::impl::CanonicalizeLLVMIRBase<
          CanonicalizeLLVMIR> {
  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<SelectConstantConditionPattern,
                 ElideFullClusterRankMaskPattern, ShuffleXorPattern>(
        &getContext());

    getContext()
        .getLoadedDialect<LLVM::LLVMDialect>()
        ->getCanonicalizationPatterns(patterns);
    for (mlir::RegisteredOperationName op :
         getContext().getRegisteredOperationsByDialect(
             LLVM::LLVMDialect::getDialectNamespace()))
      op.getCanonicalizationPatterns(patterns, &getContext());

    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace
