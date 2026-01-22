#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace triton {
namespace instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONINSTRUMENTFPSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

namespace {

Type getElementType(Type ty) {
  if (auto shaped = dyn_cast<ShapedType>(ty))
    return shaped.getElementType();
  return ty;
}

bool isFloatLike(Type ty) { return isa<FloatType>(getElementType(ty)); }

Type getIntTypeLike(Type ty) {
  auto elem = dyn_cast<FloatType>(getElementType(ty));
  if (!elem)
    return Type();

  auto *ctx = ty.getContext();
  auto intElem = IntegerType::get(ctx, elem.getWidth());
  if (auto ranked = dyn_cast<RankedTensorType>(ty))
    return RankedTensorType::get(ranked.getShape(), intElem,
                                 ranked.getEncoding());
  if (auto vec = dyn_cast<VectorType>(ty))
    return VectorType::get(vec.getShape(), intElem, vec.getScalableDims());
  if (auto shaped = dyn_cast<ShapedType>(ty))
    return shaped.clone(intElem);
  return intElem;
}

unsigned getIntBitwidth(Type ty) {
  auto elem = cast<IntegerType>(getElementType(ty));
  return elem.getWidth();
}

Value bitcastToInt(PatternRewriter &rewriter, Location loc, Value v) {
  auto intTy = getIntTypeLike(v.getType());
  return tt::BitcastOp::create(rewriter, loc, intTy, v);
}

Value bitcastToFloat(PatternRewriter &rewriter, Location loc, Value v,
                     Type floatTy) {
  return tt::BitcastOp::create(rewriter, loc, floatTy, v);
}

Value getIntConstantLike(PatternRewriter &rewriter, Location loc, Value like,
                         int64_t value) {
  auto ty = like.getType();
  if (auto shaped = dyn_cast<ShapedType>(ty)) {
    auto elem = cast<IntegerType>(shaped.getElementType());
    auto attr =
        DenseElementsAttr::get(shaped, rewriter.getIntegerAttr(elem, value));
    return arith::ConstantOp::create(rewriter, loc, attr);
  }
  auto intTy = cast<IntegerType>(ty);
  return arith::ConstantOp::create(rewriter, loc,
                                   rewriter.getIntegerAttr(intTy, value));
}

Value fpsanPow2ModInv(PatternRewriter &rewriter, Location loc, Value input) {
  auto one = getIntConstantLike(rewriter, loc, input, 1);
  auto two = getIntConstantLike(rewriter, loc, input, 2);

  auto a = arith::OrIOp::create(rewriter, loc, input, one);
  Value x = a;

  unsigned bitwidth = getIntBitwidth(input.getType());
  unsigned iters = 0;
  unsigned bits = 1;
  while (bits < bitwidth) {
    bits *= 2;
    iters += 1;
  }

  for (unsigned i = 0; i < iters; ++i) {
    auto ax = arith::MulIOp::create(rewriter, loc, a, x);
    auto twoMinusAx = arith::SubIOp::create(rewriter, loc, two, ax);
    x = arith::MulIOp::create(rewriter, loc, x, twoMinusAx);
  }
  return x;
}

Value fpsanFDiv(PatternRewriter &rewriter, Location loc, Value num, Value den) {
  auto numI = bitcastToInt(rewriter, loc, num);
  auto denI = bitcastToInt(rewriter, loc, den);
  auto inv = fpsanPow2ModInv(rewriter, loc, denI);
  auto resI = arith::MulIOp::create(rewriter, loc, numI, inv);
  return bitcastToFloat(rewriter, loc, resI, num.getType());
}

Value fpsanSRem(PatternRewriter &rewriter, Location loc, Value num, Value den) {
  auto numI = bitcastToInt(rewriter, loc, num);
  auto denI = bitcastToInt(rewriter, loc, den);
  auto one = getIntConstantLike(rewriter, loc, denI, 1);
  auto denSafe = arith::OrIOp::create(rewriter, loc, denI, one);
  auto resI = arith::RemSIOp::create(rewriter, loc, numI, denSafe);
  return bitcastToFloat(rewriter, loc, resI, num.getType());
}

template <typename OpF, typename OpI>
struct BinaryFloatToIntPattern : public OpRewritePattern<OpF> {
  using OpRewritePattern<OpF>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpF op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto loc = op.getLoc();
    auto lhsI = bitcastToInt(rewriter, loc, op.getLhs());
    auto rhsI = bitcastToInt(rewriter, loc, op.getRhs());
    auto resI = OpI::create(rewriter, loc, lhsI, rhsI);
    auto resF = bitcastToFloat(rewriter, loc, resI, op.getType());
    rewriter.replaceOp(op, resF);
    return success();
  }
};

struct DivFOpPattern : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    rewriter.replaceOp(
        op, fpsanFDiv(rewriter, op.getLoc(), op.getLhs(), op.getRhs()));
    return success();
  }
};

struct PreciseDivFOpPattern : public OpRewritePattern<tt::PreciseDivFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tt::PreciseDivFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    rewriter.replaceOp(op,
                       fpsanFDiv(rewriter, op.getLoc(), op.getX(), op.getY()));
    return success();
  }
};

struct RemFOpPattern : public OpRewritePattern<arith::RemFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    rewriter.replaceOp(
        op, fpsanSRem(rewriter, op.getLoc(), op.getLhs(), op.getRhs()));
    return success();
  }
};

struct FmaPattern : public OpRewritePattern<math::FmaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::FmaOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto loc = op.getLoc();
    auto aI = bitcastToInt(rewriter, loc, op.getA());
    auto bI = bitcastToInt(rewriter, loc, op.getB());
    auto cI = bitcastToInt(rewriter, loc, op.getC());
    auto mul = arith::MulIOp::create(rewriter, loc, aI, bI);
    auto sum = arith::AddIOp::create(rewriter, loc, mul, cI);
    auto resF = bitcastToFloat(rewriter, loc, sum, op.getType());
    rewriter.replaceOp(op, resF);
    return success();
  }
};

struct DotPattern : public OpRewritePattern<tt::DotOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tt::DotOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto loc = op.getLoc();
    auto aI = bitcastToInt(rewriter, loc, op.getA());
    auto bI = bitcastToInt(rewriter, loc, op.getB());
    auto cI = bitcastToInt(rewriter, loc, op.getC());
    auto dotI =
        tt::DotOp::create(rewriter, loc, aI, bI, cI, op.getInputPrecision(),
                          op.getMaxNumImpreciseAcc());
    auto resF = bitcastToFloat(rewriter, loc, dotI, op.getType());
    rewriter.replaceOp(op, resF);
    return success();
  }
};

struct TCGen5MMAPattern : public OpRewritePattern<ttng::TCGen5MMAOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttng::TCGen5MMAOp op,
                                PatternRewriter &rewriter) const override {
    auto aMemTy = cast<ttg::MemDescType>(op.getA().getType());
    auto bMemTy = cast<ttg::MemDescType>(op.getB().getType());
    auto dMemTy = cast<ttg::MemDescType>(op.getD().getType());

    if (!isa<FloatType>(aMemTy.getElementType()) ||
        !isa<FloatType>(bMemTy.getElementType()) ||
        !isa<FloatType>(dMemTy.getElementType()))
      return failure();

    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto numWarps = ttg::lookupNumWarps(op);

    auto getCgaLayout = [&](Attribute encoding, int rank) {
      if (auto layout = dyn_cast<ttg::LayoutEncodingTrait>(encoding))
        return layout.getCGALayout();
      return ttg::CGAEncodingAttr::get1CTALayout(ctx, rank);
    };

    auto accLayout = ttng::getDefaultLayoutForTmemLdSt(
        dMemTy, numWarps, getCgaLayout(dMemTy.getEncoding(), dMemTy.getRank()));
    auto accRegTy = RankedTensorType::get(dMemTy.getShape(),
                                          dMemTy.getElementType(), accLayout);

    Value accDep = op.getAccDep();
    auto accLoad = ttng::TMEMLoadOp::create(rewriter, loc, accRegTy, Type(),
                                            op.getD(), accDep);
    Value accReg = accLoad.getResult();

    Value accI = bitcastToInt(rewriter, loc, accReg);
    Value zeroI = getIntConstantLike(rewriter, loc, accI, 0);
    Value zeroF = bitcastToFloat(rewriter, loc, zeroI, accRegTy);
    Value accInit =
        arith::SelectOp::create(rewriter, loc, op.getUseD(), accReg, zeroF);

    auto loadOperand = [&](Value mem, bool isTensorMemory) -> Value {
      auto memTy = cast<ttg::MemDescType>(mem.getType());
      Attribute loadLayout = accLayout;
      if (isTensorMemory) {
        loadLayout = ttng::getDefaultLayoutForTmemLdSt(
            memTy, numWarps,
            getCgaLayout(memTy.getEncoding(), memTy.getRank()));
      }
      auto regTy = RankedTensorType::get(memTy.getShape(),
                                         memTy.getElementType(), loadLayout);
      if (isTensorMemory) {
        return ttng::TMEMLoadOp::create(rewriter, loc, regTy, Type(), mem,
                                        Value())
            .getResult();
      }
      return ttg::LocalLoadOp::create(rewriter, loc, regTy, mem, Value())
          .getResult();
    };

    bool aIsTmem = isa<ttng::TensorMemorySpaceAttr>(aMemTy.getMemorySpace());
    bool bIsTmem = isa<ttng::TensorMemorySpaceAttr>(bMemTy.getMemorySpace());

    if ((aIsTmem && aMemTy.getRank() != 2) ||
        (bIsTmem && bMemTy.getRank() != 2) || dMemTy.getRank() != 2)
      return failure();

    Value aReg = loadOperand(op.getA(), aIsTmem);
    Value bReg = loadOperand(op.getB(), bIsTmem);

    auto aDotEnc = ttg::DotOperandEncodingAttr::get(ctx, 0, accLayout,
                                                    aMemTy.getElementType());
    auto bDotEnc = ttg::DotOperandEncodingAttr::get(ctx, 1, accLayout,
                                                    bMemTy.getElementType());
    auto aDotTy = RankedTensorType::get(aMemTy.getShape(),
                                        aMemTy.getElementType(), aDotEnc);
    auto bDotTy = RankedTensorType::get(bMemTy.getShape(),
                                        bMemTy.getElementType(), bDotEnc);
    Value aDot =
        ttg::ConvertLayoutOp::create(rewriter, loc, aDotTy, aReg).getResult();
    Value bDot =
        ttg::ConvertLayoutOp::create(rewriter, loc, bDotTy, bReg).getResult();

    Value aDotI = bitcastToInt(rewriter, loc, aDot);
    Value bDotI = bitcastToInt(rewriter, loc, bDot);
    Value accInitI = bitcastToInt(rewriter, loc, accInit);
    auto dotI = tt::DotOp::create(rewriter, loc, accInitI.getType(), aDotI,
                                  bDotI, accInitI, tt::InputPrecision::IEEE, 0);
    Value dotF = bitcastToFloat(rewriter, loc, dotI, accRegTy);

    Type tokenTy = op.getNumResults() ? op.getToken().getType() : Type();
    auto store = ttng::TMEMStoreOp::create(rewriter, loc, tokenTy, op.getD(),
                                           Value(), dotF, op.getPred());

    if (!op.getBarriers().empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(store);
      auto barriers = op.getBarriers();
      auto barrierPreds = op.getBarrierPreds();
      for (size_t i = 0; i < barriers.size(); ++i) {
        Value pred =
            arith::AndIOp::create(rewriter, loc, op.getPred(), barrierPreds[i]);
        ttng::ArriveBarrierOp::create(rewriter, loc, barriers[i], 1, pred);
      }
    }

    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, store.getToken());
    }
    return success();
  }
};

template <typename OpTy>
struct IdentityPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

class FpSanitizerPass
    : public impl::TritonInstrumentFpSanitizerBase<FpSanitizerPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<BinaryFloatToIntPattern<arith::AddFOp, arith::AddIOp>,
                 BinaryFloatToIntPattern<arith::SubFOp, arith::SubIOp>,
                 BinaryFloatToIntPattern<arith::MulFOp, arith::MulIOp>,
                 DivFOpPattern, PreciseDivFOpPattern, RemFOpPattern, FmaPattern,
                 DotPattern, TCGen5MMAPattern, IdentityPattern<math::ExpOp>,
                 IdentityPattern<math::LogOp>, IdentityPattern<math::Exp2Op>,
                 IdentityPattern<math::Log2Op>, IdentityPattern<math::CosOp>,
                 IdentityPattern<math::SinOp>, IdentityPattern<math::SqrtOp>,
                 IdentityPattern<math::RsqrtOp>, IdentityPattern<math::ErfOp>,
                 IdentityPattern<math::FloorOp>, IdentityPattern<math::CeilOp>,
                 IdentityPattern<tt::PreciseSqrtOp>>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace instrument
} // namespace triton
} // namespace mlir
