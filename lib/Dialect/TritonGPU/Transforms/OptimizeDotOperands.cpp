#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <memory>

namespace mlir {
namespace triton {
namespace gpu {

namespace {

// Helpers

// Returns whether we can hoist DotOp Encoding through `op`.
// Roughly, whether op is elementwise and thus threads don't need
// to exchange elements. But some ops are not current supported even though
// they meet that criterion.
bool canHoistDotOpEncV2(Operation* op, DotOperandEncodingAttr& dotOpEnc) {
  // Only consider custom conversions or arith ops.
  // TODO(jlebar): Is this too restrictive?
  if (!isa<FpToFpOp, BitcastOp>(op) && !isPureUnaryInlineAsm(op) &&
      op->getDialect()->getTypeID() != TypeID::get<arith::ArithDialect>())
    return false;

  // Quick handling to fix loading issues when computing the original
  // bitwidth is unable to realize that there is a mixed-precision dot
  // (hence kWidth = 1) but wants to hoist through the type conversion.
  if (isa<arith::ExtFOp>(op) && dotOpEnc.getKWidth() == 1)
      return false;

  // Currently, these instructions are not supported during lowering of
  // shared -> dot_operand layout. Not all types and type conversions are
  // supported.
  if (isa<arith::TruncIOp, arith::TruncFOp, arith::SelectOp>(op))
    return false;

  // Don't hoist through u1 -> fp casts as they aren't supported in
  // ElementwiseOpToLLVM::reorderValues().
  if (isa<arith::UIToFPOp>(op)) {
    Type opType = getElementTypeOrSelf(op->getOperand(0));
    if (opType.isInteger(1))
      return false;
  }

  return true;
}

bool canHoistDotOpEncV3(Operation* op) {
  // Only consider custom conversions or arith ops.
  // TODO(jlebar): Is this too restrictive?
  if (!isa<FpToFpOp, BitcastOp>(op) && !isPureUnaryInlineAsm(op) &&
      op->getDialect()->getTypeID() != TypeID::get<arith::ArithDialect>())
    return false;

  // Currently, these instructions are not supported during lowering of
  // shared -> dot_operand layout. Not all types and type conversions are
  // supported.
  if (isa<arith::TruncIOp, arith::TruncFOp, arith::SelectOp>(op))
    return false;

  // Don't hoist through u1 -> fp casts as they aren't supported in
  // ElementwiseOpToLLVM::reorderValues().
  if (isa<arith::UIToFPOp>(op)) {
    Type opType = getElementTypeOrSelf(op->getOperand(0));
    if (opType.isInteger(1))
      return false;
  }

  return true;
}

// Given
//   convert(trans(src)) #dot_operand ->
//   convert(local_load(trans(alloc(src))))
// change the encoding of the inner convert to a special, swizzled shared
// encoding.
class SwizzleShmemConvert : public OpRewritePattern<ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp cvtOp,
                                PatternRewriter &rewriter) const override {
    // Match outerCvt(trans(innerCvt(x))).
    auto trans = cvtOp.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>{1, 0})
      return failure();

    auto srcTy = dyn_cast<RankedTensorType>(trans.getSrc().getType());

    if (auto srcCvt = trans.getSrc().getDefiningOp<ConvertLayoutOp>()) {
      srcTy = srcCvt.getSrc().getType();
    }
    auto sharedLoadTy = cast<RankedTensorType>(cvtOp.getType());
    auto cvtEncoding =
        dyn_cast<DotOperandEncodingAttr>(sharedLoadTy.getEncoding());
    if (!cvtEncoding)
      return failure();

    // TODO(Qingyi): need to check whether the CTALayout of innerCvtEnc should
    // be used here. For tests where numCTAs = 1, this is not a problem since
    // all CTALayouts are the same.
    //
    // Set needTrans to true here. newInnerCvtEnc is computed based on
    // argEncoding which is before the transpose. Without needTrans we will
    // compute vec and maxPhase based on incorrect m, n and k size of mma. The
    // type inference of TransOp simply swap the order but doesn't fix the vec
    // and maxPhase for the YType, hence it would causing incorrect swizzling
    // code.
    auto newInnerCvtEnc =
        SharedEncodingAttr::get(getContext(), cvtEncoding, srcTy.getShape(),
                                /*order=*/getOrder(srcTy.getEncoding()),
                                triton::gpu::getCTALayout(srcTy.getEncoding()),
                                srcTy.getElementType(), /*needTrans=*/true);
    if (newInnerCvtEnc == cvtEncoding)
      return failure();
    rewriter.setInsertionPoint(trans);
    auto sharedMemorySpace = SharedMemorySpaceAttr::get(getContext());
    auto alloc = rewriter.create<LocalAllocOp>(
        trans.getLoc(),
        MemDescType::get(srcTy.getShape(), srcTy.getElementType(),
                         newInnerCvtEnc, sharedMemorySpace),
        trans.getSrc());
    auto newTrans = rewriter.create<TransOp>(trans.getLoc(), alloc,
                                             ArrayRef<int32_t>({1, 0}));
    rewriter.replaceOpWithNewOp<LocalLoadOp>(trans, sharedLoadTy, newTrans);
    return success();
  }
};

// Move convert-to-dot-operand "up" past elementwise ops:
//
//  convert(elementwise(x)) #dot_operand ->
//  elementwise(convert(x, #dot_operand)).
//
// The goal is to put the convert right next to the originating load.  If we can
// accomplish this, then we can save a shmem round-trip:
//
//   Before:
//
//     - Load from global into shmem using an async copy.
//     - Load from shmem into a #blocked layout.
//     - Do elementwise ops over #blocked layout.
//     - Convert to #dot_operand (round-trip through shmem).
//     - Do dot.
//
//   After:
//
//     - Load from global into shmem using an async copy (same as before).
//     - Load from shmem into a #dot_operand layout.
//     - Do elementwise ops over #dot_operand layout.
//     - Do dot.
//
// Eliminating the shmem round-trip is such a big win, we're willing to do it
// even if this duplicates work because some of the elementwise ops have uses
// that don't flow into the dot.  On the other hand, we only want to do this if
// we can in fact reduce shmem round-trips: For example, simply moving a convert
// up above e.g. an `add` now means we have *two* converts.  That's worse,
// unless we can continue moving the converts upwards and eventually merge them.
// So we try to check that this will be beneficial before making any changes.
class HoistLayoutConversion : public OpRewritePattern<ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvertLayoutOp cvt,
                                PatternRewriter &rewriter) const override {
    // Only consider conversions to dot operand.
    auto cvtTy = cast<RankedTensorType>(cvt.getType());
    auto dotOpEnc = dyn_cast<DotOperandEncodingAttr>(cvtTy.getEncoding());
    if (!dotOpEnc)
      return failure();

    auto src = cvt.getSrc().getDefiningOp();
    if (!src || src->getNumOperands() == 0 || src->getNumResults() != 1)
      return failure();

    auto srcTy = dyn_cast<RankedTensorType>(src->getResult(0).getType());
    if (!srcTy)
      return failure();

    if (!all_of(src->getOperandTypes(),
                [](Type ty) { return isa<RankedTensorType>(ty); }))
      return failure();

    if (!canHoistDotOpEncV2(src, dotOpEnc))
      return failure();

    // Check that the conversion is transitively dependent on a load, and all
    // operations between the load and the conversion are layout preserving.
    //
    // TODO(jlebar): This is accidentally quadratic; we iterate over the whole
    // slice but then at the end we only modify one op!
    SetVector<Operation *> slice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    // TODO(jlebar): Is this filter redundant with omitBlockArguments == true?
    // That is, is it possible to get into a different region without going
    // through a block argument?
    opt.filter = [&](Operation *op) {
      return op->getParentRegion() == cvt->getParentRegion();
    };
    getBackwardSlice(cvt.getOperation(), &slice, opt);

    // TODO(jlebar): This is too conservative when there are multiple loads in
    // the chain (e.g. cvt(load(x) + load(y))).  The intent is to check that all
    // of the ops between the loads and the convert are elementwise.  But
    // actually we set foundLoad = true once we see the first load, and so we
    // will reject the chain if the *second* load we encounter uses a
    // non-elementwise op to calculate its pointers.
    bool foundLoad = false;
    for (Operation *currOp : slice) {
      if (isa<LoadOp>(currOp)) {
        foundLoad = true;
      } else if (foundLoad) {
        if (!canHoistDotOpEncV2(currOp, dotOpEnc))
          return failure();
      }
    }
    if (!foundLoad)
      return failure();

    SmallVector<ConvertLayoutOp> newOperands;
    for (auto operand : src->getOperands()) {
      // We checked earlier that all operands are ranked tensors.
      auto operandTy = cast<RankedTensorType>(operand.getType());
      Type newCvtTy = RankedTensorType::get(
          srcTy.getShape(), operandTy.getElementType(), cvtTy.getEncoding());
      newOperands.push_back(
          rewriter.create<ConvertLayoutOp>(cvt.getLoc(), newCvtTy, operand));
    }
    auto newRet = rewriter.clone(*src);
    for (int i = 0; i < newOperands.size(); i++)
      newRet->setOperand(i, newOperands[i]);
    newRet->getResult(0).setType(RankedTensorType::get(
        srcTy.getShape(), srcTy.getElementType(), cvtTy.getEncoding()));

    rewriter.replaceOp(cvt, newRet->getResults());
    return success();
  }
};

// Rewrite
//
//   dot(alloc(trans() #shared1) ->
//   dot(trans(alloc() #shared2))
//
// if dot is an MMAv3 (because MMAv3 allows us to fold transposes).
class FuseTransHopper : public OpRewritePattern<LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LocalAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp->hasOneUse() ||
        !allocOp->getUsers().begin()->hasTrait<OpTrait::DotLike>())
      return failure();

    auto dot = *allocOp->getUsers().begin();
    if (!allocOp.getSrc())
      return failure();

    // Match outerCvt(trans(innerCvt(x))).
    auto trans = allocOp.getSrc().getDefiningOp<TransOp>();
    if (!trans || trans.getOrder() != ArrayRef<int32_t>({1, 0}))
      return failure();

    MemDescType allocType = allocOp.getType();
    auto allocEncoding = cast<SharedEncodingAttr>(allocType.getEncoding());
    TensorOrMemDesc srcTy = trans.getSrc().getType();

    // MMAv3 with transpose only supports f16 and bf16.  Fall back to MMAv3
    // without transpose for other data types.)
    auto newInnerCvtOrder = getOrder(srcTy.getEncoding());
    if (auto cvt = trans.getSrc().getDefiningOp<ConvertLayoutOp>()) {
      newInnerCvtOrder = getOrder(cvt.getSrc().getType().getEncoding());
    }
    auto srcElemTy = allocType.getElementType();
    if (!srcElemTy.isF16() && !srcElemTy.isBF16()) {
      if (allocOp.getResult() == dot->getOperand(0)) {
        newInnerCvtOrder = {0, 1};
      } else if (allocOp.getResult() == dot->getOperand(1)) {
        newInnerCvtOrder = {1, 0};
      }
    }

    // TODO(Qingyi): need to check whether the CTALayout of innerCvtEnc should
    // be used here. For tests where numCTAs = 1, this is not a problem since
    // all CTALayouts are the same.
    auto newInnerEnc = SharedEncodingAttr::get(
        getContext(), srcTy.getShape(), newInnerCvtOrder,
        allocEncoding.getCTALayout(), srcTy.getElementType());

    MemDescType innerTy =
        MemDescType::get(srcTy.getShape(), srcTy.getElementType(), newInnerEnc,
                         allocType.getMemorySpace());
    auto newAlloc = rewriter.create<LocalAllocOp>(allocOp.getLoc(), innerTy,
                                                  trans.getSrc());
    rewriter.replaceOpWithNewOp<TransOp>(allocOp, newAlloc,
                                         ArrayRef<int32_t>({1, 0}));
    return success();
  }
};

// Rewrite
//   dot(convert(lhs #mma) #shared, rhs) #mma ->
//   dot(convert(lhs #mma) #dot_operand, rhs) #mma,
// for fp16 or bf16 MMAv3 dots.
struct MMAV3UseRegOperand
    : public OpRewritePattern<triton::nvidia_gpu::WarpGroupDotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::nvidia_gpu::WarpGroupDotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto alloc = dotOp.getOperand(0).getDefiningOp<LocalAllocOp>();
    if (!alloc || !alloc.getSrc())
      return failure();

    auto getEncoding = [](Value v) {
      return cast<TensorOrMemDesc>(v.getType()).getEncoding();
    };

    if (!isa<SharedEncodingAttr>(getEncoding(dotOp.getOperand(0))))
      return failure();
    auto srcEnc = dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(alloc.getSrc()));
    auto dstEnc =
        dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(dotOp.getResult()));
    if (!srcEnc || srcEnc.getVersionMajor() != 3 || !dstEnc ||
        dstEnc.getVersionMajor() != 3)
      return failure();
    auto srcTy = cast<RankedTensorType>(alloc.getSrc().getType());
    auto kWidth = 32 / srcTy.getElementTypeBitWidth();
    auto dotOperandEnc = DotOperandEncodingAttr::get(
        dotOp.getContext(), /*opIdx=*/0, srcEnc, /*kWidth=*/kWidth);
    auto newTy = RankedTensorType::get(srcTy.getShape(), srcTy.getElementType(),
                                       dotOperandEnc);
    if (!matchMmaV3AndDotOperandLayout(srcTy, newTy))
      return failure();

    Value newOperand =
        rewriter.create<ConvertLayoutOp>(dotOp.getLoc(), newTy, alloc.getSrc());
    rewriter.modifyOpInPlace(dotOp, [&]() { dotOp.setOperand(0, newOperand); });
    return success();
  }
};

// TODO(ggengnv) more tests (multiple elt-wise ops) and document
struct MMAV3HoistLayoutConversion
    : public OpRewritePattern<triton::nvidia_gpu::WarpGroupDotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::nvidia_gpu::WarpGroupDotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto alloc = dotOp.getOperand(0).getDefiningOp<LocalAllocOp>();
    if (!alloc || !alloc.getSrc())
      return failure();

    auto getEncoding = [](Value v) {
      return cast<TensorOrMemDesc>(v.getType()).getEncoding();
    };

    if (!isa<SharedEncodingAttr>(getEncoding(dotOp.getOperand(0))))
      return failure();

    // Performs checks for early stop
    NvidiaMmaEncodingAttr dstEnc;
    {
      auto srcEnc = dyn_cast<BlockedEncodingAttr>(getEncoding(alloc.getSrc()));
      dstEnc =
          dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(dotOp.getResult()));
      // Want: A's Encoding to be Blocked and D's encoding to be NvidiaMmA v3
      if (!srcEnc || !dstEnc || dstEnc.getVersionMajor() != 3)
        return failure();

      auto src = alloc.getSrc().getDefiningOp();

      // Value passed to alloc must have Tensor arguments and single Tensor result
      if (!src || src->getNumOperands() == 0 || src->getNumResults() != 1)
        return failure();
      if (!all_of(src->getOperandTypes(),
                  [](Type ty) { return isa<RankedTensorType>(ty); }))
        return failure();
      auto srcTy = dyn_cast<RankedTensorType>(src->getResult(0).getType());
      if (!srcTy)
        return failure();

      if (!canHoistDotOpEncV3(src))
        return failure();
    }

    SetVector<Operation *> slice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = [&](Operation *op) {
      return (op->getParentRegion() == alloc->getParentRegion()) && !isa<LoadOp, LocalLoadOp>(op)
        && (op->getNumOperands() != 0);  // Ensures all ops in slice have operands
    };

    getBackwardSlice(alloc.getOperation(), &slice, opt);

    auto isBlockedRankedTensor = [&](auto val) {
      return isa<BlockedEncodingAttr>(getEncoding(val)) && isa<RankedTensorType>(val.getType());
    };

    SmallVector<Operation *> frontierOps;
    for (Operation *currOp : slice) {
      if (!canHoistDotOpEncV3(currOp))
        return failure();

      // We previously ensured that all ops in slice have at least one operand
      bool isFrontier = false;
      for (auto operand : currOp->getOperands()) {
        auto op = operand.getDefiningOp();
        if (!slice.contains(op)) {
          // TODO that this is overly restrictive. Can add support for ConstantOp and LocalLoad
          if (!isa<LoadOp>(op))
            return failure();

          isFrontier = true;
        }
      }

      if (isFrontier) {
        if (!isa<LoadOp>(currOp->getOperand(0).getDefiningOp()))
          return failure();

        auto res = currOp->getResult(0);
        if (!isBlockedRankedTensor(res))
          return failure();

        if (!llvm::all_of(currOp->getOperands(), isBlockedRankedTensor))
          return failure();

        frontierOps.push_back(currOp);
      }
    }

    // Nothing to hoist through
    if (frontierOps.empty())
      return failure();

    auto dotOperandEnc = DotOperandEncodingAttr::get(
        dotOp.getContext(), /*opIdx=*/0, dstEnc, /*kWidth=*/0);

    // For each frontierOp:
    //  load; frontierOp; ...; warp_group_dot
    //  -> load; local_alloc; local_load; convert_layout; frontierOp; ...; warp_group_dot
    for (Operation *frontierOp : frontierOps) {
      auto frontierTy = dyn_cast<RankedTensorType>(frontierOp->getResult(0).getType());

      SmallVector<ConvertLayoutOp> newOperands;
      for (auto operand : frontierOp->getOperands()) {
        // We checked earlier that all operands are ranked tensors.
        auto operandTy = cast<RankedTensorType>(operand.getType());
        auto operandEltTy = operandTy.getElementType();

        auto oldAllocTy = alloc.getType();
        // TODO(ggengnv) previous encoding (oldAllocTy.getEncoding()) was for shared operand.
        // Is it still appropriate for loading into registers?
        auto newAllocTy = MemDescType::get(operandTy.getShape(), operandEltTy,
                                        oldAllocTy.getEncoding(), oldAllocTy.getMemorySpace());
        auto localAlloc = rewriter.create<LocalAllocOp>(alloc.getLoc(), newAllocTy, operand);
        auto localLoad = rewriter.create<LocalLoadOp>(alloc.getLoc(), operandTy, localAlloc);

        Type cvtTy = RankedTensorType::get(
            operandTy.getShape(), operandTy.getElementType(), dotOperandEnc);
        auto cvt = rewriter.create<ConvertLayoutOp>(alloc.getLoc(), cvtTy, localLoad);

        newOperands.push_back(cvt);
      }

      auto newFrontier = rewriter.clone(*frontierOp);
      for (int i = 0; i < newOperands.size(); i++)
        newFrontier->setOperand(i, newOperands[i]);
      newFrontier->getResult(0).setType(RankedTensorType::get(
          frontierTy.getShape(), frontierTy.getElementType(), dotOperandEnc));

      rewriter.replaceOp(frontierOp, newFrontier);
    }

    // replace LHS operand with its parent (in dotOpEnc)
    rewriter.modifyOpInPlace(dotOp, [&]() { dotOp.setOperand(0, alloc.getSrc()); });

    return success();
  }
};

} // namespace

#define GEN_PASS_DEF_TRITONGPUOPTIMIZEDOTOPERANDS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUOptimizeDotOperandsPass
    : public impl::TritonGPUOptimizeDotOperandsBase<
          TritonGPUOptimizeDotOperandsPass> {
public:
  using impl::TritonGPUOptimizeDotOperandsBase<
      TritonGPUOptimizeDotOperandsPass>::TritonGPUOptimizeDotOperandsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::PassManager pm(m.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    auto ret = pm.run(m);

    mlir::RewritePatternSet patterns(context);
    patterns.add<SwizzleShmemConvert>(context);
    if (this->hoistLayoutConversion.getValue()) {
      patterns.add<HoistLayoutConversion>(context);
    }
    patterns.add<FuseTransHopper>(context);
    patterns.add<MMAV3UseRegOperand>(context);
    patterns.add<MMAV3HoistLayoutConversion>(context);
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
