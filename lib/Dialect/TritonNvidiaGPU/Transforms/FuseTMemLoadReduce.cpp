#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TargetFeatures.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUFUSETMEMLOADREDUCEPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Strip away all intermediate ttg.convert_layout ops to reach the true
// producer.
static Value stripConvertLayout(Value v) {
  while (auto cvt = v.getDefiningOp<ttg::ConvertLayoutOp>())
    v = cvt.getSrc();
  return v;
}

// Combine "ttng.tmem_load" and "tt.reduce" into "ttng.tmem_load" if it
// has the `redOp` attribute.  This targets the PTX `tcgen05.ld.red`
// instruction on Blackwell (sm103+).
//
// Match:
//
//   %v = ttng.tmem_load %tmem :
//        !ttg.memdesc<MxNxf32, #tmem, ...> -> tensor<MxNxf32, #blocked>
//   [ %cvt = ttg.convert_layout %v ... ] // optional
//   %r  = "tt.reduce"(%cvt or %v) ({...max/min combiner...}) {axis = 1}
//
// And rewrite this to:
//
//   %v, %r' = ttng.tmem_load %tmem {redOp = #ttng.redOp<max|min>, NaN = ...}
//             : ... -> tensor<MxNxf32, #blocked>, tensor<Mxf32,
//             slice(#blocked)>
//   [ %r = ttg.convert_layout %r' ]
//
// I.e., the fused load operation additionally performs an
// element-wise reduction along the N-dimension of the input and produces a
// second result tensor %r'. For a input of shape [M, N], the
// reduced result has shape [M], containing one reduced value per "slice"
// of the N-dimension.

class FuseTMemLoadReducePattern : public OpRewritePattern<triton::ReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    // Instruction `tcgen05.ld.red` is only available on Blackwell sm103+.
    auto targetFeatures =
        TargetFeatures::fromModuleOp(reduceOp->getParentOfType<ModuleOp>());
    if (!targetFeatures.supportLdRed())
      return failure();

    // Bail if the region isn't a trivial shape, it should have exactly one
    // operand and one result.
    Operation *combiner = reduceOp.getSingleCombiner();
    if (!combiner)
      return failure();

    // Only support reduction along the N-dimension.
    if (reduceOp.getAxis() != 1)
      return failure();

    // Look through "convert_layout" to find the "tmem_load", which is
    // guaranteed to produce a result with rank 2.
    auto tmemLoad = stripConvertLayout(reduceOp.getOperands()[0])
                        .getDefiningOp<TMEMLoadOp>();
    if (!tmemLoad)
      return failure();

    // Skip if already a fused load.
    if (tmemLoad.getRedOp())
      return failure();

    // This is not a HW/PTX restriction, tcgen05.ld.red supports integer integer
    // types, but this is a Triton limitation: the reduction is restricted to
    // f32, element types, see the definition of TTNG_TMEMLoadOp.
    if (!tmemLoad.getType().getElementType().isF32())
      return failure();

    TMEMLoadReduceModifier redOpKind;
    bool propagateNaN;
    if (isa<arith::MaxNumFOp>(combiner)) {
      // MaxNumFOp: if one of the arguments is NaN, the result is also NaN.
      redOpKind = TMEMLoadReduceModifier::MAX;
      propagateNaN = false;
    } else if (isa<arith::MaximumFOp>(combiner)) {
      // MaximumFOp: if one of the arguments is NaN, the result is the other
      // argument.
      redOpKind = TMEMLoadReduceModifier::MAX;
      propagateNaN = true;
    } else if (isa<arith::MinNumFOp>(combiner)) {
      redOpKind = TMEMLoadReduceModifier::MIN;
      propagateNaN = false;
    } else if (isa<arith::MinimumFOp>(combiner)) {
      redOpKind = TMEMLoadReduceModifier::MIN;
      propagateNaN = true;
    } else {
      return failure();
    }

    // Verify the layout supports the fused "tcgen05.ld.red": the load must be
    // packed, each thread's register bases must span the full N axis and must
    // not advance M.
    auto maxnreg = getContextualMaxNReg(tmemLoad);
    if (!supportsTMemLoadReduce(tmemLoad.getType(), tmemLoad.getSrc().getType(),
                                maxnreg))
      return failure();

    // Now build the fused load.
    auto *ctx = tmemLoad.getContext();
    auto redOpAttr = TMEMLoadReduceModifierAttr::get(ctx, redOpKind);
    BoolAttr nanAttr = propagateNaN ? rewriter.getBoolAttr(true) : BoolAttr();
    Type tokenTy = tmemLoad.getToken() ? tmemLoad.getToken().getType() : Type();
    rewriter.setInsertionPoint(tmemLoad);
    auto newLoad = TMEMLoadOp::create(rewriter, tmemLoad.getLoc(),
                                      /*result=*/tmemLoad.getType(),
                                      /*token=*/tokenTy,
                                      /*src=*/tmemLoad.getSrc(),
                                      /*dep=*/tmemLoad.getDep(), redOpAttr,
                                      /*abs=*/BoolAttr(), nanAttr);

    // Replace original load uses (result + optional token).
    SmallVector<Value> loadReplacements{newLoad.getResult()};
    if (tmemLoad.getToken())
      loadReplacements.push_back(newLoad.getToken());
    rewriter.replaceOp(tmemLoad, loadReplacements);

    // Splice the reduce into the fused-load `red` result, inserting a layout
    // conversion if the slice encodings differ.
    Value redResult = newLoad.getRed();
    Type expectedTy = reduceOp->getResult(0).getType();
    if (redResult.getType() != expectedTy) {
      rewriter.setInsertionPoint(reduceOp);
      redResult = ttg::ConvertLayoutOp::create(rewriter, reduceOp.getLoc(),
                                               expectedTy, redResult);
    }
    rewriter.replaceOp(reduceOp, redResult);
    return success();
  }
};

} // anonymous namespace

class TritonNvidiaGPUFuseTMEMLoadReducePass
    : public impl::TritonNvidiaGPUFuseTMEMLoadReducePassBase<
          TritonNvidiaGPUFuseTMEMLoadReducePass> {
public:
  using TritonNvidiaGPUFuseTMEMLoadReducePassBase<
      TritonNvidiaGPUFuseTMEMLoadReducePass>::
      TritonNvidiaGPUFuseTMEMLoadReducePassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FuseTMemLoadReducePattern>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
