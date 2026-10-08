#include "PatternTritonGPUOpToLLVM.h"

#include <cassert>
#include <cstddef>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ReduceOpToLLVMBase.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

// Classify the source IR before dialect conversion can replace the extending
// operations. Lowering consults this analysis without following other ops.
class WidenedF32AddReductionAnalysis {
public:
  explicit WidenedF32AddReductionAnalysis(ModuleOp module) {
    module.walk([&](ReduceOp op) {
      if (isWidenedF32AddReduction(op))
        reductions.insert(op);
    });
  }

  bool contains(ReduceOp op) const { return reductions.contains(op); }

private:
  bool isWidenedF32AddReduction(ReduceOp op) const {
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return false;

    auto ext = op.getOperand(0).getDefiningOp<arith::ExtFOp>();
    if (!ext)
      return false;
    auto srcTy = dyn_cast<RankedTensorType>(ext.getIn().getType());
    auto dstTy = dyn_cast<RankedTensorType>(ext.getType());
    if (!srcTy || !dstTy || !dstTy.getElementType().isF32())
      return false;
    Type srcElemTy = srcTy.getElementType();
    if (!srcElemTy.isF16() && !srcElemTy.isBF16())
      return false;

    Block &block = op.getCombineOp().front();
    return block.getOperations().size() == 2 &&
           isa<arith::AddFOp>(block.front());
  }

  DenseSet<Operation *> reductions;
};

// Lowers f16/bf16 values widened to f32 through one in-thread add accumulator
// chain to exploit Rubin's mixed-precision add instruction to accumulate f16 /
// bf16 elements into f32.
class WidenedF32AddReduceOpConversion final : public ReduceOpConversionBase {
public:
  WidenedF32AddReduceOpConversion(
      LLVMTypeConverter &typeConverter, const NVIDIA::TargetInfo &targetInfo,
      const WidenedF32AddReductionAnalysis &analysis, PatternBenefit benefit)
      : ReduceOpConversionBase(typeConverter, targetInfo, benefit),
        targetInfo(targetInfo), analysis(analysis) {}

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!matchesWidenedPackedAdd(op))
      return failure();

    // Now perform the rewrite by reusing the common reduction lowering.
    // Its reduceInThreadValues function is overridden here, so here we
    // do the more Rubin specific lowering of the reduction using a
    // serial accumulator chain as opposed to the tree-reduction in the
    // base class.
    return ReduceOpConversionBase::matchAndRewrite(op, adaptor, rewriter);
  }

protected:
  SmallVector<Value>
  reduceInThreadValues(Location loc, ConversionPatternRewriter &rewriter,
                       Region &combineOp,
                       SmallVector<SmallVector<Value>> values,
                       unsigned /*arity*/) const override {
    return createAccumulatorChain(loc, rewriter, combineOp, std::move(values));
  }

private:
  bool matchesWidenedPackedAdd(ReduceOp op) const {
    // NVPTX can select Rubin's mixed-precision add when the reduction values
    // are f16/bf16 extended to f32 and generic in-thread lowering packs them
    // into vector<2xf32> additions.
    return analysis.contains(op) && lowersToPackedF32Add(op);
  }

  SmallVector<Value>
  createAccumulatorChain(Location loc, ConversionPatternRewriter &rewriter,
                         Region &combineOp,
                         SmallVector<SmallVector<Value>> values) const {
    assert(!values.empty());
    SmallVector<Value> accumulator = std::move(values.front());
    for (size_t i = 1; i < values.size(); ++i) {
      SmallVector<Value> next = std::move(values[i]);
      // Keep the accumulator on the RHS so NVPTX can select Rubin's
      // mixed-precision add instruction.
      accumulator = combineValues(loc, rewriter, combineOp, next, accumulator);
    }
    return accumulator;
  }

  // Determine whether generic reduction lowering vectorizes the in-thread
  // reduction as vector<2xf32> additions.
  bool lowersToPackedF32Add(ReduceOp op) const {
    ReduceOpHelper helper(op);
    auto layout = mlir::triton::gpu::toLinearLayout(helper.getSrcTy());
    auto kRegister = StringAttr::get(op.getContext(), "register");
    layout = layout.removeZeroBasesAlongDim(kRegister);
    auto linearAttr =
        mlir::triton::gpu::LinearEncodingAttr::get(op.getContext(), layout);
    auto basesPerDim =
        linearAttr.basesPerDim(kRegister, /*skipBroadcast=*/true);
    unsigned numUniqueAxisElementsPerThread = basesPerDim[op.getAxis()];
    return numUniqueAxisElementsPerThread >= 4 &&
           helper.getInThreadVectorizeOpKind(
               targetInfo.supportBitwidth16Elementwise(),
               targetInfo.supportBitwidth32Elementwise()) ==
               ReduceOpHelper::InThreadVectorizeOpKind::AddF;
  }

  const NVIDIA::TargetInfo &targetInfo;
  const WidenedF32AddReductionAnalysis analysis;
};

} // namespace

void mlir::triton::NVIDIA::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    ModuleOp module, RewritePatternSet &patterns, PatternBenefit benefit) {
  if (targetInfo.getComputeCapability() == 107 &&
      targetInfo.getPtxVersion() >= 92) {
    WidenedF32AddReductionAnalysis analysis(module);
    patterns.add<WidenedF32AddReduceOpConversion>(
        typeConverter, targetInfo, analysis, benefit.getBenefit() + 1);
  }
  mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
}
