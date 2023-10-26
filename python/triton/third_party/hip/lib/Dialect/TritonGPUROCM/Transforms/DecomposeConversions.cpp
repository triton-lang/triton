#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/AnalysisROCM/Utility.h"
#include "triton/Dialect/TritonGPUROCM/IR/Dialect.h"
#include "triton/Dialect/TritonGPUROCM/Transforms/Passes.h"
#include "triton/Dialect/TritonGPUROCM/Transforms/TritonGPUConversion.h"
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPUROCM/Transforms/Passes.h.inc"

using namespace mlir;

class TritonGPUDecomposeConversionsPass
    : public TritonGPUDecomposeConversionsBase<
          TritonGPUDecomposeConversionsPass> {
public:
  TritonGPUDecomposeConversionsPass() = default;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu_rocm::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcEncoding = srcType.getEncoding();
      if (srcEncoding.isa<triton::gpu_rocm::SharedEncodingAttr>())
        return;
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu_rocm::DotOperandEncodingAttr>();
      if (!dstDotOp)
        return;
      if (auto srcMmaEncoding =
              srcEncoding.dyn_cast<triton::gpu_rocm::MmaEncodingAttr>()) {

        if (srcMmaEncoding.getVersionMajor() == 1 ||
            (srcMmaEncoding.getWarpsPerCTA()[1] == 1 &&
             dstDotOp.getParent() == srcMmaEncoding))
          return;
      }
#if 1
      if (auto srcMfmaEncoding =
              srcEncoding.dyn_cast<triton::gpu_rocm::MfmaEncodingAttr>()) {

        if (srcMfmaEncoding.getWarpsPerCTA()[1] == 1 &&
            srcMfmaEncoding.getIsTransposed() &&
            dstDotOp.getParent() == srcMfmaEncoding)
          return;
      }
#endif
      auto tmpType = RankedTensorType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu_rocm::SharedEncodingAttr::get(
              mod.getContext(), dstDotOp, srcType.getShape(),
              triton::gpu_rocm::getOrder(srcEncoding),
              triton::gpu_rocm::getCTALayout(srcEncoding),
              srcType.getElementType()));
      auto tmp = builder.create<triton::gpu_rocm::ConvertLayoutOp>(
          cvtOp.getLoc(), tmpType, cvtOp.getOperand());
      auto newConvert = builder.create<triton::gpu_rocm::ConvertLayoutOp>(
          cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUROCMDecomposeConversionsPass() {
  return std::make_unique<TritonGPUDecomposeConversionsPass>();
}
