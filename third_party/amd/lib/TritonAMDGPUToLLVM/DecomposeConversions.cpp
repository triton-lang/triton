#include "TargetInfo.h"
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
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include <numeric>
#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;

class TritonAMDGPUDecomposeConversionsPass
    : public TritonAMDGPUDecomposeConversionsBase<
          TritonAMDGPUDecomposeConversionsPass> {
public:
  TritonAMDGPUDecomposeConversionsPass() = default;

  void runOnOperation() override {
    const int LDSSize = triton::AMD::TargetInfo::getLDSSize();
    ModuleOp mod = getOperation();
    mod.walk([&](triton::DotOp dotOp) -> void {
      OpBuilder builder(dotOp);

      uint32_t ldsUsage = 0;
      for (auto dotOperand : dotOp.getOperands()) {
        auto defOp = dotOperand.getDefiningOp();
        if (!defOp)
          continue;
        auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(defOp);

        if (!cvtOp)
          continue;

        auto srcType = cvtOp.getSrc().getType().cast<RankedTensorType>();
        auto dstType = cvtOp.getType().cast<RankedTensorType>();
        auto srcEncoding = srcType.getEncoding();
        auto dstDotOp = dstType.getEncoding()
                            .dyn_cast<triton::gpu::DotOperandEncodingAttr>();
        if (!dstDotOp)
          continue;
        auto parent = dstDotOp.getParent();
        if (parent && parent.isa<triton::gpu::MmaEncodingTrait>()) {
          // Suitable only for MMA layouts otherwise llvm scheduler is fine
          ldsUsage += mlir::triton::getCvtOpLDSUsage(cvtOp);
          // TODO: Need to define, what operand is more suitable to skip.
          if (ldsUsage > LDSSize)
            break;
        }

        if (srcEncoding.isa<triton::gpu::SharedEncodingAttr>())
          return;
        if (auto srcMfmaEncoding =
                srcEncoding.dyn_cast<triton::gpu::AMDMfmaEncodingAttr>()) {

          if (srcMfmaEncoding.getWarpsPerCTA()[1] == 1 &&
              srcMfmaEncoding.getIsTransposed() &&
              dstDotOp.getParent() == srcMfmaEncoding)
            return;
        }

        auto tmpType = triton::MemDescType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(),
                triton::gpu::getOrder(srcEncoding),
                triton::gpu::getCTALayout(srcEncoding),
                srcType.getElementType()));
        auto tmp = builder.create<triton::gpu::LocalAllocOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getSrc());
        auto newConvert = builder.create<triton::gpu::LocalLoadOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUDecomposeConversionsPass() {
  return std::make_unique<TritonAMDGPUDecomposeConversionsPass>();
}
