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
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUREDUCEDATADUPLICATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUReduceDataDuplicationPass
    : public mlir::triton::gpu::impl::TritonGPUReduceDataDuplicationBase<
          TritonGPUReduceDataDuplicationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      if (isa<triton::gpu::SharedEncodingAttr>(srcEncoding))
        return;
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (!dstDotOp)
        return;
      if (auto srcMmaEncoding =
              dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>(srcEncoding)) {

        if (srcMmaEncoding.getVersionMajor() != 2 ||
            (srcMmaEncoding.getWarpsPerCTA()[1] == 1 &&
             dstDotOp.getParent() == srcMmaEncoding))
          return;
      }
      if (auto srcMfmaEncoding =
              dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(srcEncoding)) {

        if (srcMfmaEncoding.getWarpsPerCTA()[1] == 1 &&
            srcMfmaEncoding.getIsTransposed() &&
            dstDotOp.getParent() == srcMfmaEncoding)
          return;
      }
      auto srcOrder = triton::gpu::getOrder(srcEncoding);
      auto rank = srcOrder.size();
      SmallVector<unsigned> sharedOrder;
      if (rank == 3) {
        // add all elements except the element that is zero
        for (unsigned i = 0; i < rank; ++i)
          if (srcOrder[i] != 0)
            sharedOrder.emplace_back(srcOrder[i]);
        sharedOrder.emplace_back(0);
      } else {
        sharedOrder = srcOrder;
      }
      auto tmpType = triton::MemDescType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::SharedEncodingAttr::get(
              mod.getContext(), dstDotOp, srcType.getShape(), sharedOrder,
              triton::gpu::getCTALayout(srcEncoding),
              srcType.getElementType()));
      auto tmp = builder.create<triton::gpu::LocalAllocOp>(
          cvtOp.getLoc(), tmpType, cvtOp.getSrc());
      auto newConvert = builder.create<triton::gpu::LocalLoadOp>(cvtOp.getLoc(),
                                                                 dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
