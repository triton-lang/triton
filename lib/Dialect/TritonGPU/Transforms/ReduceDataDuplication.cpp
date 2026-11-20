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
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUREDUCEDATADUPLICATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUReduceDataDuplicationPass
    : public impl::TritonGPUReduceDataDuplicationBase<
          TritonGPUReduceDataDuplicationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      if (isa<triton::gpu::SharedEncodingTrait>(srcEncoding))
        return;
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (!dstDotOp)
        return;
      if (!cvtNeedsSharedMemory(cvtOp))
        return;
      auto order = getOrderForMemory(srcType);
      auto sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
      Attribute sharedEncoding = SwizzledSharedEncodingAttr::get(
          mod.getContext(), dstDotOp, srcType.getShape(), order,
          getCGALayout(srcEncoding), srcType.getElementType());
      // Stage the original tile when it also has other users. CSE can then
      // share the allocation if those users need the same physical layout.
      Value src = cvtOp.getSrc();
      auto trans = src.getDefiningOp<TransOp>();
      bool stageBeforeTranspose = trans && !trans.getSrc().hasOneUse();
      if (stageBeforeTranspose) {
        sharedEncoding = inferSrcEncoding(trans, sharedEncoding);
        src = trans.getSrc();
      }
      auto tileType = cast<RankedTensorType>(src.getType());
      auto tmpType =
          MemDescType::get(tileType.getShape(), tileType.getElementType(),
                           sharedEncoding, sharedMemorySpace);
      Value tmp = LocalAllocOp::create(builder, cvtOp.getLoc(), tmpType, src);
      if (stageBeforeTranspose)
        tmp = MemDescTransOp::create(builder, cvtOp.getLoc(), tmp,
                                     trans.getOrder());
      auto newConvert =
          LocalLoadOp::create(builder, cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
