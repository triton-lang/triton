#include "OptimizeLDSUtility.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <numeric>

using namespace mlir;
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DECOMPOSEUNSUPPORTEDAMDCONVERSIONS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

struct DecomposeUnsupportedAMDConversions
    : public mlir::triton::impl::DecomposeUnsupportedAMDConversionsBase<
          DecomposeUnsupportedAMDConversions> {
  explicit DecomposeUnsupportedAMDConversions(StringRef targetArch) {
    this->arch = targetArch.str();
  }

  void runOnOperation() override {
    triton::AMD::TargetInfo targetInfo(this->arch.getValue());
    int sharedMemoryLimit = targetInfo.getSharedMemorySize();

    ModuleOp mod = getOperation();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    triton::gpu::decomposeSplatOpToSharedLayoutConversion(mod);

    triton::gpu::decomposeTensorCoreToDotLayoutConversion<
        triton::gpu::AMDMfmaEncodingAttr>(mod, isMfmaToDotShortcut);

    /* -------------------------------- */
    // Replace `wmma -> dot_op` with `wmma -> blocked -> dot_op`
    /* -------------------------------- */
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getSrc().getType();
      auto dstType = cvtOp.getType();
      auto srcWmma =
          dyn_cast<triton::gpu::AMDWmmaEncodingAttr>(srcType.getEncoding());
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (srcWmma && dstDotOp) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcWmma),
                getOrder(srcWmma), numWarps, threadsPerWarp, numCTAs));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
    // Try to reduce LDS usage of cvt(mfma->blocked) op by changing the shape of
    // WarpsPerCta attribute in mfma layout. The implicit LDS usage of
    // cvt(mfma->blocked) op depends on the number of warps per CTA that mfma
    // layout uses along x dimension and block layout uses across y dimension.
    //
    // clang-format off
    //
    // LDS usage of this op is roughly calculated as:
    // LDS_USAGE = getShapePerCTA(mfma_layout)[0] * getShapePerCTA(blocked_layout)[1] * sizeof(data_type)
    // LDS_USAGE = warpsPerCTA(mfma_layout)[0] * warpsPerCta(blocked_layout)[1] * C,
    // where C = 32 * sizePerWarp(blocked_layout)[1] * threadsPerWarp(blocked_layout)[1] * sizeof(data_type)
    //
    // clang-format on
    //
    // When LDS_USAGE exceeds the size of LDS, try to lower LDS usage by
    // decomposing cvt(mfma->blocked) op into 2 conversions: cvt(mfma->mfma_tmp)
    // and cvt(mfma_tmp->blocked), where mfma_tmp has WarpsPerCta attribute that
    // minimizes uses of LDS for these conversions.
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);

      auto srcType = cvtOp.getSrc().getType();
      auto dstType = cvtOp.getType();

      auto srcEnc = srcType.getEncoding();
      auto dstBlocked =
          dyn_cast<triton::gpu::BlockedEncodingAttr>(dstType.getEncoding());

      // TODO: Reduce LDS usage for WMMA dots
      if (!isa<triton::gpu::AMDMfmaEncodingAttr>(srcEnc) || !dstBlocked) {
        return;
      }

      auto currLDSUsage = mlir::triton::AMD::getCvtOpLDSUsage(cvtOp);
      if (currLDSUsage <= sharedMemoryLimit) {
        return;
      }

      unsigned numWarps = triton::gpu::getNumWarpsPerCTA(srcEnc);

      // Find all possible shapes of WarpsPerCTA by finding all possible
      // factorizations of numWarps. Pick shape for which both conversions in
      // decomposition use LDS less than sharedMemoryLimit and for which sum of
      // LDS usage is minimal. If no such shape exists, do not decompose.
      unsigned minLDSUsage = 2 * sharedMemoryLimit;
      int minIdx = -1;
      int rank = dstBlocked.getWarpsPerCTA().size();
      auto factorizedNumWarps =
          mlir::triton::AMD::factorizePowerOf2(numWarps, rank);

      SmallVector<Attribute> tmpLayouts;
      for (int i = 0; i < factorizedNumWarps.size(); i++) {
        auto warpsPerCTA = factorizedNumWarps[i];
        tmpLayouts.push_back(
            mlir::triton::AMD::createTmpLayout(srcEnc, warpsPerCTA));
      }

      for (int i = 0; i < tmpLayouts.size(); i++) {
        auto resources = mlir::triton::AMD::estimateResourcesForReplacement(
            builder, cvtOp, tmpLayouts[i]);
        if (resources.LDS <= sharedMemoryLimit && resources.LDS < minLDSUsage) {
          minLDSUsage = resources.LDS;
          minIdx = i;
        }
      }

      if (minIdx == -1 || minLDSUsage > sharedMemoryLimit) {
        return;
      }

      assert(minIdx >= 0 && minIdx < tmpLayouts.size());
      auto replacementCvts = mlir::triton::AMD::createNewConvertOps(
          builder, cvtOp, tmpLayouts[minIdx]);

      cvtOp.replaceAllUsesWith(replacementCvts.second.getResult());
      cvtOp.erase();
    });

    triton::gpu::decomposeBlockedToDotLayoutConversion(mod);
  }
};

} // namespace

namespace mlir::triton::AMD {

std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeUnsupportedConversionsPass(StringRef targetArch) {
  return std::make_unique<DecomposeUnsupportedAMDConversions>(targetArch);
}

} // namespace mlir::triton::AMD
