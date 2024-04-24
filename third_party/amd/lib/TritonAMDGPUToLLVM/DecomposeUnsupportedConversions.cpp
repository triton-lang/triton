#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
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

static int getCvtOpLDSUsage(triton::gpu::ConvertLayoutOp &cvtOp) {
  unsigned inVec = 0;
  unsigned outVec = 0;
  auto smemShape = triton::getScratchConfigForCvtLayout(cvtOp, inVec, outVec);
  unsigned elems =
      std::accumulate(smemShape.begin(), smemShape.end(), 1, std::multiplies{});
  auto srcType = cvtOp.getSrc().getType();
  auto bytes =
      srcType.getElementType().isa<triton::PointerType>()
          ? elems * kPtrBitWidth / 8
          : elems * std::max<int>(8, srcType.getElementTypeBitWidth()) / 8;

  return bytes;
}

bool isPowerOfTwo(unsigned x) { return x && (x & (x - 1)) == 0; }

static std::vector<std::pair<int, int>> factorizePowerOf2(int n) {
  assert(isPowerOfTwo(n));
  int x = log2(n);
  std::vector<std::pair<int, int>> pairs;

  for (int i = 0; i <= x / 2; ++i) {
    int j = x - i;
    pairs.push_back({pow(2, i), pow(2, j)});
    pairs.push_back({pow(2, j), pow(2, i)});
  }

  return pairs;
}

static std::pair<triton::gpu::ConvertLayoutOp, triton::gpu::ConvertLayoutOp>
createNewConvertOps(ModuleOp &mod, OpBuilder &builder,
                    triton::gpu::ConvertLayoutOp &cvtOp,
                    std::pair<unsigned, unsigned> warpsPerCta) {
  unsigned warpsPerCtaX = warpsPerCta.first;
  unsigned warpsPerCtaY = warpsPerCta.second;
  auto srcType = cvtOp.getSrc().getType();
  auto dstType = cvtOp.getType();

  auto newDstType = RankedTensorType::get(
      dstType.getShape(), dstType.getElementType(), dstType.getEncoding());
  RankedTensorType newSrcType;
  if (auto srcMfma =
          srcType.getEncoding().dyn_cast<triton::gpu::AMDMfmaEncodingAttr>()) {
    auto newMfmaEnc = triton::gpu::AMDMfmaEncodingAttr::get(
        mod.getContext(), srcMfma.getVersionMajor(), srcMfma.getVersionMinor(),
        {warpsPerCtaX, warpsPerCtaY}, srcMfma.getMDim(), srcMfma.getNDim(),
        srcMfma.getIsTransposed(), srcMfma.getCTALayout());

    newSrcType = RankedTensorType::get(srcType.getShape(),
                                       srcType.getElementType(), newMfmaEnc);
  } else if (auto srcWmma = srcType.getEncoding()
                                .dyn_cast<triton::gpu::AMDWmmaEncodingAttr>()) {
    auto newWmmaEnc = triton::gpu::AMDWmmaEncodingAttr::get(
        mod.getContext(), {warpsPerCtaX, warpsPerCtaY}, srcWmma.getCTALayout());

    newSrcType = RankedTensorType::get(srcType.getShape(),
                                       srcType.getElementType(), newWmmaEnc);
  }

  auto tmpCvt = builder.create<triton::gpu::ConvertLayoutOp>(
      cvtOp.getLoc(), newSrcType, cvtOp.getSrc());
  auto newEpilogueCvt = builder.create<triton::gpu::ConvertLayoutOp>(
      cvtOp.getLoc(), newDstType, tmpCvt);

  return std::make_pair(tmpCvt, newEpilogueCvt);
}

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
          dstType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();

      // TODO: Reduce LDS usage for WMMA dots
      if (!srcEnc.isa<triton::gpu::AMDMfmaEncodingAttr>() || !dstBlocked) {
        return;
      }

      auto currLDSUsage = getCvtOpLDSUsage(cvtOp);
      if (currLDSUsage <= LDSSize) {
        return;
      }

      unsigned numWarps = triton::gpu::getNumWarpsPerCTA(srcEnc);

      triton::gpu::ConvertLayoutOp tmpCvt;
      triton::gpu::ConvertLayoutOp newEpilogueCvt;

      // Find all possible shapes of WarpsPerCTA by finding all possible
      // factorizations of numWarps. Pick shape for which both conversions in
      // decomposition use LDS less than LDSSize and for which sum of LDS usage
      // is minimal. If no such shape exists, do not decompose.
      unsigned minLDSUsage = 2 * LDSSize;
      int minIdx = -1;
      auto factorizedNumWarps = factorizePowerOf2(numWarps);

      for (int i = 0; i < factorizedNumWarps.size(); i++) {
        auto warpsPerCTAPair = factorizedNumWarps[i];
        std::tie(tmpCvt, newEpilogueCvt) =
            createNewConvertOps(mod, builder, cvtOp, warpsPerCTAPair);

        int tmpCvtLDS = getCvtOpLDSUsage(tmpCvt);
        int newCvtLDS = getCvtOpLDSUsage(newEpilogueCvt);
        if (tmpCvtLDS <= LDSSize && newCvtLDS <= LDSSize) {
          int LDSUsage = tmpCvtLDS + newCvtLDS;
          if (LDSUsage < minLDSUsage) {
            minLDSUsage = LDSUsage;
            minIdx = i;
          }
        }
        newEpilogueCvt.erase();
        tmpCvt.erase();
      }

      if (minIdx == -1) {
        return;
      }

      assert(minIdx >= 0 && minIdx < factorizedNumWarps.size());
      auto warpsPerCTAPair = factorizedNumWarps[minIdx];
      std::tie(tmpCvt, newEpilogueCvt) =
          createNewConvertOps(mod, builder, cvtOp, warpsPerCTAPair);

      cvtOp.replaceAllUsesWith(newEpilogueCvt.getResult());
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
