/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "OptimizeLDSUtility.h"
#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Analysis/Liveness.h"
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
#define GEN_PASS_DEF_OPTIMIZEAMDLDSUSAGE
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class OptimizeAMDLDSUsage
    : public mlir::triton::impl::OptimizeAMDLDSUsageBase<OptimizeAMDLDSUsage> {

  int LDSLimit;

  // Try to reduce LDS usage of convert op by adding tmp layout in conversion:
  //
  // %1 = convert %0 (src layout -> dst layout)
  //     ->
  // %1 = convert %0 (src layout -> tmp)
  // %2 = convert %1 (tmp -> dst layout)
  //
  // The implicit LDS usage of convert op depends on src and dst layouts
  //
  // Consider mfma->blocked conversion as an example.
  //
  // tensor shape: [128, 128]
  // mfma layout: warpsPerCTA = [1, 4], instrShape = [32, 32]
  // blocked layout: sizePerThread = [1, 4], threadsPerWarp = [32, 2],
  // warpsPerCTA = [4, 1]
  //
  // minimal mfma tile is: [1*32, 4*32] = [32, 128]
  // minimal blocked tile is: [1*32*4, 4*2*1] = [128, 8]
  //
  // Roughtly scratch buffer shape for conversion is:
  // [max(32, 128), max(128, 16)] = [128, 128].
  //
  // This shape could be reduces by introducing intermediate
  // layout and replacing old convert operations with two new conversions:
  //
  // %1 = convert %0 (mfma -> blocked)
  //     ->
  // %1 = convert %0 (mfma -> tmp)
  // %2 = convert %1 (tmp -> blocked)
  //
  // Let's consider tmp as blocked layout:
  // sizePerThread = [1, 4], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4]
  // Tmp layout scratch buffer has shape: [1*32*1, 4*2*4] = [32, 32]
  //
  // With intermediate layout we have two scratch buffers:
  //
  // %1 = convert %0 (mfma -> tmp): [max(32, 32), max(128, 32)] = [32, 128]
  // %2 = convert %1 (tmp -> blocked): [max(32, 128), max(32, 32)] = [128, 32]
  //
  // Both of these buffers are 4x times smaller than original one and their live
  // times do not intersect, therefore this transformation lowers LDS
  // consumption.
  void tryFitCvtIntoLDS(triton::gpu::ConvertLayoutOp cvtOp, int targetLDSSize) {
    OpBuilder builder(cvtOp);

    auto srcType = cvtOp.getSrc().getType();
    auto dstType = cvtOp.getType();

    auto srcEnc = srcType.getEncoding();
    auto dstEnc = dstType.getEncoding();

    auto ctx = srcEnc.getContext();
    auto rank = srcType.getShape().size();
    unsigned numWarps = triton::gpu::getNumWarpsPerCTA(srcEnc);
    auto warpSize = triton::gpu::getWarpSize(srcEnc);

    // Find all possible shapes of WarpsPerCTA by finding all possible
    // factorizations of numWarps. Pick shape for which both conversions in
    // decomposition use LDS less than LDSLimit and for which sum of LDS usage
    // is minimal. If no such shape exists, do not decompose.
    auto factorizedNumWarps =
        mlir::triton::AMD::factorizePowerOf2(numWarps, rank);
    // Create a list of temporary layouts
    SmallVector<Attribute> tmpLayouts;
    for (int i = 0; i < factorizedNumWarps.size(); i++) {
      auto warpsPerCTA = factorizedNumWarps[i];
      tmpLayouts.push_back(
          mlir::triton::AMD::createTmpLayout(srcEnc, warpsPerCTA));
      tmpLayouts.push_back(
          mlir::triton::AMD::createTmpLayout(dstEnc, warpsPerCTA));
      SmallVector<unsigned> elemsPerThread(rank, 1);
      SmallVector<unsigned> threadsPerWarp(rank, 1);
      threadsPerWarp[rank - 1] = warpSize / 8;
      threadsPerWarp[rank - 2] = warpSize / threadsPerWarp[rank - 1];
      auto order = triton::gpu::getOrder(srcEnc);
      auto layoutCTA = triton::gpu::getCTALayout(srcEnc);
      auto fallbackLayout = triton::gpu::BlockedEncodingAttr::get(
          ctx, elemsPerThread, threadsPerWarp, warpsPerCTA, order, layoutCTA);
      tmpLayouts.push_back(fallbackLayout);
    }

    unsigned minLDSUsage = 2 * LDSLimit;
    int minIdx = -1;
    for (int i = 0; i < tmpLayouts.size(); i++) {
      auto resources = mlir::triton::AMD::estimateResourcesForReplacement(
          builder, cvtOp, tmpLayouts[i]);
      // TODO analyze performance along with LDS consumption
      if (resources.LDS < minLDSUsage) {
        minLDSUsage = resources.LDS;
        minIdx = i;
      }
    }

    if (minIdx == -1 || minLDSUsage > targetLDSSize) {
      return;
    }

    assert(minIdx >= 0 && minIdx < tmpLayouts.size());
    auto tmpLayout = tmpLayouts[minIdx];
    auto replacementCvts =
        mlir::triton::AMD::createNewConvertOps(builder, cvtOp, tmpLayout);

    cvtOp.replaceAllUsesWith(replacementCvts.second.getResult());
    cvtOp.erase();
  }

  struct LDSBottleneckOperation {
    triton::gpu::ConvertLayoutOp op;
    int64_t LDSSizeTarget;
  };

  /**
   * Assuming that all buffer above scratch buffer in memory space can be
   * shifted down in memory, this function gives optimistic estimation of memory
   * space available for scratch buffer.
   */
  int64_t
  computeTargetScratchBufferSize(triton::gpu::ConvertLayoutOp op,
                                 Allocation *allocation,
                                 ArrayRef<Allocation::BufferId> liveBuffers) {
    int totalSize = 0;
    auto scratchBufferId = allocation->getBufferId(op.getOperation());
    int64_t scratchBufferSize = allocation->getAllocatedSize(scratchBufferId);
    size_t totalLDSConsumption = 0;
    for (auto buf : liveBuffers) {
      totalLDSConsumption = std::max(
          totalLDSConsumption, allocation->getAllocatedInterval(buf).end());
    }
    int64_t freeRequired = totalLDSConsumption - LDSLimit;
    auto targetScratchSize = std::max(0l, scratchBufferSize - freeRequired);
    return targetScratchSize;
  }

  SmallVector<LDSBottleneckOperation>
  findLDSBottleneckLayoutConvert(ModuleAllocation &allocAnalysis,
                                 FunctionOpInterface func) {
    SmallVector<LDSBottleneckOperation> candidates;
    auto funcAnalysis = allocAnalysis.getFuncData(func);
    auto liveBuffers = funcAnalysis->getLiveBuffers();

    func.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      auto cvtBuffer = funcAnalysis->getBufferId(cvtOp.getOperation());
      assert(cvtBuffer != Allocation::InvalidBufferId);

      auto targetScratchBufferSize = computeTargetScratchBufferSize(
          cvtOp, funcAnalysis, liveBuffers[cvtOp]);
      auto currentLDSConsumption = funcAnalysis->getAllocatedSize(cvtBuffer);
      if (currentLDSConsumption > targetScratchBufferSize)
        candidates.push_back({cvtOp, targetScratchBufferSize});
    });
    return candidates;
  }

public:
  OptimizeAMDLDSUsage(StringRef targetArch, int customLDSLimit)
      : OptimizeAMDLDSUsageBase<OptimizeAMDLDSUsage>() {
    this->targetArch = targetArch.str();
    this->customLDSLimit = customLDSLimit;
  }

  void runOnOperation() override {
    if (customLDSLimit == 0) {
      if (targetArch == "") {
        std::string message =
            this->getName().str() + " requires architecture name";
        llvm_unreachable(message.c_str());
      }
      triton::AMD::TargetInfo targetInfo(targetArch.c_str());
      LDSLimit = targetInfo.getSharedMemorySize();
    } else {
      LDSLimit = customLDSLimit;
    }
    ModuleOp mod = getOperation();

    ModuleAllocation allocAnalysis(mod);
    if (allocAnalysis.getSharedMemorySize() > LDSLimit) {
      auto rootFunctions = allocAnalysis.getRoots();
      for (auto rootFunc : rootFunctions) {
        // Find operations with peak LDS consumption
        auto candidates =
            findLDSBottleneckLayoutConvert(allocAnalysis, rootFunc);
        // Try to transform candidate operations to fit them into LDS
        for (auto candidate : candidates)
          tryFitCvtIntoLDS(candidate.op, candidate.LDSSizeTarget);
      }
    }
  }
};

} // namespace

namespace mlir {

namespace triton {

namespace AMD {

std::unique_ptr<OperationPass<ModuleOp>>
createOptimizeLDSUsagePass(StringRef targetArch, int customLDSLimit) {
  return std::make_unique<OptimizeAMDLDSUsage>(targetArch, customLDSLimit);
}

} // namespace AMD

} // namespace triton

} // namespace mlir
