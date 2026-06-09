/* Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "nvidia/include/Dialect/NVWS/IR/SemaphorePendingCount.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::triton::nvws {
namespace {

std::optional<int>
getReleaseAsyncContribution(ArrayAttr asyncOps,
                            std::optional<AsyncOp> &unsupportedAsyncOp) {
  int contribution = 0;
  for (Attribute asyncOp : asyncOps) {
    auto kind = cast<AsyncOpAttr>(asyncOp).getValue();
    switch (kind) {
    case AsyncOp::TC5MMA:
    case AsyncOp::TMALoad:
    case AsyncOp::NONE:
    case AsyncOp::WGMMA:
    case AsyncOp::TMEMCopy:
      ++contribution;
      break;
    case AsyncOp::CpAsync:
      unsupportedAsyncOp = kind;
      return std::nullopt;
    default:
      llvm_unreachable("unknown async op");
    }
  }

  return contribution;
}

} // namespace

SemaphorePendingCountAnalysis
analyzeSemaphorePendingCount(SemaphoreCreateOp op) {
  SemaphorePendingCountAnalysis analysis;
  // Compute pending count for this semaphore from unique releasing partitions.
  // Different partitions contribute independently. Repeated releases from the
  // same partition do not create extra pending-count slots and must agree on
  // their async contribution.
  llvm::DenseMap<int, int> partitionContrib;
  int pendingCount = 0;

  for (Operation *user : op->getUsers()) {
    auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
    if (!releaseOp || !gpu::hasPartition(user))
      continue;

    auto partitionIds = gpu::getPartitionIds(user);
    if (partitionIds.size() != 1) {
      // Pending-count analysis is per releasing partition. If the same
      // semaphore must be released from multiple partitions, model that with
      // multiple semaphore.release ops, each carrying one partition id, with
      // associated acquire token.
      analysis.invalidPartitionArity = partitionIds.size();
      return analysis;
    }

    std::optional<AsyncOp> unsupportedAsyncOp;
    auto contribution = getReleaseAsyncContribution(releaseOp.getAsyncOps(),
                                                    unsupportedAsyncOp);
    if (!contribution) {
      analysis.unsupportedAsyncOp = unsupportedAsyncOp;
      return analysis;
    }

    int partitionId = partitionIds.front();
    auto [it, inserted] =
        partitionContrib.try_emplace(partitionId, contribution.value());
    if (inserted) {
      // First release seen for this partition. Count its contribution once.
      //
      // This is what makes:
      // - one waiting partition depend on releases from multiple other
      //   partitions, contributing once per releasing partition
      // - multiple waiting partitions observe the same semaphore stage with
      //   one releasing partition, contributing once overall
      pendingCount += contribution.value();
      continue;
    }

    // Repeated releases from the same partition are allowed only if they model
    // the same logical participant and therefore imply the same number of
    // arrivals.
    if (it->second != contribution.value()) {
      analysis.inconsistentPartitionId = partitionId;
      analysis.expectedContribution = it->second;
      analysis.actualContribution = contribution.value();
      return analysis;
    }
  }

  // Outside partitioned warp-specialized code there may be no partitioned
  // releases at all. Keep the historical fallback of one expected arrival.
  analysis.pendingCount = pendingCount == 0 ? 1 : pendingCount;
  return analysis;
}

} // namespace mlir::triton::nvws
