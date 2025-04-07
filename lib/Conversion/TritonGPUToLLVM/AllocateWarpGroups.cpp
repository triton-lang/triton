#include "mlir/IR/BuiltinOps.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUALLOCATEWARPGROUPS
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
struct AllocateWarpGroups
    : public mlir::triton::gpu::impl::TritonGPUAllocateWarpGroupsBase<
          AllocateWarpGroups> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);

    struct WarpGroupInfo {
      SmallVector<Region *> partitions;
      int maxRequestedRegs = 0;
      unsigned numWarps = 0;
    };
    struct WarpGroupPartition {
      int startId;
      Region *partition;
      int32_t estRegs;
      int numWarps;
    };

    // Compute the total number of warps required at any given time.
    int baseNumWarps = lookupNumWarps(mod);
    int maxExtraWarps = 0;
    mod.walk([&](WarpSpecializeOp op) {
      ArrayRef<int32_t> arr = op.getPartitionNumWarps();
      int req = op.getTotalPartitionWarps();
      maxExtraWarps = std::max(maxExtraWarps, req);

      // Allocate the start IDs such that the largest warpgroups have lower
      // starting warp IDs.
      // FIXME: Handle aligning warp group IDs to 4 for TMEM.
      SmallVector<std::pair<unsigned, int32_t>> idxAndSize;
      for (auto [i, size] : llvm::enumerate(arr))
        idxAndSize.emplace_back(i, size);
      llvm::sort(idxAndSize,
                 [&](auto lhs, auto rhs) { return lhs.second > rhs.second; });

      SmallVector<int32_t> startIds(arr.size());
      int startId = baseNumWarps;
      for (auto [i, size] : idxAndSize) {
        startIds[i] = startId;
        startId += size;
      }
      op.setWarpGroupStartIds(startIds);

      // Require that an estimate has been set and that we have even warpgroups.
      auto regsAttr = op.getRequestedRegisters();
      if (!regsAttr || op.getTotalPartitionWarps() % 4 != 0)
        return;

      // Group the partitions into warpgroups.
      SmallVector<WarpGroupPartition> orderedPartitions;
      for (auto [startId, partition, estRegs, numWarps] :
           llvm::zip(startIds, op.getPartitionRegions(), *regsAttr, arr))
        orderedPartitions.push_back({startId, partition, estRegs, numWarps});
      llvm::sort(orderedPartitions,
                 [&](auto lhs, auto rhs) { return lhs.startId < rhs.startId; });

      // Iterate over the partitions and assign them to warp groups. Determine
      // the maximum number of requested registers per warp group.
      SmallVector<WarpGroupInfo> warpGroups;
      for (auto [startId, partition, estRegs, numWarps] : orderedPartitions) {
        if (startId % 4 == 0) {
          warpGroups.push_back(WarpGroupInfo{});
        }
        warpGroups.back().partitions.push_back(partition);
        // Round up the nearest multiple of 8.
        int estRegsCeil8 = llvm::divideCeil(estRegs, 8) * 8;
        warpGroups.back().maxRequestedRegs =
            std::max<int>(warpGroups.back().maxRequestedRegs, estRegsCeil8);
        warpGroups.back().numWarps += numWarps;
      }

      // Determine the maximum number of registers per thread. This may have
      // been set by the user.
      int maxnreg;
      if (auto maxnregAttr =
              op->getAttrOfType<IntegerAttr>(AttrMaxRegistersName)) {
        maxnreg = maxnregAttr.getInt();
      } else {
        maxnreg = (1 << 16) / (baseNumWarps + op.getTotalPartitionWarps()) /
                  threadsPerWarp;
        maxnreg = maxnreg / 8 * 8;
      }

      // Compute the register deficit over the partition warp groups.
      int registerDeficit = 0;
      for (const WarpGroupInfo &wg : warpGroups) {
        assert(wg.numWarps % 4 == 0);
        registerDeficit +=
            (maxnreg - wg.maxRequestedRegs) * wg.numWarps * threadsPerWarp;
      }
      if (registerDeficit <= 0)
        return;

      // Determine the number of extra registers that we can distribute to the
      // default warp group.
      int leftover =
          ((baseNumWarps * threadsPerWarp * maxnreg) + registerDeficit) /
          baseNumWarps / threadsPerWarp;
      // Round down to the nearest multiple of 8.
      leftover = leftover / 8 * 8;

      // Generate setmaxnreg in each partition according to its warp group.
      SmallVector<int32_t> maxnregsPerPartition(1 + arr.size());
      for (const WarpGroupInfo &wg : warpGroups) {
        for (Region *region : wg.partitions) {
          maxnregsPerPartition[1 + region->getRegionNumber()] =
              wg.maxRequestedRegs;
        }
      }
      // Set the register usage for the default warp group.
      maxnregsPerPartition.front() = leftover;
      op.setActualRegisters(maxnregsPerPartition);

      // Set the initial max number of registers. This is needed for PTXAS to
      // cooperate.
      mod->setAttr(AttrMaxRegistersName,
                   Builder(op.getContext()).getI32IntegerAttr(maxnreg));
    });

    Builder b(&getContext());
    mod->setAttr("ttg.total-num-warps",
                 b.getI32IntegerAttr(baseNumWarps + maxExtraWarps));
  }
};
} // namespace
