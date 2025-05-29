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

// Given a `ttg.warp_specialize` with a certain number of existing warps, pad it
// with extra warps until it has the same number of full warp groups as the
// largest partitioning. This ensures that all threads can be present to
// surrender registers.
static void padToMaxWarpGroups(WarpSpecializeOp op, int numExtraWarpGroups) {
  int numExtraWarps = op.getTotalPartitionWarps();
  int warpsToAdd = numExtraWarpGroups * 4 - numExtraWarps;
  assert(warpsToAdd >= 0);

  // Fill it with powers of 2.
  SmallVector<int> paddingPartitionSizes;
  while (warpsToAdd > 0) {
    int paddingSize = llvm::NextPowerOf2(warpsToAdd) / 2;
    paddingPartitionSizes.push_back(paddingSize);
    warpsToAdd -= paddingSize;
  }

  auto partitions = cast<WarpSpecializePartitionsOp>(
      op.getPartitionOpHolder().front().front());
  OperationState state(partitions.getLoc(), partitions.getOperationName());
  for (Region *region : partitions.getRegions())
    state.addRegion()->takeBody(*region);

  SmallVector<int32_t> partitionNumWarps(op.getPartitionNumWarps());
  for (int paddingSize : paddingPartitionSizes) {
    partitionNumWarps.push_back(paddingSize);

    Block &body = state.addRegion()->emplaceBlock();
    for (Value capture : op.getExplicitCaptures())
      body.addArgument(capture.getType(), capture.getLoc());
    OpBuilder b(op.getContext());
    b.setInsertionPointToStart(&body);
    b.create<WarpReturnOp>(op.getLoc());
  }
  op.setPartitionNumWarps(partitionNumWarps);

  // Set the requested registers to low for the padded partitions that do
  // nothing.
  if (auto reqRegs = op.getRequestedRegisters()) {
    SmallVector<int32_t> newReqRegs(*reqRegs);
    newReqRegs.append(paddingPartitionSizes.size(), 16);
    op.setRequestedRegisters(newReqRegs);
  }

  OpBuilder b(partitions);
  b.create(state);
  partitions.erase();
}

namespace {
struct AllocateWarpGroups
    : public mlir::triton::gpu::impl::TritonGPUAllocateWarpGroupsBase<
          AllocateWarpGroups> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // First determine the maximum number of extra warps.
    int maxExtraWarps = 0;
    mod.walk([&](WarpSpecializeOp op) {
      maxExtraWarps = std::max<int>(maxExtraWarps, op.getTotalPartitionWarps());
    });

    // Round this up to the nearest warpgroup (multiple of 4) and then pad each
    // `ttg.warp_specialize` to the nearest warpgroup.
    int numExtraWarpGroups = llvm::divideCeil(maxExtraWarps, 4);
    mod.walk([&](WarpSpecializeOp op) {
      padToMaxWarpGroups(op, numExtraWarpGroups);
    });

    // Determine the maximum number of registers per thread. This may have
    // been set by the user.
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    int baseNumWarps = lookupNumWarps(mod);
    int maxnreg;
    if (auto maxnregAttr =
            mod->getAttrOfType<IntegerAttr>(AttrMaxRegistersName)) {
      maxnreg = maxnregAttr.getInt();
    } else {
      // Assume the user wants to use all 64K registers.
      maxnreg = (64 * 1024) / (baseNumWarps + numExtraWarpGroups * 4) /
                threadsPerWarp;
      maxnreg = maxnreg / 8 * 8;
    }

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
    mod.walk([&](WarpSpecializeOp op) {
      ArrayRef<int32_t> arr = op.getPartitionNumWarps();

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

      // Compute the register deficit over the partition warp groups.
      int registerBudget = maxnreg * baseNumWarps * threadsPerWarp;
      for (const WarpGroupInfo &wg : warpGroups) {
        assert(wg.numWarps % 4 == 0);
        registerBudget +=
            (maxnreg - wg.maxRequestedRegs) * wg.numWarps * threadsPerWarp;
      }
      if (registerBudget <= 0)
        return;

      // Determine the number of extra registers that we can distribute to the
      // default warp group.
      int leftover = registerBudget / (baseNumWarps * threadsPerWarp);
      // Round down to the nearest multiple of 8.
      leftover = leftover / 8 * 8;
      if (leftover < 24)
        return; // too few registers

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
                 b.getI32IntegerAttr(baseNumWarps + numExtraWarpGroups * 4));
  }
};
} // namespace
