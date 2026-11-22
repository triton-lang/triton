#include "mlir/IR/BuiltinOps.h"
#include "triton/Analysis/CallGraph.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUALLOCATEWARPGROUPS
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

constexpr int32_t kMinRegisters = 24;
constexpr int32_t kMinRegistersForAssertOrPrint = 32;
constexpr int32_t kMaxRegisters = 256;

static bool regionUsesAssertOrPrint(Region &region) {
  return region
      .walk([](Operation *op) {
        return isa<AssertOp, PrintOp>(op) ? WalkResult::interrupt()
                                          : WalkResult::advance();
      })
      .wasInterrupted();
}

static int32_t minRegistersForRegion(Region &region, bool instrumented,
                                     int32_t minRegisters) {
  // Work around an NVIDIA driver bug: regions that may call device system
  // functions such as assert or print need at least 32 registers.
  return instrumented || regionUsesAssertOrPrint(region)
             ? std::max(minRegisters, kMinRegistersForAssertOrPrint)
             : minRegisters;
}

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
  OperationState state(partitions.getLoc(), partitions.getOperationName(),
                       partitions.getOperands(), /*types=*/{});
  for (Region *region : partitions.getRegions())
    state.addRegion()->takeBody(*region);

  SmallVector<int32_t> partitionNumWarps(op.getPartitionNumWarps());
  for (int paddingSize : paddingPartitionSizes) {
    partitionNumWarps.push_back(paddingSize);

    Block &body = state.addRegion()->emplaceBlock();
    for (Value capture : op.getPartitionOp().getExplicitCaptures())
      body.addArgument(capture.getType(), capture.getLoc());
    OpBuilder b(op.getContext());
    b.setInsertionPointToStart(&body);
    WarpReturnOp::create(b, op.getLoc());
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

static LogicalResult annotateWarpIdOffsets(ModuleOp mod) {
  triton::CallGraph<int> callGraph(mod);
  DenseMap<FunctionOpInterface, int> offsets;
  for (auto root : callGraph.getRoots())
    offsets[root] = 0;

  LogicalResult result = success();
  callGraph.walk(
      [&](CallOpInterface call, FunctionOpInterface callee) {
        auto caller = call->getParentOfType<FunctionOpInterface>();
        int start = getWarpGroupStartWarpId(call->getBlock())
                        .value_or(offsets.lookup(caller));
        // Equal-sized partitions can share a helper when their starts have
        // the same remainder modulo the helper's warp count.
        int offset = start % lookupNumWarps(callee);
        auto [it, inserted] = offsets.try_emplace(callee, offset);
        if (!inserted && it->second != offset) {
          call.emitError("callee is used with incompatible warp ID offsets");
          result = failure();
        }
      },
      [](FunctionOpInterface) {});
  if (failed(result))
    return failure();

  Builder b(mod.getContext());
  for (auto [func, offset] : offsets)
    if (!isKernel(func))
      func->setAttr(AttrWarpIdOffsetName, b.getI32IntegerAttr(offset));
  return success();
}

namespace {
struct WarpGroupInfo {
  SmallVector<Region *> partitions;
  int maxRequestedRegs = 0;
  unsigned numWarps = 0;
  bool hasWork = false;
};

struct WarpGroupPartition {
  int startId;
  Region *partition;
  int32_t estRegs;
  int numWarps;
};

void distributeSurplusToTargets(ArrayRef<WarpGroupInfo *> targets,
                                int &surplusWarpRegs) {
  SmallVector<int> initialRegs;
  SmallVector<int> weights;
  initialRegs.reserve(targets.size());
  weights.reserve(targets.size());
  for (WarpGroupInfo *wg : targets) {
    initialRegs.push_back(wg->maxRequestedRegs);
    weights.push_back(wg->maxRequestedRegs * static_cast<int>(wg->numWarps));
  }

  while (surplusWarpRegs > 0) {
    int64_t totalWeight = 0;
    for (auto [wg, weight] : llvm::zip(targets, weights)) {
      if (wg->maxRequestedRegs < kMaxRegisters)
        totalWeight += weight;
    }
    if (totalWeight == 0)
      break;

    int distributed = 0;
    for (auto [wg, weight] : llvm::zip(targets, weights)) {
      int headroom = kMaxRegisters - wg->maxRequestedRegs;
      if (headroom <= 0)
        continue;
      int64_t shareWarpRegs =
          (static_cast<int64_t>(surplusWarpRegs) * weight) / totalWeight;
      int addRegs = static_cast<int>(shareWarpRegs / wg->numWarps) / 8 * 8;
      addRegs = std::min(addRegs, headroom);
      if (addRegs > 0) {
        wg->maxRequestedRegs += addRegs;
        distributed += addRegs * static_cast<int>(wg->numWarps);
      }
    }
    if (distributed == 0)
      break;
    surplusWarpRegs -= distributed;
  }

  // Allocate any leftover 8-register quanta truncated by rounding, cycling
  // through warpgroups in descending order of initial per-thread demand.
  SmallVector<size_t> order = llvm::to_vector(llvm::seq(targets.size()));
  llvm::stable_sort(order, [&](size_t lhs, size_t rhs) {
    return initialRegs[lhs] > initialRegs[rhs];
  });
  bool progress = true;
  while (progress && surplusWarpRegs > 0) {
    progress = false;
    for (size_t idx : order) {
      WarpGroupInfo *wg = targets[idx];
      int quantum = 8 * static_cast<int>(wg->numWarps);
      if (wg->maxRequestedRegs + 8 <= kMaxRegisters &&
          surplusWarpRegs >= quantum) {
        wg->maxRequestedRegs += 8;
        surplusWarpRegs -= quantum;
        progress = true;
      }
    }
  }
}

// Redistribute surplus CTA register budget (in registers * warps) above
// kMaxRegisters back to worker warpgroups while keeping every allocation a
// multiple of 8 capped at kMaxRegisters.
void redistributeSurplusRegisters(MutableArrayRef<WarpGroupInfo> groups,
                                  int surplusWarpRegs) {
  SmallVector<WarpGroupInfo *> activeGroups;
  SmallVector<WarpGroupInfo *> emptyGroups;
  for (WarpGroupInfo &wg : groups) {
    if (wg.hasWork)
      activeGroups.push_back(&wg);
    else
      emptyGroups.push_back(&wg);
  }
  if (!activeGroups.empty())
    distributeSurplusToTargets(activeGroups, surplusWarpRegs);
  if (!emptyGroups.empty() && surplusWarpRegs > 0)
    distributeSurplusToTargets(emptyGroups, surplusWarpRegs);
}

struct AllocateWarpGroups
    : public mlir::triton::gpu::impl::TritonGPUAllocateWarpGroupsBase<
          AllocateWarpGroups> {
  using TritonGPUAllocateWarpGroupsBase::TritonGPUAllocateWarpGroupsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool laterInstrumentation = instrumented.getValue();

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

    int baseNumWarps = lookupNumWarps(mod);

    // Compute the total number of warps required at any given time.
    mod.walk([&](WarpSpecializeOp op) {
      ArrayRef<int32_t> arr = op.getPartitionNumWarps();

      // Allocate the start IDs such that the largest warpgroups have lower
      // starting warp IDs.
      // FIXME: Handle aligning warp group IDs to 4 for TMEM.
      SmallVector<std::pair<unsigned, int32_t>> idxAndSize;
      for (auto [i, size] : llvm::enumerate(arr))
        idxAndSize.emplace_back(i, size);
      llvm::stable_sort(idxAndSize, [&](auto lhs, auto rhs) {
        return lhs.second > rhs.second;
      });

      SmallVector<int32_t> startIds(arr.size());
      int startId = baseNumWarps;
      for (auto [i, size] : idxAndSize) {
        startIds[i] = startId;
        startId += size;
      }
      op.setWarpGroupStartIds(startIds);
    });

    if (failed(annotateWarpIdOffsets(mod)))
      return signalPassFailure();

    Builder b(&getContext());
    mod->setAttr("ttg.total-num-warps",
                 b.getI32IntegerAttr(baseNumWarps + numExtraWarpGroups * 4));

    bool needsRegisterOptimization = false;
    mod.walk([&](WarpSpecializeOp op) {
      if (op.getRequestedRegisters())
        needsRegisterOptimization = true;
    });

    if (!needsRegisterOptimization)
      return;

    // Determine the maximum number of registers per thread. This may have
    // been set by the user.
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
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
    maxnreg = std::min(maxnreg, kMaxRegisters);

    // Compute register allocation for each warp specialize op.
    mod.walk([&](WarpSpecializeOp op) {
      ArrayRef<int32_t> arr = op.getPartitionNumWarps();
      auto startIds = *op.getWarpGroupStartIds();

      // Require that an estimate has been set and that we have even warpgroups.
      auto regsAttr = op.getRequestedRegisters();
      if (!regsAttr || op.getTotalPartitionWarps() % 4 != 0)
        return;

      // Group the partitions into warpgroups.
      SmallVector<WarpGroupPartition> orderedPartitions;
      for (auto [startId, partition, estRegs, numWarps] :
           llvm::zip(startIds, op.getPartitionRegions(), *regsAttr, arr)) {
        int minRegs = minRegistersForRegion(*partition, laterInstrumentation,
                                            std::max(estRegs, kMinRegisters));
        orderedPartitions.push_back({startId, partition, minRegs, numWarps});
      }
      llvm::sort(orderedPartitions,
                 [&](auto lhs, auto rhs) { return lhs.startId < rhs.startId; });

      // Iterate over the partitions and assign them to warp groups. Determine
      // the maximum number of requested registers per warp group.
      SmallVector<Region *> nonEmptyPartitions =
          op.getNonEmptyPartitionRegions();
      SmallVector<WarpGroupInfo> warpGroups;
      for (auto [startId, partition, estRegs, numWarps] : orderedPartitions) {
        if (startId % 4 == 0) {
          warpGroups.push_back(WarpGroupInfo{});
        }
        warpGroups.back().partitions.push_back(partition);
        // Round up to the nearest multiple of 8, clamped to hardware max.
        int estRegsCeil8 =
            std::min<int>(llvm::divideCeil(estRegs, 8) * 8, kMaxRegisters);
        warpGroups.back().maxRequestedRegs =
            std::max<int>(warpGroups.back().maxRequestedRegs, estRegsCeil8);
        warpGroups.back().numWarps += numWarps;
        warpGroups.back().hasWork |=
            llvm::is_contained(nonEmptyPartitions, partition);
      }

      // Compute the register budget (in registers * warps) available to the
      // default warp group after satisfying worker requests.
      int registerBudgetWarpRegs = maxnreg * baseNumWarps;
      for (const WarpGroupInfo &wg : warpGroups) {
        assert(wg.numWarps % 4 == 0);
        registerBudgetWarpRegs +=
            (maxnreg - wg.maxRequestedRegs) * static_cast<int>(wg.numWarps);
      }
      if (registerBudgetWarpRegs <= 0)
        return;

      // Determine the number of registers that we can distribute to the
      // default warp group.
      int leftover = (registerBudgetWarpRegs / baseNumWarps) / 8 * 8;
      if (leftover < minRegistersForRegion(op.getDefaultRegion(),
                                           laterInstrumentation, kMinRegisters))
        return; // too few registers

      // Hardware caps per-thread registers at kMaxRegisters (256). Rather than
      // discarding surplus above kMaxRegisters, redistribute it back to worker
      // warpgroups while preserving the total CTA register budget.
      if (registerBudgetWarpRegs > kMaxRegisters * baseNumWarps) {
        leftover = kMaxRegisters;
        redistributeSurplusRegisters(
            warpGroups, registerBudgetWarpRegs - kMaxRegisters * baseNumWarps);
      }

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
  }
};
} // namespace
