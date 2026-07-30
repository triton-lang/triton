#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

struct TMAStore {
  Operation *op;
  mlir::TypedValue<tt::TensorDescType> desc;
  mlir::TypedValue<RankedTensorType> src;
};

static SmallVector<TMAStore> getTMAStores(scf::ForOp forOp) {
  SmallVector<TMAStore> tmaStores;

  forOp.getBody()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto storeOp = dyn_cast<tt::DescriptorStoreLikeOpInterface>(op)) {
      tmaStores.push_back({storeOp, storeOp.getDesc(), storeOp.getSrc()});
      // Don't walk into nested loops.
    } else if (isa<scf::ForOp>(op)) {
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  return tmaStores;
}

static bool hasAcquireOrReleaseSemantic(tt::MemSemantic sem) {
  return sem != tt::MemSemantic::RELAXED;
}

static bool hasAcquireOrReleaseOp(scf::ForOp forOp) {
  bool hasAcquireOrRelease = false;
  forOp.getBody()->walk([&](Operation *op) {
    if (auto atomicRMW = dyn_cast<tt::AtomicRMWOp>(op)) {
      hasAcquireOrRelease = hasAcquireOrReleaseSemantic(atomicRMW.getSem());
    } else if (auto atomicCAS = dyn_cast<tt::AtomicCASOp>(op)) {
      hasAcquireOrRelease = hasAcquireOrReleaseSemantic(atomicCAS.getSem());
    } else if (auto atomicPoll = dyn_cast<tt::AtomicPollOp>(op)) {
      hasAcquireOrRelease = hasAcquireOrReleaseSemantic(atomicPoll.getSem());
    }
    return hasAcquireOrRelease ? WalkResult::interrupt()
                               : WalkResult::advance();
  });
  return hasAcquireOrRelease;
}

static Value createAlloc(scf::ForOp &forOp, const TMAStore &store) {
  OpBuilder builder(forOp);
  RankedTensorType ty = store.src.getType();
  auto encoding =
      triton::nvidia_gpu::getEncodingFromDescriptor(store.op, ty, store.desc);
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(ty.getContext());
  Type memdescType =
      ttg::MemDescType::get(ty.getShape(), ty.getElementType(), encoding,
                            sharedMemorySpace, /*mutableMemory*/ true);
  Value alloc =
      ttg::LocalAllocOp::create(builder, store.op->getLoc(), memdescType);
  return alloc;
}

static void createTMAAsyncCopy(scf::ForOp forOp, const TMAStore &store,
                               Value alloc, int pendings) {
  OpBuilder builder(store.op);
  Location loc = store.op->getLoc();

  // Put wait before the local_store to make the store truly async. We only
  // need the TMA read from the allocation to complete before reusing it.
  ttng::TMAStoreWaitOp::create(builder, loc, pendings, /*read_only=*/true);
  ttg::LocalStoreOp::create(builder, loc, store.src, alloc);
  ttng::FenceAsyncSharedOp::create(builder, loc, false);
  auto desc = store.desc;
  if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(store.op)) {
    ttng::AsyncTMACopyLocalToGlobalOp::create(builder, loc, desc,
                                              storeOp.getIndices(), alloc);
  } else if (auto reduceOp = dyn_cast<tt::DescriptorReduceOp>(store.op)) {
    ttng::AsyncTMAReduceOp::create(builder, loc, reduceOp.getKind(), desc,
                                   reduceOp.getIndices(), alloc);
  } else {
    auto scatterOp = cast<tt::DescriptorScatterOp>(store.op);
    Value xOffsets =
        ttng::sextI16ToI32Indices(scatterOp.getXOffsets(), builder, loc);
    ttng::AsyncTMAScatterOp::create(builder, loc, desc, xOffsets,
                                    scatterOp.getYOffset(), alloc);
  }

  store.op->erase();
}

static void lowerTMADescriptorCreation(scf::ForOp forOp) {
  // Use max_stage=3 to double buffer the descriptor.
  triton::CoarseSchedule schedule(3);
  triton::lowerTMADescriptors(forOp, schedule);
}

bool mlir::triton::pipelineTMAStores(scf::ForOp forOp, int numStages) {
  SmallVector<TMAStore> tmaStores = getTMAStores(forOp);
  if (tmaStores.empty())
    return false;
  if (hasAcquireOrReleaseOp(forOp))
    return false;

  DenseMap<Operation *, Value> storeAllocs;
  int numStores = static_cast<int>(tmaStores.size());
  int pendings = std::max(std::min(numStages, numStores) - 1, 0);
  int maxBuffers = pendings + 1;
  DenseMap<std::pair<ArrayRef<int64_t>, Type>, SmallVector<const TMAStore *>>
      groupedStores;
  for (const TMAStore &store : tmaStores) {
    RankedTensorType srcTy = store.src.getType();
    auto key = std::make_pair(srcTy.getShape(), srcTy.getElementType());
    groupedStores[key].push_back(&store);
  }

  for (auto &[key, stores] : groupedStores) {
    SmallVector<Value> allocs;
    // Reuse allocations for stores of the same shape and types. This allows
    // saving shared memory usage.
    int numBuffers = std::min<int>(stores.size(), maxBuffers);
    for (int i = 0; i < numBuffers; ++i)
      allocs.push_back(createAlloc(forOp, *stores[i]));

    for (auto [idx, store] : llvm::enumerate(stores)) {
      storeAllocs[store->op] = allocs[idx % numBuffers];
    }
  }

  bool hasDeviceSideTMA = llvm::any_of(tmaStores, [](const TMAStore &store) {
    return !triton::isHostSideDescriptor(store.desc);
  });
  for (const TMAStore &store : tmaStores) {
    createTMAAsyncCopy(forOp, store, storeAllocs[store.op], pendings);
  }

  // Deallocate shared memory buffers.
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  ttng::TMAStoreWaitOp::create(builder, forOp->getLoc(), 0,
                               /*read_only=*/false);
  SetVector<Value> allocs;
  for (auto it : storeAllocs)
    allocs.insert(it.second);
  for (Value alloc : allocs)
    ttg::LocalDeallocOp::create(builder, forOp->getLoc(), alloc);

  if (hasDeviceSideTMA) {
    // This is a bit coarse as it would multibuffer any descriptor in the loop
    // but it likely to not have a big impact.
    lowerTMADescriptorCreation(forOp);
  }
  return true;
}
