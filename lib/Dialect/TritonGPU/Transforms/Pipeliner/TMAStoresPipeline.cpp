#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

static SmallVector<tt::ExperimentalDescriptorStoreOp>
getTMAStores(scf::ForOp forOp) {
  SmallVector<tt::ExperimentalDescriptorStoreOp> tmaStores;

  // Do not use walk, as we don't want to walk into nested loops.
  std::function<void(Operation *)> collectTMAStores = [&](Operation *op) {
    if (auto storeOp = dyn_cast<tt::ExperimentalDescriptorStoreOp>(op)) {
      tmaStores.push_back(storeOp);
    }
    for (Region &region : op->getRegions()) {
      for (Operation &op : region.getOps()) {
        if (!isa<scf::ForOp>(op))
          collectTMAStores(&op);
      }
    }
  };
  collectTMAStores(forOp);
  return tmaStores;
}

static Value createAlloc(scf::ForOp &forOp,
                         tt::ExperimentalDescriptorStoreOp storeOp) {
  OpBuilder builder(forOp);
  auto ty = cast<RankedTensorType>(storeOp.getSrc().getType());
  auto order = ttg::getOrder(ty.getEncoding());
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  Attribute encoding =
      ttg::SharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order, ctaLayout);
  if (ty.getRank() > 1) {
    encoding = ttg::SharedEncodingAttr::get(
        ty.getContext(), ty.getShape(), order, ctaLayout, ty.getElementType());
  }
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(ty.getContext());
  Type memdescType =
      tt::MemDescType::get(ty.getShape(), ty.getElementType(), encoding,
                           sharedMemorySpace, /*mutableMemory*/ true);
  Value alloc = builder.create<ttg::LocalAllocOp>(storeOp->getLoc(),
                                                  memdescType, Value());
  return alloc;
}

static void createTMAAsyncCopy(scf::ForOp &forOp,
                               tt::ExperimentalDescriptorStoreOp storeOp,
                               Value alloc) {
  OpBuilder builder(storeOp);
  auto loc = storeOp.getLoc();
  auto ty = cast<RankedTensorType>(storeOp.getSrc().getType());
  auto order = ttg::getOrder(ty.getEncoding());
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());

  // Put wait before the local_store make the store truly async. We know
  // that we are the only user of the CopyLocalToGlobal.
  builder.create<ttng::TMAStoreWait>(loc, 0);
  builder.create<ttg::LocalStoreOp>(loc, storeOp.getSrc(), alloc);
  builder.create<ttng::FenceAsyncSharedOp>(loc, false);
  builder.create<ttng::AsyncTMACopyLocalToGlobalOp>(
      loc, storeOp.getDescPtr(), storeOp.getIndices(), alloc);

  storeOp->erase();
}

bool mlir::triton::pipelineTMAStores(scf::ForOp forOp) {
  SmallVector<tt::ExperimentalDescriptorStoreOp> tmaStores =
      getTMAStores(forOp);
  if (tmaStores.empty())
    return false;

  DenseMap<tt::ExperimentalDescriptorStoreOp, Value> storeToAlloc;
  DenseMap<std::pair<ArrayRef<int64_t>, Type>, Value> allocs;
  for (tt::ExperimentalDescriptorStoreOp op : tmaStores) {
    // Reuse allocations for stores of the same shape and types. This allows
    // saving shared memory usage. It is valid since we have a wait 0 before
    // every local_store. We could pipeline more aggressively if we didn't
    // re-use but there is a tradeoff with shared memory usage.
    auto key = std::make_pair(op.getSrc().getType().getShape(),
                              op.getSrc().getType().getElementType());
    auto it = allocs.find(key);
    if (it != allocs.end()) {
      storeToAlloc[op] = it->second;
      continue;
    }
    storeToAlloc[op] = createAlloc(forOp, op);
    allocs[key] = storeToAlloc[op];
  }

  for (tt::ExperimentalDescriptorStoreOp op : tmaStores) {
    createTMAAsyncCopy(forOp, op, storeToAlloc[op]);
  }

  // Deallocate shared memory buffers.
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  builder.create<ttng::TMAStoreWait>(forOp->getLoc(), 0);
  for (auto it : storeToAlloc) {
    builder.create<ttg::LocalDeallocOp>(forOp->getLoc(), it.second);
  }
  return true;
}
