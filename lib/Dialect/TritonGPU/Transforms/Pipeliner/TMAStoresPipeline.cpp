#include "Schedule.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static SmallVector<tt::ExperimentalDescriptorStoreOp>
getTMAStores(scf::ForOp forOp) {
  SmallVector<tt::ExperimentalDescriptorStoreOp> tmaStores;
  forOp->walk(
      [&](tt::ExperimentalDescriptorStoreOp op) { tmaStores.push_back(op); });
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

  Type memdescType = mlir::triton::MemDescType::get(
      ty.getShape(), ty.getElementType(), encoding, /*mutableMemory*/ true);
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
  auto order = mlir::triton::gpu::getOrder(ty.getEncoding());
  auto ctaLayout = mlir::triton::gpu::getCTALayout(ty.getEncoding());

  builder.create<triton::nvidia_gpu::TMAStoreWait>(loc, 0);
  builder.create<ttg::LocalStoreOp>(loc, storeOp.getSrc(), alloc);
  builder.create<triton::nvidia_gpu::FenceAsyncSharedOp>(loc, false);
  builder.create<triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(
      loc, storeOp.getDescPtr(), storeOp.getIndices(), alloc);

  storeOp->erase();
}

bool mlir::triton::pipelineTMAStores(scf::ForOp forOp) {
  SmallVector<tt::ExperimentalDescriptorStoreOp> tmaStores =
      getTMAStores(forOp);
  if (tmaStores.empty())
    return false;

  DenseMap<tt::ExperimentalDescriptorStoreOp, Value> storeToAlloc;
  for (tt::ExperimentalDescriptorStoreOp op : tmaStores) {
    storeToAlloc[op] = createAlloc(forOp, op);
  }

  for (tt::ExperimentalDescriptorStoreOp op : tmaStores) {
    createTMAAsyncCopy(forOp, op, storeToAlloc[op]);
  }
  return true;
}
