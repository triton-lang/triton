#include "amd/lib/TritonAMDGPUTransforms/PipelineUtility.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;

namespace {

// Bookkeeping for one descriptor store / scatter we want to pipeline.
struct TDMStore {
  Operation *op;
  mlir::TypedValue<tt::TensorDescType> desc;
  mlir::TypedValue<RankedTensorType> src;
};

static SmallVector<TDMStore> getTDMStores(scf::ForOp forOp) {
  SmallVector<TDMStore> stores;
  // Only stores directly in the loop body can yield their tokens at the loop
  // level; those in nested regions (e.g. scf.if) are left to
  // ConvertToTensorOps.
  for (auto storeOp :
       forOp.getBody()->getOps<tt::DescriptorStoreLikeOpInterface>())
    stores.push_back({storeOp, storeOp.getDesc(), storeOp.getSrc()});
  return stores;
}

// Lift a single LDS allocation outside the loop, sized like store.src.
static Value createAlloc(scf::ForOp &forOp, const TDMStore &store) {
  OpBuilder builder(forOp);
  RankedTensorType ty = store.src.getType();
  auto encoding = getEncodingFromDescriptor(store.op, ty, store.desc);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(ty.getContext());
  Type memdescType =
      ttg::MemDescType::get(ty.getShape(), ty.getElementType(), encoding,
                            sharedMemorySpace, /*mutableMemory=*/true);
  return ttg::LocalAllocOp::create(builder, store.op->getLoc(), memdescType);
}

// Replace one descriptor_{store,scatter} with the pipelined async TDM
// sequence:
//
//   amdg.async_tdm_wait <prevToken>    (wait for previous iter's TDM write
//                                       to release the LDS buffer)
//   ttg.local_store src, alloc         (write current iter's data into LDS)
//   amdg.async_tdm_copy_local_to_global  OR  amdg.async_tdm_scatter
//
// `prevToken` is a loop-carried token from the previous iteration's TDM op.
// Returns the token produced by the new TDM op for loop-carried threading.
static Value createTDMAsyncCopy(scf::ForOp forOp, const TDMStore &store,
                                Value alloc, Value prevToken) {
  OpBuilder builder(store.op);
  Location loc = store.op->getLoc();

  ttag::AsyncTDMWait::create(builder, loc, prevToken, 0);
  ttg::LocalStoreOp::create(builder, loc, store.src, alloc);

  Value token;
  Value desc = store.desc;
  if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(store.op)) {
    Value copyDesc = createUpdateTDMDescriptorOp(
        builder, loc, desc, storeOp.getIndices(), /*pred=*/Value{});
    auto copyOp = ttag::AsyncTDMCopyLocalToGlobalOp::create(
        builder, loc, copyDesc, alloc, /*barrier=*/Value{});
    token = copyOp.getToken();
  } else {
    auto scatterOp = cast<tt::DescriptorScatterOp>(store.op);
    // Mirror TensorScatterLowering: re-layout them to AMD's TDM
    // gather/scatter index encoding before issuing the async op.
    auto indices = scatterOp.getXOffsets();
    auto indicesType = cast<RankedTensorType>(indices.getType());
    auto idxEnc = getTDMGatherScatterIndexEncoding(scatterOp, indicesType);
    if (indicesType.getEncoding() != idxEnc) {
      auto newIdxType = RankedTensorType::get(
          indicesType.getShape(), indicesType.getElementType(), idxEnc);
      indices = ttg::ConvertLayoutOp::create(builder, loc, newIdxType, indices);
    }
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    Value scatterDesc = createUpdateTDMDescriptorOp(
        builder, loc, desc, {zero, scatterOp.getYOffset()}, /*pred=*/Value{});
    auto scatterTDMOp = ttag::AsyncTDMScatterOp::create(
        builder, loc, scatterDesc, indices, alloc, /*barrier=*/Value{});
    token = scatterTDMOp.getRetToken();
  }

  store.op->erase();
  return token;
}

} // namespace

bool mlir::pipelineTDMStores(scf::ForOp forOp) {
  SmallVector<TDMStore> stores = getTDMStores(forOp);
  if (stores.empty())
    return false;

  // Reuse a single allocation across stores with the same src shape/type to
  // save shared memory. Stores that share an allocation form one "alloc
  // group". Within an iteration, an alloc group's stores must execute
  // sequentially with respect to the LDS buffer: each `local_store` must
  // wait for the previous in-iteration TDM op on the same allocation to
  // finish reading the buffer before overwriting it.
  //
  // To get correct ordering, we maintain exactly one loop-carried token per
  // allocation that represents the *last* TDM op of the previous iteration for
  // that allocation. Inside the iteration we thread the token forward through
  // each store in the group: store 0 waits on the loop-carried token,
  // store 1 waits on store 0's token... The token of the group's last store is
  // yielded as the next iteration's loop-carried token. This ensures the last
  // TDM store of each allocation can overlap with the TDM loads inside the
  // loop.
  SmallVector<Value> uniqueAllocs;
  DenseMap<Operation *, unsigned> storeToAllocIdx;
  DenseMap<std::pair<ArrayRef<int64_t>, Type>, unsigned> allocKeyToIdx;
  for (const TDMStore &store : stores) {
    RankedTensorType srcTy = store.src.getType();
    auto key = std::make_pair(srcTy.getShape(), srcTy.getElementType());
    // Check if a key already exists, if not create a new allocation
    auto [it, inserted] = allocKeyToIdx.try_emplace(key, uniqueAllocs.size());
    if (inserted)
      uniqueAllocs.push_back(createAlloc(forOp, store));
    storeToAllocIdx[store.op] = it->second;
  }

  // Seed one "empty" loop-carried token per allocation; this is slightly
  // conservative but the best we can do as we do not have an unitialized token
  // state.
  OpBuilder preBuilder(forOp);
  SmallVector<Value> initTokens;
  initTokens.reserve(uniqueAllocs.size());
  for (size_t i = 0; i < uniqueAllocs.size(); ++i) {
    auto seedWait = ttag::AsyncTDMWait::create(preBuilder, forOp->getLoc(),
                                               ArrayRef<Value>{}, 0);
    initTokens.push_back(seedWait.getRetToken());
  }

  // Add one loop-carried token per store.  addIterArgsToLoop splices the
  // loop body (no clone), so the Operation* pointers in `storeToAllocIdx`
  // remain valid.
  unsigned firstNewArg = forOp.getBody()->getNumArguments();
  forOp = addIterArgsToLoop(preBuilder, forOp, initTokens);

  // `curTokens[i]` is the current in-flight token for `uniqueAllocs[i]`.
  // It starts at the loop-carried token and advances after each store.
  SmallVector<Value> curTokens(uniqueAllocs.size());
  for (size_t i = 0; i < uniqueAllocs.size(); ++i)
    curTokens[i] = forOp.getBody()->getArgument(firstNewArg + i);

  for (const TDMStore &store : stores) {
    unsigned aIdx = storeToAllocIdx[store.op];
    curTokens[aIdx] =
        createTDMAsyncCopy(forOp, store, uniqueAllocs[aIdx], curTokens[aIdx]);
  }

  // Yield the per-allocation tail tokens so they become the next
  // iteration's loop-carried tokens.
  appendToForOpYield(forOp, curTokens);

  // After the loop: drain the last in-flight TDM writes using the final
  // tokens, then free the allocation(s).
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  unsigned numOrigResults = forOp.getNumResults() - uniqueAllocs.size();
  SmallVector<Value> finalTokens;
  finalTokens.reserve(uniqueAllocs.size());
  for (size_t i = 0; i < uniqueAllocs.size(); ++i)
    finalTokens.push_back(forOp.getResult(numOrigResults + i));
  ttag::AsyncTDMWait::create(builder, forOp->getLoc(), finalTokens, 0);
  for (Value alloc : uniqueAllocs)
    ttg::LocalDeallocOp::create(builder, forOp->getLoc(), alloc);

  return true;
}
