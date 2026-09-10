#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::nvidia_gpu {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

bool isCrossCTALoadStore(ttg::MemDescType memDescTy, RankedTensorType regTy) {
  auto kRegister = StringAttr::get(memDescTy.getContext(), "register");
  auto kBlock = StringAttr::get(memDescTy.getContext(), "block");
  LinearLayout regLayout =
      ttg::toLinearLayout(regTy).removeZeroBasesAlongDim(kRegister);
  LinearLayout conversion = invertAndComposeLocal(
      ttg::toLinearLayoutIgnoringPadding(memDescTy), regLayout, {kBlock});
  return !conversion.isIdentityOnOutDim(kBlock);
}

bool isCrossCTAGatherScatter(ttg::MemDescType memDescTy, RankedTensorType regTy,
                             unsigned axis) {
  MLIRContext *ctx = memDescTy.getContext();
  LinearLayout sharedLayout = ttg::toLinearLayoutIgnoringPadding(memDescTy);
  SmallVector<StringAttr> allDims =
      standardOutDimNames(ctx, memDescTy.getRank());
  StringAttr axisDim = allDims[axis];
  auto kRegister = StringAttr::get(ctx, "register");
  auto kBlock = StringAttr::get(ctx, "block");

  // Runtime indices may select any shard of the indexed axis.
  if (!sharedLayout.sublayoutIsZero({kBlock}, {axisDim}))
    return true;

  LinearLayout regLayout = ttg::toLinearLayout(regTy)
                               .removeZeroBasesAlongDim(kRegister)
                               .transposeOuts(allDims);
  // Replace `axis` with a descriptor-sized input, then check whether the
  // remaining result coordinates select a remote CTA.
  SmallVector<StringAttr> nonIndexedDims = allDims;
  nonIndexedDims.erase(nonIndexedDims.begin() + axis);
  LinearLayout indexedLayout =
      regLayout.sublayout(llvm::to_vector(regLayout.getInDimNames()),
                          nonIndexedDims) *
      LinearLayout::identity1D(sharedLayout.getOutDimSize(axisDim), axisDim,
                               axisDim);
  indexedLayout = indexedLayout.transposeOuts(allDims);
  LinearLayout conversion =
      invertAndComposeLocal(sharedLayout, indexedLayout, {kBlock});
  return !conversion.isIdentityOnOutDim(kBlock);
}

std::optional<bool> hasCrossCTASharedAccess(Operation *op) {
  if (auto load = dyn_cast<ttg::LocalLoadOp>(op))
    return isCrossCTALoadStore(load.getSrc().getType(), load.getType());
  if (auto store = dyn_cast<ttg::LocalStoreOp>(op))
    return isCrossCTALoadStore(store.getDst().getType(),
                               store.getSrc().getType());
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
    return alloc.getSrc() &&
           isCrossCTALoadStore(alloc.getType(), alloc.getSrc().getType());
  if (auto gather = dyn_cast<ttg::LocalGatherOp>(op))
    return isCrossCTAGatherScatter(gather.getSrc().getType(), gather.getType(),
                                   gather.getAxis());
  if (auto scatter = dyn_cast<ttg::LocalScatterOp>(op))
    return isCrossCTAGatherScatter(scatter.getDst().getType(),
                                   scatter.getValues().getType(),
                                   scatter.getAxis());
  if (auto atomic = dyn_cast<ttg::LocalAtomicScatterRMWOp>(op))
    return isCrossCTAGatherScatter(atomic.getDst().getType(),
                                   atomic.getValues().getType(),
                                   atomic.getAxis());
  if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op)) {
    if (ttg::lookupNumCTAs(op) == 1)
      return false;
    if (tma.getMulticast())
      return true;
    auto type = cast<ttg::MemDescType>(tma.getResult().getType());
    auto encoding = type.getEncoding();
    // Subview offsets can select another CTA.
    if (ttg::dropPipeliningDim(type.getShape(), encoding) ==
        ttg::dropPipeliningDim(type.getAllocShape(), encoding))
      return false;
  }
  return std::nullopt;
}

bool hasTCGen5CommitCrossCTA(Operation *op) {
  SmallVector<Value> descs;
  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
    descs = mma.getCompletionDescs();
  else if (auto commit = dyn_cast<ttng::TCGen5CommitOp>(op))
    llvm::append_range(descs, commit.getDescs());
  else
    return false;
  return !ttng::getCTABroadcastMasks(ttng::getModuleTwoCTAs(op), descs).empty();
}

bool hasCrossCTAMBarrierUse(ttg::MBarrierOpInterface barrier) {
  Operation *op = barrier.getOperation();
  int numCTAs = ttg::lookupNumCTAs(op);
  if (numCTAs == 1)
    return false;
  if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op))
    return tma.getMulticast();
  if (isa<ttng::CLCTryCancelOp>(op))
    return true;
  if (auto store = dyn_cast<ttng::AsyncSharedStoreOp>(op))
    return isCrossCTALoadStore(store.getDst().getType(),
                               store.getSrc().getType());
  if (auto expect = dyn_cast<ttng::BarrierExpectOp>(op))
    return expect.getFromCTA().value_or(numCTAs - 1) != numCTAs - 1;
  if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op))
    return arrive.isMulticast() ||
           arrive.getFromCTA().value_or(numCTAs - 1) != numCTAs - 1;
  return hasTCGen5CommitCrossCTA(op);
}

bool requiresCrossCTAMBarrierInitSync(
    FunctionOpInterface funcOp, Value barrier, int numCTAs,
    llvm::function_ref<bool(Value)> aliasesBarrier) {
  // Barrier init sync is needed for barriers that are themselves cross-CTA,
  // and also for per-CTA barriers consumed by multi-CTA ops that multicast or
  // otherwise fan out barrier state across the cluster.
  auto barrierTy = dyn_cast<ttg::MemDescType>(barrier.getType());
  if (barrierTy && barrierTy.getShape()[0] != numCTAs)
    return true;

  // Or if it's used by a multi-CTA consumer that broadcasts barrier state
  // across CTAs even though the barrier allocation itself looks per-CTA.
  return funcOp
      ->walk<WalkOrder::PreOrder>([&](ttg::MBarrierOpInterface user) {
        return llvm::any_of(user.getBarriers(), aliasesBarrier) &&
                       hasCrossCTAMBarrierUse(user)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

} // namespace mlir::triton::nvidia_gpu
