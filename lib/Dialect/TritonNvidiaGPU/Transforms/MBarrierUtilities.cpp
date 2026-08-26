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
  LinearLayout regLayout =
      ttg::toLinearLayout(regTy).removeZeroBasesAlongDim(kRegister);
  LinearLayout conversion = invertAndComposeBlockLocal(
      ttg::toLinearLayoutIgnoringPadding(memDescTy), regLayout);
  auto kBlock = StringAttr::get(memDescTy.getContext(), "block");
  return !conversion.isIdentityOnOutDim(kBlock);
}

bool isCrossCTAGatherScatter(ttg::MemDescType memDescTy,
                            RankedTensorType regTy, unsigned axis) {
  MLIRContext *ctx = memDescTy.getContext();
  LinearLayout sharedLayout = ttg::toLinearLayoutIgnoringPadding(memDescTy);
  SmallVector<StringAttr> allDims =
      standardOutDimNames(ctx, memDescTy.getRank());
  StringAttr axisDim = allDims[axis];
  auto kBlock = StringAttr::get(ctx, "block");

  bool axisIsSharded =
      !sharedLayout.sublayoutIsZero({kBlock}, {axisDim});
  // (1) axisIsSharded: The runtime index may select any CTA shard along the
  // indexed axis.
  // (2) isCrossCTALoadStore: Access through the remaining coordinates may still
  // cross CTAs when the memdesc is a subslice of a distributed allocation.
  return axisIsSharded || isCrossCTALoadStore(memDescTy, regTy);
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
        Operation *op = user.getOperation();
        bool crossCTA = false;
        if (isa<ttng::MMAv5OpInterface, ttng::TCGen5CommitOp>(op))
          crossCTA = hasTCGen5CommitCrossCTA(op);
        else if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op))
          crossCTA = tma.getMulticast();
        else if (isa<ttng::CLCTryCancelOp>(op))
          crossCTA = true;
        else if (auto store = dyn_cast<ttng::AsyncSharedStoreOp>(op))
          crossCTA = isCrossCTALoadStore(
              store.getDst().getType(), store.getSrc().getType());
        else if (auto expect = dyn_cast<ttng::BarrierExpectOp>(op))
          crossCTA = expect.getFromCTA().has_value();
        else if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op))
          crossCTA = arrive.isMulticast() || arrive.getFromCTA().has_value();

        return crossCTA && llvm::any_of(user.getBarriers(), aliasesBarrier)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

} // namespace mlir::triton::nvidia_gpu
