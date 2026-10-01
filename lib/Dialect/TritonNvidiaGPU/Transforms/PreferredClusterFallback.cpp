#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPREFERREDCLUSTERFALLBACKPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

static bool moduleRequestsClusterSanitizer(ModuleOp mod) {
  for (StringRef attrName :
       {"ttg.instrumentation_mode", "triton.instrumentation_mode"}) {
    auto attr = mod->getAttrOfType<StringAttr>(attrName);
    if (attr && (attr.getValue().contains("consan") ||
                 attr.getValue().contains("gsan")))
      return true;
  }
  return false;
}

class TritonNvidiaGPUPreferredClusterFallbackPass
    : public impl::TritonNvidiaGPUPreferredClusterFallbackPassBase<
          TritonNvidiaGPUPreferredClusterFallbackPass> {
public:
  using impl::TritonNvidiaGPUPreferredClusterFallbackPassBase<
      TritonNvidiaGPUPreferredClusterFallbackPass>::
      TritonNvidiaGPUPreferredClusterFallbackPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod->removeAttr(AttrPreferredClusterFallbackCTAsName);

    int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
    if (computeCapability < 100 || numCTAs <= 2)
      return;

    if (moduleRequestsClusterSanitizer(mod))
      return;

    WalkResult result = mod.walk([&](Operation *op) -> WalkResult {
      auto unsupported = [&] { return WalkResult::interrupt(); };

      // Inline assembly and impure external calls can hide CTA synchronization.
      if (isa<triton::ElementwiseInlineAsmOp, ttg::InlineAsmOp>(op))
        return unsupported();
      if (auto external = dyn_cast<triton::ExternElementwiseOp>(op))
        if (!external.getPure())
          return unsupported();

      // You can synchronise CTAs with global atomic operations
      if (isa<triton::AtomicOpInterface, triton::AtomicPollOp>(op))
        return unsupported();

      // NYI: CLC can redirect a CTA to work from a different program.  To
      // support preferred fallback, ProgramCTAIdOp must be derived from the
      // canceled CTA id returned by CLC, not from the thief CTA's block id.
      // This seems tricky to implement
      if (isa<ttng::CLCTryCancelOp, ttng::CLCLoadResultOp,
              ttng::CLCIsCanceledOp, ttng::CLCGetProgramIdOp>(op))
        return unsupported();

      if (hasCrossCTAScratch(op))
        return unsupported();

      // Affine CTA offsets still lower to logical ranks, which can exceed the
      // physical fallback cluster size. CTA-local subview offsets are safe.
      for (Type type :
           llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
        auto memDescTy = dyn_cast<ttg::MemDescType>(type);
        if (!memDescTy ||
            !isa<ttg::SharedMemorySpaceAttr>(memDescTy.getMemorySpace()))
          continue;
        if (ttg::getMaskSpanOffsetsAndBlocks(memDescTy).second != 0)
          return unsupported();
      }

      // Even a CTA-local async store maps its destination and barrier using a
      // logical CTA rank that may not exist in the physical fallback cluster.
      if (isa<ttng::AsyncSharedStoreOp>(op))
        return unsupported();

      if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
        if (isCrossCTALoadStore(load.getSrc().getType(), load.getType()))
          return unsupported();
      } else if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
        if (isCrossCTALoadStore(store.getDst().getType(),
                                store.getSrc().getType()))
          return unsupported();
      } else if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
        if (alloc.getSrc() &&
            isCrossCTALoadStore(alloc.getType(), alloc.getSrc().getType()))
          return unsupported();
      } else if (auto gather = dyn_cast<ttg::LocalGatherOp>(op)) {
        if (isCrossCTAGatherScatter(gather.getSrc().getType(), gather.getType(),
                                    gather.getAxis()))
          return unsupported();
      } else if (auto scatter = dyn_cast<ttg::LocalScatterOp>(op)) {
        if (isCrossCTAGatherScatter(scatter.getDst().getType(),
                                    scatter.getValues().getType(),
                                    scatter.getAxis()))
          return unsupported();
      } else if (auto atomic = dyn_cast<ttg::LocalAtomicScatterRMWOp>(op)) {
        if (isCrossCTAGatherScatter(atomic.getDst().getType(),
                                    atomic.getValues().getType(),
                                    atomic.getAxis()))
          return unsupported();
      }

      // Larger completion counts need an explicit count for fallback clusters.
      if (auto init = dyn_cast<ttng::InitBarrierOp>(op))
        if (init.getCount() > 1 && !init.getFallbackCount())
          return unsupported();

      // Explicit routing uses physical ranks and must stay within a CTA pair.
      // fromCTA preserves the selected rank bits and broadcasts over the rest.
      if (auto expect = dyn_cast<ttng::BarrierExpectOp>(op))
        if (auto fromCTA = expect.getFromCTA())
          if (((numCTAs - 1) & ~*fromCTA) > 1)
            return unsupported();
      if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op)) {
        if (auto fromCTA = arrive.getFromCTA())
          if (((numCTAs - 1) & ~*fromCTA) > 1)
            return unsupported();
        if (arrive.getMulticastCTA() > 1)
          return unsupported();
      }

      if (auto barrier = dyn_cast<ttng::ClusterBarrierOp>(op)) {
        if (!barrier.getRelaxed())
          return unsupported();
        return WalkResult::advance();
      }

      if (auto barrierOp = dyn_cast<ttg::MBarrierOpInterface>(op)) {
        auto kBlock = StringAttr::get(barrierOp->getContext(), "block");
        for (Value barrier : barrierOp.getBarriers()) {
          auto barrierTy = cast<ttg::MemDescType>(barrier.getType());
          uint32_t cgaBroadcastMask =
              toLinearLayout(barrierTy).getFreeVariableMasks().lookup(kBlock);

          // Broadcast mbarriers use another CTA's barrier, so we only allow
          // broadcast on the first bit (i.e., CTA0 and CTA1).
          if (cgaBroadcastMask > 1)
            return unsupported();
        }
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted())
      return;

    mod->setAttr(AttrPreferredClusterFallbackCTAsName,
                 IntegerAttr::get(IntegerType::get(mod.getContext(), 32), 2));
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu
