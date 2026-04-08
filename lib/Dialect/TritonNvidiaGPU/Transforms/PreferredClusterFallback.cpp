#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPREFERREDCLUSTERFALLBACKPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

static bool hasCrossCTAConvertLayout(ttg::ConvertLayoutOp cvt) {
  auto kBlock = StringAttr::get(cvt->getContext(), "block");
  auto conversion = minimalCvtLayout(cvt.getSrc().getType(), cvt.getType());
  return conversion.hasInDim(kBlock);
}

static bool hasCrossCTAReduce(triton::ReduceOp reduce) {
  auto srcTy = reduce.getInputTypes()[0];
  auto splitNum = ttg::getCTASplitNum(srcTy.getEncoding());
  return splitNum[reduce.getAxis()] > 1;
}

static bool moduleRequestsConSan(ModuleOp mod) {
  for (StringRef attrName :
       {"ttg.instrumentation_mode", "triton.instrumentation_mode"}) {
    auto attr = mod->getAttrOfType<StringAttr>(attrName);
    if (attr && attr.getValue().contains("consan"))
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

    if (moduleRequestsConSan(mod))
      return;

    WalkResult result = mod.walk([&](Operation *op) -> WalkResult {
      auto unsupported = [&] { return WalkResult::interrupt(); };

      // You can do pretty much anything with inline asm
      if (isa<triton::ElementwiseInlineAsmOp>(op))
        return unsupported();

      // You can synchronise CTAs with global atomic operations
      if (isa<triton::AtomicRMWOp, triton::AtomicCASOp>(op))
        return unsupported();

      // NYI: CLC can redirect a CTA to work from a different program.  To
      // support preferred fallback, ProgramCTAIdOp must be derived from the
      // canceled CTA id returned by CLC, not from the thief CTA's block id.
      // This seems tricky to implement
      if (isa<ttng::CLCTryCancelOp, ttng::CLCLoadResultOp,
              ttng::CLCIsCanceledOp, ttng::CLCGetProgramIdOp>(op))
        return unsupported();

      // NYI: We could have cvt_layout in the first 2 CTAs or reduces
      if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op)) {
        if (hasCrossCTAConvertLayout(cvt))
          return unsupported();
        return WalkResult::advance();
      }

      if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
        if (hasCrossCTAReduce(reduce))
          return unsupported();
        return WalkResult::advance();
      }

      if (auto arrive = dyn_cast<ttng::ClusterArriveOp>(op)) {
        if (!arrive.getRelaxed())
          return unsupported();
        return WalkResult::advance();
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
