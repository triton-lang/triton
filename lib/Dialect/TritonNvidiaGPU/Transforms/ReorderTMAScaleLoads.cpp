#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUREORDERTMASCALELOADSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Synchronous TMA lowering materializes the tile immediately after its wait.
// Sinking the operand load shortens its register lifetime.
static DescriptorLoadOp findScaledDotLoadToSink(DotScaledOp dot, Value operand,
                                                Value scale) {
  auto cvt = operand.getDefiningOp<gpu::ConvertLayoutOp>();
  if (!cvt || !cvt->hasOneUse() || cvt->getBlock() != dot->getBlock() || !scale)
    return {};
  auto load = cvt.getSrc().getDefiningOp<DescriptorLoadOp>();
  Operation *scaleDef = scale.getDefiningOp();
  if (!load || !load->hasOneUse() || !scaleDef ||
      load->getBlock() != dot->getBlock() ||
      scaleDef->getBlock() != dot->getBlock() ||
      !load->isBeforeInBlock(scaleDef) || !scaleDef->isBeforeInBlock(cvt))
    return {};

  Operation *scaleLoad = scaleDef;
  while (isa<gpu::ConvertLayoutOp, ReshapeOp, TransOp>(scaleLoad)) {
    scaleLoad = scaleLoad->getOperand(0).getDefiningOp();
    if (!scaleLoad)
      return {};
  }
  if (!isa<DescriptorLoadOp>(scaleLoad) ||
      scaleLoad->getBlock() != dot->getBlock() ||
      !load->isBeforeInBlock(scaleLoad))
    return {};

  // The two descriptor reads can be reordered. Do not cross another memory
  // access, an operation with unknown effects, or a control-flow boundary.
  for (Operation *op = load->getNextNode(); op != scaleDef->getNextNode();
       op = op->getNextNode()) {
    if (op != scaleLoad && (op->getNumRegions() || !isMemoryEffectFree(op) ||
                            isa<DotOpInterface>(op)))
      return {};
  }
  return load;
}

class TritonNvidiaGPUReorderTMAScaleLoadsPass
    : public impl::TritonNvidiaGPUReorderTMAScaleLoadsPassBase<
          TritonNvidiaGPUReorderTMAScaleLoadsPass> {
public:
  void runOnOperation() override {
    getOperation().walk([](DotScaledOp dot) {
      auto mma = dyn_cast_or_null<gpu::NvidiaMmaEncodingAttr>(
          dot.getType().getEncoding());
      // Warp-specialized regions have partition-specific register budgets.
      if (!mma || !mma.isAmpere() || gpu::lookupNumCTAs(dot) != 1 ||
          dot->getParentOfType<gpu::WarpSpecializeOp>())
        return;
      for (auto [operand, scale] :
           {std::pair<Value, Value>{dot.getA(), dot.getAScale()},
            std::pair<Value, Value>{dot.getB(), dot.getBScale()}}) {
        if (auto load = findScaledDotLoadToSink(dot, operand, scale))
          load->moveAfter(scale.getDefiningOp());
      }
    });
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu
