#include "mlir/Analysis/Liveness.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUGLOBALSCRATCHALLOCATIONPASS
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

static int32_t roundUp(int32_t val, int32_t step) {
  auto t = val + step - 1;
  return t - (t % step);
}

struct ScratchMemoryInfo {
  int32_t offset = 0;
  uint32_t largestAlignment = 1;
};

static void assignOffset(Operation *op, OpBuilder &builder,
                         ScratchMemoryInfo &memInfo, uint32_t nbytes,
                         uint32_t align, StringRef offsetAttrName) {
  if (nbytes == 0)
    return;
  memInfo.offset = roundUp(memInfo.offset, align);
  op->setAttr(offsetAttrName, builder.getI32IntegerAttr(memInfo.offset));
  memInfo.offset += nbytes;
  memInfo.largestAlignment = std::max(memInfo.largestAlignment, align);
}

static void setModuleScratchAttrs(Operation *op, OpBuilder &builder,
                                  const ScratchMemoryInfo &globalMemInfo,
                                  const ScratchMemoryInfo &profileMemInfo) {
  int32_t totalGlobalMemorySize =
      roundUp(globalMemInfo.offset, globalMemInfo.largestAlignment);
  int32_t totalProfileMemorySize =
      roundUp(profileMemInfo.offset, profileMemInfo.largestAlignment);
  op->setAttr("ttg.global_scratch_memory_size",
              builder.getI32IntegerAttr(totalGlobalMemorySize));
  op->setAttr("ttg.global_scratch_memory_alignment",
              builder.getI32IntegerAttr(globalMemInfo.largestAlignment));
  op->setAttr("ttg.profile_scratch_memory_size",
              builder.getI32IntegerAttr(totalProfileMemorySize));
  op->setAttr("ttg.profile_scratch_memory_alignment",
              builder.getI32IntegerAttr(profileMemInfo.largestAlignment));
}

static void allocateGMem(Operation *parentOp,
                         llvm::SetVector<Operation *> &callStack) {
  // Recursively visit any dependency functions
  parentOp->walk([&](triton::CallOp call) {
    auto callable = call.resolveCallable();
    if (!callable->hasAttr("ttg.global_scratch_memory_size") ||
        !callable->hasAttr("ttg.profile_scratch_memory_size")) {
      auto inserted = callStack.insert(parentOp);
      assert(inserted && "call cycle detected");
      allocateGMem(callable, callStack);
      callStack.remove(parentOp);
    }
  });

  MLIRContext *ctx = parentOp->getContext();
  OpBuilder builder(ctx);
  ScratchMemoryInfo globalMemInfo;
  ScratchMemoryInfo profileMemInfo;

  // Dumb allocation that ignores liveness and makes no attempt to minimize
  // padding
  // TODO: Use a real algorithm
  parentOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (auto alloc = dyn_cast<triton::gpu::GlobalScratchAllocOp>(op)) {
      ScratchMemoryInfo &memInfo =
          alloc->hasAttr("3p_allocation") ? profileMemInfo : globalMemInfo;
      assignOffset(op, builder, memInfo, alloc.getNbytes(),
                   alloc.getAlignment(), "ttg.global_scratch_memory_offset");
    } else if (auto callOp = dyn_cast<triton::CallOp>(op)) {
      auto callable = callOp.resolveCallable();
      auto globalNbytesAttr = callable->getAttrOfType<IntegerAttr>(
          "ttg.global_scratch_memory_size");
      auto globalAlignAttr = callable->getAttrOfType<IntegerAttr>(
          "ttg.global_scratch_memory_alignment");
      auto profileNbytesAttr = callable->getAttrOfType<IntegerAttr>(
          "ttg.profile_scratch_memory_size");
      auto profileAlignAttr = callable->getAttrOfType<IntegerAttr>(
          "ttg.profile_scratch_memory_alignment");
      assert(globalNbytesAttr && globalAlignAttr && profileNbytesAttr &&
             profileAlignAttr);

      assignOffset(op, builder, globalMemInfo,
                   globalNbytesAttr.getValue().getZExtValue(),
                   globalAlignAttr.getValue().getZExtValue(),
                   "ttg.global_scratch_memory_offset");
      assignOffset(op, builder, profileMemInfo,
                   profileNbytesAttr.getValue().getZExtValue(),
                   profileAlignAttr.getValue().getZExtValue(),
                   "ttg.profile_scratch_memory_offset");
    }
  });
  setModuleScratchAttrs(parentOp, builder, globalMemInfo, profileMemInfo);
}

namespace {
class TritonGPUGlobalScratchAllocationPass
    : public mlir::triton::gpu::impl::TritonGPUGlobalScratchAllocationPassBase<
          TritonGPUGlobalScratchAllocationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    bool seenKernel = false;

    SetVector<Operation *> callStack;
    mod->walk([&](triton::FuncOp func) {
      allocateGMem(func, callStack);

      if (func.getVisibility() == SymbolTable::Visibility::Public) {
        assert(!seenKernel);
        seenKernel = true;
        auto size =
            func->getAttrOfType<IntegerAttr>("ttg.global_scratch_memory_size");
        auto align = func->getAttrOfType<IntegerAttr>(
            "ttg.global_scratch_memory_alignment");
        auto profileSize =
            func->getAttrOfType<IntegerAttr>("ttg.profile_scratch_memory_size");
        auto profileAlign = func->getAttrOfType<IntegerAttr>(
            "ttg.profile_scratch_memory_alignment");
        assert(size);
        assert(align);
        assert(profileSize);
        assert(profileAlign);
        mod->setAttr("ttg.global_scratch_memory_size", size);
        mod->setAttr("ttg.global_scratch_memory_alignment", align);
        mod->setAttr("ttg.profile_scratch_memory_size", profileSize);
        mod->setAttr("ttg.profile_scratch_memory_alignment", profileAlign);
      }
    });
    assert(seenKernel);
  }
};
} // namespace
