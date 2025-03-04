#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton {
#define GEN_PASS_DEF_ALLOCATEPROTONGLOBALSCRATCHBUFFER
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/Passes.h.inc"
} // namespace triton::proton
} // namespace mlir

namespace {

struct AllocateProtonGlobalScratchBuffer
    : public mlir::triton::proton::impl::AllocateProtonGlobalScratchBufferBase<
          AllocateProtonGlobalScratchBuffer> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    assert(llvm::range_size(mod.getOps<triton::FuncOp>()) == 1);
    FuncOp func = *mod.getOps<triton::FuncOp>().begin();

    int32_t totalMemorySize = 0;
    uint32_t largestAlignment = 1;

    func.walk([&](proton::gpu::GlobalScratchAllocOp op) {
      totalMemorySize += op.getNbytes();
      largestAlignment = std::max(largestAlignment, op.getAlignment());
    });
    mod->setAttr("proton.global_scratch_memory_size",
                 builder.getI32IntegerAttr(totalMemorySize));
    mod->setAttr("proton.global_scratch_memory_alignment",
                 builder.getI32IntegerAttr(largestAlignment));
  }
};

} // namespace

namespace mlir {

namespace triton::proton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createAllocateProtonGlobalScratchBufferPass() {
  return std::make_unique<AllocateProtonGlobalScratchBuffer>();
}

} // namespace gpu

} // namespace triton::proton

} // namespace mlir
