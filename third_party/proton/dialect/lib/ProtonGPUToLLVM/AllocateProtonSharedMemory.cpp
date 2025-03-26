#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton {
#define GEN_PASS_DEF_ALLOCATEPROTONSHAREDMEMORY
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/Passes.h.inc"
} // namespace triton::proton
} // namespace mlir

namespace {

struct AllocateProtonSharedMemory
    : public mlir::triton::proton::impl::AllocateProtonSharedMemoryBase<
          AllocateProtonSharedMemory> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    int sharedMemUsed = 0;
    if (mod->hasAttr("ttg.shared"))
      sharedMemUsed =
          mod->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();

    assert(llvm::range_size(mod.getOps<triton::FuncOp>()) == 1);
    FuncOp func = *mod.getOps<triton::FuncOp>().begin();

    int totalSharedMemSize = 0;
    int count = 0;
    func.walk([&](triton::gpu::LocalAllocOp alloc) {
      // We ignore the shared memory allocations that have been allocated by the
      // triton conversion pass.
      if (!alloc->hasAttr("allocation.offset")) {
        int offset =
            llvm::alignTo(sharedMemUsed, proton::gpu::getBytesPerClockEntry());
        alloc->setAttr("allocation.offset",
                       IntegerAttr::get(IntegerType::get(ctx, 32), offset));
        // Compute the proton buffer size in bytes.
        auto memDescTy =
            mlir::cast<triton::gpu::MemDescType>(alloc.getResult().getType());
        int bufferSizeInBytes =
            mlir::ShapedType::getNumElements(memDescTy.getShape()) *
            memDescTy.getElementType().getIntOrFloatBitWidth() / 8;

        totalSharedMemSize = offset + bufferSizeInBytes;
        count++;
      }
    });

    if (count == 0) {
      totalSharedMemSize = sharedMemUsed;
    }

    mod->setAttr("ttg.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                        totalSharedMemSize));
  }
};

} // namespace

namespace mlir {

namespace triton::proton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createAllocateProtonSharedMemoryPass() {
  return std::make_unique<AllocateProtonSharedMemory>();
}

} // namespace gpu

} // namespace triton::proton

} // namespace mlir
