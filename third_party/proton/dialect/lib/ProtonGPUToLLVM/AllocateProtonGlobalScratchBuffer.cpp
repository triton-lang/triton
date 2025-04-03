#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton {
#define GEN_PASS_DEF_ALLOCATEPROTONGLOBALSCRATCHBUFFER
#include "Conversion/ProtonGPUToLLVM/Passes.h.inc"
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

    assert(llvm::range_size(mod.getOps<FunctionOpInterface>()) == 1);
    auto func = *mod.getOps<FunctionOpInterface>().begin();

    int32_t cumulativeMemorySize = 0; // bytes
    std::vector<uint32_t> Alignments;

    func.walk([&](proton::gpu::GlobalScratchAllocOp op) {
      int offset = llvm::alignTo(cumulativeMemorySize,
                                 proton::gpu::getBytesPerClockEntry());
      op->setAttr("offset",
                  IntegerAttr::get(IntegerType::get(ctx, 32), offset));
      cumulativeMemorySize += op.getNbytes();
      Alignments.push_back(op.getAlignment());
    });
    assert(!Alignments.empty() &&
           "no global scratch buffer alignment values found");
    bool allAlignmentsEqual = std::equal(Alignments.begin() + 1,
                                         Alignments.end(), Alignments.begin());
    assert(allAlignmentsEqual &&
           "all global scratch buffer alignment values must be the same");
    mod->setAttr("ttg.profile_scratch_memory_size",
                 builder.getI32IntegerAttr(cumulativeMemorySize));
    mod->setAttr("ttg.profile_scratch_memory_alignment",
                 builder.getI32IntegerAttr(Alignments.front()));
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
