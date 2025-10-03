#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::proton::gpu {

#define GEN_PASS_DEF_ALLOCATEPROTONGLOBALSCRATCHBUFFERPASS
#include "Conversion/ProtonGPUToLLVM/Passes.h.inc"

struct AllocateProtonGlobalScratchBufferPass
    : public impl::AllocateProtonGlobalScratchBufferPassBase<
          AllocateProtonGlobalScratchBufferPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    auto funcOps = triton::proton::gpu::getTritonFunctions(mod);
    assert(funcOps.size() == 1 && "Expected exactly one funcOp");

    int32_t cumulativeMemorySize = 0; // bytes
    std::vector<uint32_t> alignments;

    funcOps[0].walk([&](proton::gpu::GlobalScratchAllocOp op) {
      int offset = llvm::alignTo(cumulativeMemorySize,
                                 proton::gpu::getBytesPerClockEntry());
      op->setAttr("offset",
                  IntegerAttr::get(IntegerType::get(ctx, 32), offset));
      cumulativeMemorySize += op.getNbytes();
      alignments.push_back(op.getAlignment());
    });
    if (alignments.empty())
      return;

    bool allAlignmentsEqual = std::equal(alignments.begin() + 1,
                                         alignments.end(), alignments.begin());
    assert(allAlignmentsEqual &&
           "all global scratch buffer alignment values must be the same");
    mod->setAttr("ttg.profile_scratch_memory_size",
                 builder.getI32IntegerAttr(cumulativeMemorySize));
    mod->setAttr("ttg.profile_scratch_memory_alignment",
                 builder.getI32IntegerAttr(alignments.front()));
  }
};

} // namespace mlir::triton::proton::gpu
