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

    int numFuncOps = 0;
    FunctionOpInterface func;
    mod.walk([&](FunctionOpInterface op) {
      // Ignore any intrinsic functions. On AMD the predicate load/store ops
      // are currently pseduo instrunctions at this point and will get picked up
      // here and trigger the FunctionOpInterface range based assert below
      StringRef funcName(op.getNameAttr());
      if (!funcName.contains("__")) {
        numFuncOps += 1;
        func = op;
      }
    });

    assert(numFuncOps == 1);

    int32_t cumulativeMemorySize = 0; // bytes
    std::vector<uint32_t> alignments;

    func.walk([&](proton::gpu::GlobalScratchAllocOp op) {
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

} // namespace

namespace mlir::triton::proton::gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createAllocateProtonGlobalScratchBufferPass() {
  return std::make_unique<AllocateProtonGlobalScratchBuffer>();
}

} // namespace mlir::triton::proton::gpu
