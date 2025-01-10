#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Patterns.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DECOMPOSEUNSUPPORTEDNVIDIACONVERSIONS
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

struct DecomposeUnsupportedConversions
    : public mlir::triton::impl::DecomposeUnsupportedNVIDIAConversionsBase<
          DecomposeUnsupportedConversions> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    triton::gpu::decomposeBlockedToDotLayoutConversion(mod);
  }
};
} // namespace

namespace mlir::triton::NVIDIA {

std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeUnsupportedConversionsPass() {
  return std::make_unique<DecomposeUnsupportedConversions>();
}

} // namespace mlir::triton::NVIDIA
