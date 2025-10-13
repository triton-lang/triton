#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <Python.h>


namespace mlir {
namespace triton {
namespace plugin {


#define GEN_PASS_DEF_TRITONGPUHELLOEXTENSION
#include "Passes.h.inc"

struct HelloExtensionPass :
  public impl::TritonGPUHelloExtensionBase<HelloExtensionPass> {
  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mod.walk([&](FunctionOpInterface funcOp) {
      StringAttr funcNameAttr = funcOp.getNameAttr();
      funcOp.setName("foo");
    }
    );

  }
};

} // namespace plugin
} // namespace triton
} // namespace mlir

extern "C" void addTritonPluginPass(mlir::PassManager* pm) {
  pm->addPass(mlir::triton::plugin::createTritonGPUHelloExtension());
}

extern "C" void registerTritonPluginPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::plugin::createTritonGPUHelloExtension();
  });
}
