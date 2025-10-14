#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"
#include <Python.h>
#include <unordered_map>


namespace mlir {
namespace triton {
namespace plugin {


#define GEN_PASS_DEF_TRITONGPUHELLOMLIRPLUGIN
#include "Passes.h.inc"

struct HelloMLIRPluginPass :
  public impl::TritonGPUHelloMLIRPluginBase<HelloMLIRPluginPass> {
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
  pm->addPass(mlir::triton::plugin::createTritonGPUHelloMLIRPlugin());
}

static const char *ADD_PLUGIN_PASS_NAME = "add_plugin";

static std::unordered_map<std::string, void (*)(mlir::PassManager *)> passMap = {
  {ADD_PLUGIN_PASS_NAME, addTritonPluginPass}
};

static std::vector<const char *> passNamesTable = {
  ADD_PLUGIN_PASS_NAME
};

extern "C" void tritonAddPluginPass(mlir::PassManager* pm, const char *passName) {
  std::string passNameStr(passName);
  passMap[passNameStr](pm);
}

extern "C" void tritonEnumeratePluginPasses(uint32_t *passCount, const char **passNames) {
  *passCount = passMap.size();
  if (!passNames)
    return;
  unsigned i = 0;
  for (auto passName : passNamesTable) {
    passNames[i] = passName;
  }
}

extern "C" void registerTritonPluginPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::plugin::createTritonGPUHelloMLIRPlugin();
  });
}
