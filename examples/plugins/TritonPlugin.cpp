#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/PluginUtils.h"
#include <unordered_map>

namespace mlir {
namespace triton {
namespace plugin {

#define GEN_PASS_DEF_TRITONGPUMLIRPLUGIN
#include "Passes.h.inc"

struct MLIRPluginPass : public impl::TritonGPUMLIRPluginBase<MLIRPluginPass> {
  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mod.walk([&](FunctionOpInterface funcOp) {
      StringAttr funcNameAttr = funcOp.getNameAttr();
      funcOp.setName("foo");
    });
  }
};

} // namespace plugin
} // namespace triton
} // namespace mlir

static void addTritonPluginPass(mlir::PassManager *pm) {
  pm->addPass(mlir::triton::plugin::createTritonGPUMLIRPlugin());
}

static void registerTritonPluginPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::plugin::createTritonGPUMLIRPlugin();
  });
}

static const char *ADD_PLUGIN_PASS_NAME = "add_plugin";
static std::unordered_map<std::string, void (*)(mlir::PassManager *)> passMap =
    {{ADD_PLUGIN_PASS_NAME, addTritonPluginPass}};
static std::unordered_map<std::string, void (*)()> registryMap = {
    {ADD_PLUGIN_PASS_NAME, registerTritonPluginPass}};
static std::vector<const char *> passNamesTable = {ADD_PLUGIN_PASS_NAME};

// Key APIs:

TRITON_PLUGIN_API
tritonAddPluginPass(mlir::PassManager *pm, TRITON_PLUGIN_PASS_ARGS) {
  std::string passNameStr(handle);
  if (passMap.find(passNameStr) == passMap.end())
    return TP_GENERIC_FAILURE;
  passMap[passNameStr](pm);
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonRegisterPluginPass(TRITON_PLUGIN_PASS_ARGS) {
  std::string passNameStr(handle);
  if (registryMap.find(passNameStr) == registryMap.end())
    return TP_GENERIC_FAILURE;
  registryMap[passNameStr]();
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonEnumeratePluginPasses(TRITON_PLUGIN_ENUMERATOR_ARGS) {
  if (!count)
    return TP_GENERIC_FAILURE;
  assert(passMap.size() == registryMap.size() &&
         "Expected register and add passes map size to match");
  *count = passMap.size();
  if (!handles)
    return TP_SUCCESS;
  unsigned i = 0;
  for (auto passName : passNamesTable) {
    handles[i++] = passName;
  }
  return TP_SUCCESS;
}
