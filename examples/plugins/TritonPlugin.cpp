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

#define GEN_PASS_DECL_TRITONGPUMLIRPLUGIN
#define GEN_PASS_DEF_TRITONGPUMLIRPLUGIN
#include "Passes.h.inc"

struct MLIRPluginPass : public impl::TritonGPUMLIRPluginBase<MLIRPluginPass> {
  using TritonGPUMLIRPluginBase::TritonGPUMLIRPluginBase;

  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    std::string name;
    llvm::raw_string_ostream sstr(name);
    sstr << "foo";
    if (num_warps != 4)
      sstr << "_num_warps_" << num_warps;

    mod.walk([&](FunctionOpInterface funcOp) {
      StringAttr funcNameAttr = funcOp.getNameAttr();
      funcOp.setName(name);
    });
  }
};

} // namespace plugin
} // namespace triton
} // namespace mlir

static void addTritonPluginPass(mlir::PassManager *pm,
                                const std::vector<std::string> &args) {
  if (args.empty()) {
    pm->addPass(mlir::triton::plugin::createTritonGPUMLIRPlugin());
    return;
  }

  mlir::triton::plugin::TritonGPUMLIRPluginOptions opts;
  opts.num_warps = std::atoi(args[0].c_str());
  pm->addPass(mlir::triton::plugin::createTritonGPUMLIRPlugin((opts)));
}

static void registerTritonPluginPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::plugin::createTritonGPUMLIRPlugin();
  });
}

static const char *ADD_PLUGIN_PASS_NAME = "add_plugin";
static std::unordered_map<std::string, decltype(&addTritonPluginPass)> passMap =
    {{ADD_PLUGIN_PASS_NAME, addTritonPluginPass}};
static std::unordered_map<std::string, decltype(&registerTritonPluginPass)>
    registryMap = {{ADD_PLUGIN_PASS_NAME, registerTritonPluginPass}};
static std::vector<const char *> passNamesTable = {ADD_PLUGIN_PASS_NAME};

// Key APIs:

TRITON_PLUGIN_API
tritonAddPluginPass(mlir::PassManager *pm, const char *passName,
                    const std::vector<std::string> &args) {
  std::string passNameStr(passName);
  if (passMap.find(passNameStr) == passMap.end())
    return TP_GENERIC_FAILURE;
  passMap[passNameStr](pm, args);
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonRegisterPluginPass(const char *passName) {
  std::string passNameStr(passName);
  if (registryMap.find(passNameStr) == registryMap.end())
    return TP_GENERIC_FAILURE;
  registryMap[passNameStr]();
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonEnumeratePluginPasses(uint32_t *passCount, const char **passNames) {
  if (!passCount)
    return TP_GENERIC_FAILURE;
  auto count = passMap.size();
  assert(count == registryMap.size() &&
         "Expected register and add passes map size to match");
  *passCount = count;
  if (!passNames)
    return TP_SUCCESS;
  unsigned i = 0;
  for (auto passName : passNamesTable) {
    passNames[i++] = passName;
  }
  return TP_SUCCESS;
}
