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

static const char *PLUGIN_NAME = "TritonPlugin";
static const char *PASS_NAME = "add_plugin";
static const char *VERSION = "0.1.0";

using namespace mlir::triton;

TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  static plugin::PassInfo pass = {PASS_NAME, VERSION, addTritonPluginPass,
                                  registerTritonPluginPass};
  static plugin::PassInfo passes[] = {pass};
  static plugin::PluginInfo info = {TRITON_PLUGIN_API_VERSION,
                                    PLUGIN_NAME,
                                    VERSION,
                                    passes,
                                    1,
                                    nullptr,
                                    0,
                                    nullptr,
                                    0};
  return &info;
}
