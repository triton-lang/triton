#include "RegisterTritonDialects.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "triton/Tools/PluginUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

  if (std::string filename =
          mlir::triton::tools::getStrEnv("TRITON_PASS_PLUGIN_PATH");
      !filename.empty()) {
    TritonPlugin TP(filename);
    if (auto err = TP.loadPlugin(true /*bypassTritonExtEnabledCheck=*/)) {
      llvm::report_fatal_error(std::move(err));
    }

    std::vector<const char *> passNames;
    if (auto result = TP.getPassHandles(passNames); !result)
      llvm::report_fatal_error(result.takeError());

    for (const char *passName : passNames)
      if (auto result = TP.registerPass(passName); !result)
        llvm::report_fatal_error(result.takeError());

    loadPluginDialects(TP, registry);
  }

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton (GPU) optimizer driver\n", registry));
}
