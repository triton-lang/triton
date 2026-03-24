#include "RegisterTritonDialects.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#if TRITON_EXT_ENABLED
#include "triton/Tools/PluginUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#endif

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

#if TRITON_EXT_ENABLED
  if (std::string filename =
          mlir::triton::tools::getStrEnv("TRITON_PASS_PLUGIN_PATH");
      !filename.empty()) {
    TritonPlugin TP(filename);
    loadPluginDialects(TP, registry, true /*loadPluginPasses=*/);
  }
#endif

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton (GPU) optimizer driver\n", registry));
}
