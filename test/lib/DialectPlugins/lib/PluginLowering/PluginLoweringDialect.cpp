#include "PluginLowering/PluginLoweringDialect.h"
#include "PluginLowering/PluginLoweringOps.h"
#include "PluginLowering/PluginLoweringTypes.h"

using namespace mlir;
using namespace mlir::pluginlowering;

#include "PluginLowering/PluginLoweringOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PluginLowering dialect.
//===----------------------------------------------------------------------===//

void PluginLoweringDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PluginLowering/PluginLoweringOps.cpp.inc"
      >();
  registerTypes();
}

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

#include "PluginLowering/PluginLoweringDialect.h"
#include "PluginLowering/PluginLoweringPasses.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"
#include "triton/Tools/PluginUtils.h"

using namespace mlir;

// Key APIs:

TRITON_PLUGIN_API
tritonAddPluginPass(mlir::PassManager *pm, const char *passName) {
  return TP_GENERIC_FAILURE;
}

TRITON_PLUGIN_API
tritonRegisterPluginPass(const char *passName) {
  return TP_GENERIC_FAILURE;
}

TRITON_PLUGIN_API
tritonEnumeratePluginPasses(uint32_t *passCount, const char **passNames) {
  *passCount = 0;
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonEnumeratePluginDialects(uint32_t *dialectCount, const char **dialectNames) {
  *dialectCount = 1;
  if (!dialectNames)
    return TP_SUCCESS;
  dialectNames[0] = "PluginLowering";
  return TP_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) DialectPluginLibraryInfo
tritonGetDialectPluginInfo(const char *name) {
  return {MLIR_PLUGIN_API_VERSION, "PluginLowering", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<mlir::pluginlowering::PluginLoweringDialect>();
            mlir::pluginlowering::registerPasses();
          }};
}
