#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginOps.h"
#include "DialectPlugin/DialectPluginTypes.h"

using namespace mlir;
using namespace mlir::triton::plugin;

#include "DialectPlugin/DialectPluginOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// DialectPlugin dialect.
//===----------------------------------------------------------------------===//

void DialectPluginDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DialectPlugin/DialectPluginOps.cpp.inc"
      >();
  registerTypes();
}

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginPasses.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "triton/Tools/PluginUtils.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;

static void addTritonPluginPass(mlir::PassManager *pm) {
  pm->addPass(mlir::triton::plugin::createConvertPluginGPUToLLVMPass());
}

static void registerTritonPluginPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::plugin::createConvertPluginGPUToLLVMPass();
  });
}

static const char *ADD_PLUGIN_PASS_NAME = "plugingpu_conversion";
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

TRITON_PLUGIN_API
tritonEnumeratePluginDialects(TRITON_PLUGIN_ENUMERATOR_ARGS) {
  *count = 1;
  if (!handles)
    return TP_SUCCESS;
  handles[0] = "DialectPlugin";
  return TP_SUCCESS;
}

TRITON_PLUGIN_API_TYPE(DialectPluginLibraryInfo)
tritonGetDialectPluginInfo(const char *name) {
  return {MLIR_PLUGIN_API_VERSION, "DialectPlugin", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<mlir::triton::plugin::DialectPluginDialect>();
            mlir::triton::plugin::registerpluginPasses();
          }};
}
