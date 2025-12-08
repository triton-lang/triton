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
tritonAddPluginPass(mlir::PassManager *pm, const char *passName) {
  std::string passNameStr(passName);
  if (passMap.find(passNameStr) == passMap.end())
    return TP_GENERIC_FAILURE;
  passMap[passNameStr](pm);
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
    passNames[i] = passName;
  }
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonEnumeratePluginDialects(uint32_t *dialectCount,
                              const char **dialectNames) {
  *dialectCount = 1;
  if (!dialectNames)
    return TP_SUCCESS;
  dialectNames[0] = "DialectPlugin";
  return TP_SUCCESS;
}

extern "C" __attribute__((visibility("default"))) DialectPluginLibraryInfo
tritonGetDialectPluginInfo(const char *name) {
  return {MLIR_PLUGIN_API_VERSION, "DialectPlugin", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->insert<mlir::triton::plugin::DialectPluginDialect>();
            mlir::triton::plugin::registerpluginPasses();
          }};
}
