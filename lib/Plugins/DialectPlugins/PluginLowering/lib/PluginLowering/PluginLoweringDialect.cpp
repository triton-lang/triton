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

static void addTritonPluginPass(mlir::PassManager *pm) {
  pm->addPass(mlir::pluginlowering::createPluginLoweringSwitchBarFoo());
}

static void registerTritonPluginPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::pluginlowering::createPluginLoweringSwitchBarFoo();
  });
}

static const char *ADD_PLUGIN_PASS_NAME = "pluginlowering_fooop_inserter";
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
