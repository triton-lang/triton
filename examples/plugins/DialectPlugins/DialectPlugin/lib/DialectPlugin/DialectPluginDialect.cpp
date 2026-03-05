#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginOps.h"
#include "DialectPlugin/DialectPluginTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/PluginUtils.h"
#include <cstdlib>

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

#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "triton/Tools/PluginUtils.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;

static void addTritonPluginPass(mlir::PassManager *pm, int num_warps,
                                int threadsPerWarp, int numCTAs) {
  pm->addPass(mlir::triton::plugin::createConvertPluginGPUToLLVMPass());
}

static void registerTritonPluginPass(int num_warps, int threadsPerWarp,
                                     int numCTAs) {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::plugin::createConvertPluginGPUToLLVMPass();
  });
}

static void addTritonPluginPass2(mlir::PassManager *pm, int num_warps,
                                 int threadsPerWarp, int numCTAs) {
  pm->addPass(mlir::triton::plugin::createConvertPluginGPUToTritonGPUPass(
      num_warps, threadsPerWarp, numCTAs));
}

static void registerTritonPluginPass2(int num_warps, int threadsPerWarp,
                                      int numCTAs) {
  ::mlir::registerPass([=]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::triton::plugin::createConvertPluginGPUToTritonGPUPass(
        num_warps, threadsPerWarp, numCTAs);
  });
}

static const char *ADD_PLUGIN_PASS_NAME = "plugingpu_conversion";
static const char *ADD_PLUGIN_FARITH_PASS_NAME = "plugingpu_farith_conversion";
static std::unordered_map<std::string,
                          void (*)(mlir::PassManager *, int, int, int)>
    passMap = {{ADD_PLUGIN_FARITH_PASS_NAME, addTritonPluginPass2},
               {ADD_PLUGIN_PASS_NAME, addTritonPluginPass}};
static std::unordered_map<std::string, void (*)(int, int, int)> registryMap = {
    {ADD_PLUGIN_FARITH_PASS_NAME, registerTritonPluginPass2},
    {ADD_PLUGIN_PASS_NAME, registerTritonPluginPass}};
static std::vector<const char *> passNamesTable = {ADD_PLUGIN_PASS_NAME,
                                                   ADD_PLUGIN_FARITH_PASS_NAME};

// Key APIs:

TRITON_PLUGIN_API
tritonAddPluginPass(mlir::PassManager *pm, TRITON_PLUGIN_PASS_ARGS) {
  int num_warps, threadsPerWarp, numCTAs;
  num_warps = std::atoi(args[0].c_str());
  threadsPerWarp = std::atoi(args[1].c_str());
  numCTAs = std::atoi(args[2].c_str());

  std::string passNameStr(handle);
  if (passMap.find(passNameStr) == passMap.end())
    return TP_GENERIC_FAILURE;
  passMap[passNameStr](pm, num_warps, threadsPerWarp, numCTAs);
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonRegisterPluginPass(TRITON_PLUGIN_PASS_ARGS) {
  int num_warps, threadsPerWarp, numCTAs;
  num_warps = std::atoi(args[0].c_str());
  threadsPerWarp = std::atoi(args[1].c_str());
  numCTAs = std::atoi(args[2].c_str());

  std::string passNameStr(handle);
  if (registryMap.find(passNameStr) == registryMap.end())
    return TP_GENERIC_FAILURE;
  registryMap[passNameStr](num_warps, threadsPerWarp, numCTAs);
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

TRITON_PLUGIN_API
tritonEnumeratePluginCustomOps(TRITON_PLUGIN_ENUMERATOR_ARGS) {
  if (!count)
    return TP_GENERIC_FAILURE;
  *count = 1;
  if (!handles)
    return TP_SUCCESS;
  handles[0] = "create_custom_op";
  return TP_SUCCESS;
}

TRITON_PLUGIN_API
tritonAddPluginCustomOp(TRITON_PLUGIN_CUSTOM_OP_ARGS) {
  ::mlir::Value &dst = operands[0];
  ::mlir::Value &src = operands[1];

  dst = self.create<mlir::triton::plugin::FMagicOp>(src);
  operands[0] = dst;
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
