#ifndef TRITON_PLUGIN_UTILS_H
#define TRITON_PLUGIN_UTILS_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include <cstdint>

extern "C" {
enum TritonPluginResult {
  TP_SUCCESS = 0,
  TP_GENERIC_FAILURE = 1,
};
};
#define TRITON_PLUGIN_API                                                      \
  extern "C" __attribute__((visibility("default"))) TritonPluginResult
#define TRITON_PLUGIN_API_TYPE(_TYPE)                                          \
  extern "C" __attribute__((visibility("default"))) _TYPE

struct TritonPlugin {
  TritonPlugin() = delete;
  TritonPlugin(std::string filename) : filename(filename) {}

public:
  llvm::Error checkLibraryValid(const std::string &error) const;
  const std::string ENUMERATE_PASSES = "tritonEnumeratePluginPasses";
  const std::string ENUMERATE_DIALECTS = "tritonEnumeratePluginDialects";
  const std::string DIALECT_PLUGININFO = "tritonGetDialectPluginInfo";

private:
  using enumeratePyBindHandlesType =
      std::function<TritonPluginResult(uint32_t *, const char **)>;
  using enumeratePyBindHandlesCType = TritonPluginResult (*)(uint32_t *,
                                                             const char **);

  const std::string ADD_PASS = "tritonAddPluginPass";
  using addPassType =
      std::function<TritonPluginResult(mlir::PassManager *, const char *)>;
  using addPassCType = TritonPluginResult (*)(mlir::PassManager *,
                                              const char *);

  const std::string REGISTER_PASS = "tritonRegisterPluginPass";
  using registerPassType = std::function<TritonPluginResult(const char *)>;
  using registerPassCType = TritonPluginResult (*)(const char *);

  using dialectPluginInfoType =
      std::function<::mlir::DialectPluginLibraryInfo(const char *)>;
  using dialectPluginInfoCType =
      ::mlir::DialectPluginLibraryInfo (*)(const char *);

  llvm::Expected<intptr_t> getAddressOfSymbol(const std::string &symbol) const;

  template <typename T, typename U>
  llvm::Expected<T> getAPI(const std::string &symbol) const {
    llvm::Expected<intptr_t> getDetailsFn = getAddressOfSymbol(symbol);
    if (auto Err = getDetailsFn.takeError()) {
      return Err;
    }
    auto func = reinterpret_cast<U>(*getDetailsFn);
    return func;
  }

  llvm::Expected<TritonPluginResult> checkAPIResult(TritonPluginResult result,
                                                    const char *handle) const;
  llvm::Expected<TritonPluginResult>
  enumeratePyBindHandles(enumeratePyBindHandlesType &enumeratePyBindHandles,
                         std::vector<const char *> &passNames);

public:
  std::runtime_error err2exp(llvm::Error Err);

  llvm::Error loadPlugin();

  llvm::Expected<TritonPluginResult>
  getPassHandles(std::vector<const char *> &handles);

  llvm::Expected<TritonPluginResult>
  getDialectHandles(std::vector<const char *> &handles);

  llvm::Expected<TritonPluginResult> addPass(mlir::PassManager *pm,
                                             const char *passHandle);

  llvm::Expected<TritonPluginResult> registerPass(const char *passHandle);

  llvm::Expected<::mlir::DialectPluginLibraryInfo>
  getDialectPluginInfo(const char *dialectName);

private:
  std::string filename = "";
  mutable llvm::sys::DynamicLibrary library;
  enumeratePyBindHandlesType enumeratePassesAPI;
  enumeratePyBindHandlesType enumerateDialectsAPI;
  addPassType addPassAPI;
  registerPassType registerPassAPI;
  dialectPluginInfoType dialectPluginInfoAPI;
  bool isLoaded = false;
};

#endif // TRITON_PLUGIN_UTILS_H
