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
  static constexpr char ENUMERATE_PASSES[] = "tritonEnumeratePluginPasses";
  static constexpr char ENUMERATE_DIALECTS[] = "tritonEnumeratePluginDialects";
  static constexpr char DIALECT_PLUGININFO[] = "tritonGetDialectPluginInfo";
  static constexpr char ADD_PASS[] = "tritonAddPluginPass";
  static constexpr char REGISTER_PASS[] = "tritonRegisterPluginPass";

private:
  using EnumeratePyBindHandlesType =
      std::function<TritonPluginResult(uint32_t *, const char **)>;
  using EnumeratePyBindHandlesCType = TritonPluginResult (*)(uint32_t *,
                                                             const char **);

  using AddPassType =
      std::function<TritonPluginResult(mlir::PassManager *, const char *)>;
  using AddPassCType = TritonPluginResult (*)(mlir::PassManager *,
                                              const char *);

  using RegisterPassType = std::function<TritonPluginResult(const char *)>;
  using RegisterPassCType = TritonPluginResult (*)(const char *);

  using DialectPluginInfoType =
      std::function<::mlir::DialectPluginLibraryInfo(const char *)>;
  using DialectPluginInfoCType =
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
  enumeratePyBindHandles(EnumeratePyBindHandlesType &enumeratePyBindHandles,
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
  EnumeratePyBindHandlesType enumeratePassesAPI;
  EnumeratePyBindHandlesType enumerateDialectsAPI;
  AddPassType addPassAPI;
  RegisterPassType registerPassAPI;
  DialectPluginInfoType dialectPluginInfoAPI;
  bool isLoaded = false;
};

#endif // TRITON_PLUGIN_UTILS_H
