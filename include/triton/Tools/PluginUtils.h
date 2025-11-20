#ifndef TRITON_PLUGIN_UTILS_H
#define TRITON_PLUGIN_UTILS_H

#include "mlir/Pass/PassManager.h"
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

struct TritonPlugin {
  TritonPlugin() = delete;
  TritonPlugin(std::string filename) : filename(filename) {}

private:
  const std::string ENUMERATE_PASSES = "tritonEnumeratePluginPasses";
  using enumeratePassesType =
      std::function<TritonPluginResult(uint32_t *, const char **)>;
  using enumeratePassesCType = TritonPluginResult (*)(uint32_t *,
                                                      const char **);

  const std::string ADD_PASS = "tritonAddPluginPass";
  using addPassType =
      std::function<TritonPluginResult(mlir::PassManager *, const char *)>;
  using addPassCType = TritonPluginResult (*)(mlir::PassManager *,
                                              const char *);

  const std::string REGISTER_PASS = "tritonRegisterPluginPass";
  using registerPassType = std::function<TritonPluginResult(const char *)>;
  using registerPassCType = TritonPluginResult (*)(const char *);

  llvm::Error checkLibraryValid(const std::string &error) const;

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

public:
  std::runtime_error err2exp(llvm::Error Err);

  llvm::Error loadPlugin();

  llvm::Expected<TritonPluginResult>
  getPassHandles(std::vector<const char *> &passNames);

  llvm::Expected<TritonPluginResult> addPass(mlir::PassManager *pm,
                                             const char *passHandle);

  llvm::Expected<TritonPluginResult> registerPass(const char *passHandle);

private:
  std::string filename = "";
  mutable llvm::sys::DynamicLibrary library;
  enumeratePassesType enumeratePassesAPI;
  addPassType addPassAPI;
  registerPassType registerPassAPI;
  bool isLoaded = false;
};

#endif // TRITON_PLUGIN_UTILS_H
