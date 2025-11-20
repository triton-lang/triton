#include "triton/Tools/PluginUtils.h"

llvm::Error TritonPlugin::checkLibraryValid(const std::string &error) const {
  if (!library.isValid()) {
    auto msg = llvm::Twine("Failed to load plugin library: " + error + "\n");
    return llvm::createStringError(msg);
  }
  return llvm::Error::success();
}

llvm::Expected<intptr_t>
TritonPlugin::getAddressOfSymbol(const std::string &symbol) const {
  if (auto isValid = checkLibraryValid("not loaded"))
    return isValid;
  intptr_t getDetailsFn = (intptr_t)library.getAddressOfSymbol(symbol.c_str());
  if (!getDetailsFn) {
    auto msg = llvm::Twine("Failed to get symbol: " + symbol + "\n");
    return llvm::createStringError(msg);
  }
  return getDetailsFn;
}

//   template <typename T, typename U>
//   llvm::Expected<T> TritonPlugin::getAPI(const std::string &symbol) const {
//     llvm::Expected<intptr_t> getDetailsFn = getAddressOfSymbol(symbol);
//     if (auto Err = getDetailsFn.takeError()) {
//       return Err;
//     }
//     auto func = reinterpret_cast<U>(*getDetailsFn);
//     return func;
//   }

llvm::Expected<TritonPluginResult>
TritonPlugin::checkAPIResult(TritonPluginResult result,
                             const char *handle) const {
  if (result == TP_SUCCESS)
    return TP_SUCCESS;
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "Failed to add/register plugin pass (" << handle
     << ") to pass manager, error code: " << result;
  return llvm::createStringError(msg);
}

std::runtime_error TritonPlugin::err2exp(llvm::Error Err) {
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << Err;
  return std::runtime_error(msg);
}

llvm::Error TritonPlugin::loadPlugin() {
  if (isLoaded)
    return llvm::Error::success();

  std::string error;
  library =
      llvm::sys::DynamicLibrary::getPermanentLibrary(filename.c_str(), &error);
  if (auto isValid = checkLibraryValid(error))
    return isValid;

  auto enumeratePassesAPIOrErr =
      getAPI<enumeratePassesType, enumeratePassesCType>(ENUMERATE_PASSES);
  auto addPassAPIOrErr = getAPI<addPassType, addPassCType>(ADD_PASS);
  auto registerPassAPIOrErr =
      getAPI<registerPassType, registerPassCType>(REGISTER_PASS);

  if (auto Err = enumeratePassesAPIOrErr.takeError())
    return Err;
  if (auto Err = addPassAPIOrErr.takeError())
    return Err;
  if (auto Err = registerPassAPIOrErr.takeError())
    return Err;

  enumeratePassesAPI = *enumeratePassesAPIOrErr;
  addPassAPI = *addPassAPIOrErr;
  registerPassAPI = *registerPassAPIOrErr;
  isLoaded = true;
  return llvm::Error::success();
}

llvm::Expected<TritonPluginResult>
TritonPlugin::getPassHandles(std::vector<const char *> &passNames) {
  if (auto Err = loadPlugin())
    return Err;

  uint32_t passCount = 0;
  passNames.clear();
  auto result = enumeratePassesAPI(&passCount, nullptr);
  if (result == TP_SUCCESS) {
    if (passCount == 0)
      return TP_SUCCESS;

    passNames.resize(passCount);
    result = enumeratePassesAPI(&passCount, passNames.data());
  }

  if (result == TP_SUCCESS)
    return TP_SUCCESS;
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "Failed to retrive plugin pass handles, error code: " << result;
  return llvm::createStringError(msg);
}

llvm::Expected<TritonPluginResult>
TritonPlugin::addPass(mlir::PassManager *pm, const char *passHandle) {
  if (auto Err = loadPlugin())
    return Err;
  return checkAPIResult(addPassAPI(pm, passHandle), passHandle);
}

llvm::Expected<TritonPluginResult>
TritonPlugin::registerPass(const char *passHandle) {
  if (auto Err = loadPlugin())
    return Err;
  return checkAPIResult(registerPassAPI(passHandle), passHandle);
}
