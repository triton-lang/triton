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

  // Do a quick check to see if only plugin pass symbols are present
  intptr_t isPassPluginSymbolPresent =
      (intptr_t)library.getAddressOfSymbol(ENUMERATE_PASSES.c_str());
  intptr_t isDialectPluginSymbolPresent =
      (intptr_t)library.getAddressOfSymbol(ENUMERATE_DIALECTS.c_str());

  auto enumeratePassesAPIOrErr =
      getAPI<enumeratePyBindHandlesType, enumeratePyBindHandlesCType>(
          ENUMERATE_PASSES);
  auto addPassAPIOrErr = getAPI<addPassType, addPassCType>(ADD_PASS);
  auto registerPassAPIOrErr =
      getAPI<registerPassType, registerPassCType>(REGISTER_PASS);
  if (auto Err = enumeratePassesAPIOrErr.takeError())
    return Err;
  if (auto Err = enumerateDialectsAPIOrErr.takeError())
    return Err;
  if (auto Err = addPassAPIOrErr.takeError())
    return Err;
  if (auto Err = registerPassAPIOrErr.takeError())
    return Err;
  if (auto Err = dialectPluginInfoAPIOrErr.takeError())
    return Err;

  addPassAPI = *addPassAPIOrErr;
  registerPassAPI = *registerPassAPIOrErr;

  // We require at least the pass plugin symbols to be present
  // if the pass plugin symbols are present but dialects are not just set it to
  // a nullptr and trust the user to not try and call it. However for plugin
  // dialects both sets of APIs must be implemented
  llvm::Expected<enumeratePyBindHandlesType> enumerateDialectsAPIOrErr =
      (isPassPluginSymbolPresent && !isDialectPluginSymbolPresent)
          ? llvm::Expected<enumeratePyBindHandlesType>(nullptr)
          : getAPI<enumeratePyBindHandlesType, enumeratePyBindHandlesCType>(
                ENUMERATE_DIALECTS);
  if (auto Err = enumerateDialectsAPIOrErr.takeError())
    return Err;
  enumerateDialectsAPI = *enumerateDialectsAPIOrErr;

  llvm::Expected<dialectPluginInfoType> dialectPluginInfoAPIOrErr =
      (isPassPluginSymbolPresent && !isDialectPluginSymbolPresent)
          ? llvm::Expected<dialectPluginInfoType>(nullptr)
          : getAPI<dialectPluginInfoType, dialectPluginInfoCType>(
                DIALECT_PLUGININFO);

  if (auto Err = dialectPluginInfoAPIOrErr.takeError())
    return Err;
  dialectPluginInfoAPI = *dialectPluginInfoAPIOrErr;
  enumeratePassesAPI = *enumeratePassesAPIOrErr;
  isLoaded = true;
  return llvm::Error::success();
}

llvm::Expected<TritonPluginResult> TritonPlugin::enumeratePyBindHandles(
    enumeratePyBindHandlesType &enumeratePyBindHandles,
    std::vector<const char *> &handles) {
  if (auto Err = loadPlugin())
    return Err;

  uint32_t passCount = 0;
  handles.clear();
  auto result = enumeratePyBindHandles(&passCount, nullptr);
  if (result == TP_SUCCESS) {
    if (passCount == 0)
      return TP_SUCCESS;

    handles.resize(passCount);
    result = enumeratePyBindHandles(&passCount, handles.data());
  }

  if (result == TP_SUCCESS)
    return TP_SUCCESS;
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "Failed to retrive plugin pass handles, error code: " << result;
  return llvm::createStringError(msg);
}

llvm::Expected<TritonPluginResult>
TritonPlugin::getPassHandles(std::vector<const char *> &passNames) {
  return enumeratePyBindHandles(enumeratePassesAPI, passNames);
}

llvm::Expected<TritonPluginResult>
TritonPlugin::getDialectHandles(std::vector<const char *> &dialectNames) {
  return enumeratePyBindHandles(enumerateDialectsAPI, dialectNames);
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

llvm::Expected<::mlir::DialectPluginLibraryInfo>
TritonPlugin::getDialectPluginInfo(const char *dialectName) {
  if (auto Err = loadPlugin())
    return Err;
  return dialectPluginInfoAPI(dialectName);
}
