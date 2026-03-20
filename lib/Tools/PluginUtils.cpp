#include "triton/Tools/PluginUtils.h"

llvm::Error TritonPlugin::checkLibraryValid(const std::string &error) const {
  if (!library.isValid()) {
    return llvm::createStringError(
        llvm::Twine("Failed to load plugin library: ") + error);
  }
  return llvm::Error::success();
}

llvm::Expected<intptr_t>
TritonPlugin::getAddressOfSymbol(const std::string &symbol) const {
  if (auto isValid = checkLibraryValid("not loaded"))
    return isValid;
  intptr_t getDetailsFn = (intptr_t)library.getAddressOfSymbol(symbol.c_str());
  if (!getDetailsFn) {
    return llvm::createStringError(llvm::Twine("Failed to get symbol: ") +
                                   symbol);
  }
  return getDetailsFn;
}

llvm::Expected<TritonPluginResult>
TritonPlugin::checkAPIResult(TritonPluginResult result,
                             const char *handle) const {
  if (result == TP_SUCCESS)
    return TP_SUCCESS;
  return llvm::createStringError(
      llvm::Twine("Failed to add/register a plugin pass (") + handle +
      "), error code: " + std::to_string(result));
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

  if ((intptr_t)library.getAddressOfSymbol(ENUMERATE_PASSES)) {
    auto enumeratePassesAPIOrErr =
        getAPI<EnumeratePyBindHandlesType, EnumeratePyBindHandlesCType>(
            ENUMERATE_PASSES);
    auto addPassAPIOrErr = getAPI<AddPassType, AddPassCType>(ADD_PASS);
    auto registerPassAPIOrErr =
        getAPI<RegisterPassType, RegisterPassCType>(REGISTER_PASS);

    if (auto Err = enumeratePassesAPIOrErr.takeError())
      return Err;
    if (auto Err = addPassAPIOrErr.takeError())
      return Err;
    if (auto Err = registerPassAPIOrErr.takeError())
      return Err;

    addPassAPI = *addPassAPIOrErr;
    registerPassAPI = *registerPassAPIOrErr;
    enumeratePassesAPI = *enumeratePassesAPIOrErr;
  }

  if ((intptr_t)library.getAddressOfSymbol(ENUMERATE_DIALECTS)) {
    auto enumerateDialectsAPIOrErr =
        getAPI<EnumeratePyBindHandlesType, EnumeratePyBindHandlesCType>(
            ENUMERATE_DIALECTS);
    auto dialectPluginInfoAPIOrErr =
        getAPI<DialectPluginInfoType, DialectPluginInfoCType>(
            DIALECT_PLUGININFO);

    if (auto Err = enumerateDialectsAPIOrErr.takeError())
      return Err;
    if (auto Err = dialectPluginInfoAPIOrErr.takeError())
      return Err;
    enumerateDialectsAPI = *enumerateDialectsAPIOrErr;
    dialectPluginInfoAPI = *dialectPluginInfoAPIOrErr;
  }

  if ((intptr_t)library.getAddressOfSymbol(ENUMERATE_CUSTOMOPS)) {
    auto enumerateCustomOpAPIOrErr =
        getAPI<EnumeratePyBindHandlesType, EnumeratePyBindHandlesCType>(
            ENUMERATE_CUSTOMOPS);
    auto addCustomOpAPIOrErr =
        getAPI<AddCustomOpType, AddCustomOpCType>(ADD_CUSTOMOP);

    if (auto Err = enumerateCustomOpAPIOrErr.takeError())
      return Err;
    if (auto Err = addCustomOpAPIOrErr.takeError())
      return Err;

    enumerateCustomOpAPI = *enumerateCustomOpAPIOrErr;
    addCustomOpAPI = *addCustomOpAPIOrErr;
  }

  isLoaded = true;
  return llvm::Error::success();
}

llvm::Expected<TritonPluginResult> TritonPlugin::enumeratePyBindHandles(
    EnumeratePyBindHandlesType &enumeratePyBindHandles,
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
  return llvm::createStringError(
      llvm::Twine("Failed to retrieve plugin pass handles, error code: ") +
      std::to_string(result));
}

llvm::Expected<TritonPluginResult>
TritonPlugin::getPassHandles(std::vector<const char *> &passNames) {
  if (auto Err = loadPlugin())
    return Err;
  // Do a check to see if the enumerate-passes api symbol is present, bail as
  // if there are 0 passes if not
  intptr_t isPassPluginSymbolPresent =
      (intptr_t)library.getAddressOfSymbol(ENUMERATE_PASSES);
  if (!isPassPluginSymbolPresent)
    return TP_SUCCESS;
  return enumeratePyBindHandles(enumeratePassesAPI, passNames);
}

llvm::Expected<TritonPluginResult>
TritonPlugin::getDialectHandles(std::vector<const char *> &dialectNames) {
  if (auto Err = loadPlugin())
    return Err;
  // Do a check to see if the enumerate-dialects api symbol is present, bail as
  // if there are 0 dialects if not
  intptr_t isDialectPluginSymbolPresent =
      (intptr_t)library.getAddressOfSymbol(ENUMERATE_DIALECTS);
  if (!isDialectPluginSymbolPresent)
    return TP_SUCCESS;
  return enumeratePyBindHandles(enumerateDialectsAPI, dialectNames);
}

llvm::Expected<TritonPluginResult>
TritonPlugin::getCustomOpHandles(std::vector<const char *> &customOpNames) {
  if (auto Err = loadPlugin())
    return Err;
  // Do a check to see if the enumerate-custom-ops api symbol is present, bail
  // as if there are 0 custom ops if not
  intptr_t isCustomOpSymbolPresent =
      (intptr_t)library.getAddressOfSymbol(ENUMERATE_CUSTOMOPS);
  if (!isCustomOpSymbolPresent)
    return TP_SUCCESS;
  return enumeratePyBindHandles(enumerateCustomOpAPI, customOpNames);
}

llvm::Expected<TritonPluginResult>
TritonPlugin::addPass(mlir::PassManager *pm, const char *passHandle,
                      const std::vector<std::string> &args) {
  if (auto Err = loadPlugin())
    return Err;
  return checkAPIResult(addPassAPI(pm, passHandle, args), passHandle);
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

llvm::Expected<TritonPluginResult>
TritonPlugin::addCustomOp(const char *handle, TritonOpBuilder &self,
                          std::vector<mlir::Value> &operands) {
  if (auto Err = loadPlugin())
    return Err;
  addCustomOpAPI(handle, self, operands);
  return TP_SUCCESS;
}

void registerPluginPasses(TritonPlugin &TP) {
  std::vector<const char *> passNames;
  if (auto result = TP.getPassHandles(passNames); !result)
    llvm::report_fatal_error(result.takeError());

  for (const char *passName : passNames)
    if (auto result = TP.registerPass(passName); !result)
      llvm::report_fatal_error(result.takeError());
}

void loadPluginDialects(TritonPlugin &TP, mlir::DialectRegistry &registry) {
  std::vector<const char *> dialectNames;
  if (auto result = TP.getDialectHandles(dialectNames); !result)
    llvm::report_fatal_error(result.takeError());

  for (unsigned i = 0; i < dialectNames.size(); ++i) {
    const char *dialectName = dialectNames.data()[i];
    auto result = TP.getDialectPluginInfo(dialectName);
    if (!result)
      llvm::report_fatal_error(result.takeError());
    ::mlir::DialectPluginLibraryInfo dialectPluginInfo = *result;
    dialectPluginInfo.registerDialectRegistryCallbacks(&registry);
  }
}

void loadPluginDialects(const std::string &filename,
                        mlir::DialectRegistry &registry) {
  TritonPlugin TP(filename);
  loadPluginDialects(TP, registry);
}
