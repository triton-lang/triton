#include "triton/Tools/PluginUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"

#define DEBUG_TYPE "triton-plugins"

using namespace mlir::triton::plugin;

static bool isTritonAndPluginsVersionsMatch(const std::string &pluginVersion) {
  // Here, if TRITON_PLUGIN_VERSION_CHECK is unset, then we simply do a default
  // version check. However, if it is set then we either do a full (git hash)
  // check or we skip all checking.
  auto doCheck =
      mlir::triton::tools::isEnvValueBool("TRITON_PLUGIN_VERSION_CHECK");

  // Skip check when TRITON_PLUGIN_VERSION_CHECK is set false
  if (doCheck.has_value() && !doCheck.value())
    return true;

  // Check full version string when TRITON_PLUGIN_VERSION_CHECK is set true
  if (doCheck.has_value() && doCheck.value())
    return pluginVersion == TRITON_VERSION;

  // Do partial release version check when TRITON_PLUGIN_VERSION_CHECK unset
  assert(!doCheck.has_value() && "Expected TRITON_PLUGIN_VERSION_CHECK unset");
  return llvm::StringRef(pluginVersion).split('+').first ==
         llvm::StringRef(TRITON_VERSION).split('+').first;
}

llvm::Expected<TritonPlugin> TritonPlugin::load(const std::string &filename) {
  std::string error;
  auto library =
      llvm::sys::DynamicLibrary::getPermanentLibrary(filename.c_str(), &error);
  if (!library.isValid())
    return llvm::make_error<llvm::StringError>(
        Twine("Could not load library '") + filename + "': " + error,
        llvm::inconvertibleErrorCode());

  TritonPlugin plugin{filename, library};

  // tritonGetPluginInfo should be resolved to the definition from the
  // plugin we are currently loading.
  intptr_t getInfoFn =
      (intptr_t)library.getAddressOfSymbol("tritonGetPluginInfo");
  if (!getInfoFn)
    return llvm::make_error<llvm::StringError>(
        Twine("Plugin entry point not found in '") + filename + "'.",
        llvm::inconvertibleErrorCode());

  plugin.info = reinterpret_cast<decltype(tritonGetPluginInfo) *>(getInfoFn)();

  if (plugin.info->apiVersion != TRITON_PLUGIN_API_VERSION)
    return llvm::make_error<llvm::StringError>(
        Twine("Wrong API version on plugin '") + filename + "'. Got version " +
            Twine(plugin.info->apiVersion) + ", supported version is " +
            Twine(TRITON_PLUGIN_API_VERSION) + ".",
        llvm::inconvertibleErrorCode());

  if (!isTritonAndPluginsVersionsMatch(plugin.info->tritonVersion))
    return llvm::make_error<llvm::StringError>(
        Twine("Wrong TRITON version on plugin '") + filename +
            "'. Got version " + Twine(plugin.info->tritonVersion) +
            ", supported version is " + Twine(TRITON_VERSION) + ".",
        llvm::inconvertibleErrorCode());

  return plugin;
}

const std::vector<Pass> TritonPlugin::listPasses() const {
  if (!info->passes && info->numPasses > 0)
    llvm::reportFatalUsageError(llvm::createStringError(
        llvm::Twine("Invalid pass pointer in plugin '") + filename + "'."));
  LLVM_DEBUG(llvm::dbgs() << "Listing " << info->numPasses
                          << " passes for plugin " << info->pluginName << ":"
                          << info->pluginVersion << "\n");

  std::vector<Pass> passes;
  for (auto i = 0; i < info->numPasses; ++i) {
    const auto pass = &info->passes[i];
    if (pass->addPass) {
      LLVM_DEBUG(llvm::dbgs() << "Listing pass " << pass->name << ":"
                              << pass->version << "\n");
      passes.push_back(Pass(pass->name, pass->addPass));
    }
  }
  return passes;
}

void TritonPlugin::registerPasses() const {
  if (!info->passes && info->numPasses > 0)
    llvm::reportFatalUsageError(llvm::createStringError(
        llvm::Twine("Invalid pass pointer in plugin '") + filename + "'."));
  LLVM_DEBUG(llvm::dbgs() << "Registering " << info->numPasses
                          << " passes for plugin " << info->pluginName << ":"
                          << info->pluginVersion << "\n");

  for (auto i = 0; i < info->numPasses; ++i) {
    const auto &pass = info->passes[i];
    if (pass.registerPass) {
      LLVM_DEBUG(llvm::dbgs() << "Registering pass " << pass.name << ":"
                              << pass.version << "\n");
      pass.registerPass();
    }
  }
}

void TritonPlugin::registerDialects(DialectRegistry &dialectRegistry) const {
  if (!info->dialects && info->numDialects > 0)
    llvm::reportFatalUsageError(llvm::createStringError(
        llvm::Twine("Invalid dialect pointer in plugin '") + filename + "'."));
  LLVM_DEBUG(llvm::dbgs() << "Registering " << info->numDialects
                          << " dialects for plugin " << info->pluginName << ":"
                          << info->pluginVersion << "\n");

  for (auto i = 0; i < info->numDialects; ++i) {
    const auto &dialect = info->dialects[i];
    if (dialect.registerDialect) {
      LLVM_DEBUG(llvm::dbgs() << "Registering dialect " << dialect.name << ":"
                              << dialect.version << "\n");
      dialect.registerDialect(&dialectRegistry);
    }
  }
}

const std::vector<Op> TritonPlugin::listOps() const {
  if (!info->ops && info->numOps > 0)
    llvm::reportFatalUsageError(llvm::createStringError(
        llvm::Twine("Invalid custom op pointer in plugin '") + filename +
        "'."));
  LLVM_DEBUG(llvm::dbgs() << "Listing " << info->numOps
                          << " custom ops for plugin " << info->pluginName
                          << ":" << info->pluginVersion << "\n");

  std::vector<Op> ops;
  for (auto i = 0; i < info->numOps; ++i) {
    const auto op = &info->ops[i];
    if (op->addOp) {
      LLVM_DEBUG(llvm::dbgs() << "Listing custom op " << op->name << "\n");
      ops.push_back(Op(op->name, op->addOp));
    }
  }
  return ops;
}

static std::vector<TritonPlugin> plugins;
static bool pluginsLoaded = false;
const std::vector<TritonPlugin> &mlir::triton::plugin::loadPlugins() {
  if (pluginsLoaded)
    return plugins;

  // Bailing when libtriton symbols are not visible is done to prevent
  // crashes caused by loading plugins that will never find their dependent
  // symbols (which are hidden by libtriton).
#if !defined(TRITON_EXT_ENABLED) || TRITON_EXT_ENABLED == 0
  bool skipLoading = true;
#else
  bool skipLoading = false;
#endif

  if (const char *env = std::getenv("TRITON_PLUGIN_PATHS")) {
    llvm::SmallVector<llvm::StringRef, 4> paths;
    llvm::StringRef(env).split(paths, ':');
    for (const auto &path : paths) {
      if (skipLoading) {
        llvm::errs() << "\n"
                     << "\n=================== WARNING =====================\n"
                     << "Triton will not load the following extension\n"
                     << "because it is not built with TRITON_EXT_ENABLED:\n"
                     << path
                     << "\n=================================================\n"
                     << "\n";
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "Loading plugin from path: " << path << "\n");
      auto pluginOrErr = TritonPlugin::load(path.str());
      if (auto err = pluginOrErr.takeError()) {
        llvm::Error wrappedErr = llvm::createStringError(
            llvm::Twine("Failed to load plugin from path: ") + path +
            ". Error: " + llvm::toString(std::move(err)));
        llvm::reportFatalUsageError(std::move(wrappedErr));
      }
      plugins.push_back(std::move(*pluginOrErr));
    }
  }

  pluginsLoaded = true;
  return plugins;
}

#undef DEBUG_TYPE
