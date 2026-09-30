#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::triton {

static llvm::StringMap<TargetInfoBase::Factory> &getTargetInfoFactories() {
  static llvm::StringMap<TargetInfoBase::Factory> factories;
  return factories;
}

void TargetInfoBase::registerFactory(StringRef target, Factory factory) {
  getTargetInfoFactories()[target] = std::move(factory);
}

std::unique_ptr<TargetInfoBase>
TargetInfoBase::fromModuleOp(ModuleOp moduleOp) {
  if (!moduleOp)
    return nullptr;
  auto targetAttr = moduleOp->getAttrOfType<StringAttr>(gpu::AttrTargetName);
  if (!targetAttr)
    return nullptr;
  auto &factories = getTargetInfoFactories();
  auto it = factories.find(targetAttr.getValue().split(':').first);
  return it != factories.end() ? it->second(moduleOp) : nullptr;
}

} // namespace mlir::triton
