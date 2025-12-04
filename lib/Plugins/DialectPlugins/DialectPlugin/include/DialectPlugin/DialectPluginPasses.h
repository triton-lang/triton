#ifndef DIALECTPLUGIN_DIALECTPLUGINPASSES_H
#define DIALECTPLUGIN_DIALECTPLUGINPASSES_H

#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class ModuleOp;

namespace triton {
namespace dialectplugin {
#define GEN_PASS_DECL
#include "DialectPlugin/DialectPluginPasses.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertPluginGPUToTritonGPUPass();

#define GEN_PASS_REGISTRATION
#include "DialectPlugin/DialectPluginPasses.h.inc"
} // namespace dialectplugin
} // namespace triton
} // namespace mlir

#endif
