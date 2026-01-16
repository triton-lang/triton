#ifndef DIALECTPLUGIN_DIALECTPLUGINPASSES_H
#define DIALECTPLUGIN_DIALECTPLUGINPASSES_H

#include "DialectPlugin/DialectPluginDialect.h"
#include "DialectPlugin/DialectPluginOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class ModuleOp;

namespace triton {
namespace plugin {
#define GEN_PASS_DECL
#include "DialectPlugin/DialectPluginPasses.h.inc"

std::unique_ptr<OperationPass<ModuleOp>>
createConvertPluginGPUToLLVMPass(int32_t num_warps = 80);

#define GEN_PASS_REGISTRATION
#include "DialectPlugin/DialectPluginPasses.h.inc"
} // namespace plugin
} // namespace triton
} // namespace mlir

#endif
