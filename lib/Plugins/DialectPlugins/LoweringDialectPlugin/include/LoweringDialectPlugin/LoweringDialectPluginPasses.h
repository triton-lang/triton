#ifndef LOWERINGDIALECTPLUGIN_LOWERINGDIALECTPLUGINPASSES_H
#define LOWERINGDIALECTPLUGIN_LOWERINGDIALECTPLUGINPASSES_H

#include "LoweringDialectPlugin/LoweringDialectPluginDialect.h"
#include "LoweringDialectPlugin/LoweringDialectPluginOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class ModuleOp;

namespace triton {
namespace loweringdialectplugin {
#define GEN_PASS_DECL
#include "LoweringDialectPlugin/LoweringDialectPluginPasses.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createLoweringDialectPluginMagicOpPass();

#define GEN_PASS_REGISTRATION
#include "LoweringDialectPlugin/LoweringDialectPluginPasses.h.inc"
} // namespace loweringdialectplugin
} // namespace triton
} // namespace mlir

#endif
