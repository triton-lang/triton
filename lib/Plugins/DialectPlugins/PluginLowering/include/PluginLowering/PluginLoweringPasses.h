#ifndef LOWERINGDIALECTPLUGIN_LOWERINGDIALECTPLUGINPASSES_H
#define LOWERINGDIALECTPLUGIN_LOWERINGDIALECTPLUGINPASSES_H

#include "LoweringDialectPlugin/LoweringDialectPluginDialect.h"
#include "LoweringDialectPlugin/LoweringDialectPluginOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton {
namespace loweringdialectplugin {
#define GEN_PASS_DECL
#include "LoweringDialectPlugin/LoweringDialectPluginPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "LoweringDialectPlugin/LoweringDialectPluginPasses.h.inc"
} // namespace loweringdialectplugin
} // namespace triton
} // namespace mlir

#endif
