#ifndef PLUGINLOWERING_PLUGINLOWERINGPASSES_H
#define PLUGINLOWERING_PLUGINLOWERINGPASSES_H

#include "PluginLowering/PluginLoweringDialect.h"
#include "PluginLowering/PluginLoweringOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton {
namespace pluginlowering {
#define GEN_PASS_DECL
#include "PluginLowering/PluginLoweringPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "PluginLowering/PluginLoweringPasses.h.inc"
} // namespace pluginlowering
} // namespace triton
} // namespace mlir

#endif
