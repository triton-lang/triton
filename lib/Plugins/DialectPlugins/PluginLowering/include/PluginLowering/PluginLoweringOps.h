#ifndef PLUGINLOWERING_PLUGINLOWERINGOPS_H
#define PLUGINLOWERING_PLUGINLOWERINGOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "PluginLowering/PluginLoweringOps.h.inc"

#endif // PLUGINLOWERING_PLUGINLOWERINGOPS_H
