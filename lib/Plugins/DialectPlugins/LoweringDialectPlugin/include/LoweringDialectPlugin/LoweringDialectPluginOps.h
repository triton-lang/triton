#ifndef LOWERINGDIALECTPLUGIN_LOWERINGDIALECTPLUGINOPS_H
#define LOWERINGDIALECTPLUGIN_LOWERINGDIALECTPLUGINOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "LoweringDialectPlugin/LoweringDialectPluginOps.h.inc"

#endif // LOWERINGDIALECTPLUGIN_LOWERINGDIALECTPLUGINOPS_H
