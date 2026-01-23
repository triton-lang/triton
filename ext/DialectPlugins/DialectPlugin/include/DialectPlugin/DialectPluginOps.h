#ifndef DIALECTPLUGIN_DIALECTPLUGINOPS_H
#define DIALECTPLUGIN_DIALECTPLUGINOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "DialectPlugin/DialectPluginOps.h.inc"

#endif // DIALECTPLUGIN_DIALECTPLUGINOPS_H
