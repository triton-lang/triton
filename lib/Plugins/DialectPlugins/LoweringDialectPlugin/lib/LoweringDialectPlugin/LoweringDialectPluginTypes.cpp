#include "LoweringDialectPlugin/LoweringDialectPluginTypes.h"

#include "LoweringDialectPlugin/LoweringDialectPluginDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::triton::loweringdialectplugin;

#define GET_TYPEDEF_CLASSES
#include "LoweringDialectPlugin/LoweringDialectPluginOpsTypes.cpp.inc"

void LoweringDialectPluginDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "LoweringDialectPlugin/LoweringDialectPluginOpsTypes.cpp.inc"
      >();
}
