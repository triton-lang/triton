#include "DialectPlugin/DialectPluginTypes.h"

#include "DialectPlugin/DialectPluginDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::triton::plugin;

#define GET_TYPEDEF_CLASSES
#include "DialectPlugin/DialectPluginOpsTypes.cpp.inc"

void DialectPluginDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "DialectPlugin/DialectPluginOpsTypes.cpp.inc"
      >();
}
