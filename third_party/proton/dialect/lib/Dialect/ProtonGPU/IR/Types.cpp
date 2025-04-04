#include "Dialect/ProtonGPU/IR/Types.h"

#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton::proton::gpu;

#define GET_TYPEDEF_CLASSES
#include "Dialect/ProtonGPU/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// ProtonGPU Dialect
//===----------------------------------------------------------------------===//
void ::mlir::triton::proton::gpu::ProtonGPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/ProtonGPU/IR/Types.cpp.inc"
      >();
}
