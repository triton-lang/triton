#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.cpp.inc"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "proton/dialect/include/Dialect/ProtonGPU/IR/AttrDefs.cpp.inc"

using namespace mlir;

const int mlir::triton::proton::gpu::getBytesPerClockEntry() { return 8; }
const int mlir::triton::proton::gpu::getCircularHeaderSize() { return 16; }

void mlir::triton::proton::gpu::ProtonGPUDialect::initialize() {
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/ProtonGPU/IR/AttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Dialect/ProtonGPU/IR/Ops.cpp.inc"
      >();
}

const int mlir::triton::proton::gpu::getTotalNumWarps(ModuleOp mod) {
  int numWarps = mlir::triton::gpu::lookupNumWarps(mod);
  if (auto totalNumWarps =
          mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps"))
    numWarps = totalNumWarps.getInt();
  return numWarps;
}
