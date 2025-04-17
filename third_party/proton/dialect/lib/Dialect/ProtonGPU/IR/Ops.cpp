#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "Dialect/ProtonGPU/IR/Ops.cpp.inc"

#include "Dialect/ProtonGPU/IR/OpsEnums.cpp.inc"

namespace mlir {
namespace triton {
namespace proton {
namespace gpu {

// -- StackAllocOp --
LogicalResult StackAllocOp::verify() {
  auto bufferTy = mlir::cast<triton::gpu::MemDescType>(getData().getType());
  auto elemTy = bufferTy.getElementType();
  auto rank = bufferTy.getRank();

  if (!isa<IntegerType>(elemTy) || elemTy.getIntOrFloatBitWidth() != 32)
    return emitOpError("proton stack buffer element type must be int 32");

  if (rank > 1)
    return emitOpError("proton stack currently only supports 1-D shapes");

  int stackAllocationSize =
      mlir::ShapedType::getNumElements(bufferTy.getShape());
  if (stackAllocationSize <= 0)
    return emitOpError("proton stack size must be positive and non-zero");

  return success();
}

// -- CircularRecordOp --
LogicalResult CircularStoreOp::verify() {
  auto segbaseOp =
      mlir::cast<proton::gpu::SegmentBaseOp>(getSeg().getDefiningOp());
  auto granularity = segbaseOp.getGranularity();
  auto selectedIds = segbaseOp.getSelectIdsAttr().asArrayRef();
  auto mod = getOperation()->getParentOfType<ModuleOp>();
  int segmentNum = selectedIds.empty() ? mlir::triton::gpu::lookupNumWarps(mod)
                                       : selectedIds.size();
  auto memDescTy = mlir::cast<triton::gpu::MemDescType>(getData().getType());
  const int bufferSizeInBytes =
      mlir::ShapedType::getNumElements(memDescTy.getShape()) *
      memDescTy.getElementType().getIntOrFloatBitWidth() / 8;

  if (!llvm::isPowerOf2_32(bufferSizeInBytes / segmentNum))
    return emitOpError("profiling buffer segment size must be power of 2");

  return success();
}

// -- SegmentBaseOp --
LogicalResult SegmentBaseOp::verify() {
  auto granularity = getGranularity();
  auto selectIdsAttr = getSelectIdsAttr();
  if (granularity != Granularity::WARP && selectIdsAttr.asArrayRef().size()) {
    return emitOpError(
        "only warp granularity supports non-empty selectIds for now");
  }
  return success();
}
} // namespace gpu
} // namespace proton
} // namespace triton
} // namespace mlir
