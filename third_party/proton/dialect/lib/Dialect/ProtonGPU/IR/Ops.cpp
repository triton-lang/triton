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
  auto bufferTy = mlir::cast<triton::gpu::MemDescType>(getBuffer().getType());
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
  auto scopeId = getScopeId();
  auto segmentType = getSegment().getType();
  auto granularity = segmentType.getGranularity();
  auto selectedIds = segmentType.getSelectIds();
  auto bufferSizeInBytes = segmentType.getNBytes();
  auto mod = getOperation()->getParentOfType<ModuleOp>();

  int numWarps = getTotalNumWarps(mod);

  int segmentNum = selectedIds.empty() ? numWarps : selectedIds.size();
  if (!llvm::isPowerOf2_32(bufferSizeInBytes / segmentNum))
    return emitOpError("profiling buffer segment size must be power of 2");

  if (scopeId < 0 || scopeId > 255)
    return emitOpError("scope id must be in [0, 255]");

  return success();
}

// -- SegmentAllocOp --
LogicalResult SegmentAllocOp::verify() {
  auto segmentType = getSegment().getType();
  auto granularity = segmentType.getGranularity();
  auto selectIds = segmentType.getSelectIds();
  if (granularity != Granularity::WARP && selectIds.size()) {
    return emitOpError(
        "only warp granularity supports non-empty selectIds for now");
  }
  return success();
}

// -- InitCtxOp --
LogicalResult InitCtxOp::verify() {
  if (getOperation()->getParentOfType<triton::gpu::WarpSpecializeOp>())
    return emitOpError(
        "can't initialize proton context in a warp specialized op");
  return success();
}

} // namespace gpu
} // namespace proton
} // namespace triton
} // namespace mlir
