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

namespace {
LogicalResult verifySegment(Operation *op, SegmentType segmentType) {
  auto selectedIds = segmentType.getSelectIds();
  auto bufferSizeInBytes = segmentType.getNBytes();
  auto mod = op->getParentOfType<ModuleOp>();
  int numWarps = getTotalNumWarps(mod);
  int segmentNum = selectedIds.empty() ? numWarps : selectedIds.size();
  if (!llvm::isPowerOf2_32(bufferSizeInBytes / segmentNum))
    return op->emitOpError("profiling buffer segment size must be power of 2");
  return success();
}
} // namespace

// -- CircularRecordOp --
LogicalResult CircularStoreOp::verify() {
  auto segmentType = getSegment().getType();
  if (failed(verifySegment(getOperation(), segmentType)))
    return failure();

  if (static_cast<bool>(getScopeIdAttr()) ==
      static_cast<bool>(getDynamicScopeId()))
    return emitOpError(
        "requires exactly one static or dynamic scope identifier");

  if (auto scopeId = getScopeIdAttr()) {
    auto value = scopeId.getInt();
    if (value < 0 || value > 255)
      return emitOpError("scope id must be in [0, 255]");
  }

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
