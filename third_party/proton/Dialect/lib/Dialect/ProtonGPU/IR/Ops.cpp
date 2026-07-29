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
bool isMetricTypeCompatible(Type type, MetricValueType metricType) {
  switch (metricType) {
  case MetricValueType::BOOL:
    return type.isInteger(1);
  case MetricValueType::I8:
  case MetricValueType::U8:
    return type.isInteger(8);
  case MetricValueType::I16:
  case MetricValueType::U16:
    return type.isInteger(16);
  case MetricValueType::I32:
  case MetricValueType::U32:
    return type.isInteger(32);
  case MetricValueType::F16:
    return type.isF16();
  case MetricValueType::BF16:
    return type.isBF16();
  case MetricValueType::F32:
    return type.isF32();
  case MetricValueType::NONE:
    return false;
  }
  llvm_unreachable("unknown metric value type");
}

LogicalResult verifyScopeId(Operation *op, IntegerAttr scopeId) {
  auto value = scopeId.getInt();
  if (value < 0 || value > 255)
    return op->emitOpError("scope id must be in [0, 255]");
  return success();
}
} // namespace

// -- CircularRecordOp --
LogicalResult CircularStoreOp::verify() {
  auto segmentType = getSegment().getType();
  auto selectedIds = segmentType.getSelectIds();
  auto bufferSizeInBytes = segmentType.getNBytes();
  auto mod = getOperation()->getParentOfType<ModuleOp>();
  int numWarps = getTotalNumWarps(mod);
  int segmentNum = selectedIds.empty() ? numWarps : selectedIds.size();
  if (!llvm::isPowerOf2_32(bufferSizeInBytes / segmentNum))
    return emitOpError("profiling buffer segment size must be power of 2");

  if (static_cast<bool>(getScopeIdAttr()) ==
      static_cast<bool>(getDynamicScopeId()))
    return emitOpError(
        "requires exactly one static or dynamic scope identifier");

  if (auto scopeId = getScopeIdAttr())
    if (failed(verifyScopeId(getOperation(), scopeId)))
      return failure();

  const bool hasMetric = static_cast<bool>(getMetric());
  if (hasMetric != (getMetricType() != MetricValueType::NONE))
    return emitOpError(
        "metric operand and non-none metric type must be present together");
  if (hasMetric && (!getIsStart() || getDynamicScopeId()))
    return emitOpError(
        "metrics are only supported on synchronous scope starts");
  if (hasMetric &&
      !isMetricTypeCompatible(getMetric().getType(), getMetricType()))
    return emitOpError("metric operand type does not match metric type");

  return success();
}

LogicalResult CircularMarkOp::verify() {
  auto segmentType = getSegment().getType();
  auto selectedIds = segmentType.getSelectIds();
  auto bufferSizeInBytes = segmentType.getNBytes();
  auto mod = getOperation()->getParentOfType<ModuleOp>();
  int numWarps = getTotalNumWarps(mod);
  int segmentNum = selectedIds.empty() ? numWarps : selectedIds.size();
  if (!llvm::isPowerOf2_32(bufferSizeInBytes / segmentNum))
    return emitOpError("profiling buffer segment size must be power of 2");

  return verifyScopeId(getOperation(), getScopeIdAttr());
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
