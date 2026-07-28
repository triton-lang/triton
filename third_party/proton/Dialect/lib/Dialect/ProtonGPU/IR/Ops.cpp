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

LogicalResult verifySegment(Operation *op, SegmentType segmentType) {
  auto selectedIds = segmentType.getSelectIds();
  auto bufferSizeInBytes = segmentType.getNBytes();
  auto mod = op->getParentOfType<ModuleOp>();
  int numWarps = getTotalNumWarps(mod);
  int segmentNum = selectedIds.empty() ? numWarps : selectedIds.size();
  if (!llvm::isPowerOf2_32(bufferSizeInBytes / segmentNum))
    return op->emitOpError(
        "profiling buffer segment size must be power of 2");
  return success();
}
} // namespace

// -- CircularRecordOp --
LogicalResult CircularStoreOp::verify() {
  auto scopeId = getScopeId();
  auto segmentType = getSegment().getType();
  if (failed(verifySegment(getOperation(), segmentType)))
    return failure();

  if (scopeId < 0 || scopeId > 255)
    return emitOpError("scope id must be in [0, 255]");

  const bool hasMetric = static_cast<bool>(getMetric());
  if (hasMetric != (getMetricType() != MetricValueType::NONE))
    return emitOpError(
        "metric operand and non-none metric type must be present together");
  if (hasMetric &&
      (!getIsStart() || getEventType() != EventType::SCOPE))
    return emitOpError("metrics are only supported on synchronous scope starts");
  if (hasMetric &&
      !isMetricTypeCompatible(getMetric().getType(), getMetricType()))
    return emitOpError("metric operand type does not match metric type");
  if (getEventType() == EventType::METRIC)
    return emitOpError("metric extension records cannot be emitted directly");
  if (getEventType() == EventType::MARK && !getIsStart())
    return emitOpError("markers must use the start event encoding");

  return success();
}

LogicalResult CircularStoreDynamicOp::verify() {
  if (failed(verifySegment(getOperation(), getSegment().getType())))
    return failure();
  if (getEventType() != EventType::ASYNC)
    return emitOpError("runtime scope identifiers are only valid for async events");
  if (getIsStart())
    return emitOpError("runtime scope identifiers are only valid for async ends");
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
