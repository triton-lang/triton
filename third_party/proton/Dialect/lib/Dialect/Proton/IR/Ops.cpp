#include "Dialect/Proton/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define GET_OP_CLASSES
#include "Dialect/Proton/IR/Ops.cpp.inc"

#include "Dialect/Proton/IR/OpsEnums.cpp.inc"

namespace mlir::triton::proton {

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
} // namespace

LogicalResult RecordOp::verify() {
  const bool hasMetric = static_cast<bool>(getMetric());
  const bool hasMetricName = static_cast<bool>(getMetricNameAttr());
  const bool hasMetricType = getMetricType() != MetricValueType::NONE;
  if (hasMetric != hasMetricName || hasMetric != hasMetricType)
    return emitOpError("metric operand, metric name, and metric type must be "
                       "present together");
  if (hasMetric && !getIsStart())
    return emitOpError("metrics are only supported on scope start records");
  if (hasMetric &&
      !isMetricTypeCompatible(getMetric().getType(), getMetricType()))
    return emitOpError("metric operand type does not match metric type");
  return success();
}

} // namespace mlir::triton::proton
