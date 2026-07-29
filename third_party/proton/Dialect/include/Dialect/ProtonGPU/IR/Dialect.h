#ifndef DIALECT_PROTONGPU_IR_DIALECT_H_
#define DIALECT_PROTONGPU_IR_DIALECT_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h.inc"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <cstdint>

#define GET_OP_CLASSES
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Ops.h.inc"

#define GET_ATTRDEF_CLASSES
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/AttrDefs.h.inc"

namespace mlir {
namespace triton {
namespace proton {

// Event kinds are part of the compact trace encoding, but are not represented
// as an IR attribute. The ProtonGPU operations determine the event kind from
// their semantics and whether the scope identifier is static or dynamic.
enum class EventType : uint32_t {
  SCOPE = 0,
  ASYNC = 1,
  MARK = 2,
  METRIC = 3,
};

namespace gpu {

const int getBytesPerClockEntry();

const int getCircularHeaderSize();

const int getTotalNumWarps(ModuleOp mod);

constexpr uint32_t encodeEventTag(uint32_t scopeId, bool isStart,
                                  EventType eventType,
                                  MetricValueType metricType) {
  return ((scopeId & 0xffu) << 23) | (static_cast<uint32_t>(eventType) << 20) |
         (static_cast<uint32_t>(metricType) << 16) |
         (isStart ? 0u : (1u << 31));
}

} // namespace gpu
} // namespace proton
} // namespace triton
} // namespace mlir

#endif // DIALECT_PROTONGPU_IR_DIALECT_H_
