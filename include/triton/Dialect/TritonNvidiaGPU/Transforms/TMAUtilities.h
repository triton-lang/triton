#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TMAUtils.h"

namespace mlir::triton::nvidia_gpu {

constexpr inline int TMA_SIZE_BYTES = 128;
constexpr inline int TMA_ALIGN = 128;

triton::gpu::SharedEncodingTrait
getEncodingFromDescriptor(Operation *op, RankedTensorType tensorType,
                          Value desc);

inline int getTMATransferBitWidth(Type elementType) {
  return isFP6(elementType) ? 6 : getIntOrFloatOrPtrBitWidth(elementType);
}

bool hasCGABroadcast(gpu::MemDescType memDescType);

Value sextI16ToI32Indices(Value indices, OpBuilder &builder, Location loc);

FailureOr<int> getTMASwizzleMode(Location loc, triton::TensorDescInterface ty);
FailureOr<int> getTMAElementType(Location loc, triton::TensorDescInterface ty);

bool isNarrowFP6Map(TensorDescType type);
Value updateTMADesc(OpBuilder &builder, Location loc, Value src,
                    Value base = {});

LogicalResult createTMADesc(Value tmaPtr, MakeTensorDescOp op,
                            OpBuilder &builder);

} // namespace mlir::triton::nvidia_gpu
