#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonInstrument/IR/Ops.cpp.inc"

using namespace mlir;
namespace {
// readBars should have encoding that ensures that all its elements reside
// in a single thread
bool verifyReadBarsEncoding(RankedTensorType readBarsType) {
  auto encoding =
      cast<triton::gpu::BlockedEncodingAttr>(readBarsType.getEncoding());
  int rank = encoding.getRank();
  if (rank != readBarsType.getRank() || rank != 2)
    return false;
  for (int i = 0; i < rank; ++i) {
    if (encoding.getSizePerThread()[i] != readBarsType.getShape()[i])
      return false;
  }
  return true;
}
} // namespace

namespace mlir::triton::instrument {

LogicalResult ExperimentalCheckOutstandingWritesOp::verify() {
  auto writeBarsType = cast<RankedTensorType>(getWriteBarsType());
  auto buffersType = getBuffers().getType();
  if (writeBarsType.getShape() != buffersType.getShape() ||
      writeBarsType.getEncoding() != buffersType.getEncoding())
    return emitError()
           << "writeBars and buffers must have the same shape and encoding";
  return success();
}

LogicalResult ExperimentalCheckOutstandingReadsOp::verify() {
  auto readBarsType = cast<RankedTensorType>(getReadBarsType());
  auto buffersType = getBuffers().getType();
  // readBars is 2D tensor of shape [num_buffers, num_barriers]
  if (readBarsType.getShape()[0] != buffersType.getShape()[0])
    return emitError() << "readBars dim 0 must match number of buffers";

  if (!verifyReadBarsEncoding(readBarsType))
    return emitError() << "readBars must have encoding that ensures that all "
                          "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalMarkAsWriteOp::verify() {
  auto buffersType = getBuffers().getType();
  auto writeBarsType = cast<RankedTensorType>(getWriteBarsType());
  if (writeBarsType.getShape() != buffersType.getShape() ||
      writeBarsType.getEncoding() != buffersType.getEncoding())
    return emitError()
           << "writeBars and buffers must have the same shape and encoding";
  return success();
}

LogicalResult ExperimentalMarkAsReadOp::verify() {
  auto buffersType = getBuffers().getType();
  auto barriersType = getBarriers().getType();
  auto readBarsType = cast<RankedTensorType>(getReadBarsType());
  // readBars is 2D tensor of shape [num_buffers, num_barriers]
  if (readBarsType.getShape()[0] != buffersType.getShape()[0])
    return emitError() << "readBars dim 0 must match number of buffers";
  if (readBarsType.getShape()[1] != barriersType.getShape()[0])
    return emitError() << "readBars dim 1 must match number of barriers";
  if (!verifyReadBarsEncoding(readBarsType))
    return emitError() << "readBars must have encoding that ensures that all "
                          "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalClearReadBarrierOp::verify() {
  auto readBarsType = cast<RankedTensorType>(getReadBarsType());
  auto barriersType = getBarriers().getType();
  // readBars is 2D tensor of shape [num_buffers, num_barriers]
  if (readBarsType.getShape()[1] != barriersType.getShape()[0])
    return emitError() << "readBars dim 0 must match number of barriers";
  if (!verifyReadBarsEncoding(readBarsType))
    return emitError() << "readBars must have encoding that ensures that all "
                          "its elements reside in a single thread";
  return success();
}

} // namespace mlir::triton::instrument
