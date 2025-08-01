#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonInstrument/IR/Ops.cpp.inc"

#include "triton/Dialect/TritonInstrument/IR/OpsEnums.cpp.inc"

using namespace mlir;
namespace {
// readBars/writeBars should have encoding that ensures that all its elements
// reside in a single thread
bool verifyBarsEncoding(RankedTensorType readBarsType) {
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

LogicalResult ExperimentalCheckWriteStateOp::verify() {
  auto writeStateType = cast<RankedTensorType>(getWriteStateType());
  auto buffersType = getBuffers().getType();
  if (writeStateType.getShape() != buffersType.getShape() ||
      writeStateType.getEncoding() != buffersType.getEncoding())
    return emitError()
           << "writeState and buffers must have the same shape and encoding";
  auto writeBarsType = cast<RankedTensorType>(getWriteBarsType());
  // writeBars is 2D tensor of shape [num_buffers, num_barriers]
  if (writeBarsType.getShape()[0] != buffersType.getShape()[0])
    return emitError() << "writeBars dim 0 must match number of buffers";

  if (!verifyBarsEncoding(writeBarsType))
    return emitError() << "writeBars must have encoding that ensures that all "
                          "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalCheckReadBarriersOp::verify() {
  auto readBarsType = cast<RankedTensorType>(getReadBarsType());
  auto buffersType = getBuffers().getType();
  // readBars is 2D tensor of shape [num_buffers, num_barriers]
  if (readBarsType.getShape()[0] != buffersType.getShape()[0])
    return emitError() << "readBars dim 0 must match number of buffers";

  if (!verifyBarsEncoding(readBarsType))
    return emitError() << "readBars must have encoding that ensures that all "
                          "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalSetWriteStateOp::verify() {
  auto buffersType = getBuffers().getType();
  auto writeStateType = cast<RankedTensorType>(getWriteStateType());
  if (writeStateType.getShape() != buffersType.getShape() ||
      writeStateType.getEncoding() != buffersType.getEncoding())
    return emitError()
           << "writeState and buffers must have the same shape and encoding";
  return success();
}

LogicalResult ExperimentalCommitWriteWithBarrierOp::verify() {
  auto writeBarsType = cast<RankedTensorType>(getWriteBarsType());
  auto writeStateType = cast<RankedTensorType>(getWriteStateType());
  auto barriersType = getBarriers().getType();
  if (writeBarsType.getShape()[0] != writeStateType.getShape()[0])
    return emitError()
           << "writeBars and writeState must have the same number of buffers";
  if (writeBarsType.getShape()[1] != barriersType.getShape()[0])
    return emitError() << "writeBars dim 1 must match number of barriers";
  if (!verifyBarsEncoding(writeBarsType))
    return emitError() << "writeBars must have encoding that ensures that all "
                          "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalSetReadBarrierOp::verify() {
  auto buffersType = getBuffers().getType();
  auto barriersType = getBarriers().getType();
  auto readBarsType = cast<RankedTensorType>(getReadBarsType());
  // readBars is 2D tensor of shape [num_buffers, num_barriers]
  if (readBarsType.getShape()[0] != buffersType.getShape()[0])
    return emitError() << "readBars dim 0 must match number of buffers";
  if (readBarsType.getShape()[1] != barriersType.getShape()[0])
    return emitError() << "readBars dim 1 must match number of barriers";
  if (!verifyBarsEncoding(readBarsType))
    return emitError() << "readBars must have encoding that ensures that all "
                          "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalClearWriteBarrierOp::verify() {
  auto writeBarsType = cast<RankedTensorType>(getWriteBarsType());
  auto barriersType = getBarriers().getType();
  auto writeStateType = cast<RankedTensorType>(getWriteStateType());
  if (writeBarsType.getShape()[0] != writeStateType.getShape()[0])
    return emitError()
           << "writeBars and writeState must have the same number of buffers";
  if (writeBarsType.getShape()[1] != barriersType.getShape()[0])
    return emitError() << "writeBars dim 1 must match number of barriers";
  if (!verifyBarsEncoding(writeBarsType))
    return emitError() << "writeBars must have encoding that ensures that all "
                          "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalClearReadBarrierOp::verify() {
  auto readBarsType = cast<RankedTensorType>(getReadBarsType());
  auto barriersType = getBarriers().getType();
  // readBars is 2D tensor of shape [num_buffers, num_barriers]
  if (readBarsType.getShape()[1] != barriersType.getShape()[0])
    return emitError() << "readBars dim 0 must match number of barriers";
  if (!verifyBarsEncoding(readBarsType))
    return emitError() << "readBars must have encoding that ensures that all "
                          "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalCheckBarrierWritesClearedOp::verify() {
  auto writeBarsType = cast<RankedTensorType>(getWriteBarsType());
  auto barriersType = getBarriers().getType();
  if (writeBarsType.getShape()[1] != barriersType.getShape()[0])
    return emitError() << "writeBars dim 1 must match number of barriers";
  return success();
}

LogicalResult ExperimentalStageAccessForCommitOp::verify() {
  auto buffersType = getBuffers().getType();
  auto outstandingCommitsType =
      cast<RankedTensorType>(getOutstandingCommitsType());
  if (buffersType.getShape()[0] != outstandingCommitsType.getShape()[0])
    return emitError()
           << "buffers and outstandingCommits must have the same size";
  return success();
}

LogicalResult ExperimentalCheckOutstandingCommitsOp::verify() {
  auto buffersType = getBuffers().getType();
  auto outstandingCommitsType =
      cast<RankedTensorType>(getOutstandingCommitsType());
  if (buffersType.getShape()[0] != outstandingCommitsType.getShape()[0])
    return emitError()
           << "buffers and outstandingCommits must have the same size";
  return success();
}

} // namespace mlir::triton::instrument
