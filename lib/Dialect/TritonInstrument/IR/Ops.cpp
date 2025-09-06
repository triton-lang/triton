#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonInstrument/IR/Ops.cpp.inc"

#include "triton/Dialect/TritonInstrument/IR/OpsEnums.cpp.inc"

using namespace mlir;
namespace {
// All ConSan tensors should have encoding that ensures that all its elements
// reside in a single thread
bool verifySingleThreadEncoding(RankedTensorType tensorType) {
  auto encoding =
      cast<triton::gpu::BlockedEncodingAttr>(tensorType.getEncoding());
  int rank = encoding.getRank();
  if (rank != tensorType.getRank())
    return false;
  for (int i = 0; i < rank; ++i) {
    if (encoding.getSizePerThread()[i] != tensorType.getShape()[i])
      return false;
  }
  return true;
}
} // namespace

namespace mlir::triton::instrument {

LogicalResult ExperimentalSetWriteVisibilityOp::verify() {
  auto buffersType = getBuffers().getType();
  auto writeVisibilityType = cast<RankedTensorType>(getWriteVisibilityType());
  if (writeVisibilityType.getShape() != buffersType.getShape() ||
      writeVisibilityType.getEncoding() != buffersType.getEncoding())
    return emitError() << "writeVisibility and buffers must have the same "
                          "shape and encoding";
  return success();
}

LogicalResult ExperimentalClearWriteTrackingOp::verify() {
  auto buffersType = getBuffers().getType();
  auto writeTrackingType = cast<RankedTensorType>(getWriteTrackingType());
  if (writeTrackingType.getShape()[0] != buffersType.getShape()[0])
    return emitError() << "writeTracking dim 0 must match number of buffers";
  if (!verifySingleThreadEncoding(writeTrackingType))
    return emitError()
           << "writeTracking must have encoding that ensures that all "
              "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalClearReadVisibilityOp::verify() {
  auto buffersType = getBuffers().getType();
  auto readVisibilityType = cast<RankedTensorType>(getReadVisibilityType());
  if (readVisibilityType.getShape()[0] != buffersType.getShape()[0])
    return emitError() << "readVisibility dim 0 must match number of buffers";
  if (!verifySingleThreadEncoding(readVisibilityType))
    return emitError()
           << "readVisibility must have encoding that ensures that all "
              "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalClearReadTrackingOp::verify() {
  auto buffersType = getBuffers().getType();
  auto readTrackingType = cast<RankedTensorType>(getReadTrackingType());
  if (readTrackingType.getShape()[0] != buffersType.getShape()[0])
    return emitError() << "readTracking dim 0 must match number of buffers";
  if (!verifySingleThreadEncoding(readTrackingType))
    return emitError()
           << "readTracking must have encoding that ensures that all "
              "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalSetReadVisibilityOp::verify() {
  auto buffersType = getBuffers().getType();
  auto readVisibilityType = cast<RankedTensorType>(getReadVisibilityType());
  if (readVisibilityType.getShape()[0] != buffersType.getShape()[0])
    return emitError() << "readVisibility dim 0 must match number of buffers";
  if (!verifySingleThreadEncoding(readVisibilityType))
    return emitError()
           << "readVisibility must have encoding that ensures that all "
              "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalTrackVisibleWritesOp::verify() {
  auto barriersType = getBarriers().getType();
  auto writeVisibilityType = cast<RankedTensorType>(getWriteVisibilityType());
  auto writeTrackingType = cast<RankedTensorType>(getWriteTrackingType());
  if (writeVisibilityType.getShape()[0] != writeTrackingType.getShape()[0])
    return emitError()
           << "writeVisibility dim 0 must match number of writeTracking";
  if (writeTrackingType.getShape()[1] != barriersType.getShape()[0])
    return emitError() << "writeTracking dim 1 must match number of barriers";
  if (!verifySingleThreadEncoding(writeVisibilityType))
    return emitError()
           << "writeVisibility must have encoding that ensures that all "
              "its elements reside in a single thread";
  if (!verifySingleThreadEncoding(writeTrackingType))
    return emitError()
           << "writeTracking must have encoding that ensures that all "
              "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalTrackVisibleReadsOp::verify() {
  auto barriersType = getBarriers().getType();
  auto readVisibilityType = cast<RankedTensorType>(getReadVisibilityType());
  auto readTrackingType = cast<RankedTensorType>(getReadTrackingType());
  if (readVisibilityType.getShape()[0] != readTrackingType.getShape()[0])
    return emitError()
           << "readVisibility dim 0 must match number of readTracking";
  if (readTrackingType.getShape()[1] != barriersType.getShape()[0])
    return emitError() << "readTracking dim 1 must match number of barriers";
  if (!verifySingleThreadEncoding(readVisibilityType))
    return emitError()
           << "readVisibility must have encoding that ensures that all "
              "its elements reside in a single thread";
  if (!verifySingleThreadEncoding(readTrackingType))
    return emitError()
           << "readTracking must have encoding that ensures that all "
              "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalTransferVisibleWritesOp::verify() {
  auto barriersType = getBarriers().getType();
  auto writeVisibilityType = cast<RankedTensorType>(getWriteVisibilityType());
  auto writeTrackingType = cast<RankedTensorType>(getWriteTrackingType());
  if (writeVisibilityType.getShape()[0] != writeTrackingType.getShape()[0])
    return emitError()
           << "writeVisibility dim 0 must match number of writeTracking";
  if (writeTrackingType.getShape()[1] != barriersType.getShape()[0])
    return emitError() << "writeTracking dim 1 must match number of barriers";
  if (!verifySingleThreadEncoding(writeVisibilityType))
    return emitError()
           << "writeVisibility must have encoding that ensures that all "
              "its elements reside in a single thread";
  if (!verifySingleThreadEncoding(writeTrackingType))
    return emitError()
           << "writeTracking must have encoding that ensures that all "
              "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalTransferVisibleReadsOp::verify() {
  auto barriersType = getBarriers().getType();
  auto readVisibilityType = cast<RankedTensorType>(getReadVisibilityType());
  auto readTrackingType = cast<RankedTensorType>(getReadTrackingType());
  if (readVisibilityType.getShape()[0] != readTrackingType.getShape()[0])
    return emitError()
           << "readVisibility dim 0 must match number of readTracking";
  if (readTrackingType.getShape()[1] != barriersType.getShape()[0])
    return emitError() << "readTracking dim 1 must match number of barriers";
  if (!verifySingleThreadEncoding(readVisibilityType))
    return emitError()
           << "readVisibility must have encoding that ensures that all "
              "its elements reside in a single thread";
  if (!verifySingleThreadEncoding(readTrackingType))
    return emitError()
           << "readTracking must have encoding that ensures that all "
              "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalVerifyWriteVisibilityOp::verify() {
  auto buffersType = getBuffers().getType();
  auto writeVisibilityType = cast<RankedTensorType>(getWriteVisibilityType());
  if (writeVisibilityType.getShape()[0] != buffersType.getShape()[0])
    return emitError() << "writeVisibility dim 0 must match number of buffers";
  if (!verifySingleThreadEncoding(writeVisibilityType))
    return emitError()
           << "writeVisibility must have encoding that ensures that all "
              "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalVerifyReadVisibilityOp::verify() {
  auto buffersType = getBuffers().getType();
  auto readVisibilityType = cast<RankedTensorType>(getReadVisibilityType());
  if (readVisibilityType.getShape()[0] != buffersType.getShape()[0])
    return emitError() << "readVisibility dim 0 must match number of buffers";
  if (!verifySingleThreadEncoding(readVisibilityType))
    return emitError()
           << "readVisibility must have encoding that ensures that all "
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
  if (!verifySingleThreadEncoding(writeBarsType))
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
  if (!verifySingleThreadEncoding(readBarsType))
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
  if (!verifySingleThreadEncoding(writeBarsType))
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
  if (!verifySingleThreadEncoding(readBarsType))
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
