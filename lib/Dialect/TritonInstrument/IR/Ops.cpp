#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"

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

LogicalResult ExperimentalStageAccessForCommitOp::verify() {
  auto buffersType = getBuffers().getType();
  auto outstandingCommitsType =
      cast<RankedTensorType>(getOutstandingCommitsType());
  if (buffersType.getShape()[0] != outstandingCommitsType.getShape()[0])
    return emitError()
           << "buffers and outstandingCommits must have the same size";
  if (outstandingCommitsType.getShape()[1] != NUM_THREADS)
    return emitError()
           << "outstandingCommits dim 1 must match number of threads";
  if (!verifySingleThreadEncoding(outstandingCommitsType))
    return emitError()
           << "outstandingCommits must have encoding that ensures that all "
              "its elements reside in a single thread";
  return success();
}

LogicalResult ExperimentalCheckOutstandingCommitsOp::verify() {
  auto buffersType = getBuffers().getType();
  auto outstandingCommitsType =
      cast<RankedTensorType>(getOutstandingCommitsType());
  if (buffersType.getShape()[0] != outstandingCommitsType.getShape()[0])
    return emitError()
           << "buffers and outstandingCommits must have the same size";
  if (outstandingCommitsType.getShape()[1] != NUM_THREADS)
    return emitError()
           << "outstandingCommits dim 1 must match number of threads";
  if (!verifySingleThreadEncoding(outstandingCommitsType))
    return emitError()
           << "outstandingCommits must have encoding that ensures that all "
              "its elements reside in a single thread";
  return success();
}

} // namespace mlir::triton::instrument
