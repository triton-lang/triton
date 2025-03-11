#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

//===----------------------------------------------------------------------===//
// PartitionLoop
//===----------------------------------------------------------------------===//

LogicalResult triton::gpu::partitionLoop(scf::ForOp loop) {
  FailureOr<WarpSchedule> scheduleOr = WarpSchedule::deserialize(loop);
  if (failed(scheduleOr))
    return failure();
  WarpSchedule schedule = std::move(*scheduleOr);
  if (failed(schedule.verify(loop)))
    return failure();
  return failure();
}
