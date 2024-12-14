#ifndef TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINE_ERROR_REPORTER_H_
#define TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINE_ERROR_REPORTER_H_
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::scf;

class PipelineErrorReporter {
protected:
  ForOp forOp;
  DenseSet<Operation *> rootDefiningOps;
  void findRootDefiningOp(Operation *op, unsigned int resultNumber);
  std::optional<Value> getBlockArgYieldValueFromForLoop(BlockArgument arg);
  void findRootSchedulingErrorLoopCarryDep(Operation *consumer,
                                                            Operation *producer,
                                                            Value operand);

public:
  explicit PipelineErrorReporter(ForOp forOp) : forOp(forOp) {}
  PipelineErrorReporter(const PipelineErrorReporter &) = delete;
  PipelineErrorReporter &operator=(const PipelineErrorReporter &) = delete;
  PipelineErrorReporter(PipelineErrorReporter &&) = delete;
  PipelineErrorReporter &operator=(PipelineErrorReporter &&) = delete;

  void printSchedulingError(int64_t distance, Operation *consumer,
                            Operation *producer, Value operand);
};
#endif // TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINE_ERROR_REPORTER_H_
