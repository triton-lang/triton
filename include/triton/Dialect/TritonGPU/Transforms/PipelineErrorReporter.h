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
  /// Collect the root defining ops in IfOps. There could be multiple root
  /// defining ops in IfOps, as there are then branches and else branches.
  DenseSet<Operation *> rootDefiningOps;

  /// Recursively find the root defining op of the value in IfOps.
  void findRootDefiningOp(Operation *op, unsigned int resultNumber);

  /// Return the loop argument value if the Value is an argument of the loop.
  std::optional<Value> getBlockArgYieldValueFromForLoop(BlockArgument arg);

  /// Find the loop-carried dependency that really causes the scheduling error,
  /// going into nested operations of IfOps.
  void findRootSchedulingErrorLoopCarryDep(Operation *consumer,
                                                            Operation *producer,
                                                            Value operand);

public:
  explicit PipelineErrorReporter(ForOp forOp) : forOp(forOp) {}
  PipelineErrorReporter(const PipelineErrorReporter &) = delete;
  PipelineErrorReporter &operator=(const PipelineErrorReporter &) = delete;
  PipelineErrorReporter(PipelineErrorReporter &&) = delete;
  PipelineErrorReporter &operator=(PipelineErrorReporter &&) = delete;

  /// Print the scheduling error message using MLIR's diagnostic engine.
  /// Depending on whether it is a loop-carried dependency, we print different
  /// messages.
  void printSchedulingError(int64_t distance, Operation *consumer,
                            Operation *producer, Value operand);

};
#endif // TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINE_ERROR_REPORTER_H_
