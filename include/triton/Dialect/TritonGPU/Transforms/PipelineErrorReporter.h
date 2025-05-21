#ifndef TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINE_ERROR_REPORTER_H_
#define TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINE_ERROR_REPORTER_H_
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::scf;

/// This class is used to report the scheduling error. It is used by
/// the pipeline expander.
class PipelineErrorReporter {
protected:
  ForOp forOp;

  unsigned numStages = 0;
  const DenseMap<Operation *, unsigned int> &stages;

  /// Collect the root defining ops in IfOps. There could be multiple root
  /// defining ops in IfOps, as there are then branches and else branches.
  DenseSet<Operation *> rootDefiningOps;

  /// Recursively find the root defining op of the value in IfOps by traversing
  /// back the index of an OpResult and yielded values.
  void findRootDefiningOp(Operation *op, unsigned int resultNumber);

  DenseSet<Operation *> rootUserOps;

  void findRootUserOp(Operation *op, const DenseSet<Operation *> &userOps);

  std::optional<unsigned int> findStage(Operation *op);

  /// Get the operand from the yield operation of the loop, which is the real
  /// value of the loop-carried dependency.
  std::optional<Value> getBlockArgYieldValueFromForLoop(BlockArgument arg);

  /// Find the loop-carried dependency that really causes the scheduling error,
  /// going into nested operations of IfOps.
  void findRootSchedulingErrorLoopCarryDep(Operation *consumer,
                                           Operation *producer, Value operand);

public:
  explicit PipelineErrorReporter(
      ForOp forOp, unsigned numStages,
      const DenseMap<Operation *, unsigned int> &stages)
      : forOp(forOp), numStages(numStages), stages(stages) {}
  PipelineErrorReporter(const PipelineErrorReporter &) = delete;
  PipelineErrorReporter &operator=(const PipelineErrorReporter &) = delete;
  PipelineErrorReporter(PipelineErrorReporter &&) = delete;
  PipelineErrorReporter &operator=(PipelineErrorReporter &&) = delete;

  /// Print the scheduling error message using MLIR's diagnostic engine.
  /// Depending on whether it is a loop-carried dependency, we print different
  /// messages. When distance is 0, it means the consumer and producer are in
  /// the same iteration. We are not supposed to have scheduling error in this
  /// case, as we have addressed the potential data dependency conflicts.
  ///
  /// When distance is 1, we find the root scheduling error, and print the
  /// diagnostic message.
  void printSchedulingError(int64_t distance, Operation *consumer,
                            Operation *producer, Value operand);
};
#endif // TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINE_ERROR_REPORTER_H_
