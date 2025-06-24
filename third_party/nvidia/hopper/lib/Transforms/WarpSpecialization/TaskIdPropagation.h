#ifndef NVHOPPER_ANALYSIS_TASKIDPROPAGATION_H
#define NVHOPPER_ANALYSIS_TASKIDPROPAGATION_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <optional>

using namespace mlir::dataflow;

namespace mlir::triton::gpu {

//===----------------------------------------------------------------------===//
// TaskId
//===----------------------------------------------------------------------===//

/// This lattice value represents known information on the async_task_id of a
/// lattice.
class TaskId {
public:
  /// Construct a taskId value as uninitialized.
  explicit TaskId() = default;

  /// Construct a taskId value with a known constant.
  TaskId(DenseI32ArrayAttr taskIds) : taskIds(std::move(taskIds)) {}

  /// Get the constant value. Returns null if no value was determined.
  DenseI32ArrayAttr getTaskIds() const {
    assert(!isUninitialized());
    return *taskIds;
  }

  /// Compare the taskId values.
  bool operator==(const TaskId &rhs) const { return taskIds == rhs.taskIds; }

  /// Print the taskId value.
  void print(raw_ostream &os) const;

  /// The state where the taskIds value is uninitialized. This happens when the
  /// state hasn't been set during the analysis.
  static TaskId getUninitialized() { return TaskId{}; }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !taskIds.has_value(); }

  /// Whether the state is unknown.
  bool isUnknown() const { return taskIds == nullptr; }

  /// The state where the taskId value is unknown.
  static TaskId getUnknownTaskId() { return TaskId{/*taskIds=*/nullptr}; }

  static TaskId meet(const TaskId &lhs, const TaskId &rhs);

  static TaskId join(const TaskId &lhs, const TaskId &rhs);

private:
  std::optional<DenseI32ArrayAttr> taskIds;
};

//===----------------------------------------------------------------------===//
// TaskIdLattice
//===----------------------------------------------------------------------===//

class TaskIdLattice : public Lattice<TaskId> {
public:
  using Lattice::Lattice;
};

//===----------------------------------------------------------------------===//
// TaskIdBackwardPropagation
//===----------------------------------------------------------------------===//

/// This analysis implements sparse backward propagation, which attempts to
/// determine the async_task_id of an SSA value.

class TaskIdBackwardPropagation
    : public SparseBackwardDataFlowAnalysis<TaskIdLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op, ArrayRef<TaskIdLattice *> operands,
                 ArrayRef<const TaskIdLattice *> results) override;

  void visitBranchOperand(OpOperand &operand) override;

  void visitCallOperand(OpOperand &operand) override;

  void setToExitState(TaskIdLattice *lattice) override;

  void propagateToYield(scf::YieldOp yieldOp, SmallVector<TaskId> &lattices);

  void propagateToTerminator(Operation *op,
                             ArrayRef<const TaskIdLattice *> &lattices);

  void propagateToParent(Operation *op, const TaskId &taskId);
};

} // namespace mlir::triton::gpu

#endif // NVHOPPER_ANALYSIS_TASKIDPROPAGATION_H
