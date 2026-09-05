#ifndef TRITONAMD_ANALYSIS_CONTROL_FLOW_RANGE_ANALYSIS_H
#define TRITONAMD_ANALYSIS_CONTROL_FLOW_RANGE_ANALYSIS_H

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <memory>
#include <optional>

namespace mlir {
class DominanceInfo;

namespace triton::AMD::detail {

/// Range summaries for natural loops formed by lowering structured control
/// flow. A header argument includes the final, failing condition evaluation;
/// its uses in the body and after the loop can have narrower ranges.
class ControlFlowRangeAnalysis {
public:
  using GetRangeFn = llvm::function_ref<IntegerValueRange(Value, Block *)>;

  ControlFlowRangeAnalysis(Operation *top, DominanceInfo &domInfo);
  ~ControlFlowRangeAnalysis();

  /// A missing result means the recurrence is unsupported. An uninitialized
  /// result means an input range is pending. A null useBlock requests the
  /// range of the header argument across all condition evaluations.
  std::optional<IntegerValueRange>
  getRange(BlockArgument argument, Block *useBlock, GetRangeFn getRange) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace triton::AMD::detail
} // namespace mlir

#endif
