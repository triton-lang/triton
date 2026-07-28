#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_LAYOUT_ASSIGNMENT_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_LAYOUT_ASSIGNMENT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::triton::gpu {

/// Keep the incumbent's conflict choices for compatibility, or assign layouts
/// by the physical cost of the complete producer/consumer graph.
enum class LayoutAssignmentStrategy { Legacy, Global };

/// Assign distributed tensor layouts across functions and structured control
/// flow, rematerialize profitable backward slices, hoist expensive
/// conversions, and remove dead layout conversions and loop arguments.
///
/// This is a standalone transformation rather than a nested invocation of
/// either layout optimization pass. Keeping the whole-function assignment in
/// one engine lets both pass entry points preserve the same hard boundaries and
/// control-flow invariants.
LogicalResult optimizeDistributedLayouts(ModuleOp module,
                                         bool disableRematSplitting,
                                         LayoutAssignmentStrategy strategy);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_LAYOUT_ASSIGNMENT_H_
