#ifndef TRITON_ANALYSIS_MEMBAR_PASS_H
#define TRITON_ANALYSIS_MEMBAR_PASS_H

#include "triton/Analysis/Membar/SmemModel.h"

namespace mlir::triton::membar {

enum class PlacementMode { Shared, TensorMemory, SharedAndTensorMemory };

/// Place shared-memory proxy fences on the allocated, SCF-lowered CFG.
LogicalResult placeProxyFences(ModuleOp module, BufferRegionAnalysis &regions);

/// Order peer mbarrier initialization, shared-memory reuse and kernel exit.
LogicalResult placeClusterBarriers(ModuleAllocation &allocation,
                                   BufferRegionAnalysis &regions);

/// Place region-local synchronization after physical allocation and compulsory
/// cluster rendezvous insertion. Run --convert-scf-to-cf before Membar so
/// compiler-generated thread guards are visible as CFG branches.
///
/// Build a CFG for each kernel body and warp-specialized role, following
/// its noinline calls.
/// Record each operation's memory effects, existing synchronization, and legal
/// insertion sites. Propagate producer effects through the CFG to find demands
/// for ordering or completion. A cross-warp handoff from a native TMEM load or
/// store needs its completion wait followed by a rendezvous.
/// At eligible mbarrier notifications, each warp can publish its own effects
/// when the selected synchronization is insufficient for a scalar issuer.
///
/// The solver selects sites on uncovered paths until it covers every demand.
/// A site can cover several demands. Each enclosing loop multiplies its cost
/// by four. Within an MLIR region, uniform branches divide the weight equally
/// among exclusive arms, and joins sum their shares. Each warp-specialized
/// role and the continuation inherit the launch weight.
/// Emit actions before their operations, waits first.
LogicalResult placeBarriers(ModuleAllocation &allocation,
                            BufferRegionAnalysis &regions,
                            MembarFilterFn sharedFilter, PlacementMode mode,
                            CompletionScopeFn completionScope = nullptr,
                            DenseSet<Operation *> distributable = {});

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_PASS_H
