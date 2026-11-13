#ifndef TRITON_ANALYSIS_MEMBARPLACEMENT_H
#define TRITON_ANALYSIS_MEMBARPLACEMENT_H

#include "triton/Analysis/Membar.h"

namespace mlir::triton::membar {

enum class PlacementMode { Shared, TensorMemory, SharedAndTensorMemory };

bool usesTensorMemory(ModuleOp module);

/// Place region-local synchronization after physical allocation and compulsory
/// cluster rendezvous insertion. Run --convert-scf-to-cf before Membar so
/// compiler-generated thread guards are visible as CFG branches.
LogicalResult placeBarriers(ModuleAllocation &allocation,
                            BufferRegionAnalysis &regions,
                            MembarFilterFn sharedFilter, PlacementMode mode,
                            CompletionScopeFn completionScope = nullptr);

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBARPLACEMENT_H
