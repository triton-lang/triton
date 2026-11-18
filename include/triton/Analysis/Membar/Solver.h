#ifndef TRITON_ANALYSIS_MEMBAR_SOLVER_H
#define TRITON_ANALYSIS_MEMBAR_SOLVER_H

#include "triton/Analysis/Membar/Dependencies.h"
#include "llvm/ADT/BitVector.h"
#include <array>
#include <cstdint>
#include <optional>

namespace mlir::triton::membar {

// Site IDs follow emission order so waits precede barriers at the same gap.
inline constexpr std::array ACTIONS{SyncKind::WaitLoad, SyncKind::WaitStore,
                                    SyncKind::RegionBarrier,
                                    SyncKind::WarpBarrier};

struct CutSolution {
  llvm::BitVector sites;
  uint64_t cost = 0;
  // Lower bound on total cost, independent of rendezvous scope.
  uint64_t lowerBound = 0;
};

// Site gap * actions.size() + i emits actions[i] before the gap operation.
template <typename Facts>
std::optional<CutSolution> solveMembarCut(const MembarCFG<Facts> &cfg,
                                          ArrayRef<Demand> demands,
                                          ArrayRef<SyncKind> actions = ACTIONS);

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_SOLVER_H
