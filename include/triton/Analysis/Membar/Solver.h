#ifndef TRITON_ANALYSIS_MEMBAR_SOLVER_H
#define TRITON_ANALYSIS_MEMBAR_SOLVER_H

#include "triton/Analysis/Membar/Dependencies.h"
#include "llvm/ADT/BitVector.h"
#include <array>
#include <cstdint>
#include <optional>

namespace mlir::triton::membar {

// Site IDs follow emission order so waits precede barriers at the same gap.
inline constexpr std::array ACTIONS{Action::WaitLoad, Action::WaitStore,
                                    Action::RegionBarrier, Action::WarpBarrier};

struct CutSolution {
  llvm::BitVector sites;
  uint64_t cost = 0;
  // A matching placement cost certifies optimality.
  uint64_t lowerBound = 0;
};

// Site gap * actions.size() + i emits actions[i] before the gap operation.
template <typename Facts>
std::optional<CutSolution> solveMembarCut(const MembarCFG<Facts> &cfg,
                                          ArrayRef<Demand> demands,
                                          ArrayRef<Action> actions = ACTIONS);

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_SOLVER_H
