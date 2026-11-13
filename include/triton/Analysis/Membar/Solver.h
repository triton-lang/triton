#ifndef TRITON_ANALYSIS_MEMBAR_SOLVER_H
#define TRITON_ANALYSIS_MEMBAR_SOLVER_H

#include "triton/Analysis/Membar/TmemModel.h"
#include "llvm/ADT/BitVector.h"
#include <array>
#include <cstdint>
#include <optional>

namespace mlir::triton::membar {

// Site IDs follow emission order so waits precede barriers at the same gap.
inline constexpr std::array ACTIONS{Action::WaitLoad, Action::WaitStore,
                                    Action::RegionBarrier, Action::WarpBarrier};

struct Edge {
  unsigned to;
  bool invalidatesIteration = false;
};

// Arrival observes incoming effects; departure starts outgoing dependencies.
// A fixed internal rendezvous orders paths passing through the operation.
struct Node {
  Operation *op = nullptr;
  // Zero at entry and wherever the execution group cannot rendezvous.
  unsigned insertionCost = 0;
  OpModel effects;
  SmallVector<Edge, 2> next;
};

struct MembarCFG {
  static constexpr unsigned entry = 0;
  SmallVector<Node, 0> nodes;
};

// Loop-carried shared hazards and ordered TMEM handoffs each need one extra
// state. The latter always ends with a region barrier.
enum class Path { Any, LoopCarried, WaitThenOrder };

struct Demand {
  unsigned source, sink;
  Need first;
  Path path = Path::Any;
};

struct CutSolution {
  llvm::BitVector sites;
  uint64_t cost = 0;
  // A matching placement cost certifies optimality.
  uint64_t lowerBound = 0;
};

std::optional<CutSolution> solveMembarCut(const MembarCFG &cfg,
                                          ArrayRef<Demand> demands);

} // namespace mlir::triton::membar

#endif // TRITON_ANALYSIS_MEMBAR_SOLVER_H
