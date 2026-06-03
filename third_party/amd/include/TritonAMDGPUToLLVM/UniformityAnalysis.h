#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_UNIFORMITYANALYSIS_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_UNIFORMITYANALYSIS_H_

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::AMD {

// Wave-uniformity lattice: Bottom (unknown) < Uniform < Divergent.
// Uniform = all lanes see the same value; Divergent = lanes may differ.
//
// Only query values that existed when `initializeAndRun` ran. Values
// created during pattern conversion are not in the solver state.
class UniformityValue {
public:
  enum class State { Bottom, Uniform, Divergent };

  UniformityValue() : state(State::Bottom) {}
  explicit UniformityValue(State s) : state(s) {}

  static UniformityValue uniform() { return UniformityValue(State::Uniform); }
  static UniformityValue divergent() {
    return UniformityValue(State::Divergent);
  }
  static UniformityValue bottom() { return UniformityValue(State::Bottom); }

  bool isUniform() const { return state == State::Uniform; }
  bool isDivergent() const { return state == State::Divergent; }
  bool isBottom() const { return state == State::Bottom; }

  static UniformityValue join(const UniformityValue &a,
                              const UniformityValue &b) {
    if (a.state == b.state)
      return a;
    if (a.state == State::Bottom)
      return b;
    if (b.state == State::Bottom)
      return a;
    return divergent();
  }

  bool operator==(const UniformityValue &o) const { return state == o.state; }
  bool operator!=(const UniformityValue &o) const { return state != o.state; }

  void print(llvm::raw_ostream &os) const {
    switch (state) {
    case State::Bottom:
      os << "Bottom";
      break;
    case State::Uniform:
      os << "Uniform";
      break;
    case State::Divergent:
      os << "Divergent";
      break;
    }
  }

private:
  State state;
};

using UniformityLattice = mlir::dataflow::Lattice<UniformityValue>;

// Peel extractvalue/insertvalue chains to find the scalar at a given
// struct position. Also used by OffsetUniformitySplit.
Value lookThroughExtractValue(Value v);

// Register the uniformity analysis with `solver`. Load DeadCodeAnalysis
// first, then call `solver.initializeAndRun(module)`.
void loadUniformityAnalysis(DataFlowSolver &solver);

// Returns true if `v` is wave-uniform. Conservative: returns false
// for Bottom, Divergent, or unknown values.
bool isUniformValue(Value v, const DataFlowSolver &solver);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_UNIFORMITYANALYSIS_H_
