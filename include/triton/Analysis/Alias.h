#ifndef TRITON_ANALYSIS_ALIAS_H
#define TRITON_ANALYSIS_ALIAS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlowAnalysis.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {

class AliasInfo {
public:
  AliasInfo() = default;
  AliasInfo(Value value) { insert(value); }

  void insert(Value value) { allocs.insert(value); }

  const DenseSet<Value> &getAllocs() const { return allocs; }

  bool operator==(const AliasInfo &other) const {
    return allocs == other.allocs;
  }

  /// The pessimistic value state of a value without alias
  static AliasInfo getPessimisticValueState(MLIRContext *context) {
    return AliasInfo();
  }
  static AliasInfo getPessimisticValueState(Value value) { return AliasInfo(); }

  /// The union of both arguments
  static AliasInfo join(const AliasInfo &lhs, const AliasInfo &rhs);

private:
  /// The set of allocated values that are aliased by this lattice.
  /// For now, we only consider aliased value produced by the following
  /// situations:
  /// 1. values returned by scf.yield
  /// 2. block arguments in scf.for
  /// Example:
  ///    alloc v1                  alloc v2
  ///       |                         |
  ///    |--------------|   |------------|
  ///  scf.for v3     scf.for v4       scf.for v5
  ///    |
  /// scf.yield v6
  ///
  /// v1's alloc [v1]
  /// v2's alloc [v2]
  /// v3's alloc [v1]
  /// v4's alloc [v1, v2]
  /// v5's alloc [v2]
  /// v6's alloc [v1]
  ///
  /// Therefore, v1's liveness range is the union of v3, v4, and v6
  /// v2's liveness range is the union of v4 and v5.
  DenseSet<Value> allocs;
};

//===----------------------------------------------------------------------===//
// Shared Memory Alias Analysis
//===----------------------------------------------------------------------===//
class SharedMemoryAliasAnalysis : public ForwardDataFlowAnalysis<AliasInfo> {
public:
  using ForwardDataFlowAnalysis<AliasInfo>::ForwardDataFlowAnalysis;

  /// XXX(Keren): Compatible interface with MLIR AliasAnalysis for future use.
  /// Given two values, returns their aliasing behavior.
  AliasResult alias(Value lhs, Value rhs);

  /// Returns the modify-reference behavior of `op` on `location`.
  ModRefResult getModRef(Operation *op, Value location);

  /// Computes if the alloc set of the results are changed.
  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<AliasInfo> *> operands) override;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_ALIAS_H
