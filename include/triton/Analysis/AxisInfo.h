#ifndef TRITON_ANALYSIS_AXISINFO_H
#define TRITON_ANALYSIS_AXISINFO_H

#include "mlir/Analysis/DataFlowAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// AxisInfo
//===----------------------------------------------------------------------===//

/// This lattice value represents known information on the axes of a lattice.
/// Axis information is represented by a std::map<int, int>
class AxisInfo {
public:
  typedef SmallVector<int, 4> DimVectorT;

  static const int Unknown = 0;
  static const int Default = 1;

public:
  // Default constructor
  AxisInfo() : AxisInfo({}, {}, {}) {}
  // Construct contiguity info with known contiguity
  AxisInfo(DimVectorT knownContiguity, DimVectorT knownDivisibility,
           DimVectorT knownConstancy)
      : contiguity(knownContiguity), divisibility(knownDivisibility),
        constancy(knownConstancy), rank(contiguity.size()) {
    assert(knownDivisibility.size() == (size_t)rank);
    assert(knownConstancy.size() == (size_t)rank);
  }

  // Accessors
  int getContiguity(size_t d) const { return contiguity[d]; }
  const DimVectorT &getContiguity() const { return contiguity; }

  int getDivisibility(size_t d) const { return divisibility[d]; }
  const DimVectorT &getDivisibility() const { return divisibility; }

  int getConstancy(size_t d) const { return constancy[d]; }
  const DimVectorT &getConstancy() const { return constancy; }

  int getRank() const { return rank; }

  // Comparison
  bool operator==(const AxisInfo &other) const {
    return (contiguity == other.contiguity) &&
           (divisibility == other.divisibility) &&
           (constancy == other.constancy);
  }

  /// The pessimistic value state of the contiguity is unknown.
  static AxisInfo getPessimisticValueState(MLIRContext *context) {
    return AxisInfo();
  }
  static AxisInfo getPessimisticValueState(Value value);

  // The gcd of both arguments for each dimension
  static AxisInfo join(const AxisInfo &lhs, const AxisInfo &rhs);

private:
  /// The _contiguity_ information maps the `d`-th
  /// dimension to the length of the shortest
  /// sequence of contiguous integers along it.
  /// Suppose we have an array of N elements,
  /// with a contiguity value C,
  /// the array can be divided into a list of
  /// N/C sequences of C contiguous elements.
  /// For example:
  /// [10, 11, 12, 13, 18, 19, 20, 21]
  /// [20, 21, 22, 23, 28, 29, 30, 31]
  /// Would have contiguity [1, 4].
  /// and
  /// [12, 16, 20, 24]
  /// [13, 17, 21, 25]
  /// [14, 18, 22, 26]
  /// [15, 19, 23, 27]
  /// [18, 22, 26, 30]
  /// [19, 23, 27, 31]
  /// Would have contiguity [2, 1].
  DimVectorT contiguity;

  /// The _divisibility_ information maps the `d`-th
  /// dimension to the largest power-of-two that
  /// divides the first element of all the values along it
  /// For example:
  /// [10, 11, 12, 13, 18, 19, 20, 21]
  /// [20, 21, 22, 23, 28, 29, 30, 31]
  //  would have divisibility [1, 2]
  //  and
  /// [12, 16, 20, 24]
  /// [13, 17, 21, 25]
  /// [14, 18, 22, 26]
  /// [15, 19, 23, 27]
  //  would have divisibility [4, 1]
  DimVectorT divisibility;

  /// The _constancy_ information maps the `d`-th
  /// dimension to the length of the shortest
  /// sequence of constant integer along it. This is
  /// particularly useful to infer the contiguity
  /// of operations (e.g., add) involving a constant.
  /// Suppose we have an array of N elements,
  /// with a contiguity value C,
  /// the array can be divided into a list of
  /// N/C sequences of C elements with the same value.
  /// For example
  /// [8, 8, 8, 8, 12, 12, 12, 12]
  /// [16, 16, 16, 16, 20, 20, 20, 20]
  /// would have constancy [1, 4]
  DimVectorT constancy;

  // number of dimensions of the lattice
  int rank;
};

class AxisInfoVisitor {
public:
  AxisInfoVisitor() = default;

  static bool isContiguousDim(const AxisInfo &info, ArrayRef<int64_t> shape,
                               int dim) {
    return dim < shape.size() && info.getContiguity(dim) == shape[dim];
  }

  static bool isConstantDim(const AxisInfo &info, ArrayRef<int64_t> shape,
                                 int dim) {
    return dim < shape.size() && info.getConstancy(dim) == shape[dim];
  }

  static bool isContiguityConstancyAligned(const AxisInfo &lhs,
                                           const AxisInfo &rhs,
                                           ArrayRef<int64_t> shape, int dim) {
    return dim < shape.size() &&
           (lhs.getContiguity(dim) % rhs.getConstancy(dim) == 0 ||
            rhs.getContiguity(dim) % lhs.getConstancy(dim) == 0);
  }

  static bool isContiguityAligned(const AxisInfo &lhs, const AxisInfo &rhs,
                                     ArrayRef<int64_t> shape, int dim) {
    return dim < shape.size() &&
           (lhs.getContiguity(dim) % rhs.getContiguity(dim) == 0 ||
            rhs.getContiguity(dim) % lhs.getContiguity(dim) == 0);
  }

  static bool isConstancyAligned(const AxisInfo &lhs, const AxisInfo &rhs,
                                   ArrayRef<int64_t> shape, int dim) {
    return dim < shape.size() &&
           (lhs.getConstancy(dim) % rhs.getConstancy(dim) == 0 ||
            rhs.getConstancy(dim) % lhs.getConstancy(dim) == 0);
  }

  static AxisInfo visitBinaryOp(
      Operation *op, AxisInfo lhsInfo, AxisInfo rhsInfo,
      const std::function<int(AxisInfo, AxisInfo, int)> &getContiguity,
      const std::function<int(AxisInfo, AxisInfo, int)> &getDivisibility,
      const std::function<int(AxisInfo, AxisInfo, int)> &getConstancy);

  virtual bool match(Operation *op) = 0;

  virtual AxisInfo
  getAxisInfo(Operation *op, ArrayRef<LatticeElement<AxisInfo> *> operands) = 0;
};

template <typename OpTy> class AxisInfoVisitorImpl : public AxisInfoVisitor {
public:
  AxisInfo getAxisInfo(Operation *op,
                       ArrayRef<LatticeElement<AxisInfo> *> operands) final {
    return getAxisInfo(cast<OpTy>(op), operands);
  }

  bool match(Operation *op) final { return isa<OpTy>(op); }

  virtual AxisInfo getAxisInfo(OpTy op,
                               ArrayRef<LatticeElement<AxisInfo> *> operands) {
    llvm_unreachable("Unimplemented getAxisInfo for op");
  }
};

class AxisInfoVisitorList {
public:
  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void add() {
    // Iterate over all the types in the parameter pack.
    (visitors.emplace_back(std::make_unique<Ts>()), ...);
  }

  AxisInfo apply(Operation *op, ArrayRef<LatticeElement<AxisInfo> *> operands) {
    for (auto &visitor : visitors)
      if (visitor->match(op))
        return visitor->getAxisInfo(op, operands);
    llvm_unreachable("Unimplemented getAxisInfo for op");
  }

private:
  std::vector<std::unique_ptr<AxisInfoVisitor>> visitors;
};

class AxisInfoAnalysis : public ForwardDataFlowAnalysis<AxisInfo> {
private:
  AxisInfoVisitorList visitors;

public:
  AxisInfoAnalysis(MLIRContext *context);

  ChangeResult
  visitOperation(Operation *op,
                 ArrayRef<LatticeElement<AxisInfo> *> operands) override;

  unsigned getPtrVectorSize(Value ptr);

  unsigned getPtrAlignment(Value ptr);

  unsigned getMaskAlignment(Value mask);
};

} // namespace mlir

#endif