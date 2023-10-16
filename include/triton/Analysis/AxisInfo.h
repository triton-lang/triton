#ifndef TRITON_ANALYSIS_AXISINFO_H
#define TRITON_ANALYSIS_AXISINFO_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <optional>
#include <type_traits>

namespace mlir {

//===----------------------------------------------------------------------===//
// AxisInfo
//===----------------------------------------------------------------------===//

/// This lattice value represents known information on the axes of a lattice.
class AxisInfo {
public:
  typedef SmallVector<int64_t> DimVectorT;

public:
  /// Default constructor
  AxisInfo() : AxisInfo({}, {}, {}) {}
  /// Construct contiguity info with known contiguity
  AxisInfo(DimVectorT knownContiguity, DimVectorT knownDivisibility,
           DimVectorT knownConstancy)
      : AxisInfo(knownContiguity, knownDivisibility, knownConstancy, {}) {}
  AxisInfo(DimVectorT knownContiguity, DimVectorT knownDivisibility,
           DimVectorT knownConstancy, std::optional<int64_t> knownConstantValue)
      : contiguity(knownContiguity), divisibility(knownDivisibility),
        constancy(knownConstancy), constantValue(knownConstantValue),
        rank(contiguity.size()) {
    assert(knownContiguity.size() == static_cast<size_t>(rank));
    assert(knownDivisibility.size() == static_cast<size_t>(rank));
    assert(knownConstancy.size() == static_cast<size_t>(rank));
  }

  /// Accessors
  int64_t getContiguity(size_t dim) const { return contiguity[dim]; }
  const DimVectorT &getContiguity() const { return contiguity; }

  int64_t getDivisibility(size_t dim) const { return divisibility[dim]; }
  const DimVectorT &getDivisibility() const { return divisibility; }

  int64_t getConstancy(size_t dim) const { return constancy[dim]; }
  const DimVectorT &getConstancy() const { return constancy; }

  int getRank() const { return rank; }

  std::optional<int64_t> getConstantValue() const { return constantValue; }

  template <class T>
  static void
  initPessimisticStateFromFunc(int argNumber, T funcOp, DimVectorT *contiguity,
                               DimVectorT *divisibility, DimVectorT *constancy);
  /// Comparison
  bool operator==(const AxisInfo &other) const {
    return (contiguity == other.contiguity) &&
           (divisibility == other.divisibility) &&
           (constancy == other.constancy) &&
           (constantValue == other.constantValue) && (rank == other.rank);
  }

  /// The pessimistic value state of the contiguity is unknown.
  static AxisInfo getPessimisticValueState(MLIRContext *context = nullptr) {
    return AxisInfo();
  }
  static AxisInfo getPessimisticValueState(Value value);

  /// The gcd of both arguments for each dimension
  static AxisInfo join(const AxisInfo &lhs, const AxisInfo &rhs);

  void print(raw_ostream &os) const {
    auto print = [&](StringRef name, DimVectorT vec) {
      os << name << " = [";
      llvm::interleaveComma(vec, os);
      os << "]";
    };
    print("contiguity", contiguity);
    print(", divisibility", divisibility);
    print(", constancy", constancy);
    os << ", constant_value = ";
    if (constantValue)
      os << *constantValue;
    else
      os << "<none>";
  }

private:
  /// The _contiguity_ information maps the `d`-th
  /// dimension to the length of the shortest
  /// sequence of contiguous integers along it.
  /// Suppose we have an array of N elements,
  /// with a contiguity value C,
  /// the array can be divided into a list of
  /// N/C sequences of C contiguous elements.
  /// Since we have N = 2^k, C must be a power of two.
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
  /// divides the first element of all groups of
  // _contiguity_ values along it
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
  //  On the other hand:
  //  [0, 1, 2, 0, 4, 5, 6, 7]
  //  would have divisibility 1 because
  //  _contiguity_=1
  DimVectorT divisibility;

  /// The _constancy_ information maps the `d`-th
  /// dimension to the length of the shortest
  /// sequence of constant integer along it. This is
  /// particularly useful to infer the contiguity
  /// of operations (e.g., add) involving a constant.
  /// Suppose we have an array of N elements,
  /// with a constancy value C,
  /// the array can be divided into a list of
  /// N/C sequences of C elements with the same value.
  /// Since we have N = 2^k, C must be a power of two.
  /// For example
  /// [8, 8, 8, 8, 12, 12, 12, 12]
  /// [16, 16, 16, 16, 20, 20, 20, 20]
  /// would have constancy [1, 4]
  DimVectorT constancy;

  /// The constant value of the lattice if we can infer it.
  std::optional<int64_t> constantValue;

  // number of dimensions of the lattice
  int rank{};
};

class AxisInfoVisitor {
public:
  AxisInfoVisitor() = default;
  virtual ~AxisInfoVisitor() = default;

  static bool isContiguousDim(const AxisInfo &info, ArrayRef<int64_t> shape,
                              int dim) {
    return info.getContiguity(dim) == shape[dim];
  }

  static bool isConstantDim(const AxisInfo &info, ArrayRef<int64_t> shape,
                            int dim) {
    return info.getConstancy(dim) == shape[dim];
  }

  virtual AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) = 0;

  virtual bool match(Operation *op) = 0;
};

/// Base class for all operations
template <typename OpTy> class AxisInfoVisitorImpl : public AxisInfoVisitor {
public:
  using AxisInfoVisitor::AxisInfoVisitor;

  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) final {
    return getAxisInfo(cast<OpTy>(op), operands);
  }

  bool match(Operation *op) final { return isa<OpTy>(op); }

  virtual AxisInfo
  getAxisInfo(OpTy op, ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) {
    llvm_unreachable("Unimplemented getAxisInfo");
  }
};

/// Binary operations
template <typename OpTy>
class BinaryOpVisitorImpl : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();
    auto rank = lhsInfo.getRank();
    assert(operands.size() == 2 && "Expected two operands");
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;
    auto constantValue = getConstantValue(op, lhsInfo, rhsInfo);
    for (auto d = 0; d < rank; ++d) {
      if (constantValue.has_value()) {
        contiguity.push_back(1);
        constancy.push_back(
            std::max(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d)));
        divisibility.push_back(highestPowOf2Divisor(constantValue.value()));
      } else {
        contiguity.push_back(getContiguity(op, lhsInfo, rhsInfo, d));
        constancy.push_back(getConstancy(op, lhsInfo, rhsInfo, d));
        divisibility.push_back(getDivisibility(op, lhsInfo, rhsInfo, d));
      }
    }
    return AxisInfo(contiguity, divisibility, constancy, constantValue);
  }

protected:
  virtual int64_t getContiguity(OpTy op, const AxisInfo &lhs,
                                const AxisInfo &rhs, int dim) {
    return 1;
  }

  virtual int64_t getDivisibility(OpTy op, const AxisInfo &lhs,
                                  const AxisInfo &rhs, int dim) {
    return 1;
  }

  virtual int64_t getConstancy(OpTy op, const AxisInfo &lhs,
                               const AxisInfo &rhs, int dim) {
    return 1;
  }

  virtual std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
                                                  const AxisInfo &rhs) {
    return {};
  }
};

class AxisInfoVisitorList {
public:
  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void append() {
    (visitors.emplace_back(std::make_unique<Ts>()), ...);
  }

  AxisInfo apply(Operation *op,
                 ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) {
    for (auto &visitor : visitors)
      if (visitor->match(op))
        return visitor->getAxisInfo(op, operands);
    return AxisInfo();
  }

private:
  std::vector<std::unique_ptr<AxisInfoVisitor>> visitors;
};

class AxisInfoAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                             dataflow::Lattice<AxisInfo>> {
private:
  AxisInfoVisitorList visitors;

  void setToEntryState(dataflow::Lattice<AxisInfo> *lattice) override {
    propagateIfChanged(
        lattice,
        lattice->join(AxisInfo::getPessimisticValueState(lattice->getPoint())));
  }

public:
  AxisInfoAnalysis(DataFlowSolver &solver);
  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<AxisInfo>>::getLatticeElement;
  using FuncAxisInfoMapT = DenseMap<FunctionOpInterface, AxisInfo>;

  void visitOperation(Operation *op,
                      ArrayRef<const dataflow::Lattice<AxisInfo> *> operands,
                      ArrayRef<dataflow::Lattice<AxisInfo> *> results) override;
};

/// Module level axis info analysis based on the call graph, assuming that we
/// do not have recursive functions.
/// Since each function will be called multiple times, we need to
/// calculate the axis info based on the axis info of all the callers.
/// In the future, we can perform optimization using function cloning so that
/// each call site will have unique axis info.
using AxisInfoMapT = DenseMap<Value, AxisInfo>;
class ModuleAxisInfoAnalysis : public CallGraph<AxisInfoMapT> {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp)
      : CallGraph<AxisInfoMapT>(moduleOp) {
    SmallVector<FunctionOpInterface> funcs;
    for (auto root : getRoots()) {
      walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
          // Pre-order edge walk callback
          [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
          // Post-order node walk callback
          [&](FunctionOpInterface funcOp) {
            funcs.push_back(funcOp);
            funcMap.try_emplace(funcOp, AxisInfoMapT{});
          });
    }
    SetVector<FunctionOpInterface> sortedFuncs(funcs.begin(), funcs.end());
    SymbolTableCollection symbolTable;
    for (auto funcOp : llvm::reverse(sortedFuncs)) {
      initialize(funcOp);
      funcOp.walk([&](CallOpInterface callOp) {
        auto callee =
            dyn_cast<FunctionOpInterface>(callOp.resolveCallable(&symbolTable));
        update(callOp, callee);
      });
    }
  }

  AxisInfo *getAxisInfo(Value value) {
    auto funcOp =
        value.getParentRegion()->getParentOfType<FunctionOpInterface>();
    auto *axisInfoMap = getFuncData(funcOp);
    if (!axisInfoMap) {
      return nullptr;
    }
    auto it = axisInfoMap->find(value);
    if (it == axisInfoMap->end()) {
      return nullptr;
    }
    return &(it->second);
  }

  unsigned getPtrContiguity(Value ptr);

  unsigned getPtrAlignment(Value ptr);

  unsigned getMaskAlignment(Value mask);

private:
  void initialize(FunctionOpInterface funcOp);

  void update(CallOpInterface callOp, FunctionOpInterface funcOp);
};

} // namespace mlir

#endif
