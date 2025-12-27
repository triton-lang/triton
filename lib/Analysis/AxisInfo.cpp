#include "triton/Analysis/AxisInfo.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <numeric>

#define DEBUG_TYPE "axis-info"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {
namespace {

constexpr int64_t kMaxDivisor = highestPowOf2Divisor<int64_t>(0);

template <typename... Args> int64_t gcd(int64_t a, int64_t b, Args... args) {
  if (a == 0)
    return b;
  if (b == 0)
    return a;
  if constexpr (sizeof...(args) == 0)
    return std::gcd(a, b);
  else
    return gcd(std::gcd(a, b), args...);
}

// If lhs * rhs overflows, return max value possible value for the type
int64_t multiplyDivisor(int64_t lhs, int64_t rhs) {
  if (lhs > kMaxDivisor / rhs)
    return kMaxDivisor;
  return lhs * rhs;
}

int64_t getDivisibilityFromContiguity(const AxisInfo &lhs, const AxisInfo &rhs,
                                      int d) {
  // For example if we have the following two arrays using the selectOp:
  // lhs: [[0, 1], [4, 5]]
  // rhs: [[16, 17, 18, 19]]
  // The resulting contiguity will be 2, while the divisibility will be 2
  // because 18 is not divisible by 4.
  if (lhs.getContiguity(d) == rhs.getContiguity(d) ||
      lhs.getContiguity(d) == kMaxDivisor ||
      rhs.getContiguity(d) == kMaxDivisor) {
    // Contiguity not changed or one of them is unresolved.
    // If unresolved, we can first perform a loose bound gcd since the unknown
    // contiguity will be resolved in the end.
    return gcd(lhs.getDivisibility(d), rhs.getDivisibility(d));
  } else {
    // Contiguity changed, we cannot use only divisibility.
    return gcd(lhs.getDivisibility(d), rhs.getDivisibility(d),
               lhs.getContiguity(d), rhs.getContiguity(d));
  }
}

// Base class for all operations
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
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) = 0;
};

// Binary operations
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
    assert(isa<RankedTensorType>(op.getType()) ||
           rank == 1 && "Expected ranked tensor or scalar");
    assert(operands.size() == 2 && "Expected two operands");
    auto constantValue = getConstantValue(op, lhsInfo, rhsInfo);
    if (constantValue.has_value()) {
      auto resTy = dyn_cast<RankedTensorType>(op.getType());
      AxisInfo::DimVectorT constancy =
          resTy ? to_vector(resTy.getShape()) : AxisInfo::DimVectorT(rank, 1);
      AxisInfo::DimVectorT contiguity(rank, 1);
      AxisInfo::DimVectorT divisibility(
          rank, highestPowOf2Divisor<int64_t>(constantValue.value()));
      return AxisInfo(contiguity, divisibility, constancy, constantValue);
    }
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;
    for (auto d = 0; d < rank; ++d) {
      contiguity.push_back(getContiguity(op, lhsInfo, rhsInfo, d));
      constancy.push_back(getConstancy(op, lhsInfo, rhsInfo, d));
      divisibility.push_back(getDivisibility(op, lhsInfo, rhsInfo, d));
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
    return gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
  }
  virtual std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
                                                  const AxisInfo &rhs) {
    return {};
  }
};

class AxisInfoAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                             dataflow::Lattice<AxisInfo>> {
private:
  AxisInfoVisitorList visitors;

  void setToEntryState(dataflow::Lattice<AxisInfo> *lattice) override {
    propagateIfChanged(
        lattice, lattice->join(
                     AxisInfo::getPessimisticValueState(lattice->getAnchor())));
  }

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices,
      unsigned firstIndex) override {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      visitForOpInductionVar(forOp, argLattices);
    } else if (auto ws = dyn_cast<gpu::WarpSpecializePartitionsOp>(op)) {
      visitWarpSpecializeExplicitCaptures(ws, successor, argLattices);
    } else {
      setAllToEntryStates(argLattices.take_front(firstIndex));
      setAllToEntryStates(argLattices.drop_front(
          firstIndex + successor.getSuccessorInputs().size()));
    }
  }

public:
  AxisInfoAnalysis(DataFlowSolver &solver,
                   axisinfo::CallbackType callback = nullptr);
  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<AxisInfo>>::getLatticeElement;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::Lattice<AxisInfo> *> operands,
                 ArrayRef<dataflow::Lattice<AxisInfo> *> results) override;
  void
  visitForOpInductionVar(scf::ForOp op,
                         ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices);

  void visitWarpSpecializeExplicitCaptures(
      gpu::WarpSpecializePartitionsOp ws, const RegionSuccessor &successor,
      ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices);
};

template <typename OpTy>
class CastOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

class UnrealizedConversionCastOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<mlir::UnrealizedConversionCastOp> {
public:
  using AxisInfoVisitorImpl<
      mlir::UnrealizedConversionCastOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(mlir::UnrealizedConversionCastOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto tensorType = dyn_cast<RankedTensorType>(op.getResultTypes()[0]);
    if (tensorType &&
        tensorType.getRank() != operands[0]->getValue().getRank()) {
      // Do not propagate AxisInfo with incorrect rank. This can cause a crash
      // in future visitor applications.
      return AxisInfo::getPessimisticValueState(op->getResult(0));
    }
    return operands[0]->getValue();
  }
};

class MakeRangeOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<triton::MakeRangeOp> {
public:
  using AxisInfoVisitorImpl<triton::MakeRangeOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(triton::MakeRangeOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto start = op.getStart();
    auto end = op.getEnd();
    return AxisInfo(/*contiguity=*/{end - start},
                    /*divisibility=*/{highestPowOf2Divisor(start)},
                    /*constancy=*/{1});
  }
};

class ConstantOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<arith::ConstantOp> {
public:
  using AxisInfoVisitorImpl::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(arith::ConstantOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto intAttr = dyn_cast<IntegerAttr>(op.getValue());
    auto boolAttr = dyn_cast<BoolAttr>(op.getValue());
    if (intAttr || boolAttr) {
      int64_t value{};
      if (intAttr)
        value = intAttr.getValue().getZExtValue();
      else
        value = boolAttr.getValue() ? 1 : 0;
      return AxisInfo(/*contiguity=*/{1},
                      /*divisibility=*/{highestPowOf2Divisor(value)},
                      /*constancy=*/{1},
                      /*knownConstantValue=*/{value});
    }
    // TODO: generalize to dense attr
    auto splatAttr = dyn_cast<SplatElementsAttr>(op.getValue());
    if (splatAttr && splatAttr.getElementType().isIntOrIndex()) {
      int64_t value = splatAttr.template getSplatValue<APInt>().getZExtValue();
      TensorType ty = cast<TensorType>(splatAttr.getType());
      return AxisInfo(
          /*contiguity=*/AxisInfo::DimVectorT(ty.getRank(), 1),
          /*divisibility=*/
          AxisInfo::DimVectorT(ty.getRank(), highestPowOf2Divisor(value)),
          /*constancy=*/
          AxisInfo::DimVectorT(ty.getShape().begin(), ty.getShape().end()),
          /*knownConstantValue=*/{value});
    }
    return AxisInfo();
  }
};

class PoisonOpAxisInfoVisitor final : public AxisInfoVisitorImpl<ub::PoisonOp> {
public:
  using AxisInfoVisitorImpl::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(ub::PoisonOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    unsigned rank = 1;
    if (auto shape = dyn_cast<RankedTensorType>(op.getType()))
      rank = shape.getRank();

    // Poison values are never accessed, thus assume optimistic values.
    return AxisInfo(AxisInfo::DimVectorT(rank, kMaxDivisor),
                    AxisInfo::DimVectorT(rank, kMaxDivisor),
                    AxisInfo::DimVectorT(rank, kMaxDivisor));
  }
};

template <typename OpTy>
class AddSubOpAxisInfoVisitor final : public BinaryOpVisitorImpl<OpTy> {
public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

private:
  int64_t getContiguity(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                        int dim) override {
    // Contiguity assumes an increasing sequence. So for SubIOp contiguous
    // RHS doesn't produce a contiguous result.
    if (isa<arith::SubIOp>(op))
      return gcd(lhs.getContiguity(dim), rhs.getConstancy(dim));

    return std::max(gcd(lhs.getConstancy(dim), rhs.getContiguity(dim)),
                    gcd(lhs.getContiguity(dim), rhs.getConstancy(dim)));
  }

  int64_t getDivisibility(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                          int dim) override {
    // lhs = k * d_lhs = k * k' * gcd(d_lhs, d_rhs)
    // rhs = p * d_rhs = p * p' * gcd(d_lhs, d_rhs)
    // lhs + rhs = k * d_lhs + p * d_rhs = (k * k' + p * p') * gcd(d_lhs, d_rhs)
    auto rhsDivisibility = rhs.getDivisibility(dim);
    if constexpr (std::is_same_v<OpTy, triton::AddPtrOp>) {
      //  %ptr = addptr %lhs, %rhs
      // is equivalent to
      //  %0 = mul %rhs, %elemSize
      //  %ptr = add %lhs, %0
      // The result will still be contiguous in terms of elements but not bytes
      // For example:
      // addptr [16] : !ptr<i32>, [0, 1, 2, 3] : i32 -> !ptr<i32>
      // returns:
      // [16, 20, 24, 28] : !ptr<i32>
      // with element locations:
      // [4, 5, 6, 7]
      // It is "strided contiguous" with a divisibility of 16 bytes
      auto elemSize = std::max<int64_t>(
          1, triton::getPointeeBitWidth(op.getPtr().getType()) / 8);
      rhsDivisibility = multiplyDivisor(rhs.getDivisibility(dim), elemSize);
    }
    return gcd(lhs.getDivisibility(dim), rhsDivisibility);
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
                                          const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value()) {
      if constexpr (std::is_same_v<OpTy, arith::AddIOp>) {
        return {lhs.getConstantValue().value() +
                rhs.getConstantValue().value()};
      } else if constexpr (std::is_same_v<OpTy, arith::SubIOp>) {
        return {lhs.getConstantValue().value() -
                rhs.getConstantValue().value()};
      } else if constexpr (std::is_same_v<OpTy, triton::AddPtrOp>) {
        auto elemSize = std::max<int64_t>(
            1, triton::getPointeeBitWidth(op.getPtr().getType()) / 8);
        auto rhsValue = rhs.getConstantValue().value() * elemSize;
        return {lhs.getConstantValue().value() + rhsValue};
      }
    }
    return {};
  }
};

class MulIOpAxisInfoVisitor final : public BinaryOpVisitorImpl<arith::MulIOp> {
public:
  using BinaryOpVisitorImpl<arith::MulIOp>::BinaryOpVisitorImpl;

private:
  int64_t getContiguity(arith::MulIOp op, const AxisInfo &lhs,
                        const AxisInfo &rhs, int dim) override {
    // lhs * 1 = lhs
    auto lhsContiguity =
        rhs.getConstantValue().has_value() && rhs.getConstantValue() == 1
            ? lhs.getContiguity(dim)
            : 1;
    // 1 * rhs = rhs
    auto rhsContiguity =
        lhs.getConstantValue().has_value() && lhs.getConstantValue() == 1
            ? rhs.getContiguity(dim)
            : 1;
    return std::max(lhsContiguity, rhsContiguity);
  }

  int64_t getDivisibility(arith::MulIOp op, const AxisInfo &lhs,
                          const AxisInfo &rhs, int dim) override {
    auto lhsDivisibility = lhs.getDivisibility(dim);
    if (lhs.getContiguity(dim) > 1 && rhs.getConstantValue() != 1) {
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      lhsDivisibility = 1;
    }
    auto rhsDivisibility = rhs.getDivisibility(dim);
    if (rhs.getContiguity(dim) > 1 && lhs.getConstantValue() != 1) {
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      rhsDivisibility = 1;
    }
    return multiplyDivisor(lhsDivisibility, rhsDivisibility);
  }

  std::optional<int64_t> getConstantValue(arith::MulIOp op, const AxisInfo &lhs,
                                          const AxisInfo &rhs) override {
    auto lhsConst = lhs.getConstantValue();
    auto rhsConst = rhs.getConstantValue();
    if (lhsConst.has_value() && rhsConst.has_value())
      return {lhsConst.value() * rhsConst.value()};
    if ((lhsConst.has_value() && lhsConst.value() == 0) ||
        (rhsConst.has_value() && rhsConst.value() == 0))
      return 0;
    return {};
  }
};

template <typename OpTy>
class DivOpAxisInfoVisitor final : public BinaryOpVisitorImpl<OpTy> {
public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

private:
  int64_t getContiguity(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                        int dim) override {
    // lhs / 1 = lhs
    return rhs.getConstantValue().has_value() &&
                   rhs.getConstantValue().value() == 1
               ? lhs.getContiguity(dim)
               : 1;
  }

  int64_t getConstancy(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                       int dim) override {
    auto resTy = dyn_cast<RankedTensorType>(op.getType());
    auto constancy = BinaryOpVisitorImpl<OpTy>::getConstancy(op, lhs, rhs, dim);
    if (!resTy)
      return constancy;
    auto shape = resTy.getShape();
    // Case: lhs contiguous, rhs constant.
    // lhs: d_lhs * k, d_lhs * k + 1, ..., d_lhs * k + n
    // rhs: d_rhs * p, d_rhs * p, ..., d_rhs * p
    // lhs / rhs = d_lhs * k / (d_rhs * p), (d_lhs * k + 1) / (d_rhs * p),
    // ..., (d_lhs * k + n) / (d_rhs * p)
    // Because d_lhs % d_rhs = 0 || d_rhs % d_lhs = 0,
    // the minimal constancy is gcd(d_lhs, d_rhs).
    // Since gcd(d_lhs, d_rhs) maybe > len(lhs),
    // we need to use another gcd to get the actual constancy.
    if (AxisInfoVisitor::isContiguousDim(lhs, shape, dim) &&
        AxisInfoVisitor::isConstantDim(rhs, shape, dim)) {
      constancy = std::max(constancy,
                           gcd(lhs.getContiguity(dim), lhs.getDivisibility(dim),
                               rhs.getDivisibility(dim)));
    }
    return constancy;
  }

  int64_t getDivisibility(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                          int dim) override {
    // Case 1: lhs is 0
    if (lhs.getConstantValue().has_value() &&
        lhs.getConstantValue().value() == 0)
      return lhs.getDivisibility(dim);
    // Case 2: rhs is 1
    if (rhs.getConstantValue().has_value() &&
        rhs.getConstantValue().value() == 1)
      return lhs.getDivisibility(dim);
    // Case 3: lhs has contiguity of 1 in this dimension and rhs is a power of 2
    if (rhs.getConstantValue().has_value() &&
        llvm::isPowerOf2_64(std::abs(rhs.getConstantValue().value())) &&
        lhs.getContiguity(dim) == 1) {
      int64_t absRhs = std::abs(rhs.getConstantValue().value());
      return std::max<int64_t>(1, lhs.getDivisibility(dim) / absRhs);
    }
    // otherwise: return 1
    return 1;
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
                                          const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value())
      return {lhs.getConstantValue().value() / rhs.getConstantValue().value()};
    return {};
  }
};

template <typename OpTy>
class RemOpAxisInfoVisitor final : public BinaryOpVisitorImpl<OpTy> {
public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

private:
  int64_t getContiguity(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                        int dim) override {
    auto resTy = dyn_cast<RankedTensorType>(op.getType());
    if (!resTy)
      return BinaryOpVisitorImpl<OpTy>::getContiguity(op, lhs, rhs, dim);
    auto shape = resTy.getShape();
    int64_t contiguity = 1;
    // lhs contiguous, rhs constant
    // lhs: d_lhs * k, d_lhs * k + 1, ..., d_lhs * k + n
    // rhs: d_rhs * p, d_rhs * p, ..., d_rhs * p
    // lhs % rhs = d_lhs * k % (d_rhs * p), (d_lhs * k + 1) % (d_rhs * p),
    // ..., (d_lhs * k + n) % (d_rhs * p)
    // Because d_lhs % d_rhs = 0 || d_rhs % d_lhs = 0,
    // The minimal contiguity is gcd(d_lhs, d_rhs).
    // Since gcd(d_lhs, d_rhs) maybe > len(lhs),
    // we need to use another gcd to get the actual contiguity.
    if (AxisInfoVisitor::isContiguousDim(lhs, shape, dim) &&
        AxisInfoVisitor::isConstantDim(rhs, shape, dim)) {
      contiguity = gcd(lhs.getContiguity(dim), lhs.getDivisibility(dim),
                       rhs.getDivisibility(dim));
    }
    return contiguity;
  }

  int64_t getDivisibility(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                          int dim) override {
    auto resTy = dyn_cast<RankedTensorType>(op.getType());
    if (rhs.getConstancy(dim) > 1) {
      // lhs: d_lhs * k = gcd(d_lhs, d_rhs) * k' * k = gcd(d_lhs, d_rhs) * k''
      // rhs: d_rhs * p = gcd(d_lhs, d_rhs) * p' * p = gcd(d_lhs, d_rhs) * p''
      // lhs = gcd(d_lhs, d_rhs) * k'' = gcd(d_lhs, d_rhs) * d + r
      // r must be divisible by gcd(d_lhs, d_rhs)
      return gcd(lhs.getDivisibility(dim), rhs.getDivisibility(dim));
    }
    // Otherwise we shouldn't assume any divisibility.
    // For example:
    // lhs: [2, 2, 4, 4], rhs: [0, 1, 2, 3]
    // lhs % rhs = [0, 0, 0, 1]
    return 1;
  };

  int64_t getConstancy(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                       int dim) override {
    auto constancy = BinaryOpVisitorImpl<OpTy>::getConstancy(op, lhs, rhs, dim);
    auto resTy = dyn_cast<RankedTensorType>(op.getType());
    if (!resTy)
      return constancy;
    // Case: lhs % 1 = 0
    if (rhs.getConstantValue().has_value() &&
        rhs.getConstantValue().value() == 1)
      return resTy.getDimSize(dim);
    return constancy;
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
                                          const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value())
      return {lhs.getConstantValue().value() % rhs.getConstantValue().value()};
    else if (rhs.getConstantValue().has_value() &&
             rhs.getConstantValue().value() == 1)
      return {0};
    return {};
  }
};

class SplatOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<triton::SplatOp> {
public:
  using AxisInfoVisitorImpl<triton::SplatOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(triton::SplatOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    Type _retTy = *op->result_type_begin();
    TensorType retTy = cast<TensorType>(_retTy);
    AxisInfo opInfo = operands[0]->getValue();
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;
    for (int d = 0; d < retTy.getRank(); ++d) {
      contiguity.push_back(1);
      divisibility.push_back(opInfo.getDivisibility(0));
      constancy.push_back(retTy.getShape()[d]);
    }
    return AxisInfo(contiguity, divisibility, constancy,
                    operands[0]->getValue().getConstantValue());
  }
};

class LoadOpAxisInfoVisitor final : public AxisInfoVisitorImpl<triton::LoadOp> {
public:
  using AxisInfoVisitorImpl<triton::LoadOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(triton::LoadOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    // If pointers and mask both have constancy properties, those properties
    // will also extend to output.
    AxisInfo ptrInfo = operands[0]->getValue();
    std::optional<AxisInfo> maskInfo;
    if (operands.size() > 1) {
      maskInfo = operands[1]->getValue();
    }
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;

    for (int d = 0; d < ptrInfo.getRank(); ++d) {
      contiguity.push_back(1);
      divisibility.push_back(1);
      constancy.push_back(
          gcd(ptrInfo.getConstancy(d),
              maskInfo.has_value() ? maskInfo->getConstancy(d) : 0));
    }

    return AxisInfo(contiguity, divisibility, constancy);
  }
};

class ExpandDimsOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<triton::ExpandDimsOp> {
public:
  using AxisInfoVisitorImpl<triton::ExpandDimsOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(triton::ExpandDimsOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    AxisInfo opInfo = operands[0]->getValue();
    AxisInfo::DimVectorT contiguity = opInfo.getContiguity();
    AxisInfo::DimVectorT divisibility = opInfo.getDivisibility();
    AxisInfo::DimVectorT constancy = opInfo.getConstancy();
    int64_t newDivisibility = 1;
    if (opInfo.getConstantValue().has_value()) {
      // The tensor is constant, same as ConstantOpAxisInfoVisitor
      newDivisibility = highestPowOf2Divisor(opInfo.getConstantValue().value());
    } else if (opInfo.getRank()) {
      // Otherwise, calculate the GCD as the new divisibility
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      newDivisibility =
          opInfo.getContiguity(0) > 1 ? 1 : opInfo.getDivisibility(0);
      for (int d = 1; d < opInfo.getRank(); ++d) {
        newDivisibility =
            gcd(newDivisibility,
                opInfo.getContiguity(d) > 1 ? 1 : opInfo.getDivisibility(d));
      }
    }
    contiguity.insert(contiguity.begin() + op.getAxis(), 1);
    divisibility.insert(divisibility.begin() + op.getAxis(), newDivisibility);
    constancy.insert(constancy.begin() + op.getAxis(), 1);
    return AxisInfo(contiguity, divisibility, constancy,
                    operands[0]->getValue().getConstantValue());
  }
};

class BroadcastOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<triton::BroadcastOp> {
public:
  using AxisInfoVisitorImpl<triton::BroadcastOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(triton::BroadcastOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    Type _retTy = *op->result_type_begin();
    Type _opTy = *op->operand_type_begin();
    TensorType retTy = cast<TensorType>(_retTy);
    TensorType opTy = cast<TensorType>(_opTy);
    ArrayRef<int64_t> retShape = retTy.getShape();
    ArrayRef<int64_t> opShape = opTy.getShape();
    AxisInfo opInfo = operands[0]->getValue();
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;
    for (int d = 0; d < retTy.getRank(); ++d) {
      contiguity.push_back(opShape[d] == 1 ? 1 : opInfo.getContiguity(d));
      divisibility.push_back(opInfo.getDivisibility(d));
      constancy.push_back(opShape[d] == 1 ? retShape[d]
                                          : opInfo.getConstancy(d));
    }
    return AxisInfo(contiguity, divisibility, constancy,
                    operands[0]->getValue().getConstantValue());
  }
};

template <typename OpTy>
class CmpOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto resTy = dyn_cast<RankedTensorType>(op.getType());
    if (!resTy)
      return AxisInfo();
    auto shape = resTy.getShape();
    short rank = resTy.getRank();
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();

    AxisInfo::DimVectorT contiguity, divisibility, constancy;
    std::optional<int64_t> constantValue;
    for (short d = 0; d < rank; ++d) {
      int64_t constHint;
      if (lhsInfo.getConstantValue().has_value() &&
          rhsInfo.getConstantValue().has_value()) {
        constHint = shape[d];
        constantValue =
            compare(getPredicate(op), lhsInfo.getConstantValue().value(),
                    rhsInfo.getConstantValue().value())
                ? 1
                : 0;
      } else {
        // Case 1: lhs and rhs are both partial constants
        constHint = gcd(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d));
        if ((gtPredicate(getPredicate(op)) || lePredicate(getPredicate(op))) &&
            AxisInfoVisitor::isConstantDim(lhsInfo, shape, d)) {
          // Case 2: lhs all constant, rhs all contiguous
          // NOTE:
          // lhs: 4 4 4 4
          // rhs: 4 5 6 7
          // lhs eq rhs: 1, 0, 0, 0
          // lhs ne rhs: 0, 1, 1, 1
          // lhs lt rhs: 0, 1, 1, 1
          // lhs le rhs: 1, 1, 1, 1
          // lhs ge rhs: 1, 0, 0, 0
          // lhs gt rhs: 0, 0, 0, 0
          constHint = std::max(constHint, gcd(rhsInfo.getContiguity(d),
                                              lhsInfo.getDivisibility(d),
                                              rhsInfo.getDivisibility(d)));
        } else if ((ltPredicate(getPredicate(op)) ||
                    gePredicate(getPredicate(op))) &&
                   AxisInfoVisitor::isConstantDim(rhsInfo, shape, d)) {
          // Case 3: lhs all contiguous, rhs all constant
          // NOTE
          // lhs: 4 5 6 7
          // rhs: 4 4 4 4
          // lhs eq rhs: 1, 0, 0, 0
          // lhs ne rhs: 0, 1, 1, 1
          // lhs le rhs: 1, 0, 0, 0
          // lhs lt rhs: 0, 0, 0, 0
          // lhs gt rhs: 0, 1, 1, 1
          // lhs ge rhs: 1, 1, 1, 1
          constHint = std::max(constHint, gcd(lhsInfo.getContiguity(d),
                                              lhsInfo.getDivisibility(d),
                                              rhsInfo.getDivisibility(d)));
        }
      }

      constancy.push_back(constHint);
      divisibility.push_back(1);
      contiguity.push_back(1);
    }

    return AxisInfo(contiguity, divisibility, constancy, constantValue);
  }

private:
  static arith::CmpIPredicate getPredicate(arith::CmpIOp op) {
    return op.getPredicate();
  }

  static bool gtPredicate(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::sgt ||
           predicate == arith::CmpIPredicate::ugt;
  }

  static bool gePredicate(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::sge ||
           predicate == arith::CmpIPredicate::uge;
  }

  static bool ltPredicate(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::slt ||
           predicate == arith::CmpIPredicate::ult;
  }

  static bool lePredicate(arith::CmpIPredicate predicate) {
    return predicate == arith::CmpIPredicate::sle ||
           predicate == arith::CmpIPredicate::ule;
  }

  static bool compare(arith::CmpIPredicate predicate, int64_t lhs,
                      int64_t rhs) {
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      return lhs == rhs;
    case arith::CmpIPredicate::ne:
      return lhs != rhs;
    case arith::CmpIPredicate::slt:
      return lhs < rhs;
    case arith::CmpIPredicate::sle:
      return lhs <= rhs;
    case arith::CmpIPredicate::sgt:
      return lhs > rhs;
    case arith::CmpIPredicate::sge:
      return lhs >= rhs;
    case arith::CmpIPredicate::ult:
      return (uint64_t)lhs < (uint64_t)rhs;
    case arith::CmpIPredicate::ule:
      return (uint64_t)lhs <= (uint64_t)rhs;
    case arith::CmpIPredicate::ugt:
      return (uint64_t)lhs > (uint64_t)rhs;
    case arith::CmpIPredicate::uge:
      return (uint64_t)lhs >= (uint64_t)rhs;
    default:
      break;
    }
    llvm_unreachable("unknown comparison predicate");
  }
};

template <typename OpTy>
class SelectOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto condConstancy = operands[0]->getValue().getConstancy();
    auto lhsInfo = operands[1]->getValue();
    auto rhsInfo = operands[2]->getValue();
    auto rank = lhsInfo.getRank();

    AxisInfo::DimVectorT contiguity, divisibility, constancy;
    std::optional<int64_t> constantValue;
    if (operands[0]->getValue().getConstantValue().has_value()) {
      if (operands[0]->getValue().getConstantValue() == 0) {
        contiguity = rhsInfo.getContiguity();
        divisibility = rhsInfo.getDivisibility();
        constancy = rhsInfo.getConstancy();
        constantValue = rhsInfo.getConstantValue();
      } else {
        contiguity = lhsInfo.getContiguity();
        divisibility = lhsInfo.getDivisibility();
        constancy = lhsInfo.getConstancy();
        constantValue = lhsInfo.getConstantValue();
      }
    } else {
      // The condition can be either a tensor or i1.
      // If i1 is used as the condition, the entire tensor of either
      // lhs or rhs is selected.
      bool i1Cond = isa<IntegerType>(op.getOperand(0).getType());
      for (auto d = 0; d < rank; ++d) {
        if (i1Cond) {
          constancy.push_back(
              gcd(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d)));
          divisibility.push_back(
              getDivisibilityFromContiguity(lhsInfo, rhsInfo, d));
          contiguity.push_back(
              gcd(lhsInfo.getContiguity(d), rhsInfo.getContiguity(d)));
        } else {
          constancy.push_back(gcd(lhsInfo.getConstancy(d),
                                  rhsInfo.getConstancy(d), condConstancy[d]));
          contiguity.push_back(gcd(lhsInfo.getContiguity(d),
                                   rhsInfo.getContiguity(d), condConstancy[d]));
          divisibility.push_back(
              getDivisibilityFromContiguity(lhsInfo, rhsInfo, d));
        }
      }
      if (lhsInfo.getConstantValue().has_value() &&
          rhsInfo.getConstantValue().has_value() &&
          lhsInfo.getConstantValue() == rhsInfo.getConstantValue())
        constantValue = lhsInfo.getConstantValue();

      if (constantValue.has_value()) {
        auto resTy = dyn_cast<RankedTensorType>(op.getType());
        assert(resTy || rank == 1);
        constancy =
            resTy ? to_vector(resTy.getShape()) : AxisInfo::DimVectorT(rank, 1);
      }
    }

    return AxisInfo(contiguity, divisibility, constancy, constantValue);
  }
};

template <typename OpTy>
class LogicalOpAxisInfoVisitor final : public BinaryOpVisitorImpl<OpTy> {
public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

private:
  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
                                          const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value()) {
      if constexpr (std::is_same_v<OpTy, arith::AndIOp>) {
        return {lhs.getConstantValue().value() &
                rhs.getConstantValue().value()};
      } else if constexpr (std::is_same_v<OpTy, arith::OrIOp>) {
        return {lhs.getConstantValue().value() |
                rhs.getConstantValue().value()};
      } else if constexpr (std::is_same_v<OpTy, arith::XOrIOp>) {
        return {lhs.getConstantValue().value() ^
                rhs.getConstantValue().value()};
      }
    }
    return {};
  }
};

class ShLIOpAxisInfoVisitor final : public BinaryOpVisitorImpl<arith::ShLIOp> {
public:
  using BinaryOpVisitorImpl<arith::ShLIOp>::BinaryOpVisitorImpl;

private:
  int64_t getContiguity(arith::ShLIOp op, const AxisInfo &lhs,
                        const AxisInfo &rhs, int dim) override {
    if (rhs.getConstantValue().has_value() &&
        rhs.getConstantValue().value() == 0)
      return lhs.getContiguity(dim);
    else
      return 1;
  }

  int64_t getDivisibility(arith::ShLIOp op, const AxisInfo &lhs,
                          const AxisInfo &rhs, int dim) override {
    auto shift = rhs.getConstantValue().value_or(0);
    auto lhsDivisibility = lhs.getDivisibility(dim);
    if (lhs.getContiguity(dim) > 1 && shift) {
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      lhsDivisibility = 1;
    }
    return multiplyDivisor(lhsDivisibility, 1ll << shift);
  }

  std::optional<int64_t> getConstantValue(arith::ShLIOp op, const AxisInfo &lhs,
                                          const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value())
      return {lhs.getConstantValue().value() << rhs.getConstantValue().value()};
    return {};
  }
};

template <typename OpTy>
class ShROpAxisInfoVisitor final : public BinaryOpVisitorImpl<OpTy> {
public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

private:
  int64_t getContiguity(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                        int dim) override {
    if (rhs.getConstantValue().has_value() &&
        rhs.getConstantValue().value() == 0)
      return lhs.getContiguity(dim);
    else
      return 1;
  }

  int64_t getDivisibility(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                          int dim) override {
    if (!rhs.getConstantValue().has_value())
      return 1;
    auto shift = rhs.getConstantValue().value();
    auto lhsDivisibility = lhs.getDivisibility(dim);
    if (lhs.getContiguity(dim) > 1 && shift) {
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      lhsDivisibility = 1;
    }
    return std::max<int64_t>(1, lhsDivisibility / (int64_t(1) << shift));
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
                                          const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value())
      return {lhs.getConstantValue().value() >> rhs.getConstantValue().value()};
    return {};
  }
};

template <typename OpTy>
class MaxMinOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();
    auto rank = lhsInfo.getRank();
    std::optional<int64_t> constantValue;
    if (lhsInfo.getConstantValue().has_value() &&
        rhsInfo.getConstantValue().has_value()) {
      if constexpr (std::is_same_v<OpTy, arith::MaxSIOp> ||
                    std::is_same_v<OpTy, arith::MaxUIOp>) {
        constantValue = {std::max(lhsInfo.getConstantValue().value(),
                                  rhsInfo.getConstantValue().value())};
      } else if constexpr (std::is_same_v<OpTy, arith::MinSIOp> ||
                           std::is_same_v<OpTy, arith::MinUIOp>) {
        constantValue = {std::min(lhsInfo.getConstantValue().value(),
                                  rhsInfo.getConstantValue().value())};
      }
      auto resTy = dyn_cast<RankedTensorType>(op.getType());
      assert(resTy || rank == 1);
      AxisInfo::DimVectorT constancy =
          resTy ? to_vector(resTy.getShape()) : AxisInfo::DimVectorT(rank, 1);
      AxisInfo::DimVectorT divisibility(
          rank, highestPowOf2Divisor<int64_t>(constantValue.value()));
      return AxisInfo(/*knownContiguity=*/AxisInfo::DimVectorT(rank, 1),
                      /*knownDivisibility=*/divisibility,
                      /*knownConstancy=*/constancy,
                      /*constantValue=*/constantValue);
    } else {
      AxisInfo::DimVectorT contiguity, divisibility, constancy;
      for (auto d = 0; d < rank; ++d) {
        constancy.push_back(
            gcd(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d)));
        divisibility.push_back(
            getDivisibilityFromContiguity(lhsInfo, rhsInfo, d));
        contiguity.push_back(
            gcd(lhsInfo.getContiguity(d), rhsInfo.getContiguity(d)));
      }
      return AxisInfo(contiguity, divisibility, constancy, std::nullopt);
    }
  }
};

class TransOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<triton::TransOp> {
public:
  using AxisInfoVisitorImpl<triton::TransOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(triton::TransOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    AxisInfo srcInfo = operands[0]->getValue();
    auto order = op.getOrder();
    auto rank = srcInfo.getRank();

    // Apply the transpose permutation to all axis info properties
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;

    for (int d = 0; d < rank; ++d) {
      int srcDim = order[d];
      contiguity.push_back(srcInfo.getContiguity(srcDim));
      divisibility.push_back(srcInfo.getDivisibility(srcDim));
      constancy.push_back(srcInfo.getConstancy(srcDim));
    }

    return AxisInfo(contiguity, divisibility, constancy,
                    srcInfo.getConstantValue());
  }
};

//===----------------------------------------------------------------------===//
// AxisInfoAnalysis
//===----------------------------------------------------------------------===//

AxisInfoAnalysis::AxisInfoAnalysis(DataFlowSolver &solver,
                                   axisinfo::CallbackType callback)
    : dataflow::SparseForwardDataFlowAnalysis<dataflow::Lattice<AxisInfo>>(
          solver) {
  // UnrealizedConversionCast:
  // This is needed by TritonGPUToLLVM, to get AxisInfo when the graph is
  // in the process of a PartialConversion, where UnrealizedConversionCast
  // may exist
  visitors.append<UnrealizedConversionCastOpAxisInfoVisitor>();
  visitors.append<CastOpAxisInfoVisitor<arith::ExtSIOp>,
                  CastOpAxisInfoVisitor<arith::ExtUIOp>,
                  CastOpAxisInfoVisitor<arith::TruncIOp>,
                  CastOpAxisInfoVisitor<triton::gpu::ConvertLayoutOp>,
                  CastOpAxisInfoVisitor<triton::BitcastOp>,
                  CastOpAxisInfoVisitor<triton::gluon::SetAutoLayoutOp>>();
  visitors.append<MakeRangeOpAxisInfoVisitor>();
  visitors.append<PoisonOpAxisInfoVisitor>();
  visitors.append<ConstantOpAxisInfoVisitor>();
  visitors.append<AddSubOpAxisInfoVisitor<triton::AddPtrOp>,
                  AddSubOpAxisInfoVisitor<arith::AddIOp>,
                  AddSubOpAxisInfoVisitor<arith::SubIOp>>();
  visitors.append<MulIOpAxisInfoVisitor>();
  visitors.append<DivOpAxisInfoVisitor<arith::DivSIOp>,
                  DivOpAxisInfoVisitor<arith::DivUIOp>>();
  visitors.append<RemOpAxisInfoVisitor<arith::RemSIOp>,
                  RemOpAxisInfoVisitor<arith::RemUIOp>>();
  visitors.append<BroadcastOpAxisInfoVisitor>();
  visitors.append<SplatOpAxisInfoVisitor>();
  visitors.append<ExpandDimsOpAxisInfoVisitor>();
  visitors.append<CmpOpAxisInfoVisitor<arith::CmpIOp>>();
  visitors.append<LogicalOpAxisInfoVisitor<arith::AndIOp>,
                  LogicalOpAxisInfoVisitor<arith::OrIOp>,
                  LogicalOpAxisInfoVisitor<arith::XOrIOp>>();
  visitors.append<SelectOpAxisInfoVisitor<mlir::arith::SelectOp>>();
  visitors.append<ShLIOpAxisInfoVisitor, ShROpAxisInfoVisitor<arith::ShRUIOp>,
                  ShROpAxisInfoVisitor<arith::ShRSIOp>>();
  visitors.append<MaxMinOpAxisInfoVisitor<arith::MaxSIOp>,
                  MaxMinOpAxisInfoVisitor<arith::MaxUIOp>,
                  MaxMinOpAxisInfoVisitor<arith::MinSIOp>,
                  MaxMinOpAxisInfoVisitor<arith::MinUIOp>>();
  visitors.append<LoadOpAxisInfoVisitor>();
  visitors.append<TransOpAxisInfoVisitor>();

  if (callback)
    callback(visitors);
}

LogicalResult AxisInfoAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<AxisInfo> *> operands,
    ArrayRef<dataflow::Lattice<AxisInfo> *> results) {
  // If any operands are not yet ready, skip this operation for now.
  for (auto op : operands)
    if (op->getValue().getRank() == 0)
      return success();
  AxisInfo curr = visitors.apply(op, operands);
  if (curr.getRank() == 0) {
    setAllToEntryStates(results);
    return success();
  }
  // override with hint
  auto newContiguity = curr.getContiguity();
  auto newDivisibility = curr.getDivisibility();
  auto newConstancy = curr.getConstancy();
  AxisInfo::initDimVectorFromHint(op->getDiscardableAttr("tt.contiguity"),
                                  &newContiguity);
  AxisInfo::initDimVectorFromHint(op->getDiscardableAttr("tt.divisibility"),
                                  &newDivisibility);
  AxisInfo::initDimVectorFromHint(op->getDiscardableAttr("tt.constancy"),
                                  &newConstancy);
  curr = AxisInfo(newContiguity, newDivisibility, newConstancy,
                  curr.getConstantValue());
  // join all lattice elements
  for (auto *result : results)
    propagateIfChanged(result, result->join(curr));
  return success();
}

void AxisInfoAnalysis::visitForOpInductionVar(
    scf::ForOp op, ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices) {
  ProgramPoint *programPoint = getProgramPointAfter(op);
  auto *lbLattice = getLatticeElementFor(programPoint, op.getLowerBound());
  auto *stepLattice = getLatticeElementFor(programPoint, op.getStep());
  // If lb or step is not yet ready, skip this operation for now.
  if (lbLattice->getValue().getRank() == 0 ||
      stepLattice->getValue().getRank() == 0) {
    return;
  }

  AxisInfo::DimVectorT knownContiguity(1, 1);
  AxisInfo::DimVectorT knownDivisibility(1, 1);
  AxisInfo::DimVectorT knownConstancy(1, 1);
  knownDivisibility[0] = gcd(lbLattice->getValue().getDivisibility(0),
                             stepLattice->getValue().getDivisibility(0));
  auto inductionVar =
      AxisInfo(knownContiguity, knownDivisibility, knownConstancy);
  (void)argLattices[0]->join(inductionVar);
}

void AxisInfoAnalysis::visitWarpSpecializeExplicitCaptures(
    gpu::WarpSpecializePartitionsOp ws, const RegionSuccessor &successor,
    ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices) {
  assert(!successor.isParent());
  ProgramPoint *point = getProgramPointAfter(ws);

  for (auto [capture, argLattice] :
       llvm::zip(ws.getParentOp().getExplicitCaptures(), argLattices)) {
    propagateIfChanged(
        argLattice,
        argLattice->join(getLatticeElementFor(point, capture)->getValue()));
  }
}

} // anonymous namespace

void AxisInfo::initPessimisticStateFromFunc(int argNumber,
                                            FunctionOpInterface funcOp,
                                            DimVectorT *contiguity,
                                            DimVectorT *divisibility,
                                            DimVectorT *constancy) {
  // list of attributes that we care about
  SmallVector<std::pair<DimVectorT *, std::string>> retVecs;
  retVecs.push_back({contiguity, "tt.contiguity"});
  retVecs.push_back({divisibility, "tt.divisibility"});
  retVecs.push_back({constancy, "tt.constancy"});
  // initialize attributes one by one
  for (auto [vec, attrName] : retVecs) {
    Attribute attr = funcOp.getArgAttr(argNumber, attrName);
    AxisInfo::initDimVectorFromHint(attr, vec);
  }
}

void AxisInfo::initDimVectorFromHint(Attribute attr, DimVectorT *vec) {
  if (auto int_attr = dyn_cast_or_null<IntegerAttr>(attr))
    *vec = DimVectorT(1, int_attr.getValue().getZExtValue());
  if (auto dense_attr = dyn_cast_or_null<DenseElementsAttr>(attr)) {
    auto vals = dense_attr.getValues<int>();
    *vec = DimVectorT(vals.begin(), vals.end());
  }
}

/*static*/ AxisInfo AxisInfo::getPessimisticValueState(Value value) {
  auto rank = 1;
  if (TensorType ty = dyn_cast<TensorType>(value.getType()))
    rank = ty.getRank();
  if (triton::PointerType ty = dyn_cast<triton::PointerType>(value.getType()))
    if (TensorType elemTy = dyn_cast<TensorType>(ty.getPointeeType()))
      rank = elemTy.getRank();

  DimVectorT knownContiguity(rank, 1);
  DimVectorT knownDivisibility(rank, 1);
  DimVectorT knownConstancy(rank, 1);

  BlockArgument blockArg = dyn_cast<BlockArgument>(value);

  if (blockArg && blockArg.getOwner()->isEntryBlock()) {
    Operation *op = blockArg.getOwner()->getParentOp();
    if (auto fun = dyn_cast<FunctionOpInterface>(op)) {
      initPessimisticStateFromFunc(blockArg.getArgNumber(), fun,
                                   &knownContiguity, &knownDivisibility,
                                   &knownConstancy);
    }
  } else if (Operation *op = value.getDefiningOp()) {
    // Other operations are conservatively initialized with the lowest possible
    // divisibility, contiguity, and constancy unless they have specified.
    AxisInfo::initDimVectorFromHint(op->getDiscardableAttr("tt.divisibility"),
                                    &knownDivisibility);
    AxisInfo::initDimVectorFromHint(op->getDiscardableAttr("tt.contiguity"),
                                    &knownContiguity);
    AxisInfo::initDimVectorFromHint(op->getDiscardableAttr("tt.constancy"),
                                    &knownConstancy);
  }

  return AxisInfo(knownContiguity, knownDivisibility, knownConstancy);
}

/*static*/ AxisInfo AxisInfo::join(const AxisInfo &lhs, const AxisInfo &rhs) {
  // If one argument is not initialized, return the other.
  if (lhs.getRank() == 0)
    return rhs;
  if (rhs.getRank() == 0)
    return lhs;
  assert(lhs.getRank() == rhs.getRank() && "Mismatched ranks");
  DimVectorT contiguity;
  DimVectorT divisibility;
  DimVectorT constancy;
  for (auto d = 0; d < lhs.getRank(); ++d) {
    contiguity.push_back(gcd(lhs.getContiguity(d), rhs.getContiguity(d)));
    divisibility.push_back(getDivisibilityFromContiguity(lhs, rhs, d));
    constancy.push_back(gcd(lhs.getConstancy(d), rhs.getConstancy(d)));
  }
  std::optional<int64_t> constantValue;
  if (lhs.getConstantValue().has_value() &&
      rhs.getConstantValue().has_value() &&
      lhs.getConstantValue() == rhs.getConstantValue())
    constantValue = lhs.getConstantValue();
  return AxisInfo(contiguity, divisibility, constancy, constantValue);
}

unsigned ModuleAxisInfoAnalysis::getContiguity(Value value) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy)
    return 1;
  auto elemTy = tensorTy.getElementType();
  // Get the pointee type if we have a tensor of ptrs to compute contiguity for
  if (auto ptrTy = dyn_cast<PointerType>(elemTy)) {
    elemTy = ptrTy.getPointeeType();
  }
  return getContiguity(value, elemTy.getIntOrFloatBitWidth());
}

unsigned ModuleAxisInfoAnalysis::getContiguity(Value offsetsValue,
                                               unsigned elementBitWidth) {
  // FIXME: This is not as good as it could be, as we don't need to restrict
  // the analysis to one dimension. We should determine contiguity on the
  // flattenOuts() layout
  auto tensorTy = cast<RankedTensorType>(offsetsValue.getType());
  auto linAttr = gpu::toLinearEncoding(tensorTy);
  auto order = linAttr.getOrder();
  unsigned align = getAlignment(offsetsValue, elementBitWidth);

  auto uniqueContigPerThread = linAttr.getContigPerThread();
  assert(order[0] < uniqueContigPerThread.size() &&
         "Unexpected uniqueContigPerThread size");
  unsigned contiguity = uniqueContigPerThread[order[0]];
  LDBG("getContiguity uniqueContigPerThread = " << contiguity);
  contiguity = std::min(align, contiguity);

  return contiguity;
}

unsigned ModuleAxisInfoAnalysis::getAlignment(Value value) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy)
    return 1;

  auto elemTy = tensorTy.getElementType();
  // Get the pointee type if we have a tensor of ptrs to compute contiguity for
  if (auto ptrTy = dyn_cast<PointerType>(elemTy)) {
    elemTy = ptrTy.getPointeeType();
  }
  return getAlignment(value, elemTy.getIntOrFloatBitWidth());
}

unsigned ModuleAxisInfoAnalysis::getAlignment(Value offsetsValue,
                                              unsigned elementBitWidth) {
  auto tensorTy = cast<RankedTensorType>(offsetsValue.getType());
  auto *axisInfo = getAxisInfo(offsetsValue);
  if (!axisInfo)
    return 1;
  auto linAttr = gpu::toLinearEncoding(tensorTy);
  auto order = linAttr.getOrder();

  auto divisibility = axisInfo->getDivisibility(order[0]);
  auto elemNumBytes = std::max<unsigned>(elementBitWidth / 8, 1);
  auto elemTy = tensorTy.getElementType();
  auto maxMultiple = isa<PointerType>(elemTy)
                         ? std::max<int64_t>(divisibility / elemNumBytes, 1)
                         : divisibility;

  auto maxContig = axisInfo->getContiguity(order[0]);
  unsigned alignment = std::min(maxMultiple, maxContig);
  LDBG("getAlignment order[0] " << order[0] << " maxContig = " << maxContig
                                << " elemNumBits = " << elementBitWidth
                                << " maxMultiple = " << maxMultiple
                                << " alignment " << alignment);
  LLVM_DEBUG({
    std::string axisStr;
    llvm::raw_string_ostream os(axisStr);
    axisInfo->print(os);
    LDBG("-- " << axisStr);
  });
  return alignment;
}

unsigned ModuleAxisInfoAnalysis::getMaskAlignment(Value mask) {
  auto tensorTy = dyn_cast<RankedTensorType>(mask.getType());
  if (!tensorTy)
    return 1;
  auto *axisInfo = getAxisInfo(mask);
  if (!axisInfo)
    return 1;
  auto linAttr = gpu::toLinearEncoding(tensorTy);
  auto maskOrder = linAttr.getOrder();
  auto alignment = std::max<unsigned>(axisInfo->getConstancy(maskOrder[0]), 1);
  LDBG("getMaskAlignment maskOrder[0] " << maskOrder[0] << " alignment "
                                        << alignment);
  LLVM_DEBUG({
    std::string axisStr;
    llvm::raw_string_ostream os(axisStr);
    axisInfo->print(os);
    LDBG("-- " << axisStr);
  });
  return alignment;
}

void ModuleAxisInfoAnalysis::initialize(FunctionOpInterface funcOp,
                                        axisinfo::CallbackType callback) {
  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  AxisInfoAnalysis *analysis = solver->load<AxisInfoAnalysis>(callback);
  if (failed(solver->initializeAndRun(funcOp)))
    return;

  auto *axisInfoMap = getFuncData(funcOp);
  auto updateAxisInfoMap = [&](Value value) {
    auto axisInfo = analysis->getLatticeElement(value)->getValue();
    // If we could not determine the AxisInfo for this value, assume the
    // pessimistic state.
    if (axisInfo.getRank() == 0)
      axisInfo = AxisInfo::getPessimisticValueState(value);
    auto &valInfo = (*axisInfoMap)[value];
    valInfo = AxisInfo::join(axisInfo, valInfo);
  };
  funcOp.walk([&](Operation *op) {
    for (auto value : op->getResults()) {
      updateAxisInfoMap(value);
    }
  });
  funcOp.walk([&](Block *block) {
    for (auto value : block->getArguments()) {
      updateAxisInfoMap(value);
    }
  });
}

void ModuleAxisInfoAnalysis::update(CallOpInterface callOp,
                                    FunctionOpInterface callee) {
  auto caller = callOp->getParentOfType<FunctionOpInterface>();
  auto *axisInfoMap = getFuncData(caller);
  for (auto entry : llvm::enumerate(callOp->getOperands())) {
    auto index = entry.index();
    auto value = entry.value();
    auto setAttrFn = [&](StringRef attrName, int64_t prevValue) {
      auto curValue = kMaxDivisor;
      if (callee.getArgAttrOfType<IntegerAttr>(index, attrName)) {
        curValue =
            callee.getArgAttrOfType<IntegerAttr>(index, attrName).getInt();
      }
      auto attr = IntegerAttr::get(IntegerType::get(callee.getContext(), 64),
                                   gcd(prevValue, curValue));
      callee.setArgAttr(index, attrName, attr);
    };
    auto axisInfo = axisInfoMap->lookup(value);
    // Only scalar arguments are supported. Do not forward multi-dimensional
    // AxisInfo to the callee.
    if (axisInfo.getRank() != 1)
      continue;
    setAttrFn("tt.contiguity", axisInfo.getContiguity(0));
    setAttrFn("tt.divisibility", axisInfo.getDivisibility(0));
    setAttrFn("tt.constancy", axisInfo.getConstancy(0));
  }
}

} // namespace mlir::triton
