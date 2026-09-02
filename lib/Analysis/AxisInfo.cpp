#include "triton/Analysis/AxisInfo.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <numeric>

#define DEBUG_TYPE "axis-info"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {
namespace {

constexpr int64_t kMaxDivisor = highestPowOf2Divisor<int64_t>(0);
constexpr unsigned kMaxDivisorLog2 = std::numeric_limits<int64_t>::digits - 1;

using ConstantInt = std::optional<APInt>;

unsigned getIntegerBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type)))
    return intType.getWidth();
  return 0;
}

// Keep constants exact on defined executions. Generic operation folders can
// refine poison, so check the integer operations' poison-producing flags here.
class ConstantEvaluator {
public:
  ConstantEvaluator(Operation *operation,
                    ArrayRef<const dataflow::Lattice<AxisInfo> *> operands)
      : operation(operation), operands(operands),
        resultWidth(operation->getNumResults() == 1
                        ? getIntegerBitWidth(operation->getResult(0).getType())
                        : 0) {
    assert(operands.size() == operation->getNumOperands());
  }

  ConstantInt run() const {
    if (!resultWidth)
      return std::nullopt;

    return llvm::TypeSwitch<Operation *, ConstantInt>(operation)
        .Case<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp,
              arith::TruncIOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
              arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp,
              arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::ShLIOp,
              arith::ShRSIOp, arith::ShRUIOp, arith::MinSIOp, arith::MaxSIOp,
              arith::MinUIOp, arith::MaxUIOp, arith::CmpIOp, arith::SelectOp>(
            [&](auto op) { return evaluate(op); })
        .Default([](Operation *) -> ConstantInt { return std::nullopt; });
  }

private:
  const APInt *constantOperand(unsigned index) const {
    assert(index < operands.size());
    const auto &constant = operands[index]->getValue().getConstantValue();
    assert((!constant ||
            constant->getBitWidth() ==
                getIntegerBitWidth(operation->getOperand(index).getType())) &&
           "constant width must match the operand type");
    return constant ? &*constant : nullptr;
  }

  template <typename Fn> ConstantInt evaluateBinary(Fn &&fn) const {
    const APInt *lhs = constantOperand(0);
    const APInt *rhs = constantOperand(1);
    if (!lhs || !rhs || lhs->getBitWidth() != rhs->getBitWidth())
      return std::nullopt;
    return fn(*lhs, *rhs);
  }

  using OverflowOp = APInt (APInt::*)(const APInt &, bool &) const;

  template <typename OpTy>
  ConstantInt evaluateOverflowing(OpTy op, OverflowOp signedOp,
                                  OverflowOp unsignedOp) const {
    return evaluateBinary([&](const APInt &a, const APInt &b) -> ConstantInt {
      bool overflow;
      APInt result = (a.*unsignedOp)(b, overflow);
      if (op.hasNoUnsignedWrap() && overflow)
        return std::nullopt;
      if (op.hasNoSignedWrap()) {
        (void)(a.*signedOp)(b, overflow);
        if (overflow)
          return std::nullopt;
      }
      return result;
    });
  }

  template <typename OpTy> ConstantInt evaluateDivision(OpTy op) const {
    return evaluateBinary([&](const APInt &a, const APInt &b) -> ConstantInt {
      if (b.isZero())
        return std::nullopt;
      if constexpr (std::is_same_v<OpTy, arith::DivSIOp>) {
        bool overflow;
        APInt result = a.sdiv_ov(b, overflow);
        if (overflow || (op.getIsExact() && !a.srem(b).isZero()))
          return std::nullopt;
        return result;
      } else {
        if (op.getIsExact() && !a.urem(b).isZero())
          return std::nullopt;
        return a.udiv(b);
      }
    });
  }

  template <typename OpTy> ConstantInt evaluateRemainder(OpTy op) const {
    const APInt *rhs = constantOperand(1);
    // lhs % 1 = 0; signed remainder by -1 is also zero.
    if (rhs && (rhs->isOne() || (isa<arith::RemSIOp>(op) && rhs->isAllOnes())))
      return APInt(resultWidth, 0);
    return evaluateBinary([&](const APInt &a, const APInt &b) -> ConstantInt {
      if (b.isZero())
        return std::nullopt;
      if constexpr (std::is_same_v<OpTy, arith::RemSIOp>)
        return a.srem(b);
      else
        return a.urem(b);
    });
  }

  template <typename OpTy> ConstantInt evaluateRightShift(OpTy op) const {
    return evaluateBinary([&](const APInt &a, const APInt &b) -> ConstantInt {
      if (b.uge(a.getBitWidth()))
        return std::nullopt;
      unsigned shift = b.getZExtValue();
      if (op.getIsExact() && a.countr_zero() < shift)
        return std::nullopt;
      if constexpr (std::is_same_v<OpTy, arith::ShRSIOp>)
        return a.ashr(shift);
      else
        return a.lshr(shift);
    });
  }

  ConstantInt evaluate(arith::ConstantOp op) const {
    APInt value;
    if (matchPattern(op.getValue(), m_ConstantInt(&value)))
      return value;
    return std::nullopt;
  }

  ConstantInt evaluate(arith::ExtSIOp) const {
    const APInt *value = constantOperand(0);
    return value ? ConstantInt(value->sext(resultWidth)) : std::nullopt;
  }

  ConstantInt evaluate(arith::ExtUIOp op) const {
    const APInt *value = constantOperand(0);
    if (!value || (op.getNonNeg() && value->isNegative()))
      return std::nullopt;
    return value->zext(resultWidth);
  }

  ConstantInt evaluate(arith::TruncIOp op) const {
    const APInt *value = constantOperand(0);
    if (!value || (op.hasNoUnsignedWrap() && !value->isIntN(resultWidth)) ||
        (op.hasNoSignedWrap() && !value->isSignedIntN(resultWidth)))
      return std::nullopt;
    return value->trunc(resultWidth);
  }

  ConstantInt evaluate(arith::AddIOp op) const {
    return evaluateOverflowing(op, &APInt::sadd_ov, &APInt::uadd_ov);
  }

  ConstantInt evaluate(arith::SubIOp op) const {
    return evaluateOverflowing(op, &APInt::ssub_ov, &APInt::usub_ov);
  }

  ConstantInt evaluate(arith::MulIOp op) const {
    const APInt *lhs = constantOperand(0);
    const APInt *rhs = constantOperand(1);
    if ((lhs && lhs->isZero()) || (rhs && rhs->isZero()))
      return APInt(resultWidth, 0);
    return evaluateOverflowing(op, &APInt::smul_ov, &APInt::umul_ov);
  }

  ConstantInt evaluate(arith::DivSIOp op) const { return evaluateDivision(op); }

  ConstantInt evaluate(arith::DivUIOp op) const { return evaluateDivision(op); }

  ConstantInt evaluate(arith::RemSIOp op) const {
    return evaluateRemainder(op);
  }

  ConstantInt evaluate(arith::RemUIOp op) const {
    return evaluateRemainder(op);
  }

  ConstantInt evaluate(arith::AndIOp) const {
    return evaluateBinary([](const APInt &a, const APInt &b) { return a & b; });
  }

  ConstantInt evaluate(arith::OrIOp) const {
    return evaluateBinary([](const APInt &a, const APInt &b) { return a | b; });
  }

  ConstantInt evaluate(arith::XOrIOp) const {
    return evaluateBinary([](const APInt &a, const APInt &b) { return a ^ b; });
  }

  ConstantInt evaluate(arith::ShLIOp op) const {
    const APInt *lhs = constantOperand(0);
    const APInt *rhs = constantOperand(1);
    if (!lhs || !rhs || rhs->uge(lhs->getBitWidth()))
      return std::nullopt;
    return evaluateOverflowing(op, &APInt::sshl_ov, &APInt::ushl_ov);
  }

  ConstantInt evaluate(arith::ShRSIOp op) const {
    return evaluateRightShift(op);
  }

  ConstantInt evaluate(arith::ShRUIOp op) const {
    return evaluateRightShift(op);
  }

  ConstantInt evaluate(arith::MinSIOp) const {
    return evaluateBinary(llvm::APIntOps::smin);
  }

  ConstantInt evaluate(arith::MaxSIOp) const {
    return evaluateBinary(llvm::APIntOps::smax);
  }

  ConstantInt evaluate(arith::MinUIOp) const {
    return evaluateBinary(llvm::APIntOps::umin);
  }

  ConstantInt evaluate(arith::MaxUIOp) const {
    return evaluateBinary(llvm::APIntOps::umax);
  }

  ConstantInt evaluate(arith::CmpIOp op) const {
    return evaluateBinary([&](const APInt &a, const APInt &b) {
      return APInt(1, arith::applyCmpPredicate(op.getPredicate(), a, b));
    });
  }

  ConstantInt evaluate(arith::SelectOp) const {
    const APInt *condition = constantOperand(0);
    const APInt *trueValue = constantOperand(1);
    const APInt *falseValue = constantOperand(2);
    if (condition) {
      const APInt *selected = condition->isZero() ? falseValue : trueValue;
      return selected ? ConstantInt(*selected) : std::nullopt;
    }
    if (trueValue && falseValue &&
        trueValue->getBitWidth() == falseValue->getBitWidth() &&
        *trueValue == *falseValue)
      return *trueValue;
    return std::nullopt;
  }

  Operation *operation;
  ArrayRef<const dataflow::Lattice<AxisInfo> *> operands;
  unsigned resultWidth;
};

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
    const AxisInfo &lhsInfo = operands[0]->getValue();
    const AxisInfo &rhsInfo = operands[1]->getValue();
    auto rank = lhsInfo.getRank();
    assert(isa<RankedTensorType>(op.getType()) ||
           rank == 1 && "Expected ranked tensor or scalar");
    assert(operands.size() == 2 && "Expected two operands");
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;
    for (auto d = 0; d < rank; ++d) {
      contiguity.push_back(getContiguity(op, lhsInfo, rhsInfo, d));
      constancy.push_back(getConstancy(op, lhsInfo, rhsInfo, d));
      divisibility.push_back(getDivisibility(op, lhsInfo, rhsInfo, d));
    }
    return AxisInfo(contiguity, divisibility, constancy);
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
};

template <typename OpTy>
class IntegerCastOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    const AxisInfo &info = operands[0]->getValue();
    // Exact casts are handled before the bound visitors. An unproven or poison
    // cast must not retain its operand's constant value.
    return AxisInfo(info.getContiguity(), info.getDivisibility(),
                    info.getConstancy());
  }
};

template <typename OpTy>
class IdentityOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

class BitcastOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<triton::BitcastOp> {
public:
  using AxisInfoVisitorImpl<triton::BitcastOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(triton::BitcastOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    AxisInfo info = operands[0]->getValue();
    if (!isa<PointerType>(getElementTypeOrSelf(op.getSrc().getType())))
      return info;

    int64_t srcBytes =
        std::max<int64_t>(1, getPointeeBitWidth(op.getSrc().getType()) / 8);
    int64_t dstBytes =
        std::max<int64_t>(1, getPointeeBitWidth(op.getType()) / 8);
    if (srcBytes == dstBytes)
      return info;

    AxisInfo::DimVectorT divisibility = info.getDivisibility();
    // Every former element becomes a group base after the bitcast.
    for (unsigned dim = 0; dim < divisibility.size(); ++dim)
      if (info.getContiguity(dim) > 1)
        divisibility[dim] = gcd(divisibility[dim], srcBytes);
    return AxisInfo(AxisInfo::DimVectorT(info.getRank(), 1), divisibility,
                    info.getConstancy(), info.getConstantValue());
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
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return {};
    auto tensorType = dyn_cast<RankedTensorType>(op.getResultTypes()[0]);
    if (tensorType &&
        tensorType.getRank() != operands[0]->getValue().getRank()) {
      // Do not propagate AxisInfo with incorrect rank. This can cause a crash
      // in future visitor applications.
      return AxisInfo::getPessimisticValueState(op->getResult(0));
    }
    const AxisInfo &info = operands[0]->getValue();
    if (op.getOperand(0).getType() == op.getResult(0).getType())
      return info;
    return AxisInfo(info.getContiguity(), info.getDivisibility(),
                    info.getConstancy());
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
    if (isa<arith::SubIOp>(op)) {
      // Case 1: If contiguity(lhs) > 1 and contiguity(rhs) > 1,
      // x_t - y_t = (base_x + t) - (base_y + t) = base_x - base_y for any
      // 0 <= t < min(contig_x, contig_y), so contiguity is 1.
      // Case 2: If contiguity(lhs) > 1 and contiguity(rhs) == 1,
      // x_t - y = (base_x + t) - base_y = base_x - base_y + t for any
      // 0 <= t < contig_x,
      // the contiguity depends on the constancy of rhs.
      // Case 3: If contiguity(lhs) == 1 and contiguity(rhs) > 1,
      // x - y_t = base_x - (base_y + t) = base_x - base_y - t for any
      // 0 <= t < contig_y. The result is decreasing within the contiguous
      // block, so contiguity is 1.
      // Case 4: If contiguity(lhs) == 1 and contiguity(rhs) == 1,
      // x - y = base_x - base_y, so contiguity is 1.
      return gcd(lhs.getContiguity(dim), rhs.getConstancy(dim));
    }
    // For AddIOp and AddPtrOp
    // Case 1: If contiguity(lhs) > 1 and contiguity(rhs) > 1,
    // x_t + y_t = (base_x + t) + (base_y + t) = base_x + base_y + 2t for any
    // 0 <= t < min(contig_x, contig_y),
    // so contiguity is 1.
    // Case 2: If contiguity(lhs) > 1 and contiguity(rhs) == 1,
    // x_t + y = (base_x + t) + base_y = base_x + base_y + t for any
    // 0 <= t < contig_x, so contiguity depends on constancy of rhs.
    // Case 3: If contiguity(lhs) == 1 and contiguity(rhs) > 1,
    // It's symmetric to case B.
    // Case 4: If contiguity(lhs) == 1 and contiguity(rhs) == 1,
    // It's trivial that contiguity is 1
    return std::max(gcd(lhs.getConstancy(dim), rhs.getContiguity(dim)),
                    gcd(lhs.getContiguity(dim), rhs.getConstancy(dim)));
  }

  int64_t getDivisibility(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                          int dim) override {
    int64_t elemSize = 1;
    auto lhsDivisibility = lhs.getDivisibility(dim);
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
      elemSize = std::max<int64_t>(
          1, triton::getPointeeBitWidth(op.getPtr().getType()) / 8);
      rhsDivisibility = multiplyDivisor(rhs.getDivisibility(dim), elemSize);
    }
    if (lhs.getContiguity(dim) > 1 && rhs.getContiguity(dim) > 1) {
      // If both operands are contiguous, the in-group offsets are:
      // Let lhs_t = base_lhs + t and rhs_t = base_rhs + t for any
      // 0 <= t < min(contig_lhs, contig_rhs).
      // For addition:
      //   lhs_t + rhs_t = base_lhs + base_rhs + 2t
      // For subtraction:
      //   lhs_t - rhs_t = base_lhs - base_rhs
      if constexpr (std::is_same_v<OpTy, arith::SubIOp>) {
        if (lhs.getContiguity(dim) == rhs.getContiguity(dim))
          return gcd(lhsDivisibility, rhsDivisibility);
      }
      if ((lhsDivisibility % 2 == 0 && rhsDivisibility % 2 == 0)) {
        // Both even -> result divisible by 2.
        return 2;
      } else {
        // At least one is odd -> the "lower bound" of divisibility is 1.
        return 1;
      }
    } else {
      // At least one operand is partially constant.
      // Divisibility is defined on the *first element* of a contiguity
      // group. When an operand has contiguity larger than the result
      // contiguity, the "first element of a result group" can fall inside an
      // operand's contiguity group, so we must clamp the operand divisibility
      // accordingly (otherwise we can overestimate alignment).
      if (lhs.getContiguity(dim) > 1 || rhs.getContiguity(dim) > 1) {
        auto resContiguity = getContiguity(op, lhs, rhs, dim);
        return gcd(lhsDivisibility, rhsDivisibility,
                   multiplyDivisor(resContiguity, elemSize));
      } else {
        return gcd(lhsDivisibility, rhsDivisibility);
      }
    }
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
        rhs.getConstantValue() && rhs.getConstantValue()->isOne()
            ? lhs.getContiguity(dim)
            : 1;
    // 1 * rhs = rhs
    auto rhsContiguity =
        lhs.getConstantValue() && lhs.getConstantValue()->isOne()
            ? rhs.getContiguity(dim)
            : 1;
    return std::max(lhsContiguity, rhsContiguity);
  }

  int64_t getDivisibility(arith::MulIOp op, const AxisInfo &lhs,
                          const AxisInfo &rhs, int dim) override {
    auto lhsDivisibility = lhs.getDivisibility(dim);
    if (lhs.getContiguity(dim) > 1 &&
        (!rhs.getConstantValue() || !rhs.getConstantValue()->isOne())) {
      // If the operand is contiguous, the divisibility of the
      // sequence drops to 1.
      // Example: [4, 5, 6, 7] (base 4 divisible by 4).
      // Multiplying by 2 yields [8, 10, 12, 14] (GCD=2).
      // Preserving divisibility=4 implies result align 8 (unsafe).
      lhsDivisibility = 1;
    }
    auto rhsDivisibility = rhs.getDivisibility(dim);
    if (rhs.getContiguity(dim) > 1 &&
        (!lhs.getConstantValue() || !lhs.getConstantValue()->isOne())) {
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      rhsDivisibility = 1;
    }
    return multiplyDivisor(lhsDivisibility, rhsDivisibility);
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
    return rhs.getConstantValue() && rhs.getConstantValue()->isOne()
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
    if (lhs.getConstantValue() && lhs.getConstantValue()->isZero())
      return lhs.getDivisibility(dim);
    // Case 2: rhs is 1
    if (rhs.getConstantValue() && rhs.getConstantValue()->isOne())
      return lhs.getDivisibility(dim);
    // Case 3: lhs has contiguity of 1 in this dimension and rhs is a power of 2
    if (rhs.getConstantValue() && lhs.getContiguity(dim) == 1) {
      APInt divisor = *rhs.getConstantValue();
      if constexpr (std::is_same_v<OpTy, arith::DivSIOp>)
        if (divisor.isNegative())
          divisor = -divisor;
      if (divisor.isPowerOf2()) {
        unsigned shift = std::min(divisor.countr_zero(), kMaxDivisorLog2);
        return std::max<int64_t>(1, lhs.getDivisibility(dim) >> shift);
      }
    }
    // otherwise: return 1
    return 1;
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
    auto contiguity = getContiguity(op, lhs, rhs, dim);
    auto divisibility = gcd(lhs.getDivisibility(dim), rhs.getDivisibility(dim));
    // New group bases inside an operand's contiguous group have offsets that
    // are multiples of the result contiguity.
    if (lhs.getContiguity(dim) > contiguity ||
        rhs.getContiguity(dim) > contiguity)
      divisibility = gcd(divisibility, contiguity);
    return divisibility;
  };
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
    // The tensor is constant.
    if (opInfo.getConstantValue())
      return AxisInfo::getConstantValueState(op.getType(),
                                             *opInfo.getConstantValue());
    AxisInfo::DimVectorT contiguity = opInfo.getContiguity();
    AxisInfo::DimVectorT divisibility = opInfo.getDivisibility();
    AxisInfo::DimVectorT constancy = opInfo.getConstancy();
    int64_t newDivisibility = 1;
    if (opInfo.getRank()) {
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
    return AxisInfo(contiguity, divisibility, constancy);
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

class ReshapeOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<triton::ReshapeOp> {
public:
  using AxisInfoVisitorImpl<triton::ReshapeOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(triton::ReshapeOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    AxisInfo srcInfo = operands[0]->getValue();
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = cast<RankedTensorType>(op.getType());
    auto dstShape = dstTy.getShape();

    // Constant tensor stays constant
    if (srcInfo.getConstantValue())
      return AxisInfo::getConstantValueState(dstTy,
                                             *srcInfo.getConstantValue());

    auto srcShape = srcTy.getShape();
    // `suffixProducts[d + 1]` is the flat stride of axis `d` in row-major
    // order.
    auto getSuffixProducts = [](ArrayRef<int64_t> shape) {
      SmallVector<int64_t> suffixProducts(shape.size() + 1, 1);
      for (int d = shape.size() - 1; d >= 0; --d)
        suffixProducts[d] = suffixProducts[d + 1] * shape[d];
      return suffixProducts;
    };
    auto srcSuffixProducts = getSuffixProducts(srcShape);
    auto dstSuffixProducts = getSuffixProducts(dstShape);

    // Unit contiguity makes every element a group base, so divisibility from
    // such a dimension applies globally. AxisInfo normalization propagates
    // this fact within one value, but reshape can lose the source axis that
    // carries it, so seed every destination axis before refining it below.
    int64_t globalDivisibility = srcInfo.getGlobalDivisibility();

    AxisInfo::DimVectorT contiguity(dstTy.getRank(), 1);
    AxisInfo::DimVectorT divisibility(dstTy.getRank(), globalDivisibility);
    AxisInfo::DimVectorT constancy(dstTy.getRank(), 1);

    for (int dstDim = 0; dstDim < dstTy.getRank(); ++dstDim) {
      int64_t dstStride = dstSuffixProducts[dstDim + 1];
      // Main idea:
      // Let m = dstDim, and Q = dstSuffixProducts, P = srcSuffixProducts
      // Q[m + 1] \in [P[i + 1], P[i]).
      // This means that dimension m is splitting dimension i, so it often
      // inherits the properties of this dimension
      // Note that the "off by one" indexing comes from the fact that the
      // stride for dimension m is in Q[m + 1].
      int srcDim = 0;
      for (; srcDim < srcTy.getRank(); ++srcDim) {
        int64_t srcStride = srcSuffixProducts[srcDim + 1];
        if (srcStride <= dstStride && dstStride < srcSuffixProducts[srcDim])
          break;
      }

      if (srcDim == srcTy.getRank()) {
        // If there are 1-sized axes at the beginning, we do not have
        // dstStride < srcSuffixProducts[srcDim] but we can still reuse
        // the outermost source axis.
        assert(dstShape[dstDim] == 1);
        srcDim = 0;
      }

      int64_t srcStride = srcSuffixProducts[srcDim + 1];
      int64_t srcContiguity = srcInfo.getContiguity(srcDim);
      int64_t srcDivisibility = srcInfo.getDivisibility(srcDim);
      int64_t srcConstancy = srcInfo.getConstancy(srcDim);

      if (srcContiguity > 1) {
        // Contiguity only survives when reshape lands on the low boundary of
        // the source axis. Starting inside the axis loses the unit-stride run.
        if (dstStride == srcStride) {
          int64_t dstContiguity = std::min(srcContiguity, dstShape[dstDim]);
          contiguity[dstDim] = dstContiguity;
          // If the whole contiguous run survives, the group bases are
          // unchanged. When the run is truncated, later group bases can start
          // inside the original run, so divisibility must be clamped
          // accordingly.
          int64_t dstDivisibility =
              dstContiguity == srcContiguity
                  ? srcDivisibility
                  : std::min(srcDivisibility, dstContiguity);
          divisibility[dstDim] = std::max(globalDivisibility, dstDivisibility);
        }
        continue;
      }

      int64_t constancyEnd = srcStride * srcConstancy;
      if (dstStride <= constancyEnd) {
        // If we land inside a constant axis, the constancy is the minimum
        // between the shape and how much constancy survives.
        int64_t dstConstancy =
            std::min<int64_t>(dstShape[dstDim], constancyEnd / dstStride);

        // Several constant dimensions can merge into a single constant
        // dimension
        int64_t remainingSize = dstShape[dstDim] / dstConstancy;
        for (int dim = srcDim - 1;
             srcConstancy == srcShape[srcDim] && dim >= 0 && remainingSize > 1;
             --dim) {
          int64_t pieceSize = std::min(srcShape[dim], remainingSize);
          int64_t pieceConstancy =
              std::min(srcInfo.getConstancy(dim), pieceSize);
          dstConstancy *= pieceConstancy;
          if (pieceConstancy < pieceSize)
            break;
          remainingSize /= pieceSize;
        }
        constancy[dstDim] = dstConstancy;
      }
    }

    return AxisInfo(contiguity, divisibility, constancy);
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
    for (short d = 0; d < rank; ++d) {
      // Case 1: lhs and rhs are both partial constants
      int64_t constHint = gcd(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d));
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

      constancy.push_back(constHint);
      divisibility.push_back(1);
      contiguity.push_back(1);
    }

    return AxisInfo(contiguity, divisibility, constancy);
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
};

template <typename OpTy>
class SelectOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    const AxisInfo &condInfo = operands[0]->getValue();
    const AxisInfo &lhsInfo = operands[1]->getValue();
    const AxisInfo &rhsInfo = operands[2]->getValue();
    if (const auto &condition = condInfo.getConstantValue())
      return condition->isZero() ? rhsInfo : lhsInfo;

    // The condition can be either a tensor or i1.
    // If i1 is used as the condition, the entire tensor of either
    // lhs or rhs is selected.
    if (isa<IntegerType>(op.getOperand(0).getType()))
      return AxisInfo::join(lhsInfo, rhsInfo);

    AxisInfo::DimVectorT contiguity, divisibility, constancy;
    for (int d = 0; d < lhsInfo.getRank(); ++d) {
      constancy.push_back(gcd(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d),
                              condInfo.getConstancy(d)));
      contiguity.push_back(gcd(lhsInfo.getContiguity(d),
                               rhsInfo.getContiguity(d),
                               condInfo.getConstancy(d)));
      // getDivisibilityFromContiguity does not see condConstancy; clamp
      // by the just-computed output contiguity so the result remains
      // sound when condConstancy reduces it below the input contiguities.
      divisibility.push_back(
          gcd(getDivisibilityFromContiguity(lhsInfo, rhsInfo, d),
              contiguity.back()));
    }

    return AxisInfo(contiguity, divisibility, constancy);
  }
};

class ShLIOpAxisInfoVisitor final : public BinaryOpVisitorImpl<arith::ShLIOp> {
public:
  using BinaryOpVisitorImpl<arith::ShLIOp>::BinaryOpVisitorImpl;

private:
  int64_t getContiguity(arith::ShLIOp op, const AxisInfo &lhs,
                        const AxisInfo &rhs, int dim) override {
    if (rhs.getConstantValue() && rhs.getConstantValue()->isZero())
      return lhs.getContiguity(dim);
    else
      return 1;
  }

  int64_t getDivisibility(arith::ShLIOp op, const AxisInfo &lhs,
                          const AxisInfo &rhs, int dim) override {
    const auto &amount = rhs.getConstantValue();
    auto lhsDivisibility = lhs.getDivisibility(dim);
    if (lhs.getContiguity(dim) > 1 && (!amount || !amount->isZero())) {
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      lhsDivisibility = 1;
    }
    if (!amount)
      return lhsDivisibility;
    if (amount->uge(getIntegerBitWidth(op.getType())))
      return 1;
    unsigned shift = amount->getLimitedValue(kMaxDivisorLog2);
    return multiplyDivisor(lhsDivisibility, int64_t{1} << shift);
  }
};

template <typename OpTy>
class ShROpAxisInfoVisitor final : public BinaryOpVisitorImpl<OpTy> {
public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

private:
  int64_t getContiguity(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                        int dim) override {
    if (rhs.getConstantValue() && rhs.getConstantValue()->isZero())
      return lhs.getContiguity(dim);
    else
      return 1;
  }

  int64_t getDivisibility(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                          int dim) override {
    const auto &amount = rhs.getConstantValue();
    if (!amount || amount->uge(getIntegerBitWidth(op.getType())))
      return 1;
    auto lhsDivisibility = lhs.getDivisibility(dim);
    if (lhs.getContiguity(dim) > 1 && !amount->isZero()) {
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      lhsDivisibility = 1;
    }
    unsigned shift = amount->getLimitedValue(kMaxDivisorLog2);
    return std::max<int64_t>(1, lhsDivisibility >> shift);
  }
};

template <typename OpTy>
class MaxMinOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    return AxisInfo::join(operands[0]->getValue(), operands[1]->getValue());
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

} // anonymous namespace

//===----------------------------------------------------------------------===//
// AxisInfoAnalysis
//===----------------------------------------------------------------------===//

AxisInfoAnalysis::AxisInfoAnalysis(DataFlowSolver &solver)
    : dataflow::SparseForwardDataFlowAnalysis<dataflow::Lattice<AxisInfo>>(
          solver) {
  // UnrealizedConversionCast:
  // This is needed by TritonGPUToLLVM, to get AxisInfo when the graph is
  // in the process of a PartialConversion, where UnrealizedConversionCast
  // may exist
  visitors.append<UnrealizedConversionCastOpAxisInfoVisitor>();
  visitors.append<IntegerCastOpAxisInfoVisitor<arith::ExtSIOp>,
                  IntegerCastOpAxisInfoVisitor<arith::ExtUIOp>,
                  IntegerCastOpAxisInfoVisitor<arith::TruncIOp>,
                  IdentityOpAxisInfoVisitor<triton::gpu::ConvertLayoutOp>,
                  BitcastOpAxisInfoVisitor,
                  IdentityOpAxisInfoVisitor<triton::gluon::SetAutoLayoutOp>>();
  visitors.append<MakeRangeOpAxisInfoVisitor>();
  visitors.append<PoisonOpAxisInfoVisitor>();
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
  visitors.append<ReshapeOpAxisInfoVisitor>();
  visitors.append<CmpOpAxisInfoVisitor<arith::CmpIOp>>();
  visitors.append<BinaryOpVisitorImpl<arith::AndIOp>,
                  BinaryOpVisitorImpl<arith::OrIOp>,
                  BinaryOpVisitorImpl<arith::XOrIOp>>();
  visitors.append<SelectOpAxisInfoVisitor<mlir::arith::SelectOp>>();
  visitors.append<ShLIOpAxisInfoVisitor, ShROpAxisInfoVisitor<arith::ShRUIOp>,
                  ShROpAxisInfoVisitor<arith::ShRSIOp>>();
  visitors.append<MaxMinOpAxisInfoVisitor<arith::MaxSIOp>,
                  MaxMinOpAxisInfoVisitor<arith::MaxUIOp>,
                  MaxMinOpAxisInfoVisitor<arith::MinSIOp>,
                  MaxMinOpAxisInfoVisitor<arith::MinUIOp>>();
  visitors.append<LoadOpAxisInfoVisitor>();
  visitors.append<TransOpAxisInfoVisitor>();
}

void AxisInfoAnalysis::setToEntryState(dataflow::Lattice<AxisInfo> *lattice) {
  propagateIfChanged(lattice, lattice->join(AxisInfo::getPessimisticValueState(
                                  lattice->getAnchor())));
}

void AxisInfoAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor & /*successor*/,
    ValueRange /*nonSuccessorInputs*/,
    ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    visitForOpInductionVar(forOp, argLattices);
  } else {
    setAllToEntryStates(argLattices);
  }
}

LogicalResult AxisInfoAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<AxisInfo> *> operands,
    ArrayRef<dataflow::Lattice<AxisInfo> *> results) {
  if (results.empty())
    return success();
  // If any operands are not yet ready, skip this operation for now.
  for (auto op : operands)
    if (op->getValue().getRank() == 0)
      return success();
  AxisInfo curr;
  if (ConstantInt constant = ConstantEvaluator(op, operands).run())
    curr = AxisInfo::getConstantValueState(op->getResult(0).getType(),
                                           std::move(*constant));
  else {
    curr = visitors.apply(op, operands);
    if (const auto &constant = curr.getConstantValue()) {
      if (op->getNumResults() == 1 &&
          getIntegerBitWidth(op->getResult(0).getType()) ==
              constant->getBitWidth()) {
        // Shape and layout operations can forward a known splat. Recover its
        // strongest bounds for the result shape.
        curr = AxisInfo::getConstantValueState(op->getResult(0).getType(),
                                               *constant);
      } else {
        curr = AxisInfo(curr.getContiguity(), curr.getDivisibility(),
                        curr.getConstancy());
      }
    }
  }
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

/*static*/ AxisInfo AxisInfo::getConstantValueState(Type type, APInt value) {
  assert(cast<IntegerType>(getElementTypeOrSelf(type)).getWidth() ==
             value.getBitWidth() &&
         "constant width must match the result type");
  auto tensorType = dyn_cast<RankedTensorType>(type);
  DimVectorT constancy =
      tensorType ? to_vector(tensorType.getShape()) : DimVectorT(1, 1);
  unsigned rank = constancy.size();
  int64_t divisibility =
      value.isZero()
          ? kMaxDivisor
          : int64_t{1} << std::min(value.countr_zero(), kMaxDivisorLog2);
  return AxisInfo(DimVectorT(rank, 1), DimVectorT(rank, divisibility),
                  constancy, std::move(value));
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
  ConstantInt constantValue;
  const auto &lhsConstant = lhs.getConstantValue();
  const auto &rhsConstant = rhs.getConstantValue();
  if (lhsConstant && rhsConstant &&
      lhsConstant->getBitWidth() == rhsConstant->getBitWidth() &&
      *lhsConstant == *rhsConstant)
    constantValue = lhsConstant;
  return AxisInfo(contiguity, divisibility, constancy,
                  std::move(constantValue));
}

AxisInfoAnalysis *
AxisInfoAnalysis::loadDefaultAnalysis(DataFlowSolver *solver) {
  return solver->load<AxisInfoAnalysis>();
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
  auto order = gpu::getOrder(tensorTy);
  unsigned align = getAlignment(offsetsValue, elementBitWidth);

  auto uniqueContigPerThread = gpu::getContigPerThread(tensorTy);
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
  auto order = gpu::getOrder(tensorTy);

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
  auto maskOrder = gpu::getOrder(tensorTy);
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

void ModuleAxisInfoAnalysis::initialize(
    FunctionOpInterface funcOp, AxisInfoAnalysis::LoadCallback loadAnalysis) {
  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  AxisInfoAnalysis *analysis = loadAnalysis(solver.get());
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
