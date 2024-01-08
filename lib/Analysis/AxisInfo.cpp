#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

// Function for extended Euclidean Algorithm
static int64_t gcdImpl(int64_t a, int64_t b, int64_t *x, int64_t *y) {
  // Base Case
  if (a == 0) {
    *x = 0;
    *y = 1;
    return b;
  }
  int64_t x1, y1; // To store results of recursive call
  int64_t gcd = gcdImpl(b % a, a, &x1, &y1);
  // Update x and y using results of
  // recursive call
  *x = y1 - (b / a) * x1;
  *y = x1;
  return gcd;
}

static int64_t gcd(int64_t a, int64_t b) {
  if (a == 0)
    return b;
  if (b == 0)
    return a;
  int64_t x, y;
  return gcdImpl(a, b, &x, &y);
}

static constexpr int log2Int(int64_t num) {
  return (num > 1) ? 1 + log2Int(num / 2) : 0;
}

//===----------------------------------------------------------------------===//
// AxisInfo
//===----------------------------------------------------------------------===//

template <class T>
void AxisInfo::initPessimisticStateFromFunc(int argNumber, T funcOp,
                                            DimVectorT *contiguity,
                                            DimVectorT *divisibility,
                                            DimVectorT *constancy) {
  // liast of attributes that we care about
  SmallVector<std::pair<DimVectorT *, std::string>> retVecs;
  retVecs.push_back({contiguity, "tt.contiguity"});
  retVecs.push_back({divisibility, "tt.divisibility"});
  retVecs.push_back({constancy, "tt.constancy"});
  // initialize attributes one by one
  for (auto [vec, attrName] : retVecs) {
    Attribute attr = funcOp.getArgAttr(argNumber, attrName);
    if (auto int_attr = attr.dyn_cast_or_null<IntegerAttr>())
      *vec = DimVectorT(contiguity->size(), int_attr.getValue().getZExtValue());
    if (auto dense_attr = attr.dyn_cast_or_null<DenseElementsAttr>()) {
      auto vals = dense_attr.getValues<int>();
      *vec = DimVectorT(vals.begin(), vals.end());
    }
  }
}

AxisInfo AxisInfo::getPessimisticValueState(Value value) {
  auto rank = 1;
  if (TensorType ty = value.getType().dyn_cast<TensorType>())
    rank = ty.getRank();
  if (triton::PointerType ty = value.getType().dyn_cast<triton::PointerType>())
    if (TensorType elemTy = ty.getPointeeType().dyn_cast<TensorType>())
      rank = elemTy.getRank();

  DimVectorT knownContiguity(rank, 1);
  DimVectorT knownDivisibility(rank, 1);
  DimVectorT knownConstancy(rank, 1);

  BlockArgument blockArg = value.dyn_cast<BlockArgument>();

  if (blockArg && blockArg.getOwner()->isEntryBlock()) {
    Operation *op = blockArg.getOwner()->getParentOp();
    if (auto fun = dyn_cast<FunctionOpInterface>(op))
      initPessimisticStateFromFunc(blockArg.getArgNumber(), fun,
                                   &knownContiguity, &knownDivisibility,
                                   &knownConstancy);
    // llvm codegen check alignment to generate vector load/store
    // would be nice if this wasn't the case
    else if (auto fun = dyn_cast<LLVM::LLVMFuncOp>(op))
      initPessimisticStateFromFunc(blockArg.getArgNumber(), fun,
                                   &knownContiguity, &knownDivisibility,
                                   &knownConstancy);
  } else if (Operation *op = value.getDefiningOp()) {
    if (Attribute attr = op->getDiscardableAttr("tt.divisibility")) {
      auto vals = attr.cast<DenseElementsAttr>().getValues<int>();
      knownDivisibility = DimVectorT(vals.begin(), vals.end());
    }
    if (Attribute attr = op->getDiscardableAttr("tt.contiguity")) {
      auto vals = attr.cast<DenseElementsAttr>().getValues<int>();
      knownContiguity = DimVectorT(vals.begin(), vals.end());
    }
    if (Attribute attr = op->getDiscardableAttr("tt.constancy")) {
      auto vals = attr.cast<DenseElementsAttr>().getValues<int>();
      knownConstancy = DimVectorT(vals.begin(), vals.end());
    }
  }

  return AxisInfo(knownContiguity, knownDivisibility, knownConstancy);
}

// The gcd of both arguments for each dimension
AxisInfo AxisInfo::join(const AxisInfo &lhs, const AxisInfo &rhs) {
  // If one argument is not initialized, return the other.
  if (lhs.getRank() == 0)
    return rhs;
  if (rhs.getRank() == 0)
    return lhs;
  DimVectorT contiguity;
  DimVectorT divisibility;
  DimVectorT constancy;
  for (auto d = 0; d < lhs.getRank(); ++d) {
    contiguity.push_back(gcd(lhs.getContiguity(d), rhs.getContiguity(d)));
    divisibility.push_back(gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));
    constancy.push_back(gcd(lhs.getConstancy(d), rhs.getConstancy(d)));
  }
  std::optional<int64_t> constantValue;
  if (lhs.getConstantValue().has_value() &&
      rhs.getConstantValue().has_value() &&
      lhs.getConstantValue() == rhs.getConstantValue())
    constantValue = lhs.getConstantValue();
  return AxisInfo(contiguity, divisibility, constancy, constantValue);
}

//===----------------------------------------------------------------------===//
// AxisInfoVisitor
//===----------------------------------------------------------------------===//

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

template <typename OpTy>
class ConstantOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto intAttr = op.getValue().template dyn_cast<IntegerAttr>();
    auto boolAttr = op.getValue().template dyn_cast<BoolAttr>();
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
    auto splatAttr = op.getValue().template dyn_cast<SplatElementsAttr>();
    if (splatAttr && splatAttr.getElementType().isIntOrIndex()) {
      int64_t value = splatAttr.template getSplatValue<APInt>().getZExtValue();
      TensorType ty = splatAttr.getType().template cast<TensorType>();
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

template <typename OpTy>
class AddSubOpAxisInfoVisitor final : public BinaryOpVisitorImpl<OpTy> {
public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

private:
  int64_t getContiguity(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                        int dim) override {
    return std::max(gcd(lhs.getConstancy(dim), rhs.getContiguity(dim)),
                    gcd(lhs.getContiguity(dim), rhs.getConstancy(dim)));
  }

  int64_t getDivisibility(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                          int dim) override {
    // lhs = k * d_lhs = k * k' * gcd(d_lhs, d_rhs)
    // rhs = p * d_rhs = p * p' * gcd(d_lhs, d_rhs)
    // lhs + rhs = k * d_lhs + p * d_rhs = (k * d_lhs + p * d_rhs) *
    // gcd(d_lhs, d_rhs)
    auto elemSize = 1;
    if constexpr (std::is_same_v<OpTy, triton::AddPtrOp>) {
      //  %ptr = addptr %lhs, %rhs
      // is equivalent to
      //  %0 = mul %lhs, %elemSize
      //  %ptr = add %0, %rhs
      elemSize = std::max<unsigned int>(
          1, triton::getPointeeBitWidth(op.getPtr().getType()) / 8);
    }
    return gcd(lhs.getDivisibility(dim), rhs.getDivisibility(dim) * elemSize);
  }

  int64_t getConstancy(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                       int dim) override {
    return gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
                                          const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value()) {
      if constexpr (std::is_same_v<OpTy, arith::AddIOp> ||
                    std::is_same_v<OpTy, triton::AddPtrOp> ||
                    std::is_same_v<OpTy, LLVM::AddOp>) {
        return {lhs.getConstantValue().value() +
                rhs.getConstantValue().value()};
      } else if constexpr (std::is_same_v<OpTy, arith::SubIOp>) {
        return {lhs.getConstantValue().value() -
                rhs.getConstantValue().value()};
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

  int64_t getConstancy(arith::MulIOp op, const AxisInfo &lhs,
                       const AxisInfo &rhs, int dim) override {
    return gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
  }

  int64_t getDivisibility(arith::MulIOp op, const AxisInfo &lhs,
                          const AxisInfo &rhs, int dim) override {
    auto lhsDivisibility = lhs.getDivisibility(dim);
    if (lhs.getContiguity(dim) > 1 &&
        !(rhs.getConstantValue().has_value() && rhs.getConstantValue() == 1)) {
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      lhsDivisibility = 1;
    }
    auto rhsDivisibility = rhs.getDivisibility(dim);
    if (rhs.getContiguity(dim) > 1 &&
        !(lhs.getConstantValue().has_value() && lhs.getConstantValue() == 1)) {
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      rhsDivisibility = 1;
    }
    return lhsDivisibility * rhsDivisibility;
  }

  std::optional<int64_t> getConstantValue(arith::MulIOp op, const AxisInfo &lhs,
                                          const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value())
      return {lhs.getConstantValue().value() * rhs.getConstantValue().value()};
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
    auto resTy = op.getResult().getType().template dyn_cast<RankedTensorType>();
    if (!resTy)
      return BinaryOpVisitorImpl<OpTy>::getConstancy(op, lhs, rhs, dim);
    auto shape = resTy.getShape();
    // Case 1: both lhs and rhs are constants.
    auto constancy = gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
    // Case 2: lhs contiguous, rhs constant.
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
      constancy = std::max(constancy, gcd(lhs.getContiguity(dim),
                                          gcd(lhs.getDivisibility(dim),
                                              rhs.getDivisibility(dim))));
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
    auto resTy = op.getResult().getType().template dyn_cast<RankedTensorType>();
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
      contiguity = std::max(contiguity, gcd(lhs.getContiguity(dim),
                                            gcd(lhs.getDivisibility(dim),
                                                rhs.getDivisibility(dim))));
    }
    return contiguity;
  }

  int64_t getDivisibility(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                          int dim) override {
    // lhs: d_lhs * k = gcd(d_lhs, d_rhs) * k' * k = gcd(d_lhs, d_rhs) * k''
    // rhs: d_rhs * p = gcd(d_lhs, d_rhs) * p' * p = gcd(d_lhs, d_rhs) * p''
    // lhs = gcd(d_lhs, d_rhs) * k'' = gcd(d_lhs, d_rhs) * d + r
    // r must be divisible by gcd(d_lhs, d_rhs)
    return gcd(lhs.getDivisibility(dim), rhs.getDivisibility(dim));
  };

  int64_t getConstancy(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                       int dim) override {
    auto resTy = op.getResult().getType().template dyn_cast<RankedTensorType>();
    if (!resTy)
      return BinaryOpVisitorImpl<OpTy>::getConstancy(op, lhs, rhs, dim);
    auto shape = resTy.getShape();
    // lhs % 1 = 0
    return rhs.getConstantValue().has_value() &&
                   rhs.getConstantValue().value() == 1
               ? shape[dim]
               : gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
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
    TensorType retTy = _retTy.cast<TensorType>();
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
    TensorType retTy = _retTy.cast<TensorType>();
    TensorType opTy = _opTy.cast<TensorType>();
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
    auto resTy = op.getResult().getType().template dyn_cast<RankedTensorType>();
    if (!resTy)
      return AxisInfo();
    auto shape = resTy.getShape();
    short rank = resTy.getRank();
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();

    AxisInfo::DimVectorT contiguity, divisibility, constancy;
    std::optional<int64_t> constantValue;
    for (short d = 0; d < rank; ++d) {
      int64_t constHint = 1;
      if (lhsInfo.getConstantValue().has_value() &&
          rhsInfo.getConstantValue().has_value()) {
        constHint = lhsInfo.getConstancy(d);
        constantValue =
            compare(getPredicate(op), lhsInfo.getConstantValue().value(),
                    rhsInfo.getConstantValue().value())
                ? 1
                : 0;
      } else {
        // Case 1: lhs and rhs are both partial constants
        constHint = gcd(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d));
        // Case 2: lhs all constant, rhs all contiguous
        // NOTE:
        // lhs: 4 4 4 4
        // rhs: 4 5 6 7
        // lhs ge rhs: 1, 0, 0, 0
        // Case 3: lhs all contiguous, rhs all constant
        // NOTE
        // lhs: 4 5 6 7
        // rhs: 4 4 4 4
        // lhs sle rhs: 1, 0, 0, 0
        if (/*Case 2=*/(
                notGePredicate(getPredicate(op)) &&
                (AxisInfoVisitor::isConstantDim(lhsInfo, shape, d) &&
                 AxisInfoVisitor::isContiguousDim(rhsInfo, shape, d))) ||
            /*Case 3=*/(notLePredicate(getPredicate(op)) &&
                        (AxisInfoVisitor::isContiguousDim(lhsInfo, shape, d) &&
                         AxisInfoVisitor::isConstantDim(rhsInfo, shape, d)))) {
          constHint = std::max(constHint, gcd(lhsInfo.getContiguity(d),
                                              gcd(lhsInfo.getDivisibility(d),
                                                  rhsInfo.getDivisibility(d))));
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

  static bool notGePredicate(arith::CmpIPredicate predicate) {
    return predicate != arith::CmpIPredicate::sge &&
           predicate != arith::CmpIPredicate::uge;
  }

  static bool notLePredicate(arith::CmpIPredicate predicate) {
    return predicate != arith::CmpIPredicate::sle &&
           predicate != arith::CmpIPredicate::ule;
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
      bool i1Cond = op.getOperand(0).getType().template isa<IntegerType>();
      for (auto d = 0; d < rank; ++d) {
        if (i1Cond) {
          constancy.push_back(
              std::min(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d)));
          divisibility.push_back(
              std::min(lhsInfo.getDivisibility(d), rhsInfo.getDivisibility(d)));
          contiguity.push_back(
              std::min(lhsInfo.getContiguity(d), rhsInfo.getContiguity(d)));
        } else {
          constancy.push_back(
              std::min(gcd(lhsInfo.getConstancy(d), condConstancy[d]),
                       gcd(rhsInfo.getConstancy(d), condConstancy[d])));
          contiguity.push_back(
              std::min(gcd(lhsInfo.getContiguity(d), condConstancy[d]),
                       gcd(rhsInfo.getContiguity(d), condConstancy[d])));
          if (contiguity.back() == lhsInfo.getContiguity(d) &&
              contiguity.back() == rhsInfo.getContiguity(d)) {
            // Contiguity not changed
            divisibility.push_back(
                gcd(lhsInfo.getDivisibility(d), rhsInfo.getDivisibility(d)));
          } else {
            // Contiguity changed, we cannot use only divisibility.
            // For example, the following example should have contiguity 2 and
            // divisibility 2
            // [[0, 1], [4, 5]]
            // [[16, 17, 18, 19]]
            divisibility.push_back(
                std::min(gcd(lhsInfo.getDivisibility(d), contiguity.back()),
                         gcd(rhsInfo.getDivisibility(d), contiguity.back())));
          }
        }
      }
      if (lhsInfo.getConstantValue().has_value() &&
          rhsInfo.getConstantValue().has_value() &&
          lhsInfo.getConstantValue() == rhsInfo.getConstantValue())
        constantValue = lhsInfo.getConstantValue();
    }

    return AxisInfo(contiguity, divisibility, constancy, constantValue);
  }
};

template <typename OpTy>
class LogicalOpAxisInfoVisitor final : public BinaryOpVisitorImpl<OpTy> {
public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

private:
  int64_t getConstancy(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                       int dim) override {
    return gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
  }

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
    auto shift = rhs.getConstantValue().has_value()
                     ? rhs.getConstantValue().value()
                     : rhs.getDivisibility(dim);
    auto lhsDivisibility = lhs.getDivisibility(dim);
    if (lhs.getContiguity(dim) > 1 && shift) {
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      lhsDivisibility = 1;
    }
    auto numBits = log2Int(lhsDivisibility);
    auto maxBits = log2Int(highestPowOf2Divisor<int64_t>(0));
    // Make sure the return value doesn't exceed highestPowOf2Divisor<int64>(0)
    if (shift + numBits > maxBits)
      return highestPowOf2Divisor<int64_t>(0);
    return lhsDivisibility << shift;
  }

  int64_t getConstancy(arith::ShLIOp op, const AxisInfo &lhs,
                       const AxisInfo &rhs, int dim) override {
    return gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
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
    auto shift = rhs.getConstantValue().has_value()
                     ? rhs.getConstantValue().value()
                     : rhs.getDivisibility(dim);
    auto lhsDivisibility = lhs.getDivisibility(dim);
    if (lhs.getContiguity(dim) > 1 && shift) {
      // Treat [2^n,2^n+1,...]'s divisibility as 1 instead of 2^n
      lhsDivisibility = 1;
    }
    return std::max<int64_t>(1, lhsDivisibility / (1 << shift));
  }

  int64_t getConstancy(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                       int dim) override {
    return gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
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
      return AxisInfo(/*knownContiguity=*/AxisInfo::DimVectorT(rank, 1),
                      /*knownDivisibility=*/AxisInfo::DimVectorT(rank, 1),
                      /*knownConstancy=*/AxisInfo::DimVectorT(rank, 1),
                      /*constantValue=*/constantValue);
    } else {
      AxisInfo::DimVectorT contiguity, divisibility, constancy;
      for (auto d = 0; d < rank; ++d) {
        constancy.push_back(
            std::min(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d)));
        divisibility.push_back(
            std::min(lhsInfo.getDivisibility(d), rhsInfo.getDivisibility(d)));
        contiguity.push_back(
            std::min(lhsInfo.getContiguity(d), rhsInfo.getContiguity(d)));
      }
      return AxisInfo(contiguity, divisibility, constancy, std::nullopt);
    }
  }
};

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
  visitors.append<CastOpAxisInfoVisitor<arith::ExtSIOp>,
                  CastOpAxisInfoVisitor<arith::ExtUIOp>,
                  CastOpAxisInfoVisitor<arith::TruncIOp>,
                  CastOpAxisInfoVisitor<arith::IndexCastOp>,
                  CastOpAxisInfoVisitor<triton::PtrToIntOp>,
                  CastOpAxisInfoVisitor<triton::IntToPtrOp>,
                  CastOpAxisInfoVisitor<triton::gpu::ConvertLayoutOp>,
                  CastOpAxisInfoVisitor<mlir::UnrealizedConversionCastOp>,
                  CastOpAxisInfoVisitor<triton::BitcastOp>>();
  // TODO: Remove rules for LLVM::ConstantOp, LLVM::AddOp
  // when scf.for supports integers induction variable
  visitors.append<MakeRangeOpAxisInfoVisitor>();
  visitors.append<ConstantOpAxisInfoVisitor<arith::ConstantOp>,
                  ConstantOpAxisInfoVisitor<LLVM::ConstantOp>>();
  visitors.append<AddSubOpAxisInfoVisitor<triton::AddPtrOp>,
                  AddSubOpAxisInfoVisitor<arith::AddIOp>,
                  AddSubOpAxisInfoVisitor<arith::SubIOp>,
                  AddSubOpAxisInfoVisitor<LLVM::AddOp>>();
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
}

void AxisInfoAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<AxisInfo> *> operands,
    ArrayRef<dataflow::Lattice<AxisInfo> *> results) {
  // TODO: For sure not the right way to do this
  // but why is scf.if not initialized otherwise?
  for (auto op : operands)
    if (op->getValue().getRank() == 0)
      setToEntryState((dataflow::Lattice<AxisInfo> *)op);
  AxisInfo curr = visitors.apply(op, operands);
  if (curr.getRank() == 0)
    return setAllToEntryStates(results);
  // override with hint
  auto newContiguity = curr.getContiguity();
  auto newDivisibility = curr.getDivisibility();
  auto newConstancy = curr.getConstancy();
  if (Attribute attr = op->getDiscardableAttr("tt.contiguity")) {
    auto vals = attr.cast<DenseElementsAttr>().getValues<int>();
    newContiguity = AxisInfo::DimVectorT(vals.begin(), vals.end());
  }
  if (Attribute attr = op->getDiscardableAttr("tt.divisibility")) {
    auto vals = attr.cast<DenseElementsAttr>().getValues<int>();
    newDivisibility = AxisInfo::DimVectorT(vals.begin(), vals.end());
  }
  if (Attribute attr = op->getDiscardableAttr("tt.constancy")) {
    auto vals = attr.cast<DenseElementsAttr>().getValues<int>();
    newConstancy = AxisInfo::DimVectorT(vals.begin(), vals.end());
  }
  curr = mlir::AxisInfo(newContiguity, newDivisibility, newConstancy,
                        curr.getConstantValue());
  // join all lattice elements
  for (auto *result : results)
    propagateIfChanged(result, result->join(curr));
}

unsigned ModuleAxisInfoAnalysis::getPtrContiguity(Value ptr) {
  auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return 1;
  auto layout = tensorTy.getEncoding();

  // Here order should be ordered by contiguous first, so the first element
  // should have the largest contiguous.
  auto order = triton::gpu::getOrder(layout);
  unsigned align = getPtrAlignment(ptr);

  auto uniqueContigPerThread =
      triton::gpu::getUniqueContigPerThread(layout, tensorTy.getShape());
  assert(order[0] < uniqueContigPerThread.size() &&
         "Unexpected uniqueContigPerThread size");
  unsigned contiguity = uniqueContigPerThread[order[0]];
  contiguity = std::min(align, contiguity);

  return contiguity;
}

void AxisInfoAnalysis::visitForOpInductionVar(
    scf::ForOp op, ArrayRef<dataflow::Lattice<AxisInfo> *> argLattices) {
  auto lb = getLatticeElementFor(op, op.getLowerBound())->getValue();
  auto step = getLatticeElementFor(op, op.getStep())->getValue();

  AxisInfo::DimVectorT knownContiguity(1, 1);
  AxisInfo::DimVectorT knownDivisibility(1, 1);
  AxisInfo::DimVectorT knownConstancy(1, 1);
  knownDivisibility[0] = gcd(lb.getDivisibility(0), step.getDivisibility(0));
  auto inductionVar =
      AxisInfo(knownContiguity, knownDivisibility, knownConstancy);
  (void)argLattices[0]->join(inductionVar);
}

unsigned ModuleAxisInfoAnalysis::getPtrAlignment(Value ptr) {
  auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return 1;
  auto *axisInfo = getAxisInfo(ptr);
  if (!axisInfo)
    return 1;
  auto layout = tensorTy.getEncoding();
  auto order = triton::gpu::getOrder(layout);
  auto maxMultipleBytes = axisInfo->getDivisibility(order[0]);
  auto maxContig = axisInfo->getContiguity(order[0]);
  auto elemNumBits = triton::getPointeeBitWidth(tensorTy);
  auto elemNumBytes = std::max<unsigned>(elemNumBits / 8, 1);
  auto maxMultiple = std::max<int64_t>(maxMultipleBytes / elemNumBytes, 1);
  unsigned alignment = std::min(maxMultiple, maxContig);
  return alignment;
}

unsigned ModuleAxisInfoAnalysis::getMaskAlignment(Value mask) {
  auto tensorTy = mask.getType().dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return 1;
  auto *axisInfo = getAxisInfo(mask);
  if (!axisInfo)
    return 1;
  auto maskOrder = triton::gpu::getOrder(tensorTy.getEncoding());
  auto alignment = std::max<unsigned>(axisInfo->getConstancy(maskOrder[0]), 1);
  return alignment;
}

void ModuleAxisInfoAnalysis::initialize(FunctionOpInterface funcOp) {
  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  AxisInfoAnalysis *analysis = solver->load<AxisInfoAnalysis>();
  if (failed(solver->initializeAndRun(funcOp)))
    return;
  auto *axisInfoMap = getFuncData(funcOp);
  auto updateAxisInfoMap = [&](Value value) {
    auto axisInfo = analysis->getLatticeElement(value)->getValue();
    AxisInfo curAxisInfo;
    if (axisInfoMap->count(value)) {
      curAxisInfo = AxisInfo::join(axisInfo, axisInfoMap->lookup(value));
    } else {
      curAxisInfo = axisInfo;
    }
    (*axisInfoMap)[value] = curAxisInfo;
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
      auto curValue = highestPowOf2Divisor<int64_t>(0);
      if (callee.getArgAttrOfType<IntegerAttr>(index, attrName)) {
        curValue =
            callee.getArgAttrOfType<IntegerAttr>(index, attrName).getInt();
      }
      auto attr = IntegerAttr::get(IntegerType::get(callee.getContext(), 64),
                                   gcd(prevValue, curValue));
      callee.setArgAttr(index, attrName, attr);
    };
    auto axisInfo = axisInfoMap->lookup(value);
    assert(axisInfo.getRank() == 1 && "only scalar arguments are supported");
    setAttrFn("tt.contiguity", axisInfo.getContiguity(0));
    setAttrFn("tt.divisibility", axisInfo.getDivisibility(0));
    setAttrFn("tt.constancy", axisInfo.getConstancy(0));
  }
}

} // namespace mlir
