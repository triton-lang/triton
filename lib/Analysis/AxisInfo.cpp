#include "mlir/Analysis/DataFlowAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

// Function for extended Euclidean Algorithm
static int gcdImpl(int a, int b, int *x, int *y) {
  // Base Case
  if (a == 0) {
    *x = 0;
    *y = 1;
    return b;
  }
  int x1, y1; // To store results of recursive call
  int gcd = gcdImpl(b % a, a, &x1, &y1);
  // Update x and y using results of
  // recursive call
  *x = y1 - (b / a) * x1;
  *y = x1;
  return gcd;
}

static int gcd(int a, int b) {
  int x, y;
  return gcdImpl(a, b, &x, &y);
}

static bool isValueConstantInt(Value value, int intValue) {
  int v;
  if (auto constantIntOp =
          dyn_cast<arith::ConstantIntOp>(value.getDefiningOp())) {
    v = constantIntOp.value();
  } else if (auto constantIndexOp =
                 dyn_cast<arith::ConstantIndexOp>(value.getDefiningOp())) {
    v = constantIntOp.value();
  } else {
    return false;
  }
  return v == intValue;
}

//===----------------------------------------------------------------------===//
// AxisInfo
//===----------------------------------------------------------------------===//

AxisInfo AxisInfo::getPessimisticValueState(Value value) {
  size_t rank = 1;
  if (TensorType ty = value.getType().dyn_cast<TensorType>())
    rank = ty.getRank();
  int contiHint = AxisInfo::Default;
  int divHint = AxisInfo::Default;
  int constHint = AxisInfo::Default;
  BlockArgument blockArg = value.dyn_cast<BlockArgument>();
  if (blockArg && blockArg.getOwner()->isEntryBlock()) {
    // If the block argument is the entry block, we first call
    // AxisInfo::getPessimisticValueState, then immedidately call AxisInfo::join
    // to join the hint with the init block argument.
    // So in the join function, gcd(0, x) = x
    contiHint = AxisInfo::Unknown;
    divHint = AxisInfo::Unknown;
    constHint = AxisInfo::Unknown;
    Operation *op = blockArg.getOwner()->getParentOp();
    if (FuncOp fun = dyn_cast<FuncOp>(op)) {
      Attribute attr =
          fun.getArgAttr(blockArg.getArgNumber(), "tt.divisibility");
      if (attr)
        divHint = attr.cast<IntegerAttr>().getValue().getZExtValue();
    } else if (auto fun = dyn_cast<LLVM::LLVMFuncOp>(op)) {
      Attribute attr =
          fun.getArgAttr(blockArg.getArgNumber(), "tt.divisibility");
      if (attr)
        divHint = attr.cast<IntegerAttr>().getValue().getZExtValue();
    }
  }

  DimVectorT contiguity(rank, contiHint);
  DimVectorT divisibility(rank, divHint);
  DimVectorT constancy(rank, constHint);
  return AxisInfo(contiguity, divisibility, constancy);
}

// The gcd of both arguments for each dimension
AxisInfo AxisInfo::join(const AxisInfo &lhs, const AxisInfo &rhs) {
  DimVectorT retContiguity;
  DimVectorT retDivisibility;
  DimVectorT retConstancy;
  for (int d = 0; d < lhs.getRank(); ++d) {
    retContiguity.push_back(gcd(lhs.getContiguity(d), rhs.getContiguity(d)));
    retDivisibility.push_back(
        gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));
    retConstancy.push_back(gcd(lhs.getConstancy(d), rhs.getConstancy(d)));
  }
  return AxisInfo(retContiguity, retDivisibility, retConstancy);
}

//===----------------------------------------------------------------------===//
// AxisInfoVisitor
//===----------------------------------------------------------------------===//

AxisInfo AxisInfoVisitor::visitBinaryOp(
    Operation *op, AxisInfo lhsInfo, AxisInfo rhsInfo,
    const std::function<int(AxisInfo, AxisInfo, int)> &getContiguity,
    const std::function<int(AxisInfo, AxisInfo, int)> &getDivisibility,
    const std::function<int(AxisInfo, AxisInfo, int)> &getConstancy) {
  int rank = lhsInfo.getRank();
  AxisInfo::DimVectorT newContiguity;
  AxisInfo::DimVectorT newDivisibility;
  AxisInfo::DimVectorT newConstancy;
  for (int d = 0; d < rank; ++d) {
    newContiguity.push_back(getContiguity(lhsInfo, rhsInfo, d));
    newDivisibility.push_back(getDivisibility(lhsInfo, rhsInfo, d));
    newConstancy.push_back(getConstancy(lhsInfo, rhsInfo, d));
  }
  return AxisInfo(newContiguity, newDivisibility, newConstancy);
}

template <typename OpTy>
class CastOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
 public: 
  AxisInfo getAxisInfo(Operation *op, ArrayRef<LatticeElement<AxisInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

class MakeRangeOpAxisInfoVisitor final : public AxisInfoVisitorImpl<triton::MakeRangeOp> {
 public:
  AxisInfo getAxisInfo(triton::MakeRangeOp op, ArrayRef<LatticeElement<AxisInfo> *> operands) override {
    int start = op.start();
    int end = op.end();
    AxisInfo::DimVectorT contiguity = {end - start};
    AxisInfo::DimVectorT divisibility = {highestPowOf2Divisor(start)};
    AxisInfo::DimVectorT constancy = {AxisInfo::Default};
    return AxisInfo(contiguity, divisibility, constancy);
  }
};

class ConstantOpAxisInfoVisitor final : public AxisInfoVisitorImpl<ConstantOp> {
 public:
  AxisInfo getAxisInfo(ConstantOp op, ArrayRef<LatticeElement<AxisInfo> *> operands) override {
    auto intAttr = op.getValue().dyn_cast<IntegerAttr>();
    if (intAttr) {
      int val = intAttr.getValue().getZExtValue();
      return AxisInfo({AxisInfo::Default}, {highestPowOf2Divisor(val)}, {AxisInfo::Default});
    }
    // TODO: generalize to dense attr
    // XXX: what does it mean?
    auto splatAttr = op.getValue().dyn_cast<SplatElementsAttr>();
    if (splatAttr && splatAttr.getElementType().isIntOrIndex()) {
      // XXX: is it safe to cast to int? what about i64? ptr can be 64bit
      auto value = splatAttr.getSplatValue<int>();
      TensorType ty = splatAttr.getType().cast<TensorType>();
      return AxisInfo(
          AxisInfo::DimVectorT(ty.getRank(), AxisInfo::Default),
          AxisInfo::DimVectorT(ty.getRank(), highestPowOf2Divisor(value)),
          AxisInfo::DimVectorT(ty.getShape().begin(), ty.getShape().end()));
    }
    llvm_unreachable("unsupported constant type");
    return AxisInfo();
  }
};

template <typename OpTy>
class AddOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
 public:
  AxisInfo getAxisInfo(Operation *op, ArrayRef<LatticeElement<AxisInfo> *> operands) override {
    auto tensorTy = op->getOperand(0).getType().cast<RankedTensorType>();
    auto shape = tensorTy.getShape();
    auto newContiguity = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      auto contig = AxisInfo::Default;
      if (AxisInfoVisitor::isContiguityConstancyAligned(lhs, rhs, shape, d))
        contig = std::max(gcd(lhs.getConstancy(d), rhs.getContiguity(d)),
                          gcd(lhs.getContiguity(d), rhs.getConstancy(d)));
      return contig;
    };
    auto newConstancy = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      auto constancy = AxisInfo::Default;
      if (AxisInfoVisitor::isConstancyAligned(lhs, rhs, shape, d))
        constancy = gcd(lhs.getConstancy(d), rhs.getConstancy(d));
      return constancy;
    };
    auto newDivisibility = [](AxisInfo lhs, AxisInfo rhs, int d) {
      // clang-format off
      // lhs = k * d_lhs = k * k' * gcd(d_lhs, d_rhs)
      // rhs = p * d_rhs = p * p' * gcd(d_lhs, d_rhs)
      // lhs + rhs = k * d_lhs + p * d_rhs = (k * d_lhs + p * d_rhs) * gcd(d_lhs, d_rhs)
      // clang-format on
      return gcd(lhs.getDivisibility(d), rhs.getDivisibility(d));
    };
    return visitBinaryOp(op, operands[0]->getValue(), operands[1]->getValue(),
                         newContiguity, newDivisibility, newConstancy);
  }
};

class MulIOpAxisInfoVisitor final : public AxisInfoVisitorImpl<arith::MulIOp> {
 public:
  AxisInfo getAxisInfo(arith::MulIOp op, ArrayRef<LatticeElement<AxisInfo> *> operands) override {
    auto tensorTy = op->getOperand(0).getType().cast<RankedTensorType>();
    auto shape = tensorTy.getShape();
    auto newContiguity = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      // lhs * 1 = lhs
      auto lhsContiguity =
          isValueConstantInt(op.getOperand(1), 1) ? lhs.getContiguity(d) : 1;
      // 1 * rhs = rhs
      auto rhsContiguity =
          isValueConstantInt(op.getOperand(0), 1) ? rhs.getContiguity(d) : 1;
      return std::max(lhsContiguity, rhsContiguity);
    };
    auto newConstancy = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      auto constancy = AxisInfo::Default;
      if (AxisInfoVisitor::isConstancyAligned(lhs, rhs, shape, d))
        constancy = gcd(lhs.getConstancy(d), rhs.getConstancy(d));
      return constancy;
    };
    auto newDivisibility = [](AxisInfo lhs, AxisInfo rhs, int d) {
      // clang-format off
      // lhs = k * d_lhs
      // rhs = p * d_rhs
      // lhs * rhs = k * d_lhs * p * d_rhs = k * p * d_lhs * d_rhs
      // clang-format on
      return lhs.getDivisibility(d) * rhs.getDivisibility(d);
    };
    return visitBinaryOp(op, operands[0]->getValue(), operands[1]->getValue(),
                         newContiguity, newDivisibility, newConstancy);
  }
};

template <typename OpTy>
class DivOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
  public:
    AxisInfo getAxisInfo(Operation *op, ArrayRef<LatticeElement<AxisInfo> *> operands) override {
      auto tensorTy = op->getOperand(0).getType().cast<RankedTensorType>();
      auto shape = tensorTy.getShape();
      auto newContiguity = [&](AxisInfo lhs, AxisInfo rhs, int d) {
        // lhs / 1 = lhs
        return isValueConstantInt(op->getOperand(1), 1) == 1
                   ? lhs.getContiguity(d)
                   : 1;
      };
      auto newConstancy = [&](AxisInfo lhs, AxisInfo rhs, int d) {
        auto constancy = gcd(lhs.getConstancy(d), rhs.getConstancy(d));
        // Case 1: lhs full contiguous, rhs full constant
        // lhs = d_lhs * k, d_lhs * k + 1, ..., d_lhs * k + n
        // rhs = d_rhs * p, d_rhs * p, ..., d_rhs * p
        // At some point t, d_lhs * k + t == d_rhs * p
        // 10, 11, 12, 13, ... 
        // 8, 8, 8, 8, ...
        // 88, 88

        // gcd * p, gcd * p

        // n / p, n / p + 1 / (gcd * p), ... k / (gcd * p)
        // k == gcd * p
        // k >= gcd
        // gcd * n, gcd * n, ...
        // gcd * p, gcd * p + 1
        // n / p, n / (p + 1/gcd), ..., n / (p + k/gcd)
        // k / gcd > 1 => 1 > gcd
        if ((isContigAlongDim(lhs, lhsShape, d) &&
             isConstantAlongDim(rhs, rhsShape, d)) ||
            (isConstantAlongDim(lhs, lhsShape, d) &&
             isContigAlongDim(rhs, rhsShape, d))) {
          constancy = std::max(
              constancy, gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));
        }
        return constancy;
      };
      auto newDivisibility = [](AxisInfo lhs, AxisInfo rhs, int d) {
        // lhs = k * d_lhs = k * k' * gcd(d_lhs, d_rhs)
        // rhs = p * d_rhs = p * p' * gcd(d_lhs, d_rhs)
        // lhs / rhs = k * k' * gcd(d_lhs, d_rhs) / (p * p' * gcd(d_lhs, d_rhs)) 
        //           = k / p * k' / p'
        // gcd(k', p') = gcd(d_lhs / gcd(d_lhs, d_rhs), d_rhs / gcd(d_lhs, d_rhs))
        auto lhsDivisibility = lhs.getDivisibility(d);
        auto rhsDivisibility = rhs.getDivisibility(d);
        auto initGcd = gcd(lhsDivisibility, rhsDivisibility);
        return gcd(lhsDivisibility / initGcd, rhsDivisibility / initGcd);
      };
      return visitBinaryOp(op, operands[0]->getValue(), operands[1]->getValue(),
                           newContiguity, newDivisibility, newConstancy);
    }
};

//===----------------------------------------------------------------------===//
// AxisInfoAnalysis
//===----------------------------------------------------------------------===//

AxisInfoAnalysis::AxisInfoAnalysis(MLIRContext *context)
    : ForwardDataFlowAnalysis<AxisInfo>(context) {}


ChangeResult AxisInfoAnalysis::visitOperation(
    Operation *op, ArrayRef<LatticeElement<AxisInfo> *> operands) {
  AxisInfo curr;
  // This preserves the input axes (e.g., cast):
  if (llvm::isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
                arith::IndexCastOp, triton::PtrToIntOp, triton::IntToPtrOp,
                triton::gpu::ConvertLayoutOp>(op))
    curr = operands[0]->getValue();
  // Constant ranges
  if (triton::MakeRangeOp make_range =
          llvm::dyn_cast<triton::MakeRangeOp>(op)) {
    int start = make_range.start();
    int end = make_range.end();
    AxisInfo::DimVectorT contiguity = {end - start};
    AxisInfo::DimVectorT divisibility = {highestPowOf2Divisor(start)};
    AxisInfo::DimVectorT constancy = {1};
    curr = AxisInfo(contiguity, divisibility, constancy);
  }
  // Constant
  if (arith::ConstantOp constant = llvm::dyn_cast<arith::ConstantOp>(op)) {
    auto intAttr = constant.getValue().dyn_cast<IntegerAttr>();
    if (intAttr) {
      int val = intAttr.getValue().getZExtValue();
      curr = AxisInfo({1}, {highestPowOf2Divisor(val)}, {1});
    }
    // TODO: generalize to dense attr
    auto splatAttr = constant.getValue().dyn_cast<SplatElementsAttr>();
    if (splatAttr && splatAttr.getElementType().isIntOrIndex()) {
      auto value = splatAttr.getSplatValue<int>();
      TensorType ty = splatAttr.getType().cast<TensorType>();
      curr = AxisInfo(
          AxisInfo::DimVectorT(ty.getRank(), 1),
          AxisInfo::DimVectorT(ty.getRank(), highestPowOf2Divisor(value)),
          AxisInfo::DimVectorT(ty.getShape().begin(), ty.getShape().end()));
    }
  }
  // TODO: refactor & complete binary ops
  // Addition
  if (llvm::isa<arith::AddIOp, triton::AddPtrOp>(op)) {
    auto newContiguity = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      return std::max(gcd(lhs.getContiguity(d), rhs.getConstancy(d)),
                      gcd(lhs.getConstancy(d), rhs.getContiguity(d)));
    };
    auto newConstancy = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      return gcd(lhs.getConstancy(d), rhs.getConstancy(d));
    };
    auto newDivisibility = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      return gcd(lhs.getDivisibility(d), rhs.getDivisibility(d));
    };
    curr = visitBinaryOp(op, operands[0]->getValue(), operands[1]->getValue(),
                         newContiguity, newDivisibility, newConstancy);
  }
  // Multiplication
  if (llvm::isa<arith::MulIOp>(op)) {
    auto newContiguity = [&](AxisInfo lhs, AxisInfo rhs, int d) { 
      auto lhs = op->getOperand(0);
      auto rhs = op->getOperand(1);
      auto lhsValue = 1;
      auto rhsValue = 1;
      if (auto constantIntOp = dyn_cast<arith::ConstantIntOp>(lhs)) {
        lhsValue = constantIntOp.value();
      } else if (auto constantIndexOp = dyn_cast<arith::ConstantIndexOp>(lhs)) {
        lhsValue = constantIndexOp.value();
      }
      if (auto constantIntOp = dyn_cast<arith::ConstantIntOp>(rhs)) {
        rhsValue = constantIntOp.value();
      } else if (auto constantIndexOp = dyn_cast<arith::ConstantIndexOp>(rhs)) {
        rhsValue = constantIndexOp.value();
      }
      auto lhsContiguity = rhsValue == 1 ? lhs.getContiguity(d) : 1;
      auto rhsContiguity = lhsValue == 1 ? rhs.getContiguity(d) : 1;
      return std::max(lhsContiguity, rhsContiguity);
    };
    auto newConstancy = [](AxisInfo lhs, AxisInfo rhs, int d) {
      return gcd(lhs.getConstancy(d), rhs.getConstancy(d));
    };
    auto newDivisibility = [](AxisInfo lhs, AxisInfo rhs, int d) {
      return lhs.getDivisibility(d) * rhs.getDivisibility(d);
    };
    curr = visitBinaryOp(op, operands[0]->getValue(), operands[1]->getValue(),
                         newContiguity, newDivisibility, newConstancy);
  }
  // Division
  if (llvm::isa<arith::DivSIOp, arith::DivUIOp>(op)) {
    auto newContiguity = [&](AxisInfo lhs, AxisInfo rhs, int d) { 
      auto rhs = op->getOperand(1);
      auto rhsValue = 1;
      if (auto constantIntOp = dyn_cast<arith::ConstantIntOp>(rhs)) {
        rhsValue = constantIntOp.value();
      } else if (auto constantIndexOp = dyn_cast<arith::ConstantIndexOp>(rhs)) {
        rhsValue = constantIndexOp.value();
      }
      return rhsValue == 1 ? lhs.getContiguity(d) : 1; 
    };
    auto newDivisibility = [](AxisInfo lhs, AxisInfo rhs, int d) {
      // gcd * n / gcd * p -> gcd(n, p)
      auto lhsDivisibility = lhs.getDivisibility(d);
      auto rhsDivisibility = rhs.getDivisibility(d);
      auto initGcd = gcd(lhsDivisibility, rhsDivisibility);
      return gcd(lhsDivisibility / initGcd, rhsDivisibility / initGcd);
    };
    auto newConstancy = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      auto lhsShape =
          op->getOperand(0).getType().cast<RankedTensorType>().getShape();
      auto rhsShape =
          op->getOperand(1).getType().cast<RankedTensorType>().getShape();
      auto constancy = gcd(lhs.getConstancy(d), rhs.getConstancy(d));
      // gcd * n, gcd * n + 1
      // gcd * p, gcd * p
      // n / p, n / p + 1 / (gcd * p), ... k / (gcd * p)
      // k == gcd * p
      // k >= gcd
      // gcd * n, gcd * n, ...
      // gcd * p, gcd * p + 1
      // n / p, n / (p + 1/gcd), ..., n / (p + k/gcd)
      // k / gcd > 1 => 1 > gcd
      if ((isContigAlongDim(lhs, lhsShape, d) &&
           isConstantAlongDim(rhs, rhsShape, d)) ||
          (isConstantAlongDim(lhs, lhsShape, d) &&
           isContigAlongDim(rhs, rhsShape, d))) {
        constancy = std::max(
            constancy, gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));
      }
      return constancy;
    };
    curr = visitBinaryOp(op, operands[0]->getValue(), operands[1]->getValue(),
                         newContiguity, newDivisibility, newConstancy);
  }
  // Remainder
  if (llvm::isa<arith::RemSIOp, arith::RemUIOp>(op)) {
    auto newContiguity = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      auto lhsShape =
          op->getOperand(0).getType().cast<RankedTensorType>().getShape();
      auto rhsShape =
          op->getOperand(1).getType().cast<RankedTensorType>().getShape();
      // gcd * n, gcd * n + 1
      // gcd * p, gcd * p
      // (gcd * n) % (gcd * p) = n % p
      // (gcd * n + 1) % (gcd * p) = n % p + 1
      // gcd * n, gcd * n
      // gcd * p, gcd * p + 1
      // (gcd * n) % (gcd * p) = n % p
      // (gcd * n) % (gcd * p + 1) = n % p + n
      // (gcd * n) % (gcd * p + 2) = n % p + n % 2
      // ...
      // (gcd * n) % (gcd * p + 2) = n % p + n % p
      if ((isContigAlongDim(lhs, lhsShape, d) &&
           isContigAlongDim(rhs, rhsShape, d)) ||
          (isContigAlongDim(lhs, lhsShape, d) &&
           isConstantAlongDim(rhs, rhsShape, d))) {
        return lhs.getContiguity(d);
      }
      // 12 12 12 12 12 12 12 12 
      // 4 5 6 7 8 9 10 11 12 12 
      if (isConstantAlongDim(lhs, lhsShape, d) &&
          isContigAlongDim(rhs, rhsShape, d) &&
          lhs.getDivisibility(d) % rhs.getDivisibility(d) == 0) {
        return gcd(lhs.getDivisibility(d), rhs.getDivisibility(d));
      }
      return 1;
    };
    auto newDivisibility = [](AxisInfo lhs, AxisInfo rhs, int d) {
      // (gcd * p) % (gcd * n) = p % n = (p1 * p2) % (n1 * n2)
      // (2^p % 2^n) * (p2 % n2)
      return gcd(lhs.getDivisibility(d), rhs.getDivisibility(d));
    };
    auto newConstancy = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      auto lhsShape =
          op->getOperand(0).getType().cast<RankedTensorType>().getShape();
      auto rhsShape =
          op->getOperand(1).getType().cast<RankedTensorType>().getShape();
      auto constancy = gcd(lhs.getConstancy(d), rhs.getConstancy(d));
      // gcd * n, gcd * n + 1
      // gcd * p, gcd * p
      // n / p, n / p + 1 / (gcd * p), ... k / (gcd * p)
      // k == gcd * p
      // k >= gcd
      // gcd * n, gcd * n, ...
      // gcd * p, gcd * p + 1
      // n / p, n / (p + 1/gcd), ..., n / (p + k/gcd)
      // k / gcd > 1 => 1 > gcd
      if ((isContigAlongDim(lhs, lhsShape, d) &&
           isConstantAlongDim(rhs, rhsShape, d)) ||
          (isConstantAlongDim(lhs, lhsShape, d) &&
           isContigAlongDim(rhs, rhsShape, d))) {
        constancy = std::max(
            constancy, gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));
      }
      return constancy;
    };
    curr = visitBinaryOp(op, operands[0]->getValue(), operands[1]->getValue(),
                         newContiguity, newDivisibility, newConstancy);
  }
  if (llvm::isa<arith::AndIOp, arith::OrIOp>(op)) {
    // TODO: Apply constant (0) opt for or
    auto newContiguity = [](AxisInfo lhs, AxisInfo rhs, int d) { return 1; };
    auto newDivisibility = [](AxisInfo lhs, AxisInfo rhs, int d) { return 1; };
    auto newConstancy = [](AxisInfo lhs, AxisInfo rhs, int d) {
      return gcd(lhs.getConstancy(d), rhs.getConstancy(d));
    };
    curr = visitBinaryOp(op, operands[0]->getValue(), operands[1]->getValue(),
                         newContiguity, newDivisibility, newConstancy);
  }
  // Splat
  if (llvm::isa<triton::SplatOp>(op)) {
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
    curr = AxisInfo(contiguity, divisibility, constancy);
  }
  // expandDims
  if (auto expandDims = llvm::dyn_cast<triton::ExpandDimsOp>(op)) {
    AxisInfo opInfo = operands[0]->getValue();
    AxisInfo::DimVectorT contiguity = opInfo.getContiguity();
    AxisInfo::DimVectorT divisibility = opInfo.getDivisibility();
    AxisInfo::DimVectorT constancy = opInfo.getConstancy();
    contiguity.insert(contiguity.begin() + expandDims.axis(), 1);
    divisibility.insert(divisibility.begin() + expandDims.axis(), 1);
    constancy.insert(constancy.begin() + expandDims.axis(), 1);
    curr = AxisInfo(contiguity, divisibility, constancy);
  }
  // Broadcast
  if (llvm::isa<triton::BroadcastOp>(op)) {
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
    curr = AxisInfo(contiguity, divisibility, constancy);
  }

  // CmpI
  if (isa<arith::CmpIOp, triton::gpu::CmpIOp>(op) &&
      op->getResult(0).getType().dyn_cast<TensorType>()) {
    auto resTy = op->getResult(0).getType().cast<TensorType>();
    short rank = resTy.getRank();
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();
    auto shape = resTy.getShape();

    AxisInfo::DimVectorT contiguity, divisibility, constancy;
    for (short d = 0; d < rank; ++d) {
      auto contig = 1;
      if (isContigAlongDim(lhsInfo, shape, d) && isContigAlongDim(rhsInfo, shape, d)) {
        contig = shape[d];
      } else {
        contig = gcd(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d));
        // 16 * p, 16 * p + 1
        // 8 * n, 8 * n
        if ((isContigAlongDim(lhsInfo, shape, d) &&
             isConstantAlongDim(rhsInfo, shape, d)) ||
            isContigAlongDim(rhsInfo, shape, d) &&
                isConstantAlongDim(lhsInfo, shape, d)) {
          contig = std::max(contig, gcd(lhsInfo.getDivisibility(d),
                                      rhsInfo.getDivisibility(d)));
        }
      }

      constancy.push_back(contig);
      divisibility.push_back(1);
      contiguity.push_back(1);
    }

    curr = AxisInfo(contiguity, divisibility, constancy);
  }

  // Select
  if (isa<mlir::SelectOp, triton::gpu::SelectOp>(op) &&
      op->getResult(0).getType().dyn_cast<TensorType>()) {
    auto rank = op->getResult(0).getType().cast<TensorType>().getRank();
    auto condConstancy = operands[0]->getValue().getConstancy();  
    auto lhsInfo = operands[1]->getValue();
    auto rhsInfo = operands[2]->getValue();

    AxisInfo::DimVectorT contiguity, divisibility, constancy;
    for (short d = 0; d < rank; ++d) {
      constancy.push_back(
          std::min(gcd(lhsInfo.getConstancy(d), condConstancy[d]),
                   gcd(rhsInfo.getConstancy(d), condConstancy[d])));
      divisibility.push_back(
          std::min(lhsInfo.getDivisibility(d), rhsInfo.getDivisibility(d)));
      contiguity.push_back(
          std::min(gcd(lhsInfo.getContiguity(d), condConstancy[d]),
                   gcd(rhsInfo.getContiguity(d), condConstancy[d])));
    }

    curr = AxisInfo(contiguity, divisibility, constancy);
  }

  // UnrealizedConversionCast
  // This is needed by TritonGPUToLLVM, to get AxisInfo when the graph is
  // in the process of a PartialConversion, where UnrealizedConversionCast
  // may exist
  if (llvm::isa<mlir::UnrealizedConversionCastOp>(op)) {
    curr = operands[0]->getValue();
  }
  if (curr.getRank() == 0) {
    return markAllPessimisticFixpoint(op->getResults());
  }
  llvm::errs() << *op << "\n";
  for (size_t d = 0; d < curr.getRank(); ++d) {
    llvm::errs() << "d: " << d << " contiguity: " << curr.getContiguity(d)
                 << " divisibility: " << curr.getDivisibility(d)
                 << " constancy: " << curr.getConstancy(d) << "\n";
  }


  // join all lattice elements
  ChangeResult result = ChangeResult::NoChange;
  for (Value value : op->getResults()) {
    result |= getLatticeElement(value).join(curr);
  }
  return result;
}

unsigned AxisInfoAnalysis::getPtrVectorSize(Value ptr) {
  auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return 1;
  auto layout = tensorTy.getEncoding();
  auto shape = tensorTy.getShape();

  // Here order should be ordered by contiguous first, so the first element
  // should have the largest contiguous.
  auto order = triton::gpu::getOrder(layout);
  unsigned align = getPtrAlignment(ptr);

  unsigned contigPerThread = triton::gpu::getSizePerThread(layout)[order[0]];
  unsigned vec = std::min(align, contigPerThread);
  vec = std::min<unsigned>(shape[order[0]], vec);

  return vec;
}

unsigned AxisInfoAnalysis::getPtrAlignment(Value ptr) {
  auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return 1;
  auto axisInfo = lookupLatticeElement(ptr)->getValue();
  auto layout = tensorTy.getEncoding();
  auto order = triton::gpu::getOrder(layout);
  unsigned maxMultiple = axisInfo.getDivisibility(order[0]);
  unsigned maxContig = axisInfo.getContiguity(order[0]);
  unsigned alignment = std::min(maxMultiple, maxContig);
  return alignment;
}

unsigned AxisInfoAnalysis::getMaskAlignment(Value mask) {
  auto tensorTy = mask.getType().dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return 1;
  auto maskOrder = triton::gpu::getOrder(tensorTy.getEncoding());
  auto maskAxis = lookupLatticeElement(mask)->getValue();
  auto alignment = std::max<unsigned>(maskAxis.getConstancy(maskOrder[0]), 1);
  return alignment;
}

} // namespace mlir
