#include "third_party/amd/include/Analysis/ConstantTensorValueAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::AMD {

namespace {

// Recursion guard: deep IR chains shouldn't drag the analysis into pathological
// recursion. 64 is well above anything Triton currently emits for offsets.
constexpr unsigned kMaxDepth = 64;

// Strip integer cast ops we treat as identity for evaluation purposes.
std::optional<int64_t> eval(Value v, ArrayRef<int64_t> coord, int64_t subst,
                            unsigned depth);

std::optional<int64_t> evalConstant(arith::ConstantOp cst,
                                    ArrayRef<int64_t> coord) {
  auto attr = cst.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();
  auto dense = dyn_cast<DenseElementsAttr>(attr);
  if (!dense || !dense.getElementType().isIntOrIndex())
    return std::nullopt;
  if (dense.isSplat())
    return dense.getSplatValue<APInt>().getSExtValue();
  auto tensorTy = dyn_cast<RankedTensorType>(cst.getType());
  if (!tensorTy || tensorTy.getRank() != (int64_t)coord.size())
    return std::nullopt;
  // Look up the element at `coord` in a non-splat dense tensor.
  auto shape = tensorTy.getShape();
  int64_t flat = 0;
  for (int i = 0; i < tensorTy.getRank(); ++i) {
    if (coord[i] < 0 || coord[i] >= shape[i])
      return std::nullopt;
    flat = flat * shape[i] + coord[i];
  }
  auto values = dense.getValues<APInt>();
  if (flat >= (int64_t)values.size())
    return std::nullopt;
  return (*(values.begin() + flat)).getSExtValue();
}

std::optional<int64_t> evalBinary(Operation *op, ArrayRef<int64_t> coord,
                                  int64_t subst, unsigned depth) {
  auto lhs = eval(op->getOperand(0), coord, subst, depth + 1);
  if (!lhs)
    return std::nullopt;
  auto rhs = eval(op->getOperand(1), coord, subst, depth + 1);
  if (!rhs)
    return std::nullopt;
  int64_t a = *lhs, b = *rhs;
  if (isa<arith::AddIOp>(op))
    return a + b;
  if (isa<arith::SubIOp>(op))
    return a - b;
  if (isa<arith::MulIOp>(op))
    return a * b;
  if (isa<arith::DivSIOp>(op))
    return b == 0 ? std::nullopt : std::optional<int64_t>(a / b);
  if (isa<arith::DivUIOp>(op))
    return b == 0 ? std::nullopt
                  : std::optional<int64_t>((uint64_t)a / (uint64_t)b);
  if (isa<arith::RemSIOp>(op))
    return b == 0 ? std::nullopt : std::optional<int64_t>(a % b);
  if (isa<arith::RemUIOp>(op))
    return b == 0 ? std::nullopt
                  : std::optional<int64_t>((uint64_t)a % (uint64_t)b);
  if (isa<arith::ShLIOp>(op))
    return a << (b & 63);
  if (isa<arith::ShRSIOp>(op))
    return a >> (b & 63);
  if (isa<arith::ShRUIOp>(op))
    return (int64_t)((uint64_t)a >> (b & 63));
  if (isa<arith::AndIOp>(op))
    return a & b;
  if (isa<arith::OrIOp>(op))
    return a | b;
  if (isa<arith::XOrIOp>(op))
    return a ^ b;
  return std::nullopt;
}

std::optional<int64_t> eval(Value v, ArrayRef<int64_t> coord, int64_t subst,
                            unsigned depth) {
  if (depth > kMaxDepth)
    return std::nullopt;

  auto tensorTy = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorTy) {
    if (auto cst = v.getDefiningOp<arith::ConstantOp>())
      return evalConstant(cst, /*coord=*/{});
    // Unknown scalar (kernel arg, block arg, ...) -- substitute.
    return subst;
  }

  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;

  if (auto cst = dyn_cast<arith::ConstantOp>(op))
    return evalConstant(cst, coord);

  if (auto rng = dyn_cast<tt::MakeRangeOp>(op)) {
    // 1D tensor with values [start, start+1, ...).
    if (coord.size() != 1)
      return std::nullopt;
    return (int64_t)rng.getStart() + coord[0];
  }

  if (auto splat = dyn_cast<tt::SplatOp>(op))
    return eval(splat.getSrc(), /*coord=*/{}, subst, depth + 1);

  if (auto bcast = dyn_cast<tt::BroadcastOp>(op)) {
    auto srcTy = cast<RankedTensorType>(bcast.getSrc().getType());
    // For every axis where srcShape == 1, the coord collapses to 0.
    SmallVector<int64_t> srcCoord(coord.begin(), coord.end());
    for (auto [i, d] : llvm::enumerate(srcTy.getShape()))
      if (d == 1)
        srcCoord[i] = 0;
    return eval(bcast.getSrc(), srcCoord, subst, depth + 1);
  }

  if (auto ex = dyn_cast<tt::ExpandDimsOp>(op)) {
    unsigned axis = ex.getAxis();
    if (axis >= coord.size())
      return std::nullopt;
    SmallVector<int64_t> srcCoord;
    srcCoord.reserve(coord.size() - 1);
    for (auto [i, c] : llvm::enumerate(coord))
      if (i != axis)
        srcCoord.push_back(c);
    return eval(ex.getSrc(), srcCoord, subst, depth + 1);
  }

  if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op))
    return eval(cvt.getSrc(), coord, subst, depth + 1);

  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
    return eval(op->getOperand(0), coord, subst, depth + 1);

  if (op->getNumOperands() == 2 && op->getNumResults() == 1)
    return evalBinary(op, coord, subst, depth);

  return std::nullopt;
}

} // namespace

std::optional<int64_t> evaluateAt(Value value, ArrayRef<int64_t> coord,
                                  int64_t unknownScalarSubst) {
  return eval(value, coord, unknownScalarSubst, 0);
}

unsigned getPerThreadContiguityFromLinearLayout(
    Value offsetsValue, mlir::triton::ModuleAxisInfoAnalysis &axisAnalysis) {
  auto tensorTy = dyn_cast<RankedTensorType>(offsetsValue.getType());
  if (!tensorTy)
    return 1;

  // register -> tensor-coord map, straight from the tensor's LinearLayout.
  LinearLayout ll = ttg::toLinearLayout(tensorTy);
  MLIRContext *ctx = tensorTy.getContext();
  StringAttr kReg = StringAttr::get(ctx, "register");
  if (!llvm::is_contained(ll.getInDimNames(), kReg))
    return 1;
  unsigned numRegs = ll.getInDimSize(kReg);
  if (numRegs <= 1)
    return 1;
  unsigned numRegBits = llvm::Log2_32(numRegs);

  auto outDims = llvm::to_vector(ll.getOutDimNames());
  unsigned rank = tensorTy.getRank();
  if (outDims.size() != rank)
    return 1;

  // Tensor coord reached by register value `regIdx` (all other in-dims = 0).
  auto coordAt = [&](int32_t regIdx) {
    SmallVector<std::pair<StringAttr, int32_t>> ins;
    ins.reserve(ll.getNumInDims());
    for (auto dim : ll.getInDimNames())
      ins.push_back({dim, dim == kReg ? regIdx : 0});
    auto outs = ll.apply(ins);
    SmallVector<int64_t> coord(rank, 0);
    for (auto [i, kv] : llvm::enumerate(outs))
      coord[i] = kv.second;
    return coord;
  };

  // Recover the register->offset map for one unknown-scalar substitution:
  // basis[b] = offset(2^b) - offset(0). Returns std::nullopt if the chain has
  // an unsupported op, or if the map is NOT GF(2)-linear over the register
  // subspace (some composite register value disagrees with the XOR-free sum of
  // its set-bit basis deltas -- e.g. an offset with carries between bits).
  auto recoverBasis =
      [&](int64_t subst) -> std::optional<SmallVector<int64_t>> {
    auto base = evaluateAt(offsetsValue, coordAt(0), subst);
    if (!base)
      return std::nullopt;
    SmallVector<int64_t> basis(numRegBits, 0);
    for (unsigned b = 0; b < numRegBits; ++b) {
      auto v = evaluateAt(offsetsValue, coordAt(1u << b), subst);
      if (!v)
        return std::nullopt;
      basis[b] = *v - *base;
    }
    // Linearity check across the full register subspace.
    for (unsigned r = 1; r < numRegs; ++r) {
      auto v = evaluateAt(offsetsValue, coordAt(r), subst);
      if (!v)
        return std::nullopt;
      int64_t predicted = 0;
      for (unsigned b = 0; b < numRegBits; ++b)
        if (r & (1u << b))
          predicted += basis[b];
      if (*v - *base != predicted)
        return std::nullopt; // not linear -> not a valid register->offset LL.
    }
    return basis;
  };

  // Structural independence from kernel/loop scalars: require the recovered
  // basis images to be identical across several substitutions. Any term whose
  // register-stride depends on an unknown scalar shows up as a basis mismatch.
  unsigned divisibility = std::max<unsigned>(
      1, axisAnalysis.getAlignment(offsetsValue, /*elementBitWidth=*/8));
  const int64_t substs[] = {0,
                            1,
                            3,
                            7,
                            static_cast<int64_t>(divisibility),
                            static_cast<int64_t>(divisibility) - 1,
                            0x40000001};
  std::optional<SmallVector<int64_t>> ref;
  for (int64_t s : substs) {
    auto basis = recoverBasis(s);
    if (!basis)
      return 1;
    if (!ref)
      ref = basis;
    else if (*ref != *basis)
      return 1; // register-stride depends on an unknown scalar.
  }

  // Read contiguity straight off the verified linear map: registers [0..2^k)
  // hit offsets [0..2^k) iff basis bit b maps to 2^b for every b < k, taken
  // from bit 0 upward.
  unsigned contigBits = 0;
  while (contigBits < numRegBits &&
         (*ref)[contigBits] == (int64_t)(1u << contigBits))
    ++contigBits;
  return 1u << contigBits;
}

} // namespace mlir::triton::AMD
