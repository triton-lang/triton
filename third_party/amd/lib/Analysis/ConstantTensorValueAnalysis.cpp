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

unsigned getPerThreadConsecutiveContiguity(
    Value offsetsValue, mlir::triton::ModuleAxisInfoAnalysis &axisAnalysis) {
  auto tensorTy = dyn_cast<RankedTensorType>(offsetsValue.getType());
  if (!tensorTy)
    return 1;

  // Map register bases -> tensor-coord deltas via the offsets tensor's linear
  // layout. lane=0, warp=0, block=0 picks a single thread; register basis i
  // contributes a delta on (out)dims.
  LinearLayout ll = ttg::toLinearLayout(tensorTy);
  MLIRContext *ctx = tensorTy.getContext();
  StringAttr kReg = StringAttr::get(ctx, "register");
  if (!llvm::is_contained(ll.getInDimNames(), kReg))
    return 1;
  unsigned numRegs = ll.getInDimSize(kReg);
  if (numRegs <= 1)
    return 1;

  auto outDims = llvm::to_vector(ll.getOutDimNames());
  unsigned rank = tensorTy.getRank();
  if (outDims.size() != rank)
    return 1;

  // ll.apply returns outs in getOutDimNames() order, which matches `outDims`.
  auto coordAt = [&](int32_t regIdx) {
    SmallVector<std::pair<StringAttr, int32_t>> ins;
    ins.reserve(ll.getNumInDims());
    for (auto dim : ll.getInDimNames())
      ins.push_back({dim, dim == kReg ? regIdx : 0});
    auto outs = ll.apply(ins);
    assert(outs.size() == rank);
    SmallVector<int64_t> coord(rank, 0);
    for (auto [i, kv] : llvm::enumerate(outs)) {
      assert(kv.first == outDims[i]);
      coord[i] = kv.second;
    }
    return coord;
  };

  // Per-register memory deltas relative to register 0.
  auto deltasFor = [&](int64_t subst) -> std::optional<SmallVector<int64_t>> {
    auto base = evaluateAt(offsetsValue, coordAt(0), subst);
    if (!base)
      return std::nullopt;
    SmallVector<int64_t> deltas{0};
    deltas.reserve(numRegs);
    for (unsigned r = 1; r < numRegs; ++r) {
      auto v = evaluateAt(offsetsValue, coordAt(r), subst);
      if (!v)
        return std::nullopt;
      deltas.push_back(*v - *base);
    }
    return deltas;
  };

  // To prove the contiguity is independent of unknown scalars (kernel args,
  // block args), probe with two distinct substitutions and require the
  // per-register deltas to agree. Use 0 and the AxisInfo divisibility of the
  // offsets value.
  unsigned divisibility = std::max<unsigned>(
      1, axisAnalysis.getAlignment(offsetsValue, /*elementBitWidth=*/8));
  auto deltasA = deltasFor(0);
  auto deltasB = deltasFor(divisibility);
  if (!deltasA || !deltasB || *deltasA != *deltasB)
    return 1;

  // Largest prefix where deltas[r] == r (i.e. registers [0..N) access
  // consecutive offsets), rounded down to a power of two.
  unsigned consecutive = 1;
  while (consecutive < numRegs &&
         (*deltasA)[consecutive] == (int64_t)consecutive)
    ++consecutive;
  return llvm::bit_floor(consecutive);
}

} // namespace mlir::triton::AMD
