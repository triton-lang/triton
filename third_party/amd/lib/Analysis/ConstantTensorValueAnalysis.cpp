#include "third_party/amd/include/Analysis/ConstantTensorValueAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <map>
#include <optional>
#include <string>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::AMD {

namespace {

constexpr unsigned kMaxDepth = 96;

int64_t floorDiv(int64_t a, int64_t b) {
  int64_t q = a / b, r = a % b;
  if (r != 0 && ((r < 0) != (b < 0)))
    --q;
  return q;
}
int64_t floorMod(int64_t a, int64_t b) {
  int64_t r = a % b;
  if (r != 0 && ((r < 0) != (b < 0)))
    r += b;
  return r;
}

// ---- Range (sound interval over-approximation; gates floordiv/mod/min/max) --
struct Range {
  int64_t lo = 0, hi = 0;
  bool loInf = true, hiInf = true;
  static Range full() { return {}; }
  static Range point(int64_t c) { return {c, c, false, false}; }
  bool finite() const { return !loInf && !hiInf; }
};
Range rAdd(const Range &a, const Range &b) {
  Range r;
  r.loInf = a.loInf || b.loInf;
  r.hiInf = a.hiInf || b.hiInf;
  if (!r.loInf)
    r.lo = a.lo + b.lo;
  if (!r.hiInf)
    r.hi = a.hi + b.hi;
  return r;
}
Range rScale(const Range &a, int64_t k) {
  if (k == 0)
    return Range::point(0);
  Range r;
  if (k > 0) {
    r.loInf = a.loInf;
    r.hiInf = a.hiInf;
    if (!r.loInf)
      r.lo = a.lo * k;
    if (!r.hiInf)
      r.hi = a.hi * k;
  } else {
    r.loInf = a.hiInf;
    r.hiInf = a.loInf;
    if (!r.loInf)
      r.lo = a.hi * k;
    if (!r.hiInf)
      r.hi = a.lo * k;
  }
  return r;
}
Range rHull(const Range &a, const Range &b) {
  Range r;
  r.loInf = a.loInf || b.loInf;
  r.hiInf = a.hiInf || b.hiInf;
  if (!r.loInf)
    r.lo = std::min(a.lo, b.lo);
  if (!r.hiInf)
    r.hi = std::max(a.hi, b.hi);
  return r;
}
// If the range lies inside a single band [m*c, (m+1)*c) return m.
std::optional<int64_t> rBand(const Range &a, int64_t c) {
  if (!a.finite())
    return std::nullopt;
  if (floorDiv(a.lo, c) == floorDiv(a.hi, c))
    return floorDiv(a.lo, c);
  return std::nullopt;
}

// ---- Abstract value: exact function of the unknowns at a fixed register -----
// V = cst + sum lin[sym]*sym + sum opq[atomKey]*atom.  `fatal` is bottom-truth
// (an unanalyzable tensor whose register variance is unknown).
struct Val {
  bool fatal = false;
  int64_t cst = 0;
  llvm::SmallDenseMap<const void *, int64_t, 2> lin; // scalar symbol -> coeff
  std::map<std::string, int64_t> opq;                // opaque atom key -> coeff
  Range rng;

  static Val ground() {
    Val v;
    v.fatal = true;
    return v;
  }
  static Val constant(int64_t c) {
    Val v;
    v.cst = c;
    v.rng = Range::point(c);
    return v;
  }
  bool pureConst() const { return !fatal && lin.empty() && opq.empty(); }
  void prune() {
    SmallVector<const void *> dl;
    for (auto &kv : lin)
      if (kv.second == 0)
        dl.push_back(kv.first);
    for (auto k : dl)
      lin.erase(k);
    for (auto it = opq.begin(); it != opq.end();)
      it = (it->second == 0) ? opq.erase(it) : std::next(it);
  }
};

// Per-analysis context: AxisInfo divisibility for scalar symbols + atom metadata.
struct Ctx {
  ModuleAxisInfoAnalysis &ai;
  llvm::SmallDenseMap<const void *, int64_t, 8> symDiv;
  std::map<std::string, int64_t> atomDiv;
  std::map<std::string, Range> atomRng;
  explicit Ctx(ModuleAxisInfoAnalysis &a) : ai(a) {}

  int64_t symDivOf(Value v) {
    auto *k = v.getAsOpaquePointer();
    auto it = symDiv.find(k);
    if (it != symDiv.end())
      return it->second;
    int64_t d = 1;
    if (AxisInfo *info = ai.getAxisInfo(v))
      d = std::max<int64_t>(1, info->getDivisibility(0));
    symDiv[k] = d;
    return d;
  }
  int64_t divOfLin(const void *symPtr, int64_t coeff) {
    auto it = symDiv.find(symPtr);
    int64_t d = it != symDiv.end() ? it->second : 1;
    return coeff * d;
  }
  int64_t divOfOpq(const std::string &key, int64_t coeff) {
    auto it = atomDiv.find(key);
    int64_t d = it != atomDiv.end() ? it->second : 1;
    return coeff * d;
  }
};

std::string keyOf(const Val &v) {
  if (v.fatal)
    return "F";
  SmallVector<std::pair<const void *, int64_t>> ls(v.lin.begin(), v.lin.end());
  llvm::sort(ls, [](auto &a, auto &b) { return a.first < b.first; });
  std::string s = "c" + std::to_string(v.cst);
  for (auto &kv : ls)
    s += "|s" + std::to_string((uintptr_t)kv.first) + ":" +
         std::to_string(kv.second);
  for (auto &kv : v.opq) // std::map already ordered
    s += "|o" + kv.first + ":" + std::to_string(kv.second);
  return s;
}

Val addSub(const Val &a, const Val &b, bool sub) {
  if (a.fatal || b.fatal)
    return Val::ground();
  Val r = a;
  r.cst += sub ? -b.cst : b.cst;
  for (auto &kv : b.lin)
    r.lin[kv.first] += sub ? -kv.second : kv.second;
  for (auto &kv : b.opq)
    r.opq[kv.first] += sub ? -kv.second : kv.second;
  r.rng = rAdd(a.rng, sub ? rScale(b.rng, -1) : b.rng);
  r.prune();
  return r;
}

Val scaleBy(Val a, int64_t k) {
  if (a.fatal)
    return a;
  if (k == 0)
    return Val::constant(0);
  a.cst *= k;
  for (auto &kv : a.lin)
    kv.second *= k;
  for (auto &kv : a.opq)
    kv.second *= k;
  a.rng = rScale(a.rng, k);
  return a;
}

// Wrap a nonlinear result as a single opaque atom (coeff 1), registering its
// divisibility/range so enclosing mod/floordiv can reason about it.
Val makeAtom(Ctx &ctx, const std::string &key, int64_t div, Range rng) {
  ctx.atomDiv[key] = div;
  ctx.atomRng[key] = rng;
  Val v;
  v.opq[key] = 1;
  v.rng = rng;
  return v;
}

Val mulV(const Val &a, const Val &b, Ctx &ctx) {
  if (a.fatal || b.fatal)
    return Val::ground();
  if (a.pureConst())
    return scaleBy(b, a.cst);
  if (b.pureConst())
    return scaleBy(a, b.cst);
  // unknown * unknown -> opaque
  int64_t div = 1; // conservative
  return makeAtom(ctx, "*(" + keyOf(a) + "," + keyOf(b) + ")", div,
                  Range::full());
}

// (sum a_i u_i + sum b_k t_k + cst) mod c.  Each symbolic term drops when
// c | coeff*divisibility(term). Else fall back to an opaque atom (Range [0,c)).
Val modC(const Val &a, int64_t c, Ctx &ctx) {
  if (a.fatal || c <= 0)
    return a.fatal ? a : Val::ground();
  // M1: every symbolic term is a multiple of c.
  bool allMultiple = true;
  for (auto &kv : a.lin)
    allMultiple &= (ctx.divOfLin(kv.first, kv.second) % c == 0);
  for (auto &kv : a.opq)
    allMultiple &= (ctx.divOfOpq(kv.first, kv.second) % c == 0);
  if (allMultiple)
    return Val::constant(floorMod(a.cst, c));
  // M2: range fits a single band.
  if (auto m = rBand(a.rng, c))
    return addSub(a, Val::constant(*m * c), /*sub=*/true);
  // Fallback: opaque, keyed by the whole value (cancels iff register-invariant).
  Range r;
  r.loInf = r.hiInf = false;
  r.lo = 0;
  r.hi = c - 1;
  return makeAtom(ctx, "%" + std::to_string(c) + "(" + keyOf(a) + ")",
                  /*div=*/1, r);
}

// floordiv c.  Pull terms whose *coefficient* is divisible by c into the exact
// quotient; the remaining residual must fit a single band (via range) to be
// exact, otherwise fall back to an opaque atom.
Val divC(const Val &a, int64_t c, Ctx &ctx) {
  if (a.fatal || c <= 0)
    return a.fatal ? a : Val::ground();

  Val q;            // exact quotient of the c-divisible part
  Val rest;         // residual (must land in one band)
  q.rng = Range::point(0);
  rest.rng = Range::point(0);
  // constant part
  q.cst = floorDiv(a.cst, c);
  rest.cst = floorMod(a.cst, c);
  rest.rng = rAdd(rest.rng, Range::point(rest.cst));
  q.rng = rAdd(q.rng, Range::point(q.cst));
  for (auto &kv : a.lin) {
    if (kv.second % c == 0) {
      q.lin[kv.first] += kv.second / c;
    } else {
      rest.lin[kv.first] += kv.second;
      // residual range from this symbol's range (unknown -> full)
      rest.rng = Range::full();
    }
  }
  for (auto &kv : a.opq) {
    if (kv.second % c == 0) {
      q.opq[kv.first] += kv.second / c;
    } else {
      rest.opq[kv.first] += kv.second;
      auto it = ctx.atomRng.find(kv.first);
      Range ar = it != ctx.atomRng.end() ? it->second : Range::full();
      rest.rng = rAdd(rest.rng, rScale(ar, kv.second));
    }
  }
  // D1: residual within a single band [m*c,(m+1)*c) -> quotient += m, exact.
  if (auto m = rBand(rest.rng, c)) {
    q.cst += *m;
    q.rng = rAdd(q.rng, Range::point(*m));
    q.prune();
    return q;
  }
  // Fallback: opaque DIV of the whole value.
  Range r = rScale(a.rng, 1); // floor(range/c)
  if (r.finite()) {
    r.lo = floorDiv(a.rng.lo, c);
    r.hi = floorDiv(a.rng.hi, c);
  } else {
    r = Range::full();
  }
  return makeAtom(ctx, "/" + std::to_string(c) + "(" + keyOf(a) + ")",
                  /*div=*/1, r);
}

Val minMax(const Val &a, const Val &b, bool isMin, Ctx &ctx) {
  if (a.fatal || b.fatal)
    return Val::ground();
  if (a.pureConst() && b.pureConst())
    return Val::constant(isMin ? std::min(a.cst, b.cst)
                               : std::max(a.cst, b.cst));
  // range-decidable ordering
  if (a.rng.finite() && b.rng.finite()) {
    if (a.rng.hi <= b.rng.lo)
      return isMin ? a : b;
    if (b.rng.hi <= a.rng.lo)
      return isMin ? b : a;
  }
  return makeAtom(ctx, (isMin ? "min(" : "max(") + keyOf(a) + "," + keyOf(b) +
                           ")",
                  /*div=*/1, rHull(a.rng, b.rng));
}

Val eval(Value v, ArrayRef<int64_t> coord, Ctx &ctx, unsigned depth);

std::optional<int64_t> evalConstInt(Value v, ArrayRef<int64_t> coord) {
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return std::nullopt;
  auto attr = cst.getValue();
  if (auto i = dyn_cast<IntegerAttr>(attr))
    return i.getInt();
  auto dense = dyn_cast<DenseElementsAttr>(attr);
  if (!dense || !dense.getElementType().isIntOrIndex())
    return std::nullopt;
  if (dense.isSplat())
    return dense.getSplatValue<APInt>().getSExtValue();
  auto ty = dyn_cast<RankedTensorType>(cst.getType());
  if (!ty || ty.getRank() != (int64_t)coord.size())
    return std::nullopt;
  auto shape = ty.getShape();
  int64_t flat = 0;
  for (int i = 0; i < ty.getRank(); ++i) {
    if (coord[i] < 0 || coord[i] >= shape[i])
      return std::nullopt;
    flat = flat * shape[i] + coord[i];
  }
  auto vals = dense.getValues<APInt>();
  if (flat >= (int64_t)vals.size())
    return std::nullopt;
  return (*(vals.begin() + flat)).getSExtValue();
}

Val evalBinary(Operation *op, ArrayRef<int64_t> coord, Ctx &ctx,
               unsigned depth) {
  Val a = eval(op->getOperand(0), coord, ctx, depth + 1);
  Val b = eval(op->getOperand(1), coord, ctx, depth + 1);
  if (isa<arith::AddIOp>(op))
    return addSub(a, b, false);
  if (isa<arith::SubIOp>(op))
    return addSub(a, b, true);
  if (isa<arith::MulIOp>(op))
    return mulV(a, b, ctx);
  if (isa<arith::DivSIOp, arith::DivUIOp>(op))
    return b.pureConst() ? divC(a, b.cst, ctx) : Val::ground();
  if (isa<arith::RemSIOp, arith::RemUIOp>(op))
    return b.pureConst() ? modC(a, b.cst, ctx) : Val::ground();
  if (isa<arith::ShLIOp>(op))
    return (b.pureConst() && b.cst >= 0 && b.cst < 62)
               ? scaleBy(a, int64_t(1) << b.cst)
               : Val::ground();
  if (isa<arith::ShRSIOp, arith::ShRUIOp>(op))
    return (b.pureConst() && b.cst >= 0 && b.cst < 62)
               ? divC(a, int64_t(1) << b.cst, ctx)
               : Val::ground();
  if (isa<arith::AndIOp>(op)) {
    auto mask = [&](const Val &x, const Val &m) -> std::optional<Val> {
      if (!m.pureConst())
        return std::nullopt;
      int64_t mv = m.cst;
      if (mv > 0 && (mv & (mv + 1)) == 0)
        return modC(x, mv + 1, ctx);
      if (x.pureConst())
        return Val::constant(x.cst & mv);
      return std::nullopt;
    };
    if (auto r = mask(a, b))
      return *r;
    if (auto r = mask(b, a))
      return *r;
    return Val::ground();
  }
  if (isa<arith::MinSIOp, arith::MinUIOp>(op))
    return minMax(a, b, /*isMin=*/true, ctx);
  if (isa<arith::MaxSIOp, arith::MaxUIOp>(op))
    return minMax(a, b, /*isMin=*/false, ctx);
  return Val::ground();
}

// Treat an opaque scalar (program-id, function arg, unhandled scalar op) as a
// single register-invariant symbol -- sound because a scalar does not depend on
// the register/coordinate.
Val scalarSymbol(Value v, Ctx &ctx) {
  ctx.symDivOf(v); // cache divisibility
  Val s;
  s.lin[v.getAsOpaquePointer()] = 1;
  s.rng = Range::full();
  return s;
}

Val eval(Value v, ArrayRef<int64_t> coord, Ctx &ctx, unsigned depth) {
  bool isTensor = isa<RankedTensorType>(v.getType());
  if (depth > kMaxDepth)
    return isTensor ? Val::ground() : scalarSymbol(v, ctx);

  if (auto c = evalConstInt(v, coord))
    return Val::constant(*c);

  Operation *op = v.getDefiningOp();
  if (!op) // block/function argument
    return isTensor ? Val::ground() : scalarSymbol(v, ctx);

  // tensor-shape ops
  if (auto rng = dyn_cast<tt::MakeRangeOp>(op)) {
    if (coord.size() != 1)
      return Val::ground();
    return Val::constant((int64_t)rng.getStart() + coord[0]);
  }
  if (auto splat = dyn_cast<tt::SplatOp>(op))
    return eval(splat.getSrc(), /*coord=*/{}, ctx, depth + 1);
  if (auto bcast = dyn_cast<tt::BroadcastOp>(op)) {
    auto srcTy = cast<RankedTensorType>(bcast.getSrc().getType());
    SmallVector<int64_t> sc(coord.begin(), coord.end());
    for (auto [i, d] : llvm::enumerate(srcTy.getShape()))
      if (d == 1)
        sc[i] = 0;
    return eval(bcast.getSrc(), sc, ctx, depth + 1);
  }
  if (auto ex = dyn_cast<tt::ExpandDimsOp>(op)) {
    unsigned axis = ex.getAxis();
    if (axis >= coord.size())
      return Val::ground();
    SmallVector<int64_t> sc;
    for (auto [i, c] : llvm::enumerate(coord))
      if (i != axis)
        sc.push_back(c);
    return eval(ex.getSrc(), sc, ctx, depth + 1);
  }
  if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op))
    return eval(cvt.getSrc(), coord, ctx, depth + 1);
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
    return eval(op->getOperand(0), coord, ctx, depth + 1);

  // arithmetic (recurses through scalar AND tensor ops, exposing literal
  // coefficients like pid*128 needed for the floordiv divisibility pull)
  if (op->getNumOperands() == 2 && op->getNumResults() == 1) {
    Val r = evalBinary(op, coord, ctx, depth);
    // A scalar result is register-invariant no matter how it is computed: if we
    // failed to decompose it (e.g. divsi by a non-constant inside the program-id
    // math), keep it as one opaque symbol rather than poisoning to bottom.
    if (r.fatal && !isTensor)
      return scalarSymbol(v, ctx);
    return r;
  }

  // Unhandled: a scalar is still register-invariant; a tensor is not.
  return isTensor ? Val::ground() : scalarSymbol(v, ctx);
}

} // namespace

unsigned getPerThreadContiguityFromLinearLayout(
    Value offsetsValue, mlir::triton::ModuleAxisInfoAnalysis &axisAnalysis) {
  auto tensorTy = dyn_cast<RankedTensorType>(offsetsValue.getType());
  if (!tensorTy)
    return 1;

  LinearLayout ll = ttg::toLinearLayout(tensorTy);
  MLIRContext *mctx = tensorTy.getContext();
  StringAttr kReg = StringAttr::get(mctx, "register");
  if (!llvm::is_contained(ll.getInDimNames(), kReg))
    return 1;
  unsigned numRegs = ll.getInDimSize(kReg);
  if (numRegs <= 1)
    return 1;
  unsigned rank = tensorTy.getRank();
  if (llvm::range_size(ll.getOutDimNames()) != rank)
    return 1;

  auto coordAt = [&](int32_t regIdx) {
    SmallVector<std::pair<StringAttr, int32_t>> ins;
    for (auto dim : ll.getInDimNames())
      ins.push_back({dim, dim == kReg ? regIdx : 0});
    auto outs = ll.apply(ins);
    SmallVector<int64_t> coord(rank, 0);
    for (auto [i, kv] : llvm::enumerate(outs))
      coord[i] = kv.second;
    return coord;
  };

  Ctx ctx(axisAnalysis);
  Val base = eval(offsetsValue, coordAt(0), ctx, 0);
  if (base.fatal)
    return 1;

  // March registers; trust r only when offset(r) - offset(0) reduces to the
  // exact constant r (no surviving unknown/opaque terms) -- a proof over all
  // unknown valuations, no substitution.
  unsigned consecutive = 1;
  while (consecutive < numRegs) {
    Val v = eval(offsetsValue, coordAt(consecutive), ctx, 0);
    Val delta = addSub(v, base, /*sub=*/true);
    if (delta.fatal || !delta.lin.empty() || !delta.opq.empty())
      break;
    if (delta.cst != (int64_t)consecutive)
      break;
    ++consecutive;
  }
  return llvm::bit_floor(consecutive);
}

} // namespace mlir::triton::AMD
