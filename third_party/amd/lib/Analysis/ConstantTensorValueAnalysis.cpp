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
int64_t gcdPos(int64_t a, int64_t b) {
  a = a < 0 ? -a : a;
  b = b < 0 ? -b : b;
  while (b) {
    int64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
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
  if (!r.loInf && __builtin_add_overflow(a.lo, b.lo, &r.lo))
    r.loInf = true; // overflow -> conservatively unbounded
  if (!r.hiInf && __builtin_add_overflow(a.hi, b.hi, &r.hi))
    r.hiInf = true;
  return r;
}
Range rScale(const Range &a, int64_t k) {
  if (k == 0)
    return Range::point(0);
  Range r;
  if (k > 0) {
    r.loInf = a.loInf;
    r.hiInf = a.hiInf;
    if (!r.loInf && __builtin_mul_overflow(a.lo, k, &r.lo))
      r.loInf = true;
    if (!r.hiInf && __builtin_mul_overflow(a.hi, k, &r.hi))
      r.hiInf = true;
  } else {
    r.loInf = a.hiInf;
    r.hiInf = a.loInf;
    if (!r.loInf && __builtin_mul_overflow(a.hi, k, &r.lo))
      r.loInf = true;
    if (!r.hiInf && __builtin_mul_overflow(a.lo, k, &r.hi))
      r.hiInf = true;
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
  int64_t loBand = floorDiv(a.lo, c);
  if (loBand == floorDiv(a.hi, c))
    return loBand;
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
  // Set when an explicit IR subtraction may have driven the value negative.
  // floor-based mod/div folding is only valid on non-negative operands
  // (signed remsi/divsi truncate toward zero; unsigned remui/divui wrap).
  // Unknown scalars are treated as non-negative (memory-index convention).
  bool tainted = false;

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
  // (value, coord) -> Val memo. Makes each register's walk linear in DAG size
  // (no re-walking shared/CSE'd subterms) and lets coord-independent scalar
  // subtrees -- evaluated with empty coord -- be computed once and reused across
  // all registers. Keyed exactly (not by hash) to stay sound.
  std::map<std::string, Val> memo;
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
  // coeff * divisibility, std::nullopt on int64 overflow (so the drop check
  // conservatively treats the term as not provably a multiple of c).
  std::optional<int64_t> divOfLin(const void *symPtr, int64_t coeff) {
    auto it = symDiv.find(symPtr);
    int64_t d = it != symDiv.end() ? it->second : 1, r;
    if (__builtin_mul_overflow(coeff, d, &r))
      return std::nullopt;
    return r;
  }
  std::optional<int64_t> divOfOpq(const std::string &key, int64_t coeff) {
    auto it = atomDiv.find(key);
    int64_t d = it != atomDiv.end() ? it->second : 1, r;
    if (__builtin_mul_overflow(coeff, d, &r))
      return std::nullopt;
    return r;
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
  r.tainted = a.tainted || b.tainted;
  r.prune();
  return r;
}

Val scaleBy(Val a, int64_t k) {
  if (a.fatal)
    return a;
  if (k == 0)
    return Val::constant(0);
  // Overflow -> cannot represent the scaled value soundly.
  if (__builtin_mul_overflow(a.cst, k, &a.cst))
    return Val::ground();
  for (auto &kv : a.lin)
    if (__builtin_mul_overflow(kv.second, k, &kv.second))
      return Val::ground();
  for (auto &kv : a.opq)
    if (__builtin_mul_overflow(kv.second, k, &kv.second))
      return Val::ground();
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
  Val o = makeAtom(ctx, "*(" + keyOf(a) + "," + keyOf(b) + ")", /*div=*/1,
                   Range::full());
  o.tainted = a.tainted || b.tainted;
  return o;
}

// (sum a_i u_i + sum b_k t_k + cst) mod c.  Each symbolic term drops when
// c | coeff*divisibility(term). Else fall back to an opaque atom (Range [0,c)).
// Opaque result for a value floor-folding cannot soundly simplify (e.g. a
// possibly-negative operand under truncating/ wrapping mod-div).
Val taintedOpaque(const Val &a, char op, int64_t c, Ctx &ctx) {
  Val o = makeAtom(ctx, std::string(1, op) + std::to_string(c) + "!(" +
                            keyOf(a) + ")",
                   /*div=*/1, Range::full());
  o.tainted = true;
  return o;
}

Val modC(const Val &a, int64_t c, Ctx &ctx) {
  if (a.fatal || c <= 0)
    return a.fatal ? a : Val::ground();
  if (a.tainted) // possibly-negative operand: floor-mod != remsi/remui
    return taintedOpaque(a, '%', c, ctx);
  // Drop every term provably a multiple of c (c | coeff*divisibility); the
  // remaining residual (cst + surviving terms) must fit a single band
  // [m*c,(m+1)c) for the result to be exact = residual - m*c. This combines the
  // old M1 (all-drop) and M2 (whole-range-band) and crucially handles the mixed
  // case: e.g. (pid*128 + row + L_row) % 128 -> pid*128 drops, residual
  // row+L_row in [0,128) -> exact, even though pid's range is unknown.
  Val resid;
  resid.cst = floorMod(a.cst, c); // multiple-of-c part of the constant drops too
  resid.rng = Range::point(resid.cst);
  for (auto &kv : a.lin) {
    auto d = ctx.divOfLin(kv.first, kv.second);
    if (d && *d % c == 0)
      continue; // provably a multiple of c -> drops
    resid.lin[kv.first] += kv.second;
    resid.rng = Range::full(); // surviving symbol: range unknown
  }
  for (auto &kv : a.opq) {
    auto d = ctx.divOfOpq(kv.first, kv.second);
    if (d && *d % c == 0)
      continue; // provably a multiple of c -> drops
    resid.opq[kv.first] += kv.second;
    auto it = ctx.atomRng.find(kv.first);
    Range ar = it != ctx.atomRng.end() ? it->second : Range::full();
    resid.rng = rAdd(resid.rng, rScale(ar, kv.second));
  }
  resid.prune();
  if (auto m = rBand(resid.rng, c)) {
    int64_t mc;
    if (!__builtin_mul_overflow(*m, c, &mc))
      return addSub(resid, Val::constant(mc), /*sub=*/true);
  }
  // Fallback: opaque, keyed by the REDUCED residual (multiples dropped, cst
  // reduced mod c) so register-invariant operands that differ only by a
  // multiple of c -- e.g. (row+L)%16 at row=0 vs row=64 -- get the same key and
  // cancel in offset(r) - offset(0).
  Range r;
  r.loInf = r.hiInf = false;
  r.lo = 0;
  r.hi = c - 1;
  return makeAtom(ctx, "%" + std::to_string(c) + "(" + keyOf(resid) + ")",
                  /*div=*/1, r);
}

// floordiv c.  Pull terms whose *coefficient* is divisible by c into the exact
// quotient; the remaining residual must fit a single band (via range) to be
// exact, otherwise fall back to an opaque atom.
Val divC(const Val &a, int64_t c, Ctx &ctx) {
  if (a.fatal || c <= 0)
    return a.fatal ? a : Val::ground();
  if (a.tainted) // possibly-negative operand: floor-div != divsi/divui
    return taintedOpaque(a, '/', c, ctx);

  Val q;            // exact quotient of the c-divisible part
  Val rest;         // residual (must land in one band)
  // constant part
  q.cst = floorDiv(a.cst, c);
  q.rng = Range::point(q.cst);
  rest.cst = floorMod(a.cst, c);
  rest.rng = Range::point(rest.cst);
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
  Range r = a.rng;
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
  Val o = makeAtom(ctx, (isMin ? "min(" : "max(") + keyOf(a) + "," + keyOf(b) +
                            ")",
                   /*div=*/1, rHull(a.rng, b.rng));
  o.tainted = a.tainted || b.tainted;
  return o;
}

// Coordinates are now symbolic Vals: a coordinate axis = concrete register
// contribution + a register-invariant lane/warp/block atom (so contiguity is
// proven over ALL lanes, not just lane 0).
Val eval(Value v, ArrayRef<Val> coord, Ctx &ctx, unsigned depth);
Val evalImpl(Value v, ArrayRef<Val> coord, Ctx &ctx, unsigned depth);

// Memoizing wrapper: dedups shared DAG subterms within a register and reuses
// coord-independent (scalar) results across registers. Results produced via the
// depth cap are NOT memoized, so a truncated value can never be reused at a
// shallower depth.
Val eval(Value v, ArrayRef<Val> coord, Ctx &ctx, unsigned depth) {
  if (depth > kMaxDepth)
    return evalImpl(v, coord, ctx, depth);
  std::string key = std::to_string((uintptr_t)v.getAsOpaquePointer());
  key += '@';
  for (const Val &c : coord) {
    key += keyOf(c);
    key += ';';
  }
  auto it = ctx.memo.find(key);
  if (it != ctx.memo.end())
    return it->second;
  Val r = evalImpl(v, coord, ctx, depth);
  ctx.memo.emplace(std::move(key), r);
  return r;
}

// Constant fold; for a non-splat dense tensor we can only index it when the
// coordinate is fully concrete (no lane/warp symbolic part).
std::optional<int64_t> evalConstInt(Value v, ArrayRef<Val> coord) {
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
    if (!coord[i].pureConst())
      return std::nullopt; // symbolic lane component -> cannot index
    int64_t ci = coord[i].cst;
    if (ci < 0 || ci >= shape[i])
      return std::nullopt;
    flat = flat * shape[i] + ci;
  }
  auto vals = dense.getValues<APInt>();
  if (flat >= (int64_t)vals.size())
    return std::nullopt;
  return (*(vals.begin() + flat)).getSExtValue();
}

Val evalBinary(Operation *op, ArrayRef<Val> coord, Ctx &ctx, unsigned depth) {
  Val a = eval(op->getOperand(0), coord, ctx, depth + 1);
  Val b = eval(op->getOperand(1), coord, ctx, depth + 1);
  if (isa<arith::AddIOp>(op))
    return addSub(a, b, false);
  if (isa<arith::SubIOp>(op)) {
    Val r = addSub(a, b, true);
    r.tainted = true; // explicit subtraction may go negative
    return r;
  }
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

Val evalImpl(Value v, ArrayRef<Val> coord, Ctx &ctx, unsigned depth) {
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
    // start + coordinate (coordinate carries the symbolic lane/warp part)
    return addSub(Val::constant((int64_t)rng.getStart()), coord[0],
                  /*sub=*/false);
  }
  if (auto splat = dyn_cast<tt::SplatOp>(op))
    return eval(splat.getSrc(), /*coord=*/{}, ctx, depth + 1);
  if (auto bcast = dyn_cast<tt::BroadcastOp>(op)) {
    auto srcTy = cast<RankedTensorType>(bcast.getSrc().getType());
    SmallVector<Val> sc(coord.begin(), coord.end());
    for (auto [i, d] : llvm::enumerate(srcTy.getShape()))
      if (d == 1)
        sc[i] = Val::constant(0);
    return eval(bcast.getSrc(), sc, ctx, depth + 1);
  }
  if (auto ex = dyn_cast<tt::ExpandDimsOp>(op)) {
    unsigned axis = ex.getAxis();
    if (axis >= coord.size())
      return Val::ground();
    SmallVector<Val> sc;
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

  auto outDims = llvm::to_vector(ll.getOutDimNames());
  Ctx ctx(axisAnalysis);

  // Per-axis lane/warp/block contribution, modeled as a register-invariant
  // bounded atom: divisibility = gcd of the non-register basis components along
  // that axis, range = [sum of negatives, sum of positives]. This makes the
  // contiguity a theorem over ALL lanes/warps -- the existing mod/div rules
  // either drop or band-resolve this atom (MXFP4 pow2-aligned tilings), or it
  // survives and correctly caps contiguity (non-aligned / non-pow2 offsets).
  SmallVector<Val> laneAtom(rank);
  for (unsigned j = 0; j < rank; ++j) {
    int64_t g = 0, lo = 0, hi = 0;
    for (auto dim : ll.getInDimNames()) {
      if (dim == kReg)
        continue;
      for (int p = 0, e = ll.getInDimSizeLog2(dim); p < e; ++p) {
        int64_t comp = ll.getBasis(dim, p, outDims[j]);
        if (comp == 0)
          continue;
        g = gcdPos(g, comp);
        if (comp > 0)
          hi += comp;
        else
          lo += comp;
      }
    }
    if (lo == 0 && hi == 0) {
      laneAtom[j] = Val::constant(0);
    } else {
      Range r;
      r.loInf = r.hiInf = false;
      r.lo = lo;
      r.hi = hi;
      laneAtom[j] =
          makeAtom(ctx, "lane#" + std::to_string(j), g > 0 ? g : 1, r);
    }
  }

  auto coordAt = [&](int32_t regIdx) {
    SmallVector<std::pair<StringAttr, int32_t>> ins;
    for (auto dim : ll.getInDimNames())
      ins.push_back({dim, dim == kReg ? regIdx : 0});
    auto outs = ll.apply(ins);
    SmallVector<Val> coord(rank, Val::constant(0));
    for (auto [i, kv] : llvm::enumerate(outs))
      coord[i] = addSub(Val::constant(kv.second), laneAtom[i], /*sub=*/false);
    return coord;
  };

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
