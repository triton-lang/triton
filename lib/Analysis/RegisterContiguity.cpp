#include "triton/Analysis/RegisterContiguity.h"

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
#include <string>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton {

namespace {

constexpr unsigned kMaxDepth = 64;

// Per-analysis context: AxisInfo (for sound per-scalar divisibility) plus a
// cache of each unknown scalar symbol's divisibility.
struct Ctx {
  ModuleAxisInfoAnalysis &ai;
  llvm::SmallDenseMap<const void *, int64_t, 4> symDiv;
  explicit Ctx(ModuleAxisInfoAnalysis &a) : ai(a) {}
  int64_t divisibilityOf(Value v) {
    auto it = symDiv.find(v.getAsOpaquePointer());
    if (it != symDiv.end())
      return it->second;
    int64_t d = 1;
    if (AxisInfo *info = ai.getAxisInfo(v))
      d = std::max<int64_t>(1, info->getDivisibility(0));
    symDiv[v.getAsOpaquePointer()] = d;
    return d;
  }
};

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

// Abstract value for an integer at one tensor coordinate:
//
//     cst  +  sum_i symCoeff_i * symbol_i  +  sum_j opqCoeff_j * opaque_j
//
// `symbol_i` are unknown *scalar* SSA values (kernel args, loop IVs) -- always
// register-invariant. `opaque_j` are register-invariant-or-not nonlinear
// subterms (e.g. symbol*symbol, mod/div of symbolic values) identified by a
// structural key. Two opaque atoms with equal keys are the *same* runtime value
// and therefore cancel in offset(r) - offset(0); a key built from
// register-varying operands differs across registers and correctly survives the
// subtraction (flagging a scalar/coord-dependent stride). `fatal` is true ground
// (an unanalyzable tensor whose register-variance is unknown).
struct Val {
  bool fatal = false;
  int64_t cst = 0;
  llvm::SmallDenseMap<const void *, int64_t, 2> sym; // symbol id -> coeff
  std::map<std::string, int64_t> opq;                // opaque key -> coeff

  static Val ground() {
    Val v;
    v.fatal = true;
    return v;
  }
  static Val constant(int64_t c) {
    Val v;
    v.cst = c;
    return v;
  }
  bool pureConst() const { return !fatal && sym.empty() && opq.empty(); }
  bool affineFoldable() const { return !fatal && opq.empty(); }
  void prune() {
    SmallVector<const void *> ds;
    for (auto &kv : sym)
      if (kv.second == 0)
        ds.push_back(kv.first);
    for (auto k : ds)
      sym.erase(k);
    for (auto it = opq.begin(); it != opq.end();)
      it = (it->second == 0) ? opq.erase(it) : std::next(it);
  }
};

std::string keyOf(const Val &v) {
  if (v.fatal)
    return "F";
  // Deterministic within one analysis run (symbol ids are stable pointers).
  SmallVector<std::pair<const void *, int64_t>> syms(v.sym.begin(),
                                                     v.sym.end());
  llvm::sort(syms, [](auto &a, auto &b) { return a.first < b.first; });
  std::string s = "c" + std::to_string(v.cst);
  for (auto &kv : syms)
    s += "|s" + std::to_string((uintptr_t)kv.first) + ":" +
         std::to_string(kv.second);
  for (auto &kv : v.opq) // std::map is already ordered
    s += "|o" + kv.first + ":" + std::to_string(kv.second);
  return s;
}

Val scaleBy(Val a, int64_t k) {
  if (a.fatal)
    return a;
  if (k == 0)
    return Val::constant(0);
  a.cst *= k;
  for (auto &kv : a.sym)
    kv.second *= k;
  for (auto &kv : a.opq)
    kv.second *= k;
  return a;
}

Val addSub(const Val &a, const Val &b, bool sub) {
  if (a.fatal || b.fatal)
    return Val::ground();
  Val r = a;
  r.cst += sub ? -b.cst : b.cst;
  for (auto &kv : b.sym)
    r.sym[kv.first] += sub ? -kv.second : kv.second;
  for (auto &kv : b.opq)
    r.opq[kv.first] += sub ? -kv.second : kv.second;
  r.prune();
  return r;
}

// Wrap a non-affine result as a single opaque atom (coeff 1). Deterministic in
// its operand identities, so register-invariant inputs yield a canceling atom.
Val opaque(const std::string &key) {
  Val v;
  v.opq[key] = 1;
  return v;
}

Val mul(const Val &a, const Val &b) {
  if (a.fatal || b.fatal)
    return Val::ground();
  if (a.pureConst())
    return scaleBy(b, a.cst);
  if (b.pureConst())
    return scaleBy(a, b.cst);
  return opaque("*(" + keyOf(a) + "," + keyOf(b) + ")");
}

Val divByConst(const Val &a, int64_t c) {
  if (a.fatal || c <= 0)
    return a.fatal ? a : Val::ground();
  if (a.affineFoldable()) {
    bool ok = true;
    for (auto &kv : a.sym)
      ok &= (kv.second % c == 0);
    if (ok) {
      Val r;
      r.cst = floorDiv(a.cst, c);
      for (auto &kv : a.sym)
        r.sym[kv.first] = kv.second / c;
      r.prune();
      return r;
    }
  }
  return opaque("/" + std::to_string(c) + "(" + keyOf(a) + ")");
}

// (sum_i a_i*sym_i + cst) % c == cst % c, valid iff c divides each
// a_i*divisibility(sym_i) -- i.e. each symbolic term is provably a multiple of
// c (AxisInfo's alignment facts supply divisibility(sym_i)).
Val modByConst(const Val &a, int64_t c, Ctx &ctx) {
  if (a.fatal || c <= 0)
    return a.fatal ? a : Val::ground();
  if (a.opq.empty()) {
    bool ok = true;
    for (auto &kv : a.sym) {
      Value sym = Value::getFromOpaquePointer(kv.first);
      ok &= ((kv.second * ctx.divisibilityOf(sym)) % c == 0);
    }
    if (ok)
      return Val::constant(floorMod(a.cst, c));
  }
  return opaque("%" + std::to_string(c) + "(" + keyOf(a) + ")");
}

Val eval(Value v, ArrayRef<int64_t> coord, unsigned depth, Ctx &ctx);

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

Val evalBinary(Operation *op, ArrayRef<int64_t> coord, unsigned depth,
               Ctx &ctx) {
  Val a = eval(op->getOperand(0), coord, depth + 1, ctx);
  Val b = eval(op->getOperand(1), coord, depth + 1, ctx);
  if (isa<arith::AddIOp>(op))
    return addSub(a, b, false);
  if (isa<arith::SubIOp>(op))
    return addSub(a, b, true);
  if (isa<arith::MulIOp>(op))
    return mul(a, b);
  if (isa<arith::DivSIOp, arith::DivUIOp>(op))
    return b.pureConst() ? divByConst(a, b.cst) : Val::ground();
  if (isa<arith::RemSIOp, arith::RemUIOp>(op))
    return b.pureConst() ? modByConst(a, b.cst, ctx) : Val::ground();
  if (isa<arith::ShLIOp>(op))
    return (b.pureConst() && b.cst >= 0 && b.cst < 62)
               ? scaleBy(a, int64_t(1) << b.cst)
               : Val::ground();
  if (isa<arith::ShRSIOp, arith::ShRUIOp>(op))
    return (b.pureConst() && b.cst >= 0 && b.cst < 62)
               ? divByConst(a, int64_t(1) << b.cst)
               : Val::ground();
  if (isa<arith::AndIOp>(op)) {
    auto tryMask = [&](const Val &x, const Val &m) -> std::optional<Val> {
      if (!m.pureConst())
        return std::nullopt;
      int64_t mv = m.cst;
      if (mv > 0 && (mv & (mv + 1)) == 0) // mv == 2^k - 1
        return modByConst(x, mv + 1, ctx);
      if (x.pureConst())
        return Val::constant(x.cst & mv);
      return opaque("&" + std::to_string(mv) + "(" + keyOf(x) + ")");
    };
    if (auto r = tryMask(a, b))
      return *r;
    if (auto r = tryMask(b, a))
      return *r;
    return Val::ground();
  }
  if (isa<arith::OrIOp, arith::XOrIOp>(op)) {
    bool isOr = isa<arith::OrIOp>(op);
    if (a.pureConst() && b.pureConst())
      return Val::constant(isOr ? (a.cst | b.cst) : (a.cst ^ b.cst));
    return opaque((isOr ? "|(" : "^(") + keyOf(a) + "," + keyOf(b) + ")");
  }
  return Val::ground();
}

Val eval(Value v, ArrayRef<int64_t> coord, unsigned depth, Ctx &ctx) {
  if (depth > kMaxDepth)
    return Val::ground();

  auto tensorTy = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorTy) {
    if (auto c = evalConstInt(v, /*coord=*/{}))
      return Val::constant(*c);
    // Unknown *scalar* -> register-invariant symbol; cache its divisibility.
    ctx.divisibilityOf(v);
    Val s;
    s.sym[v.getAsOpaquePointer()] = 1;
    return s;
  }

  Operation *op = v.getDefiningOp();
  if (!op)
    return Val::ground();

  if (isa<arith::ConstantOp>(op)) {
    if (auto c = evalConstInt(v, coord))
      return Val::constant(*c);
    return Val::ground();
  }
  if (auto rng = dyn_cast<tt::MakeRangeOp>(op)) {
    if (coord.size() != 1)
      return Val::ground();
    return Val::constant((int64_t)rng.getStart() + coord[0]);
  }
  if (auto splat = dyn_cast<tt::SplatOp>(op))
    return eval(splat.getSrc(), /*coord=*/{}, depth + 1, ctx);
  if (auto bcast = dyn_cast<tt::BroadcastOp>(op)) {
    auto srcTy = cast<RankedTensorType>(bcast.getSrc().getType());
    SmallVector<int64_t> srcCoord(coord.begin(), coord.end());
    for (auto [i, d] : llvm::enumerate(srcTy.getShape()))
      if (d == 1)
        srcCoord[i] = 0;
    return eval(bcast.getSrc(), srcCoord, depth + 1, ctx);
  }
  if (auto ex = dyn_cast<tt::ExpandDimsOp>(op)) {
    unsigned axis = ex.getAxis();
    if (axis >= coord.size())
      return Val::ground();
    SmallVector<int64_t> srcCoord;
    for (auto [i, c] : llvm::enumerate(coord))
      if (i != axis)
        srcCoord.push_back(c);
    return eval(ex.getSrc(), srcCoord, depth + 1, ctx);
  }
  if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(op))
    return eval(cvt.getSrc(), coord, depth + 1, ctx);
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
    return eval(op->getOperand(0), coord, depth + 1, ctx);
  if (op->getNumOperands() == 2 && op->getNumResults() == 1)
    return evalBinary(op, coord, depth, ctx);

  return Val::ground();
}

} // namespace

unsigned getPerThreadContiguityAlongRegisters(Value offsetsValue,
                                              ModuleAxisInfoAnalysis &axisInfo) {
  auto tensorTy = dyn_cast<RankedTensorType>(offsetsValue.getType());
  if (!tensorTy)
    return 1;

  LinearLayout ll = ttg::toLinearLayout(tensorTy);
  MLIRContext *ctx = tensorTy.getContext();
  StringAttr kReg = StringAttr::get(ctx, "register");
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

  Ctx ectx(axisInfo);
  Val base = eval(offsetsValue, coordAt(0), 0, ectx);
  if (base.fatal)
    return 1;

  // Trust register r only when offset(r) - offset(0) is a scalar-INDEPENDENT
  // constant: all symbol and opaque terms must cancel (proving the stride does
  // not depend on any unknown scalar or coord-varying nonlinear subterm).
  unsigned consecutive = 1;
  while (consecutive < numRegs) {
    Val v = eval(offsetsValue, coordAt(consecutive), 0, ectx);
    Val delta = addSub(v, base, /*sub=*/true);
    if (delta.fatal || !delta.sym.empty() || !delta.opq.empty())
      break;
    if (delta.cst != (int64_t)consecutive)
      break;
    ++consecutive;
  }
  return llvm::bit_floor(consecutive);
}

} // namespace mlir::triton
