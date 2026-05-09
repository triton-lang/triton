/// IndexSimplification.cpp — Z3-backed index arithmetic simplification
///
/// Problem
/// -------
/// PyTorch Inductor generates Triton POI kernels with multi-level mixed-radix
/// index expressions such as:
///
///   x0 + 512*(tmp4%614400) + 1228800*((x0 + 512*(tmp4%2400)) // 1228800)
///
/// where x0 = xindex%512 and tmp4 = xindex//512.  This expression equals
/// xindex, but the redundant modulo/division chain creates a long dependency
/// that causes high VRF bank-conflict rates on some hardware.
///
/// Architecture
/// ------------
/// The pass runs in three phases:
///
/// Phase 1 — Pure algebraic patterns (always valid, no solver):
///
///   (a) (x % N) // N  →  0
///   (b) (a%M) + M*((a//M)%N)  →  a%(M*N)   [mixed-radix identity]
///
/// Phase 2 — Z3-backed range simplification (requires TRITON_HAVE_Z3):
///
///   A Z3IndexProver builds a symbolic model of the MLIR SSA graph using
///   unbounded Z3 integer variables.  It collects three kinds of constraints:
///
///     1. tt.make_range {start, end}  →  start ≤ x < end
///     2. tt.get_program_id           →  x ≥ 0
///     3. arith.cmpi slt(v, const)    →  v < const  (xmask guard assumptions)
///     4. arith.remsi(v, N)           →  result = mod(v, N)
///
///   For each arith.divsi / arith.remsi in the IR it asks Z3:
///   "Under the collected constraints, is the LHS always in [0, N)?
///    If yes: divsi → 0 / remsi → identity."
///
/// Phase 3 — Re-apply algebraic patterns to exploit newly simplified ops.
///
/// Graceful degradation
/// --------------------
/// When libz3 is not available at build time (TRITON_HAVE_Z3 not defined),
/// Phase 2 is skipped.  Phase 1 + Phase 3 algebraic rewrites still apply.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#ifdef TRITON_HAVE_Z3
#include "z3++.h"
#endif
#include <unordered_map>

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONINDEXSIMPLIFICATION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helpers shared by patterns and the Z3 prover
//===----------------------------------------------------------------------===//

/// Return the constant integer value of a scalar arith.constant or a
/// dense-splat tensor arith.constant, including through tt.splat.
static std::optional<int64_t> getSplatIntConst(Value v) {
  if (auto splatOp = v.getDefiningOp<triton::SplatOp>())
    return getSplatIntConst(splatOp.getSrc());
  auto constOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;
  if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
    return intAttr.getInt();
  if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue()))
    if (denseAttr.isSplat())
      return denseAttr.getSplatValue<APInt>().getSExtValue();
  return std::nullopt;
}

/// Create a zero constant with the given type (scalar or tensor).
static Value makeZeroConst(OpBuilder &b, Location loc, Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return b.create<arith::ConstantOp>(loc, IntegerAttr::get(intTy, 0));
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    auto elemTy = cast<IntegerType>(tensorTy.getElementType());
    return b.create<arith::ConstantOp>(
        loc, DenseIntElementsAttr::get(tensorTy, APInt(elemTy.getWidth(), 0)));
  }
  llvm_unreachable("makeZeroConst: unsupported type");
}

/// Create a splat integer constant with the given value and type.
static Value makeSplatIntConst(OpBuilder &b, Location loc, Type type,
                               int64_t val) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return b.create<arith::ConstantOp>(loc, IntegerAttr::get(intTy, val));
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    auto elemTy = cast<IntegerType>(tensorTy.getElementType());
    return b.create<arith::ConstantOp>(
        loc, DenseIntElementsAttr::get(
                 tensorTy, APInt(elemTy.getWidth(), val, /*isSigned=*/true)));
  }
  llvm_unreachable("makeSplatIntConst: unsupported type");
}

/// Strip arith type-cast ops (extsi / trunci / index_cast) recursively.
/// Used to compare the "base" integer value across type-widened SSA chains.
static Value stripTypeCasts(Value v) {
  while (Operation *defOp = v.getDefiningOp()) {
    if (auto op = dyn_cast<arith::ExtSIOp>(defOp)) { v = op.getIn(); continue; }
    if (auto op = dyn_cast<arith::TruncIOp>(defOp)) { v = op.getIn(); continue; }
    if (auto op = dyn_cast<arith::IndexCastOp>(defOp)) { v = op.getIn(); continue; }
    break;
  }
  return v;
}

//===----------------------------------------------------------------------===//
// Z3IndexProver  (requires TRITON_HAVE_Z3)
//===----------------------------------------------------------------------===//

#ifdef TRITON_HAVE_Z3

/// Cache key for Z3 range queries: (value, lo, hi).
struct RangeQueryKey {
  void *valPtr;
  int64_t lo, hi;
  bool operator==(const RangeQueryKey &o) const {
    return valPtr == o.valPtr && lo == o.lo && hi == o.hi;
  }
};
struct RangeQueryKeyHash {
  size_t operator()(const RangeQueryKey &k) const {
    size_t h = std::hash<void *>{}(k.valPtr);
    h ^= std::hash<int64_t>{}(k.lo) + 0x9e3779b9u + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.hi) + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h;
  }
};

class Z3IndexProver {
public:
  explicit Z3IndexProver(unsigned timeoutMs = 500)
      : ctx_(), timeoutMs_(timeoutMs) {}

  Z3IndexProver(const Z3IndexProver &) = delete;
  Z3IndexProver &operator=(const Z3IndexProver &) = delete;

  /// Walk `root`, translate every SSA result to a Z3 expression, and collect
  /// all intrinsic and guard constraints.  Must be called before canProveInRange.
  void collectConstraints(Operation *root) {
    // Pass 1: translate all results (builds cache + adds remsi/range/pid
    // constraints as a side effect of translation).
    root->walk([&](Operation *op) {
      for (Value result : op->getResults())
        (void)translate(result);
    });

    // Pass 2: collect guard constraints from comparison predicates.
    root->walk([&](arith::CmpIOp cmpOp) {
      if (cmpOp.getPredicate() != arith::CmpIPredicate::slt)
        return;
      auto bound = getSplatIntConst(cmpOp.getRhs());
      if (!bound || *bound <= 0)
        return;
      z3::expr lhsExpr = translate(cmpOp.getLhs());
      constraints_.push_back(lhsExpr >= ctx_.int_val((int64_t)0));
      constraints_.push_back(lhsExpr < ctx_.int_val(*bound));
    });

    // Pass 3: inject scf.for induction variable bounds.
    root->walk([&](scf::ForOp forOp) {
      Value iv = forOp.getInductionVar();
      z3::expr ivExpr = translate(iv);
      auto lb = getSplatIntConst(forOp.getLowerBound());
      auto ub = getSplatIntConst(forOp.getUpperBound());
      if (lb)
        constraints_.push_back(ivExpr >= ctx_.int_val(*lb));
      else
        constraints_.push_back(ivExpr >= translate(forOp.getLowerBound()));
      if (ub)
        constraints_.push_back(ivExpr < ctx_.int_val(*ub));
      else
        constraints_.push_back(ivExpr < translate(forOp.getUpperBound()));
    });
  }

  /// Returns true iff Z3 proves, under the collected constraints, that
  /// `v` is always in [lo, hi).  Returns false on timeout or SAT.
  bool canProveInRange(Value v, int64_t lo, int64_t hi) {
    // Fast path: arith.remsi result is always in [0, divisor).
    if (lo == 0) {
      if (auto remOp = v.getDefiningOp<arith::RemSIOp>()) {
        auto N = getSplatIntConst(remOp.getRhs());
        if (N && *N == hi)
          return true;
      }
    }

    RangeQueryKey key{v.getAsOpaquePointer(), lo, hi};
    auto it = queryCache_.find(key);
    if (it != queryCache_.end())
      return it->second;

    z3::solver solver(ctx_);
    z3::params params(ctx_);
    params.set("timeout", timeoutMs_);
    solver.set(params);

    for (const z3::expr &c : constraints_)
      solver.add(c);

    z3::expr vExpr = translate(v);
    solver.add((vExpr < ctx_.int_val(lo)) || (vExpr >= ctx_.int_val(hi)));

    bool result = (solver.check() == z3::unsat);
    queryCache_.emplace(key, result);
    return result;
  }

private:
  z3::context ctx_;
  std::unordered_map<void *, z3::expr> cache_;
  std::vector<z3::expr> constraints_;
  unsigned varCount_ = 0;
  unsigned timeoutMs_;
  std::unordered_map<RangeQueryKey, bool, RangeQueryKeyHash> queryCache_;

  z3::expr freshVar(const char *prefix = "v") {
    std::string name = std::string(prefix) + std::to_string(varCount_++);
    return ctx_.int_const(name.c_str());
  }

  z3::expr translate(Value v) {
    auto *key = v.getAsOpaquePointer();
    auto it = cache_.find(key);
    if (it != cache_.end())
      return it->second;
    z3::expr placeholder = freshVar("cyc");
    cache_.emplace(key, placeholder);
    z3::expr result = translateImpl(v);
    cache_.find(key)->second = result;
    return result;
  }

  z3::expr translateImpl(Value v) {
    Operation *defOp = v.getDefiningOp();

    if (!defOp)
      return freshVar("arg");

    if (auto op = dyn_cast<triton::SplatOp>(defOp))
      return translate(op.getSrc());
    if (auto op = dyn_cast<triton::BroadcastOp>(defOp))
      return translate(op.getSrc());
    if (auto op = dyn_cast<arith::ExtSIOp>(defOp))
      return translate(op.getIn());
    if (auto op = dyn_cast<arith::TruncIOp>(defOp))
      return translate(op.getIn());
    if (auto op = dyn_cast<arith::IndexCastOp>(defOp))
      return translate(op.getIn());

    if (auto op = dyn_cast<arith::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue()))
        return ctx_.int_val(intAttr.getInt());
      if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(op.getValue()))
        if (denseAttr.isSplat())
          return ctx_.int_val(denseAttr.getSplatValue<APInt>().getSExtValue());
    }

    if (auto op = dyn_cast<triton::MakeRangeOp>(defOp)) {
      z3::expr r = freshVar("range");
      constraints_.push_back(r >= ctx_.int_val((int64_t)op.getStart()));
      constraints_.push_back(r < ctx_.int_val((int64_t)op.getEnd()));
      return r;
    }

    if (isa<triton::GetProgramIdOp>(defOp)) {
      z3::expr pid = freshVar("pid");
      constraints_.push_back(pid >= ctx_.int_val((int64_t)0));
      return pid;
    }

    if (auto op = dyn_cast<arith::AddIOp>(defOp))
      return translate(op.getLhs()) + translate(op.getRhs());
    if (auto op = dyn_cast<arith::SubIOp>(defOp))
      return translate(op.getLhs()) - translate(op.getRhs());
    if (auto op = dyn_cast<arith::MulIOp>(defOp))
      return translate(op.getLhs()) * translate(op.getRhs());
    if (auto op = dyn_cast<arith::DivSIOp>(defOp))
      return translate(op.getLhs()) / translate(op.getRhs());

    if (auto op = dyn_cast<arith::RemSIOp>(defOp)) {
      z3::expr dividend = translate(op.getLhs());
      z3::expr divisor = translate(op.getRhs());
      z3::expr remVar = freshVar("rem");
      constraints_.push_back(remVar == z3::mod(dividend, divisor));
      return remVar;
    }

    if (auto op = dyn_cast<arith::AndIOp>(defOp)) {
      Value lhsVal = op.getLhs(), rhsVal = op.getRhs();
      auto tryMask = [&](Value base, Value maskVal) -> std::optional<z3::expr> {
        auto m = getSplatIntConst(maskVal);
        if (!m || *m <= 0)
          return std::nullopt;
        if ((*m & (*m + 1)) == 0) {
          z3::expr andVar = freshVar("and");
          constraints_.push_back(
              andVar == z3::mod(translate(base), ctx_.int_val(*m + 1)));
          return andVar;
        }
        return std::nullopt;
      };
      if (auto e = tryMask(lhsVal, rhsVal)) return *e;
      if (auto e = tryMask(rhsVal, lhsVal)) return *e;
      z3::expr lhsE = translate(lhsVal), rhsE = translate(rhsVal);
      z3::expr andVar = freshVar("and");
      constraints_.push_back(andVar >= ctx_.int_val((int64_t)0));
      constraints_.push_back(andVar <= lhsE);
      constraints_.push_back(andVar <= rhsE);
      return andVar;
    }

    if (auto op = dyn_cast<arith::ShLIOp>(defOp)) {
      auto shift = getSplatIntConst(op.getRhs());
      if (shift && *shift >= 0 && *shift < 63)
        return translate(op.getLhs()) * ctx_.int_val((int64_t)(1LL << *shift));
      return freshVar("shl");
    }

    if (auto op = dyn_cast<arith::ShRSIOp>(defOp)) {
      auto shift = getSplatIntConst(op.getRhs());
      if (shift && *shift >= 0 && *shift < 63)
        return translate(op.getLhs()) / ctx_.int_val((int64_t)(1LL << *shift));
      return freshVar("shr");
    }

    if (auto op = dyn_cast<arith::MaxSIOp>(defOp)) {
      z3::expr lhsE = translate(op.getLhs());
      z3::expr rhsE = translate(op.getRhs());
      z3::expr maxVar = freshVar("max");
      constraints_.push_back(maxVar == ite(lhsE >= rhsE, lhsE, rhsE));
      return maxVar;
    }

    return freshVar();
  }
};

#endif // TRITON_HAVE_Z3

//===----------------------------------------------------------------------===//
// Phase 1: Pure algebraic rewrite patterns
//===----------------------------------------------------------------------===//

/// (x % N) // N  →  0
struct RemDivToZero : OpRewritePattern<arith::DivSIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivSIOp divOp,
                                PatternRewriter &rw) const override {
    auto remOp = divOp.getLhs().getDefiningOp<arith::RemSIOp>();
    if (!remOp)
      return failure();
    auto N1 = getSplatIntConst(remOp.getRhs());
    auto N2 = getSplatIntConst(divOp.getRhs());
    if (!N1 || !N2 || *N1 != *N2 || *N1 <= 0)
      return failure();
    rw.replaceOp(divOp, makeZeroConst(rw, divOp.getLoc(), divOp.getType()));
    return success();
  }
};

/// (a % M) + M * ((a // M) % N)  →  a % (M * N)
///
/// Mixed-radix decomposition identity for non-negative integers.
/// Example: (xindex%512) + 512*((xindex//512)%2400) = xindex % 1228800
struct ComposeModMul : OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp addOp,
                                PatternRewriter &rw) const override {
    Value lhs = addOp.getLhs(), rhs = addOp.getRhs();
    for (auto [termA, termB] :
         {std::pair<Value, Value>{lhs, rhs},
          std::pair<Value, Value>{rhs, lhs}}) {

      auto remA = termA.getDefiningOp<arith::RemSIOp>();
      if (!remA)
        continue;
      auto M = getSplatIntConst(remA.getRhs());
      if (!M || *M <= 0)
        continue;
      Value a = remA.getLhs();
      Value aBase = stripTypeCasts(a);

      Value factorVal;
      if (!matchMulByConst(termB, *M, factorVal))
        continue;

      auto remB = factorVal.getDefiningOp<arith::RemSIOp>();
      if (!remB)
        continue;
      auto N = getSplatIntConst(remB.getRhs());
      if (!N || *N <= 0)
        continue;

      auto divOp = remB.getLhs().getDefiningOp<arith::DivSIOp>();
      if (!divOp)
        continue;
      auto M2 = getSplatIntConst(divOp.getRhs());
      if (!M2 || *M2 != *M || stripTypeCasts(divOp.getLhs()) != aBase)
        continue;

      int64_t MN = (*M) * (*N);
      Value MNconst =
          makeSplatIntConst(rw, addOp.getLoc(), addOp.getType(), MN);
      rw.replaceOpWithNewOp<arith::RemSIOp>(addOp, a, MNconst);
      return success();
    }
    return failure();
  }

private:
  bool matchMulByConst(Value val, int64_t M, Value &other) const {
    auto mulOp = val.getDefiningOp<arith::MulIOp>();
    if (!mulOp)
      return false;
    auto c1 = getSplatIntConst(mulOp.getLhs());
    if (c1 && *c1 == M) {
      other = mulOp.getRhs();
      return true;
    }
    auto c2 = getSplatIntConst(mulOp.getRhs());
    if (c2 && *c2 == M) {
      other = mulOp.getLhs();
      return true;
    }
    return false;
  }
};

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

class TritonIndexSimplificationPass
    : public mlir::triton::TritonIndexSimplificationBase<
          TritonIndexSimplificationPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();

    // Phase 1: Pure algebraic patterns.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<RemDivToZero>(ctx);
      patterns.add<ComposeModMul>(ctx);
      GreedyRewriteConfig cfg;
      cfg.maxIterations = 8;
      (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns), cfg);
    }

    // Phase 2: Z3-backed range simplification.
#ifdef TRITON_HAVE_Z3
    {
      Z3IndexProver prover;
      prover.collectConstraints(moduleOp);

      SmallVector<arith::DivSIOp> divsToZero;
      SmallVector<arith::RemSIOp> remsToIdentity;

      moduleOp.walk([&](Operation *op) {
        if (auto divOp = dyn_cast<arith::DivSIOp>(op)) {
          auto N = getSplatIntConst(divOp.getRhs());
          if (N && *N > 0 && prover.canProveInRange(divOp.getLhs(), 0, *N))
            divsToZero.push_back(divOp);
        } else if (auto remOp = dyn_cast<arith::RemSIOp>(op)) {
          auto N = getSplatIntConst(remOp.getRhs());
          if (N && *N > 0 && prover.canProveInRange(remOp.getLhs(), 0, *N))
            remsToIdentity.push_back(remOp);
        }
      });

      IRRewriter rewriter(ctx);
      for (arith::DivSIOp divOp : divsToZero) {
        rewriter.setInsertionPoint(divOp);
        rewriter.replaceOp(divOp,
                           makeZeroConst(rewriter, divOp.getLoc(),
                                         divOp.getType()));
      }
      for (arith::RemSIOp remOp : remsToIdentity) {
        rewriter.replaceOp(remOp, remOp.getLhs());
      }
    }
#endif // TRITON_HAVE_Z3

    // Phase 3: Re-apply algebraic patterns to exploit newly simplified ops.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<RemDivToZero>(ctx);
      patterns.add<ComposeModMul>(ctx);
      GreedyRewriteConfig cfg;
      cfg.maxIterations = 4;
      (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns), cfg);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::triton::createTritonIndexSimplification() {
  return std::make_unique<TritonIndexSimplificationPass>();
}
