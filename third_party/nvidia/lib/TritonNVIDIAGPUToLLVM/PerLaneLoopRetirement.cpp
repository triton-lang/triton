//===- PerLaneLoopRetirement.cpp - retire lanes from divergent loops -----===//
//
// Rewrites warp-uniform (lock-step) while-loop latches into per-lane latches
// so that each lane of a warp retires from the loop as soon as its own exit
// condition holds, instead of all lanes iterating to the warp's maximum trip
// count.
//
// Triton lowers a data-dependent loop such as
//
//   while tl.max((j < trip).to(tl.int32)) > 0: ...
//
// to a latch of the shape
//
//   ^header(...):
//     %pred = llvm.icmp "slt" %j, %trip           // per-lane i1
//     %z    = llvm.zext %pred : i1 to i32
//     %r    = nvvm.redux.sync max %z, C(-1)       // warp-collective
//     %any  = llvm.icmp "sgt" %r, C(0)            // warp-uniform
//     llvm.cond_br %any, ^body, ^exit
//
// Every lane iterates until the *slowest* lane in its warp finishes: the
// per-warp cost is 32 x max(trip) instead of sum(trip), and the latch pays a
// warp-collective reduction per iteration.  On Volta+ (independent thread
// scheduling) the lock-step schedule is a codegen choice, not a hardware
// requirement.  This pass redirects the latch branch to the per-lane
// predicate so each lane retires independently, deletes the then-dead
// cross-lane reduction, and reconverges the lanes that entered the loop
// (`activemask` captured in the preheader) with `nvvm.bar.warp.sync` at the
// loop exit.
//
// The rewrite is observationally equivalent to the lock-step schedule iff a
// lane's loop-carried state and side effects are unaffected by the
// iterations it would have spent masked-off.  The pass *verifies* this
// structurally and refuses the rewrite otherwise:
//
//  (1) Latch shape: the branch condition is `any lane active`, i.e. an
//      icmp sgt|ne 0 of a full-warp redux (max/umax/or/add) of a
//      zext/select-normalized i1.
//  (2) Body safety: the loop body contains no operation that requires the
//      retired lane's participation (any NVVM op: collectives, barriers,
//      nested redux latches) or whose side effects would be lost relative
//      to the masked schedule (calls, atomics, unpredicated llvm.store).
//  (3) Single exit: the latch is the only way out of the loop, so every
//      entering lane passes the reconvergence point.
//  (4) Live-out freezing: every loop-carried value used after the loop is
//      *frozen* on lane-inactive iterations -- updated only through
//      `select(pred, x, old)` or a masked-identity form such as
//      `old + select(pred, x, 0)` (the lowered form of Triton's
//      `tl.where(active, ...)` idioms).
//  (5) Predicate monotonicity: once a lane's predicate turns false it can
//      never turn true again under continued lock-step execution (otherwise
//      the masked schedule would resume the lane while the retired lane is
//      gone).  Each icmp operand must be frozen/invariant, or be an
//      induction update moving away from re-satisfying the comparison
//      (e.g. `j = j + c, c >= 0` against `slt`).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVPERLANELOOPRETIREMENT
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

// Return the integer value of an LLVM dialect constant, if `v` is one.
static std::optional<int64_t> getConstInt(Value v) {
  if (auto cst = v.getDefiningOp<LLVM::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return attr.getInt();
  return std::nullopt;
}

// Peel per-lane packing/extension wrappers: zext/sext of i1, bitcasts, and
// extractvalue-of-insertvalue / extractelement-of-insertelement pairs.
// These are value-preserving plumbing that Triton's tile lowering inserts
// around per-lane scalars.
static Value peelWrappers(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto z = dyn_cast<LLVM::ZExtOp>(def)) {
      v = z.getArg();
      continue;
    }
    if (auto s = dyn_cast<LLVM::SExtOp>(def)) {
      v = s.getArg();
      continue;
    }
    if (auto b = dyn_cast<LLVM::BitcastOp>(def)) {
      v = b.getArg();
      continue;
    }
    if (auto ev = dyn_cast<LLVM::ExtractValueOp>(def)) {
      Value cont = ev.getContainer();
      Value replaced;
      while (auto iv = cont.getDefiningOp<LLVM::InsertValueOp>()) {
        if (iv.getPosition() == ev.getPosition()) {
          replaced = iv.getValue();
          break;
        }
        cont = iv.getContainer();
      }
      if (!replaced)
        break;
      v = replaced;
      continue;
    }
    if (auto ee = dyn_cast<LLVM::ExtractElementOp>(def)) {
      auto pos = getConstInt(ee.getPosition());
      if (!pos)
        break;
      Value cont = ee.getVector();
      Value replaced;
      while (auto ie = cont.getDefiningOp<LLVM::InsertElementOp>()) {
        auto ipos = getConstInt(ie.getPosition());
        if (!ipos)
          break;
        if (*ipos == *pos) {
          replaced = ie.getValue();
          break;
        }
        cont = ie.getVector();
      }
      if (!replaced)
        break;
      v = replaced;
      continue;
    }
    break;
  }
  return v;
}

// How a loop-carried value evolves on iterations where the lane's predicate
// is false (see the file comment).
enum class UpdateKind {
  Invariant, // carried through unchanged
  Frozen,    // masked update: unchanged whenever the predicate is false
  Monotone,  // unmasked induction: v = v + c on every iteration
  Unknown,
};

struct UpdateInfo {
  UpdateKind kind = UpdateKind::Unknown;
  int64_t step = 0; // for Monotone: the (constant) increment
};

// A loop-level value root.  Triton's tile lowering carries per-lane scalars
// as fields of struct-typed block arguments: the scalar is
// `extractvalue(%arg)[path]` and the back edge rebuilds the struct with
// `insertvalue`.  A root therefore names a base value plus a projection
// path (empty for plain scalars).
struct Root {
  Value base;
  SmallVector<int64_t, 2> path;
  bool operator==(const Root &o) const {
    return base == o.base && path == o.path;
  }
  bool operator!=(const Root &o) const { return !(*this == o); }
};

class LoopAnalysis {
public:
  LoopAnalysis(LLVM::CondBrOp latch) : latch(latch) {
    header = latch->getBlock();
    exitBlk = latch.getFalseDest();
  }

  // Discover the loop: body blocks reachable from the latch's true
  // destination without passing through the header, single exit (the
  // latch), a back edge to the header, and a unique preheader.
  bool structure() {
    llvm::SmallPtrSet<Block *, 8> seen;
    bool reachesHeader = false;
    SmallVector<Block *, 8> worklist{latch.getTrueDest()};
    while (!worklist.empty()) {
      Block *b = worklist.pop_back_val();
      if (b == header) {
        reachesHeader = true;
        continue;
      }
      if (b == exitBlk)
        return false; // side exit: lanes could bypass the reconvergence
      if (!seen.insert(b).second)
        continue;
      bodyBlocks.push_back(b);
      for (Block *succ : b->getSuccessors())
        worklist.push_back(succ);
    }
    if (!reachesHeader)
      return false;
    inLoop = std::move(seen);
    inLoop.insert(header);
    for (Block *pred : header->getPredecessors()) {
      if (inLoop.contains(pred))
        latchPreds.push_back(pred);
      else if (!preheader)
        preheader = pred;
      else
        return false; // multiple preheaders
    }
    if (!preheader || latchPreds.empty())
      return false;
    // The exit reconvergence syncs on the activemask captured in the
    // preheader, so every lane observed there must actually enter the loop
    // (and hence reach the exit): require an unconditional branch.  This is
    // what `scf.while` lowering produces.
    return isa<LLVM::BrOp>(preheader->getTerminator());
  }

  bool isInLoop(Block *b) const { return inLoop.contains(b); }
  Block *getHeader() const { return header; }
  Block *getExit() const { return exitBlk; }
  Block *getPreheader() const { return preheader; }
  ArrayRef<Block *> getBodyBlocks() const { return bodyBlocks; }
  ArrayRef<Block *> getLatchPreds() const { return latchPreds; }

  // The operand a predecessor's terminator passes for `argIdx` of `dest`.
  // Returns null for terminators we do not model.
  static Value incomingOperand(Block *pred, Block *dest, unsigned argIdx) {
    Operation *term = pred->getTerminator();
    if (auto br = dyn_cast<LLVM::BrOp>(term))
      return br.getDest() == dest ? br.getDestOperands()[argIdx] : Value();
    if (auto cbr = dyn_cast<LLVM::CondBrOp>(term)) {
      // A conditional branch may reach `dest` on either edge; require a
      // unique one so the incoming value is well defined.
      Value v;
      if (cbr.getTrueDest() == dest)
        v = cbr.getTrueDestOperands()[argIdx];
      if (cbr.getFalseDest() == dest) {
        if (v)
          return Value();
        v = cbr.getFalseDestOperands()[argIdx];
      }
      return v;
    }
    return Value();
  }

  // Normalize a value to its loop-level root: peel wrappers, accumulate
  // extractvalue projections whose container does not peel further, and map
  // block arguments of body blocks through their (unique-valued) incoming
  // edges.  Fixpoints at header arguments, loop-invariant values, or opaque
  // definitions.
  Root resolveRoot(Value v) {
    SmallVector<int64_t, 2> path;
    llvm::SmallPtrSet<void *, 8> visited;
    for (;;) {
      v = peelWrappers(v);
      if (auto ev = v.getDefiningOp<LLVM::ExtractValueOp>()) {
        // peelWrappers already resolved matched insertvalue pairs; what
        // remains is a genuine projection of an opaque container.
        path.insert(path.begin(), ev.getPosition().begin(),
                    ev.getPosition().end());
        v = ev.getContainer();
        continue;
      }
      auto ba = dyn_cast<BlockArgument>(v);
      if (!ba)
        break;
      Block *owner = ba.getOwner();
      if (owner == header || !inLoop.contains(owner))
        break;
      if (!visited.insert(v.getAsOpaquePointer()).second)
        break;
      Value merged;
      for (Block *pred : owner->getPredecessors()) {
        Value in = incomingOperand(pred, owner, ba.getArgNumber());
        if (!in) {
          merged = Value();
          break;
        }
        in = peelWrappers(in);
        if (!merged)
          merged = in;
        else if (merged != in) {
          merged = Value();
          break;
        }
      }
      if (!merged)
        break;
      v = merged;
    }
    return {v, path};
  }

  // Project a (struct) value onto `path`, walking insertvalue chains.
  // Returns the scalar carried at `path`, or null if it cannot be resolved.
  Value project(Value v, ArrayRef<int64_t> path) {
    while (!path.empty()) {
      v = peelWrappers(v);
      auto iv = v.getDefiningOp<LLVM::InsertValueOp>();
      if (!iv) {
        // Projection of a block argument or opaque value: not resolvable
        // here; the caller falls back to root comparison.
        return Value();
      }
      ArrayRef<int64_t> ipos = iv.getPosition();
      if (ipos == path)
        return iv.getValue();
      if (path.size() > ipos.size() && path.take_front(ipos.size()) == ipos) {
        v = iv.getValue();
        path = path.drop_front(ipos.size());
        continue;
      }
      // Disjoint position: the field we want lives in the container.
      v = iv.getContainer();
    }
    return v;
  }

  // Loop-invariant: defined outside the loop (constants, function
  // arguments, preheader computations).
  bool isInvariant(const Root &r) {
    if (auto ba = dyn_cast<BlockArgument>(r.base))
      return !inLoop.contains(ba.getOwner());
    Operation *def = r.base.getDefiningOp();
    return def && !inLoop.contains(def->getBlock());
  }

  bool isHeaderRoot(const Root &r) {
    auto ba = dyn_cast<BlockArgument>(r.base);
    return ba && ba.getOwner() == header;
  }

  // Structural equivalence of `c` with the latch's per-lane predicate: the
  // same SSA value, or an icmp with the same kind whose operands share
  // roots with the latch icmp (a body might recompute `active = j < trip`).
  bool equivToLatchPredicate(Value c, LLVM::ICmpOp latchCmp) {
    c = peelWrappers(c);
    if (c == latchCmp.getResult())
      return true;
    auto icmp = c.getDefiningOp<LLVM::ICmpOp>();
    if (!icmp || icmp.getPredicate() != latchCmp.getPredicate())
      return false;
    return resolveRoot(icmp.getLhs()) == resolveRoot(latchCmp.getLhs()) &&
           resolveRoot(icmp.getRhs()) == resolveRoot(latchCmp.getRhs());
  }

  // Classify how the loop-carried value named by the header root `r`
  // changes on each back edge.
  UpdateInfo classifyUpdate(const Root &r, LLVM::ICmpOp latchCmp) {
    auto key =
        std::make_pair(r.base.getAsOpaquePointer(),
                       std::vector<int64_t>(r.path.begin(), r.path.end()));
    auto it = updateCache.find(key);
    if (it != updateCache.end())
      return it->second;
    // Seed the cache to keep any (impossible) recursion terminating.
    updateCache[key] = {UpdateKind::Unknown, 0};
    auto arg = cast<BlockArgument>(r.base);
    UpdateInfo result;
    bool first = true;
    for (Block *pred : latchPreds) {
      Value in = incomingOperand(pred, header, arg.getArgNumber());
      UpdateInfo edge{UpdateKind::Unknown, 0};
      if (in) {
        Value scalar = project(peelWrappers(in), r.path);
        if (scalar)
          edge = classifyEdge(peelWrappers(scalar), r, latchCmp);
        else {
          // The projection did not resolve through insertvalue chains; if
          // it roots back to the same header slot it is carried through.
          if (resolveRoot(in) == r)
            edge = {UpdateKind::Invariant, 0};
        }
      }
      if (first) {
        result = edge;
        first = false;
      } else if (edge.kind != result.kind ||
                 (edge.kind == UpdateKind::Monotone &&
                  (edge.step < 0) != (result.step < 0))) {
        result = {UpdateKind::Unknown, 0};
      }
    }
    updateCache[key] = result;
    return result;
  }

private:
  UpdateInfo classifyEdge(Value in, const Root &r, LLVM::ICmpOp latchCmp) {
    auto rootsToSelf = [&](Value v) { return resolveRoot(v) == r; };

    if (rootsToSelf(in))
      return {UpdateKind::Invariant, 0};

    // select(pred, x, self): frozen whenever pred is false.
    if (auto sel = in.getDefiningOp<LLVM::SelectOp>())
      if (rootsToSelf(sel.getFalseValue()) &&
          equivToLatchPredicate(sel.getCondition(), latchCmp))
        return {UpdateKind::Frozen, 0};

    // self + select(pred, x, 0) (also |, ^): masked-identity update, frozen
    // whenever pred is false.  self + C: unmasked monotone induction.
    auto maskedIdentity = [&](Value a, Value b) -> std::optional<UpdateInfo> {
      if (!rootsToSelf(a))
        return std::nullopt;
      if (auto sel = peelWrappers(b).getDefiningOp<LLVM::SelectOp>()) {
        auto fv = getConstInt(peelWrappers(sel.getFalseValue()));
        if (fv && *fv == 0 &&
            equivToLatchPredicate(sel.getCondition(), latchCmp))
          return UpdateInfo{UpdateKind::Frozen, 0};
      }
      return std::nullopt;
    };

    if (auto add = in.getDefiningOp<LLVM::AddOp>()) {
      for (auto [a, b] : {std::pair(add.getLhs(), add.getRhs()),
                          std::pair(add.getRhs(), add.getLhs())}) {
        if (auto res = maskedIdentity(a, b))
          return *res;
        if (rootsToSelf(a))
          if (auto c = getConstInt(peelWrappers(b)))
            return {*c == 0 ? UpdateKind::Invariant : UpdateKind::Monotone, *c};
      }
    }
    if (auto orOp = in.getDefiningOp<LLVM::OrOp>()) {
      for (auto [a, b] : {std::pair(orOp.getLhs(), orOp.getRhs()),
                          std::pair(orOp.getRhs(), orOp.getLhs())})
        if (auto res = maskedIdentity(a, b))
          return *res;
    }
    if (auto xorOp = in.getDefiningOp<LLVM::XOrOp>()) {
      for (auto [a, b] : {std::pair(xorOp.getLhs(), xorOp.getRhs()),
                          std::pair(xorOp.getRhs(), xorOp.getLhs())})
        if (auto res = maskedIdentity(a, b))
          return *res;
    }
    return {UpdateKind::Unknown, 0};
  }

  LLVM::CondBrOp latch;
  Block *header = nullptr;
  Block *exitBlk = nullptr;
  Block *preheader = nullptr;
  SmallVector<Block *, 8> bodyBlocks;
  SmallVector<Block *, 4> latchPreds;
  llvm::SmallPtrSet<Block *, 8> inLoop;
  std::map<std::pair<void *, std::vector<int64_t>>, UpdateInfo> updateCache;
};

// Normalize the redux input down to the per-lane i1 predicate: peel wrappers
// and a `select(pred, c, 0)` (c > 0) if present.
static Value matchPerLanePredicate(Value reduxInput) {
  Value v = peelWrappers(reduxInput);
  if (auto sel = v.getDefiningOp<LLVM::SelectOp>()) {
    auto tv = getConstInt(peelWrappers(sel.getTrueValue()));
    auto fv = getConstInt(peelWrappers(sel.getFalseValue()));
    if (tv && fv && *tv > 0 && *fv == 0)
      v = peelWrappers(sel.getCondition());
  }
  if (!v.getType().isInteger(1))
    return nullptr;
  return v;
}

// A retired lane no longer executes the loop body, so the body must not
// contain an operation whose semantics require the retired lane's
// participation (warp collectives, barriers) or whose side effects would be
// lost relative to the masked lock-step schedule (calls, atomics,
// unpredicated stores).
static bool isBodySafe(ArrayRef<Block *> bodyBlocks) {
  for (Block *b : bodyBlocks) {
    for (Operation &op : *b) {
      if (isa<NVVM::NVVMDialect>(op.getDialect()))
        return false;
      if (isa<LLVM::CallOp, LLVM::InvokeOp, LLVM::AtomicRMWOp,
              LLVM::AtomicCmpXchgOp, LLVM::FenceOp, LLVM::StoreOp>(op))
        return false;
      // Predicated gathers (Triton's masked tl.load lowering) are safe: a
      // lane whose predicate is false performs no access either way.  Any
      // asm that can write memory or synchronize is not.
      if (auto asmOp = dyn_cast<LLVM::InlineAsmOp>(op)) {
        StringRef str = asmOp.getAsmString();
        if (str.contains("st.") || str.contains("atom") ||
            str.contains("red.") || str.contains("bar.") ||
            str.contains("membar") || str.contains("fence"))
          return false;
      }
    }
  }
  return true;
}

// Once a lane's predicate turns false it must stay false under continued
// lock-step execution: each icmp operand is frozen/invariant, or an
// induction moving away from re-satisfying the comparison.
static bool isPredicateMonotone(LLVM::ICmpOp cmp, LoopAnalysis &loop) {
  auto sideOk = [&](Value operand, bool isLhs) {
    Root root = loop.resolveRoot(operand);
    if (loop.isInvariant(root))
      return true;
    if (!loop.isHeaderRoot(root))
      return false;
    UpdateInfo u = loop.classifyUpdate(root, cmp);
    if (u.kind == UpdateKind::Invariant || u.kind == UpdateKind::Frozen)
      return true;
    if (u.kind != UpdateKind::Monotone)
      return false;
    switch (cmp.getPredicate()) {
    case LLVM::ICmpPredicate::slt:
    case LLVM::ICmpPredicate::sle:
    case LLVM::ICmpPredicate::ult:
    case LLVM::ICmpPredicate::ule:
      // loop-while-below: lhs may only grow, rhs may only shrink
      return isLhs ? u.step >= 0 : u.step <= 0;
    case LLVM::ICmpPredicate::sgt:
    case LLVM::ICmpPredicate::sge:
    case LLVM::ICmpPredicate::ugt:
    case LLVM::ICmpPredicate::uge:
      return isLhs ? u.step <= 0 : u.step >= 0;
    default:
      // eq/ne: only frozen/invariant operands keep the predicate stable.
      return false;
    }
  };
  return sideOk(cmp.getLhs(), /*isLhs=*/true) &&
         sideOk(cmp.getRhs(), /*isLhs=*/false);
}

struct NVPerLaneLoopRetirement
    : public mlir::triton::impl::NVPerLaneLoopRetirementBase<
          NVPerLaneLoopRetirement> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    struct Match {
      LLVM::CondBrOp condBr;
      LLVM::ICmpOp anyCmp;
      NVVM::ReduxOp redux;
      Value perLanePred;
      Block *preheader;
      unsigned liveOut;
    };
    SmallVector<Match> matches;

    mod.walk([&](LLVM::CondBrOp condBr) {
      // (1) Latch shape: %any = icmp sgt|ne (redux max|umax|or|add zext(i1),
      // full mask), 0.
      auto anyCmp = condBr.getCondition().getDefiningOp<LLVM::ICmpOp>();
      if (!anyCmp)
        return;
      if (anyCmp.getPredicate() != LLVM::ICmpPredicate::sgt &&
          anyCmp.getPredicate() != LLVM::ICmpPredicate::ne)
        return;
      auto rhs = getConstInt(peelWrappers(anyCmp.getRhs()));
      if (!rhs || *rhs != 0)
        return;
      auto redux = peelWrappers(anyCmp.getLhs()).getDefiningOp<NVVM::ReduxOp>();
      if (!redux)
        return;
      // Over the {0,1} range of the normalized input, max/umax/or/add all
      // compute "any lane active" when compared against zero.
      switch (redux.getKind()) {
      case NVVM::ReductionKind::MAX:
      case NVVM::ReductionKind::UMAX:
      case NVVM::ReductionKind::OR:
      case NVVM::ReductionKind::ADD:
        break;
      default:
        return;
      }
      if (redux.getAbs() || redux.getNan())
        return;
      // Full-warp member mask only: a partial mask means the loop is already
      // executing under known divergence this rewrite does not model.
      auto mask = getConstInt(peelWrappers(redux.getMaskAndClamp()));
      if (!mask || (*mask != -1 && *mask != 0xffffffff))
        return;
      Value perLane = matchPerLanePredicate(redux.getVal());
      if (!perLane)
        return;
      auto perLaneCmp = perLane.getDefiningOp<LLVM::ICmpOp>();
      if (!perLaneCmp)
        return; // monotonicity reasoning (5) needs the comparison structure

      // (3) Loop structure: single exit, unique preheader.
      LoopAnalysis loop(condBr);
      if (!loop.structure())
        return;
      // (2) Body safety.
      if (!isBodySafe(loop.getBodyBlocks()))
        return;
      // (5) Predicate monotonicity.
      if (!isPredicateMonotone(perLaneCmp, loop))
        return;
      // (4) Live-out freezing: every loop-carried value used after the loop
      // must be frozen (or invariant) on lane-inactive iterations.  Uses
      // outside the loop are projections (extractvalue) of the struct-typed
      // header arguments; any other escaping use is refused conservatively.
      unsigned liveOut = 0;
      for (BlockArgument arg : loop.getHeader()->getArguments()) {
        bool counted = false;
        for (Operation *user : arg.getUsers()) {
          if (loop.isInLoop(user->getBlock()))
            continue;
          Root root{arg, {}};
          if (auto ev = dyn_cast<LLVM::ExtractValueOp>(user))
            root.path.assign(ev.getPosition().begin(), ev.getPosition().end());
          UpdateInfo u = loop.classifyUpdate(root, perLaneCmp);
          if (u.kind != UpdateKind::Frozen && u.kind != UpdateKind::Invariant)
            return;
          if (!counted) {
            ++liveOut;
            counted = true;
          }
        }
      }

      matches.push_back(
          {condBr, anyCmp, redux, perLane, loop.getPreheader(), liveOut});
    });

    for (Match &m : matches) {
      m.condBr->emitRemark()
          << "applying per-lane loop retirement (verified: " << m.liveOut
          << " live-out value(s) frozen on inactive iterations, predicate "
             "monotone)";

      Location loc = m.condBr.getLoc();

      // Capture the set of lanes entering the loop; they are exactly the
      // lanes that must reconverge at the exit.
      OpBuilder pb(m.preheader->getTerminator());
      auto activeMask = LLVM::InlineAsmOp::create(
          pb, loc, pb.getI32Type(), /*operands=*/ValueRange(),
          /*asm_string=*/"activemask.b32 $0;", /*constraints=*/"=r",
          /*has_side_effects=*/true, /*is_align_stack=*/false,
          LLVM::TailCallKind::None,
          LLVM::AsmDialectAttr::get(pb.getContext(), LLVM::AsmDialect::AD_ATT),
          /*operand_attrs=*/ArrayAttr());

      // Branch on the per-lane predicate: each lane retires independently
      // under independent thread scheduling.
      m.condBr.getConditionMutable().assign(m.perLanePred);

      // Reconverge the entering lanes at the loop exit before any
      // subsequent warp-collective operation.
      Block *exit = m.condBr.getFalseDest();
      OpBuilder b(exit, exit->begin());
      NVVM::SyncWarpOp::create(b, loc, activeMask.getRes());

      // Delete the dead warp-collective latch computation.
      if (m.anyCmp.use_empty())
        m.anyCmp.erase();
      if (m.redux.use_empty())
        m.redux.erase();
    }
  }
};

} // namespace
