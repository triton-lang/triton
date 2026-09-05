#include "ControlFlowRangeAnalysis.h"

#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <tuple>

using namespace mlir;
using namespace mlir::triton::AMD::detail;

namespace {

using GetRangeFn = ControlFlowRangeAnalysis::GetRangeFn;

// All arithmetic below uses signed, widened APInts. This also represents the
// unsigned input domain and the sentinel past an inclusive loop bound.
struct Interval {
  APInt min;
  APInt max;
};

template <typename T> struct Evaluation {
  std::optional<T> value;
  bool pending = false;
};

APInt min(const APInt &a, const APInt &b) { return a.slt(b) ? a : b; }
APInt max(const APInt &a, const APInt &b) { return a.sgt(b) ? a : b; }

std::optional<APInt> addProduct(const APInt &initial, const APInt &count,
                                const APInt &delta) {
  bool overflow;
  APInt product = count.smul_ov(delta, overflow);
  if (overflow)
    return std::nullopt;
  APInt result = initial.sadd_ov(product, overflow);
  if (overflow)
    return std::nullopt;
  return result;
}

APInt ceilPositive(const APInt &numerator, const APInt &denominator) {
  if (numerator.isNonPositive())
    return APInt(numerator.getBitWidth(), 0);
  return numerator.udiv(denominator) +
         APInt(numerator.getBitWidth(), !numerator.urem(denominator).isZero());
}

Interval widen(const ConstantIntRanges &range, unsigned width, bool isSigned) {
  if (isSigned)
    return {range.smin().sext(width), range.smax().sext(width)};
  return {range.umin().zext(width), range.umax().zext(width)};
}

Interval domain(unsigned valueWidth, unsigned width, bool isSigned) {
  if (isSigned)
    return {APInt::getSignedMinValue(valueWidth).sext(width),
            APInt::getSignedMaxValue(valueWidth).sext(width)};
  return {APInt(width, 0), APInt::getMaxValue(valueWidth).zext(width)};
}

bool contains(Interval outer, Interval inner) {
  return outer.min.sle(inner.min) && outer.max.sge(inner.max);
}

ConstantIntRanges narrow(Interval range, unsigned width, bool isSigned) {
  return ConstantIntRanges::range(range.min.trunc(width),
                                  range.max.trunc(width), isSigned);
}

Block *getAncestorBlock(Block *block, Region *region) {
  while (block && block->getParent() != region)
    block = block->getParentOp()->getBlock();
  return block;
}

// Structural dependencies are independent of lattice updates. The optional
// allowed value is the enclosing IV when checking a nested recurrence.
struct Dependencies {
  using Key = std::tuple<Value, Block *, Value, Value>;
  DenseMap<Key, bool> known;
  unsigned depth = 0;

  bool check(Value value, const CFGLoop &loop, Value allowed = {},
             Value excluded = {}) {
    if (value == excluded)
      return false;
    if (value == allowed)
      return true;
    Block *block =
        getAncestorBlock(value.getParentBlock(), loop.getHeader()->getParent());
    if (!block || !loop.contains(block))
      return true;
    Key key{value, loop.getHeader(), allowed, excluded};
    if (auto it = known.find(key); it != known.end())
      return it->second;
    auto *op = value.getDefiningOp();
    if (!op || op->getNumRegions() || !isPure(op) ||
        !isa<InferIntRangeInterface>(op) || depth >= 64)
      return false;
    known.try_emplace(key, false);
    ++depth;
    bool result = llvm::all_of(op->getOperands(), [&](Value operand) {
      return check(operand, loop, allowed, excluded);
    });
    --depth;
    // A depth-limit failure may succeed when reached through a shorter path.
    if (result)
      known[key] = true;
    else
      known.erase(key);
    return result;
  }
};

Value getIncoming(Block::pred_iterator edge, BlockArgument argument) {
  auto branch = dyn_cast<BranchOpInterface>((*edge)->getTerminator());
  if (!branch)
    return {};
  return branch.getSuccessorOperands(
      edge.getSuccessorIndex())[argument.getArgNumber()];
}

struct LoopDescription {
  CFGLoop *loop;
  Block *entry;
  Block *exit;
  BlockArgument induction;
  Value bound;
  bool isSigned;
  bool increasing;
  bool inclusive;
};

std::optional<LoopDescription> describeLoop(CFGLoop *loop,
                                            Dependencies &dependencies) {
  Block *header = loop->getHeader();
  auto branch = dyn_cast<cf::CondBranchOp>(header->getTerminator());
  if (!branch || loop->getExitingBlock() != header)
    return std::nullopt;
  bool trueContinues = loop->contains(branch.getTrueDest());
  if (trueContinues == loop->contains(branch.getFalseDest()))
    return std::nullopt;
  // Self-edges are post-test loops; their header and exit counts differ.
  if (branch.getTrueDest() == header || branch.getFalseDest() == header)
    return std::nullopt;
  Block *entry = loop->getLoopPredecessor();
  if (!entry)
    return std::nullopt;
  auto cmp = branch.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp->getBlock() != header)
    return std::nullopt;
  arith::CmpIPredicate predicate = cmp.getPredicate();
  if (!trueContinues)
    predicate = arith::invertPredicate(predicate);
  using P = arith::CmpIPredicate;
  if (predicate == P::eq || predicate == P::ne)
    return std::nullopt;
  bool isLhs = true;
  if (auto argument = dyn_cast<BlockArgument>(cmp.getRhs());
      argument && argument.getOwner() == header &&
      dependencies.check(cmp.getLhs(), *loop))
    isLhs = false;
  Value iv = isLhs ? cmp.getLhs() : cmp.getRhs();
  Value bound = isLhs ? cmp.getRhs() : cmp.getLhs();
  auto induction = dyn_cast<BlockArgument>(iv);
  if (!induction || induction.getOwner() != header ||
      !isa<IntegerType>(induction.getType()) ||
      !dependencies.check(bound, *loop))
    return std::nullopt;
  bool isSigned = predicate == P::slt || predicate == P::sle ||
                  predicate == P::sgt || predicate == P::sge;
  bool less = predicate == P::slt || predicate == P::sle ||
              predicate == P::ult || predicate == P::ule;
  bool inclusive = predicate == P::sle || predicate == P::sge ||
                   predicate == P::ule || predicate == P::uge;
  return LoopDescription{loop,
                         entry,
                         trueContinues ? branch.getFalseDest()
                                       : branch.getTrueDest(),
                         induction,
                         bound,
                         isSigned,
                         less == isLhs,
                         inclusive};
}

IntegerValueRange readRangeImpl(Value value, Block *block, GetRangeFn getRange,
                                DenseMap<Value, IntegerValueRange> &known,
                                unsigned depth) {
  if (auto it = known.find(value); it != known.end())
    return it->second;
  // Constants inside an unvisited diamond arm need not wait for its liveness.
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return IntegerValueRange(ConstantIntRanges::constant(constant));
  IntegerValueRange range = getRange(value, block);
  if (!range.isUninitialized() || depth >= 64)
    return range;
  auto *op = value.getDefiningOp();
  auto infer = dyn_cast_or_null<InferIntRangeInterface>(op);
  if (!infer || !isPure(op) || op->getNumRegions())
    return range;
  SmallVector<IntegerValueRange> operands;
  for (Value operand : op->getOperands())
    operands.push_back(
        readRangeImpl(operand, block, getRange, known, depth + 1));
  infer.inferResultRangesFromOptional(
      operands, [&](Value result, const IntegerValueRange &inferred) {
        if (result == value)
          range = inferred;
      });
  known.try_emplace(value, range);
  return range;
}

IntegerValueRange readRange(Value value, Block *block, GetRangeFn getRange) {
  DenseMap<Value, IntegerValueRange> known;
  return readRangeImpl(value, block, getRange, known, 0);
}

IntegerValueRange getInitialRange(BlockArgument argument,
                                  const LoopDescription &loop,
                                  GetRangeFn getRange) {
  IntegerValueRange initial;
  for (auto edge = argument.getOwner()->pred_begin(),
            end = argument.getOwner()->pred_end();
       edge != end; ++edge) {
    if (*edge != loop.entry)
      continue;
    Value incoming = getIncoming(edge, argument);
    if (!incoming)
      return IntegerValueRange::getMaxRange(argument);
    IntegerValueRange range = readRange(incoming, loop.entry, getRange);
    if (range.isUninitialized())
      return range;
    initial = IntegerValueRange::join(initial, range);
  }
  return initial;
}

struct LoopBounds {
  APInt minTrips;
  APInt maxTrips;
  Interval header;
  Interval body;
  Interval exit;
};

std::optional<LoopBounds> getLoopBounds(const LoopDescription &loop,
                                        Interval initial, Interval bound,
                                        Interval delta, unsigned valueWidth) {
  unsigned width = initial.min.getBitWidth();
  APInt one(width, 1);
  // Reflect descending loops so both directions advance toward an upper bound.
  auto orient = [&](Interval range) {
    return loop.increasing ? range : Interval{-range.max, -range.min};
  };
  initial = orient(initial);
  bound = orient(bound);
  delta = orient(delta);
  if (!delta.min.isStrictlyPositive() || !delta.max.isStrictlyPositive())
    return std::nullopt;
  if (loop.inclusive) {
    bound.min += one;
    bound.max += one;
  }
  APInt minTrips = ceilPositive(bound.min - initial.max, delta.max);
  APInt maxTrips = ceilPositive(bound.max - initial.min, delta.min);
  if (maxTrips.isZero()) {
    initial = orient(initial);
    return LoopBounds{minTrips, maxTrips, initial, initial, initial};
  }

  auto last = addProduct(initial.max, maxTrips - one, delta.max);
  auto firstExit = addProduct(initial.min, minTrips, delta.min);
  if (!last || !firstExit)
    return std::nullopt;
  Interval body{initial.min, min(bound.max - one, *last)};
  auto sentinel = addProduct(body.max, one, delta.max);
  if (!sentinel)
    return std::nullopt;
  Interval header{initial.min, max(initial.max, *sentinel)};
  Interval exit{max(max(initial.min, bound.min), *firstExit), header.max};
  header = orient(header);
  // In particular, the update that first fails the condition must not wrap.
  if (!contains(domain(valueWidth, width, loop.isSigned), header))
    return std::nullopt;
  return LoopBounds{minTrips, maxTrips, header, orient(body), orient(exit)};
}

enum class UseKind { Header, Body, Exit };

UseKind getUseKind(const LoopDescription &loop, Block *useBlock,
                   DominanceInfo &domInfo) {
  Block *header = loop.loop->getHeader();
  Block *block = getAncestorBlock(useBlock, header->getParent());
  if (!block || block == header)
    return UseKind::Header;
  if (loop.loop->contains(block))
    return UseKind::Body;
  if (loop.exit->getSinglePredecessor() == header &&
      domInfo.dominates(loop.exit, block))
    return UseKind::Exit;
  return UseKind::Header;
}

std::optional<Interval> getRecurrenceBounds(Interval initial, Interval delta,
                                            const APInt &minTrips,
                                            const APInt &maxTrips) {
  auto lower = addProduct(
      initial.min, delta.min.isNegative() ? maxTrips : minTrips, delta.min);
  auto upper = addProduct(
      initial.max, delta.max.isNegative() ? minTrips : maxTrips, delta.max);
  if (!lower || !upper)
    return std::nullopt;
  return Interval{*lower, *upper};
}

} // namespace

struct ControlFlowRangeAnalysis::Impl {
  DominanceInfo &domInfo;
  SmallVector<std::unique_ptr<CFGLoopInfo>> loopInfos;
  DenseMap<Block *, LoopDescription> loops;
  mutable Dependencies dependencies;
  unsigned maxIVWidth = 1;

  Impl(Operation *top, DominanceInfo &domInfo) : domInfo(domInfo) {
    top->walk([&](Operation *op) {
      for (Region &region : op->getRegions()) {
        if (region.empty() || region.hasOneBlock() ||
            !domInfo.hasSSADominance(&region))
          continue;
        auto info = std::make_unique<CFGLoopInfo>(domInfo.getDomTree(&region));
        for (CFGLoop *loop : info->getLoopsInPreorder())
          if (auto description = describeLoop(loop, dependencies)) {
            maxIVWidth =
                std::max(maxIVWidth, ConstantIntRanges::getStorageBitwidth(
                                         description->induction.getType()));
            loops.try_emplace(loop->getHeader(), std::move(*description));
          }
        loopInfos.push_back(std::move(info));
      }
    });
  }

  struct Query {
    const Impl &analysis;
    GetRangeFn getRange;
    unsigned width;
    unsigned depth = 0;
    using Result = Evaluation<Interval>;
    using ExpressionKey = std::tuple<Value, Value, Block *, Value, unsigned>;
    using DeltaKey = std::tuple<Value, Value, unsigned>;
    DenseMap<ExpressionKey, Result> expressions;
    DenseMap<DeltaKey, Result> deltas;
    DenseMap<std::pair<Block *, Value>, Evaluation<LoopBounds>> loopBounds;

    Query(const Impl &analysis, GetRangeFn getRange, unsigned width)
        : analysis(analysis), getRange(getRange), width(width) {}

    template <typename Key, typename T, typename Fn>
    Evaluation<T> memo(DenseMap<Key, Evaluation<T>> &cache, Key key,
                       Fn compute) {
      if (auto it = cache.find(key); it != cache.end())
        return it->second;
      if (depth >= 64)
        return {};
      // Re-entering an unfinished expression is an unsupported cycle.
      cache.try_emplace(key);
      ++depth;
      auto result = compute();
      --depth;
      cache[key] = result;
      return result;
    }

    static bool unsupported(const Result &result) {
      return !result.value && !result.pending;
    }

    template <typename Fn>
    static Result combine(Result lhs, Result rhs, Fn fn) {
      if (unsupported(lhs) || unsupported(rhs))
        return {};
      if (lhs.pending || rhs.pending)
        return {std::nullopt, true};
      return {fn(*lhs.value, *rhs.value), false};
    }

    static Result add(Result lhs, Result rhs, bool subtract = false) {
      return combine(lhs, rhs,
                     [&](Interval a, Interval b) -> std::optional<Interval> {
                       bool loOverflow, hiOverflow;
                       APInt lo = subtract ? a.min.ssub_ov(b.max, loOverflow)
                                           : a.min.sadd_ov(b.min, loOverflow);
                       APInt hi = subtract ? a.max.ssub_ov(b.min, hiOverflow)
                                           : a.max.sadd_ov(b.max, hiOverflow);
                       if (loOverflow || hiOverflow)
                         return std::nullopt;
                       return Interval{lo, hi};
                     });
    }

    template <typename Include, typename Read>
    Result incoming(BlockArgument argument, Include include, Read read) {
      std::optional<Result> result;
      for (auto edge = argument.getOwner()->pred_begin(),
                end = argument.getOwner()->pred_end();
           edge != end; ++edge) {
        if (!include(*edge))
          continue;
        Value value = getIncoming(edge, argument);
        Result next = value ? read(value, *edge) : Result{};
        // Pending information never hides an unsupported structural arm.
        result = result ? combine(*result, next,
                                  [](Interval a, Interval b) {
                                    return Interval{min(a.min, b.min),
                                                    max(a.max, b.max)};
                                  })
                        : next;
      }
      return result.value_or(Result{});
    }

    bool independent(Value value, Value excluded) {
      if (!excluded)
        return true;
      const auto &outer =
          analysis.loops.find(excluded.getParentBlock())->second;
      return analysis.dependencies.check(value, *outer.loop, outer.induction,
                                         excluded);
    }

    Result invariant(Value value, const LoopDescription &loop, Value excluded,
                     bool isSigned) {
      if (!analysis.dependencies.check(value, *loop.loop) ||
          !independent(value, excluded))
        return {};
      auto range = readRange(value, loop.loop->getHeader(), getRange);
      return range.isUninitialized()
                 ? Result{std::nullopt, true}
                 : Result{widen(range.getValue(), width, isSigned), false};
    }

    Result loopExit(BlockArgument argument, BlockArgument base,
                    const LoopDescription &child, Value excluded,
                    bool isSigned) {
      auto initial = incoming(
          argument, [&](Block *p) { return p == child.entry; },
          [&](Value v, Block *p) {
            return expression(v, base, p, excluded, isSigned);
          });
      Value outer = excluded ? excluded : base;
      auto change = delta(argument, outer, isSigned);
      if (unsupported(initial))
        return {};
      // An unchanged child needs no finite trip count to describe its exit.
      if (change.value && change.value->min.isZero() &&
          change.value->max.isZero())
        return initial;
      auto count = bounds(child, outer);
      if (!count.value)
        return {std::nullopt, count.pending || change.pending};
      if (count.value->maxTrips.isZero())
        return initial;
      if (!change.value)
        return change;
      Interval zero{APInt(width, 0), APInt(width, 0)};
      return add(initial, {getRecurrenceBounds(zero, *change.value,
                                               count.value->minTrips,
                                               count.value->maxTrips),
                           false});
    }

    Result expression(Value value, BlockArgument base, Block *useBlock,
                      Value excluded, bool isSigned) {
      if (value == base)
        return {Interval{APInt(width, 0), APInt(width, 0)}, false};
      if (auto *op = value.getDefiningOp())
        useBlock = op->getBlock();
      return memo(
          expressions, ExpressionKey{value, base, useBlock, excluded, isSigned},
          [&]() -> Result {
            const auto &loop = analysis.loops.find(base.getOwner())->second;
            Value previous, increment;
            bool subtract = false;
            if (auto op = value.getDefiningOp<arith::AddIOp>()) {
              previous = op.getLhs();
              increment = op.getRhs();
              if (!analysis.dependencies.check(increment, *loop.loop))
                std::swap(previous, increment);
            } else if (auto op = value.getDefiningOp<arith::SubIOp>()) {
              previous = op.getLhs();
              increment = op.getRhs();
              subtract = true;
            }
            if (increment) {
              auto operand =
                  expression(previous, base, useBlock, excluded, isSigned);
              auto change = invariant(increment, loop, excluded, isSigned);
              return add(operand, change, subtract);
            }
            auto argument = dyn_cast<BlockArgument>(value);
            if (!argument || argument.getOwner() == base.getOwner() ||
                !loop.loop->contains(argument.getOwner()))
              return {};
            auto nested = analysis.loops.find(argument.getOwner());
            // Keep the controlling IV's update independent of nested loops.
            if (base != loop.induction && nested != analysis.loops.end() &&
                getUseKind(nested->second, useBlock, analysis.domInfo) ==
                    UseKind::Exit) {
              auto result =
                  loopExit(argument, base, nested->second, excluded, isSigned);
              if (!unsupported(result))
                return result;
            }
            return incoming(
                argument, [](Block *) { return true; },
                [&](Value v, Block *p) {
                  return loop.loop->contains(p)
                             ? expression(v, base, p, excluded, isSigned)
                             : Result{};
                });
          });
    }

    Result delta(BlockArgument argument, Value excluded, bool isSigned) {
      return memo(deltas, DeltaKey{argument, excluded, isSigned}, [&] {
        const auto &loop = analysis.loops.find(argument.getOwner())->second;
        return incoming(
            argument, [&](Block *p) { return loop.loop->contains(p); },
            [&](Value v, Block *p) {
              return expression(v, argument, p, excluded, isSigned);
            });
      });
    }

    Evaluation<LoopBounds> bounds(const LoopDescription &loop,
                                  Value excluded = {}) {
      return memo(
          loopBounds,
          std::pair<Block *, Value>{loop.loop->getHeader(), excluded},
          [&]() -> Evaluation<LoopBounds> {
            if (!independent(loop.bound, excluded))
              return {};
            for (auto edge = loop.loop->getHeader()->pred_begin(),
                      end = loop.loop->getHeader()->pred_end();
                 edge != end; ++edge) {
              if (*edge != loop.entry)
                continue;
              Value initial = getIncoming(edge, loop.induction);
              if (!initial || !independent(initial, excluded))
                return {};
            }
            auto initial = getInitialRange(loop.induction, loop, getRange);
            auto limit =
                readRange(loop.bound, loop.loop->getHeader(), getRange);
            bool pending = false;
            for (bool isSigned : {true, false}) {
              auto step = delta(loop.induction, excluded, isSigned);
              if (unsupported(step))
                continue;
              if (step.pending || initial.isUninitialized() ||
                  limit.isUninitialized()) {
                pending = true;
                continue;
              }
              auto result = getLoopBounds(
                  loop, widen(initial.getValue(), width, loop.isSigned),
                  widen(limit.getValue(), width, loop.isSigned), *step.value,
                  ConstantIntRanges::getStorageBitwidth(
                      loop.induction.getType()));
              if (result)
                return {result, false};
            }
            return {std::nullopt, pending};
          });
    }
  };

  std::optional<IntegerValueRange>
  getRange(BlockArgument argument, Block *useBlock, GetRangeFn getRange) const {
    auto it = loops.find(argument.getOwner());
    unsigned valueWidth =
        ConstantIntRanges::getStorageBitwidth(argument.getType());
    if (it == loops.end() || !valueWidth)
      return std::nullopt;
    const auto &loop = it->second;
    // All products are checked, including products across nested loops.
    unsigned width = 2 * std::max(valueWidth, maxIVWidth) + 16;
    Query query{*this, getRange, width};
    SmallVector<Interval, 2> deltas;
    bool pending = false;
    for (bool isSigned : {true, false}) {
      auto delta = query.delta(argument, {}, isSigned);
      if (delta.value && delta.value->min.isZero() && delta.value->max.isZero())
        return getInitialRange(argument, loop, getRange);
      pending |= delta.pending;
      if (delta.value)
        deltas.push_back(*delta.value);
    }
    auto count = query.bounds(loop);
    if (count.value && count.value->maxTrips.isZero())
      return getInitialRange(argument, loop, getRange);
    if (!count.value)
      return (count.pending || pending)
                 ? std::optional<IntegerValueRange>(IntegerValueRange())
                 : std::nullopt;
    if (deltas.empty())
      return pending ? std::optional<IntegerValueRange>(IntegerValueRange())
                     : std::nullopt;
    UseKind use = getUseKind(loop, useBlock, domInfo);
    if (argument == loop.induction) {
      Interval range = use == UseKind::Body   ? count.value->body
                       : use == UseKind::Exit ? count.value->exit
                                              : count.value->header;
      return IntegerValueRange(narrow(range, valueWidth, loop.isSigned));
    }
    auto initial = getInitialRange(argument, loop, getRange);
    if (initial.isUninitialized())
      return initial;
    APInt zero(width, 0), one(width, 1);
    APInt minTrips = use == UseKind::Exit ? count.value->minTrips : zero;
    APInt maxTrips = use == UseKind::Body
                         ? max(zero, count.value->maxTrips - one)
                         : count.value->maxTrips;
    std::optional<ConstantIntRanges> result;
    for (bool isSigned : {true, false}) {
      Interval init = widen(initial.getValue(), width, isSigned);
      for (const Interval &delta : deltas) {
        auto all =
            getRecurrenceBounds(init, delta, zero, count.value->maxTrips);
        if (!all || !contains(domain(valueWidth, width, isSigned), *all))
          continue;
        auto refined = getRecurrenceBounds(init, delta, minTrips, maxTrips);
        if (!refined)
          continue;
        auto range = narrow(*refined, valueWidth, isSigned);
        result = result ? result->intersection(range) : range;
      }
    }
    if (result)
      return IntegerValueRange(*result);
    return pending ? std::optional<IntegerValueRange>(IntegerValueRange())
                   : std::nullopt;
  }
};

ControlFlowRangeAnalysis::ControlFlowRangeAnalysis(Operation *top,
                                                   DominanceInfo &domInfo)
    : impl(std::make_unique<Impl>(top, domInfo)) {}

ControlFlowRangeAnalysis::~ControlFlowRangeAnalysis() = default;

std::optional<IntegerValueRange>
ControlFlowRangeAnalysis::getRange(BlockArgument argument, Block *useBlock,
                                   GetRangeFn getRange) const {
  return impl->getRange(argument, useBlock, getRange);
}
