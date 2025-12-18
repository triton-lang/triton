//===----------------------------------------------------------------------===//
//
// This pass tries to prefetch operands (a and b) of tt.dot.
// Those ConvertLayoutOps will be lowered to shared memory loads.
//
// For example:
// %a: tensor<128x32xf16, #enc>
// scf.for %iv = ... iter_args(%a_arg = %a, ...) {
//   %d = tt.dot %a_arg, %b, %c
//   ...
//   scf.yield %a_next, ...
// }
//
// will be translated to
//
// %a: tensor<128x32xf16, #enc>
// %a_tmp = tensor.subview %a[0, 0] [128, 16]
// %a_prefetch = ttg.local_load %a_tmp
// scf.for %iv = ... iter_args(%a_buf = %a, ..., %a_prefetch_arg = %a_prefetch)
// {
//   %x = tt.dot %a_prefetch_arg, %b, %c
//   %a_tmp_rem = tensor.subview %a_buf[0, 16] [128, 16]
//   %a_prefetch_next = ttg.local_load %a_tmp_rem
//   ...
//   scf.yield %next_a, ..., %a_prefetch_next
// }
//===----------------------------------------------------------------------===//

#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-prefetch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPREFETCH
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static SmallVector<Value>
getPrefetchSrc(Value v, scf::ForOp forContext = nullptr,
               SmallPtrSetImpl<Value> *visitedArgs = nullptr) {
  Operation *op = v.getDefiningOp();
  bool foundConvertFromShared = false;
  SmallVector<Value> rets;
  if (!op) {
    if (!forContext)
      return rets;
    auto barg = dyn_cast<BlockArgument>(v);
    if (!barg)
      return rets;
    if (barg.getOwner()->getParentOp() != forContext.getOperation())
      return rets;
    if (barg.getArgNumber() < forContext.getNumInductionVars())
      return rets;
    SmallPtrSet<Value, 8> localVisited;
    if (!visitedArgs)
      visitedArgs = &localVisited;
    if (!visitedArgs->insert(barg).second)
      return rets;
    auto yieldOp = cast<scf::YieldOp>(forContext.getBody()->getTerminator());
    unsigned iterIdx = barg.getArgNumber() - forContext.getNumInductionVars();
    return getPrefetchSrc(yieldOp.getOperand(iterIdx), forContext, visitedArgs);
  }
  rets.push_back(v);
  LDBG("Prefetch src: " << *op);
  SmallPtrSet<Operation *, 8> visited;
  Operation *curr = op;
  while (curr) {
    if (!visited.insert(curr).second)
      break;
    Value nextVal;
    for (Value operand : curr->getOperands()) {
      if (isa<triton::gpu::AsyncTokenType>(operand.getType()))
        continue;
      nextVal = operand;
      break;
    }
    if (!nextVal)
      break;
    rets.push_back(nextVal);
    if (auto load = dyn_cast<triton::gpu::LocalLoadOp>(curr))
      if (isa<DotOperandEncodingAttr>(load.getType().getEncoding()))
        foundConvertFromShared = true;
    if (auto barg = dyn_cast<BlockArgument>(nextVal)) {
      if (forContext &&
          barg.getOwner()->getParentOp() == forContext.getOperation() &&
          barg.getArgNumber() >= forContext.getNumInductionVars()) {
        SmallPtrSet<Value, 8> localVisited;
        if (!visitedArgs)
          visitedArgs = &localVisited;
        if (!visitedArgs->insert(barg).second)
          break;
        auto yieldOp =
            cast<scf::YieldOp>(forContext.getBody()->getTerminator());
        unsigned iterIdx =
            barg.getArgNumber() - forContext.getNumInductionVars();
        SmallVector<Value> tail = getPrefetchSrc(yieldOp.getOperand(iterIdx),
                                                 forContext, visitedArgs);
        rets.append(tail.begin(), tail.end());
        if (!tail.empty())
          foundConvertFromShared = true;
        break;
      }
      break;
    }
    curr = nextVal.getDefiningOp();
    if (curr)
      LDBG("op: " << *curr);
  }
  std::reverse(rets.begin(), rets.end());

  if (foundConvertFromShared)
    return rets;
  return {};
}

static bool isValueFromInductionVar(scf::ForOp forOp, Value v) {
  if (auto barg = dyn_cast<BlockArgument>(v)) {
    if (barg.getOwner()->getParentOp() == forOp.getOperation())
      return barg.getArgNumber() < forOp.getNumInductionVars();
  }
  return false;
}

static bool collectOpsForLoad(triton::gpu::LocalLoadOp load, scf::ForOp forOp,
                              SmallVector<Operation *> &orderedOps) {
  DenseSet<Operation *> visited;
  std::function<bool(Operation *)> dfs = [&](Operation *op) -> bool {
    if (op->getBlock() != forOp.getBody())
      return true;
    if (!visited.insert(op).second)
      return true;

    if (isa<triton::gpu::AsyncWaitOp>(op)) {
      orderedOps.push_back(op);
      return true;
    }

    if (op->getNumResults() == 0) {
      LDBG("Cannot clone op with no results: " << *op);
      return false;
    }

    if (op != load.getOperation()) {
      if (!isa<triton::gpu::AsyncWaitOp>(op) && !isMemoryEffectFree(op)) {
        LDBG("Cannot clone op with side effects: " << *op);
        return false;
      }
    }

    for (Value operand : op->getOperands()) {
      if (auto asyncWait = operand.getDefiningOp<triton::gpu::AsyncWaitOp>()) {
        if (!dfs(asyncWait)) {
          LDBG("Failed to clone async wait: " << *asyncWait);
          return false;
        }
        continue;
      }
      if (isa<triton::gpu::AsyncTokenType>(operand.getType()))
        continue;
      if (isValueFromInductionVar(forOp, operand))
        continue;
      if (auto barg = dyn_cast<BlockArgument>(operand)) {
        if (barg.getOwner()->getParentOp() == forOp.getOperation())
          continue;
      }
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      if (!dfs(def)) {
        LDBG("Failed to clone operand def: " << *def);
        return false;
      }
    }
    orderedOps.push_back(op);
    return true;
  };

  return dfs(load.getOperation());
}

struct LoadRotationInfo {
  triton::gpu::LocalLoadOp loadOp;
  SmallVector<Operation *> opsInOrder;
  SmallVector<Value> carriedValues;
  SmallVector<Value> initialValues;
  SmallVector<Value> rotatedValues;
  unsigned iterArgBase = 0;
};

static scf::ForOp rotateLocalLoadChains(scf::ForOp forOp) {
  Block *loopBody = forOp.getBody();
  auto yieldOp = cast<scf::YieldOp>(loopBody->getTerminator());

  SmallVector<LoadRotationInfo, 4> rotations;
  DenseSet<Operation *> seenLoads;

  auto considerOperand = [&](Value operand) {
    SmallVector<Value> vals = getPrefetchSrc(operand, forOp);
    if (vals.empty())
      return;
    triton::gpu::LocalLoadOp load = nullptr;
    for (Value candidate : vals) {
      load = candidate.getDefiningOp<triton::gpu::LocalLoadOp>();
      if (load)
        break;
    }
    if (!load || load->getParentOfType<scf::ForOp>() != forOp)
      return;
    if (!seenLoads.insert(load.getOperation()).second)
      return;
    if (llvm::is_contained(yieldOp.getOperands(), load.getResult()))
      return;

    LoadRotationInfo info;
    info.loadOp = load;
    if (!collectOpsForLoad(load, forOp, info.opsInOrder))
      return;
    LDBG("Collected ops for load: " << info.opsInOrder.size());
    rotations.push_back(std::move(info));
  };

  for (Operation &op : loopBody->without_terminator()) {
    if (auto dot = dyn_cast<triton::DotOp>(&op)) {
      considerOperand(dot.getA());
      considerOperand(dot.getB());
    }
  }

  LDBG("Candidate load chains: " << rotations.size());

  if (rotations.empty())
    return forOp;

  for (auto &info : rotations) {
    DenseSet<Operation *> sliceOps;
    for (Operation *op : info.opsInOrder)
      sliceOps.insert(op);
    DenseSet<Value> seen;
    info.carriedValues.clear();
    for (Operation *op : info.opsInOrder) {
      for (Value res : op->getResults()) {
        bool needsCarry = false;
        for (OpOperand &use : res.getUses()) {
          Operation *user = use.getOwner();
          if (sliceOps.contains(user))
            continue;
          needsCarry = true;
          break;
        }
        if (needsCarry && seen.insert(res).second)
          info.carriedValues.push_back(res);
      }
    }
    LDBG("Carried values count: " << info.carriedValues.size());
  }

  llvm::erase_if(rotations, [](const LoadRotationInfo &info) {
    return info.carriedValues.empty();
  });

  LDBG("Retained load chains after carry analysis: " << rotations.size());

  if (rotations.empty())
    return forOp;

  DenseSet<Operation *> opsToRotate;
  for (auto &info : rotations)
    for (Operation *rotOp : info.opsInOrder)
      opsToRotate.insert(rotOp);

  OpBuilder preheaderBuilder(forOp);
  preheaderBuilder.setInsertionPoint(forOp);

  for (auto &info : rotations) {
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), forOp.getLowerBound());
    for (auto [arg, init] :
         llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs()))
      mapping.map(arg, init);
    for (Operation *op : info.opsInOrder) {
      Operation *clone = preheaderBuilder.clone(*op, mapping);
      for (auto [origRes, newRes] :
           llvm::zip(op->getResults(), clone->getResults()))
        mapping.map(origRes, newRes);
    }
    info.initialValues.clear();
    info.initialValues.reserve(info.carriedValues.size());
    for (Value carry : info.carriedValues)
      info.initialValues.push_back(mapping.lookupOrDefault(carry));
  }

  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                 forOp.getInitArgs().end());
  for (auto &info : rotations)
    newInitArgs.append(info.initialValues.begin(), info.initialValues.end());

  unsigned baseIterArgCount = forOp.getInitArgs().size();
  unsigned extraArgCursor = baseIterArgCount;
  for (auto &info : rotations) {
    info.iterArgBase = extraArgCursor;
    extraArgCursor += info.carriedValues.size();
  }

  OpBuilder builder(forOp);
  auto newFor =
      scf::ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), newInitArgs,
                         [&](OpBuilder &nestedBuilder, Location loc,
                             Value /*iv*/, ValueRange iterArgs) {
                           scf::YieldOp::create(nestedBuilder, loc, iterArgs);
                         });

  IRMapping bodyMap;
  bodyMap.map(forOp.getInductionVar(), newFor.getInductionVar());

  auto newArgs = newFor.getRegionIterArgs();
  for (auto [oldArg, newArg] : llvm::zip(forOp.getRegionIterArgs(),
                                         newArgs.take_front(baseIterArgCount)))
    bodyMap.map(oldArg, newArg);

  for (auto &info : rotations)
    for (auto [idx, value] : llvm::enumerate(info.carriedValues))
      bodyMap.map(value, newArgs[info.iterArgBase + idx]);

  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(newFor.getBody());

  for (Operation &op : loopBody->without_terminator()) {
    if (opsToRotate.contains(&op))
      continue;
    Operation *clone = bodyBuilder.clone(op, bodyMap);
    for (auto [origRes, newRes] :
         llvm::zip(op.getResults(), clone->getResults()))
      bodyMap.map(origRes, newRes);
  }

  Operation *defaultYield = newFor.getBody()->getTerminator();
  defaultYield->erase();

  OpBuilder tailBuilder = OpBuilder::atBlockEnd(newFor.getBody());
  for (auto &info : rotations) {
    info.rotatedValues.clear();
    info.rotatedValues.reserve(info.carriedValues.size());
    for (Operation *op : info.opsInOrder) {
      Operation *clone = tailBuilder.clone(*op, bodyMap);
      for (auto [origRes, newRes] :
           llvm::zip(op->getResults(), clone->getResults()))
        bodyMap.map(origRes, newRes);
    }
    for (Value carry : info.carriedValues)
      info.rotatedValues.push_back(bodyMap.lookupOrDefault(carry));
  }

  SmallVector<Value> yieldValues;
  unsigned extraYieldCount = 0;
  for (auto &info : rotations)
    extraYieldCount += info.rotatedValues.size();
  yieldValues.reserve(yieldOp->getNumOperands() + extraYieldCount);
  for (Value operand : yieldOp.getOperands())
    yieldValues.push_back(bodyMap.lookupOrDefault(operand));
  for (auto &info : rotations)
    yieldValues.append(info.rotatedValues.begin(), info.rotatedValues.end());

  scf::YieldOp::create(tailBuilder, forOp.getLoc(), yieldValues);

  auto newResults = newFor.getResults();
  for (auto [oldRes, newRes] : llvm::zip(
           forOp.getResults(), newResults.take_front(forOp.getNumResults())))
    oldRes.replaceAllUsesWith(newRes);

  forOp.erase();
  return newFor;
}

class Prefetcher {
  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  ///
  // TODO: add a hook to infer prefetchWidth
  unsigned prefetchWidth = 32;

  /// dots to be prefetched
  SetVector<triton::DotOp> dots;
  /// dot => dot operand
  DenseMap<Value, Value> dot2aLoopArg;
  DenseMap<Value, Value> dot2aHeaderDef;
  DenseMap<Value, Value> dot2bLoopArg;
  DenseMap<Value, Value> dot2bHeaderDef;
  DenseMap<Value, Value> dot2aYield;
  DenseMap<Value, Value> dot2bYield;
  DenseMap<Value, SmallVector<Value>> dot2aVals;
  DenseMap<Value, SmallVector<Value>> dot2bVals;
  /// operand => defining
  DenseMap<Value, Value> operand2headPrefetch;

  LogicalResult isForOpOperand(Value v);

  Value generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                         Attribute dotEncoding, OpBuilder &builder,
                         std::optional<int64_t> offsetK = std::nullopt,
                         std::optional<int64_t> shapeK = std::nullopt);

  void cloneElementwiseOps(Value &bRem, const SmallVector<Value> &vals,
                           OpBuilder &builder);

public:
  Prefetcher() = delete;

  Prefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();
};

void Prefetcher::cloneElementwiseOps(Value &ret, const SmallVector<Value> &vals,
                                     OpBuilder &builder) {
  IRMapping mapping;
  mapping.map(vals[1], ret);
  for (int i = 2; i < vals.size(); i++) {
    Value v = vals[i];
    Value curr = builder.clone(*v.getDefiningOp(), mapping)->getResult(0);
    if (isa<RankedTensorType>(curr.getType())) {
      auto retType = RankedTensorType::get(
          cast<RankedTensorType>(ret.getType()).getShape(),
          cast<RankedTensorType>(curr.getType()).getElementType(),
          cast<RankedTensorType>(curr.getDefiningOp()->getOperand(0).getType())
              .getEncoding());
      curr.setType(retType);
    }
    mapping.map(v, curr);
  }
  if (vals.size() > 1)
    ret = mapping.lookup(vals.back());
}

Value Prefetcher::generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                                   Attribute dotEncoding, OpBuilder &builder,
                                   std::optional<int64_t> offsetK,
                                   std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<triton::gpu::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  auto rank = shape.size();
  SmallVector<int32_t> offset(rank, 0);
  Type elementType = type.getElementType();

  // k => (prefetchWidth, k - prefetchWidth)
  int64_t kIdx = opIdx == 0 ? rank - 1 : rank - 2;

  offset[kIdx] = isPrologue ? 0 : prefetchWidth;
  shape[kIdx] = isPrologue ? prefetchWidth : (shape[kIdx] - prefetchWidth);

  if (shapeK)
    shape[kIdx] = *shapeK;
  if (offsetK)
    offset[kIdx] = *offsetK;

  Value newSmem = triton::gpu::MemDescSubsliceOp::create(
      builder, v.getLoc(),
      triton::gpu::MemDescType::get(
          shape, elementType, type.getEncoding(), type.getMemorySpace(),
          type.getMutableMemory(), type.getAllocShape()),
      v, offset);

  auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding, prefetchWidth / 8);
  Value prefetchSlice = triton::gpu::LocalLoadOp::create(
      builder, v.getLoc(),
      RankedTensorType::get(shape, elementType, dotOperandEnc), newSmem);

  return prefetchSlice;
}

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<TensorOrMemDesc>(v.getType()).getEncoding();
  };

  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // Only accepts dotOps encoded as Nvidia MMA v2 or AMD MFMA
      auto dstMmaEnc =
          dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(dotOp.getResult()));
      auto dstMfmaEnc =
          dyn_cast<AMDMfmaEncodingAttr>(getEncoding(dotOp.getResult()));
      if (!dstMfmaEnc && (!dstMmaEnc || dstMmaEnc.getVersionMajor() != 2))
        // Don't rewrite if any other type is found.
        return failure();
      dotsInFor.push_back(dotOp);
    }

  if (dotsInFor.empty())
    return failure();

  // TODO: segfault (original for still has uses)
  // when used in flash attention that has 2 dots in the loop
  if (dotsInFor.size() > 1)
    return failure();

  auto getIncomingOp = [this](Value v) -> Value {
    if (auto arg = mlir::dyn_cast<BlockArgument>(v))
      if (arg.getOwner()->getParentOp() == forOp.getOperation())
        return forOp.getTiedLoopInit(arg)->get();
    return Value();
  };

  auto getYieldOperand = [this](BlockArgument arg) -> Value {
    unsigned yieldIdx = arg.getArgNumber() - forOp.getNumInductionVars();
    return yieldOp.getOperand(yieldIdx);
  };

  auto resolveMemDescFromInit = [&](BlockArgument arg) -> Value {
    unsigned iterIdx = arg.getArgNumber() - forOp.getNumInductionVars();
    Value initVal = forOp.getInitArgs()[iterIdx];
    auto chain = getPrefetchSrc(initVal);
    if (chain.empty())
      return Value();
    return chain.front();
  };

  auto resolveMemDescFromYield = [&](BlockArgument arg) -> Value {
    unsigned iterIdx = arg.getArgNumber() - forOp.getNumInductionVars();
    Value yieldVal = yieldOp.getOperand(iterIdx);
    auto chain = getPrefetchSrc(yieldVal, forOp);
    if (chain.empty())
      return Value();
    return chain.front();
  };

  auto selectMemDesc = [](ArrayRef<Value> vals) -> Value {
    for (Value candidate : vals) {
      auto memDescType =
          dyn_cast<triton::gpu::MemDescType>(candidate.getType());
      if (!memDescType)
        continue;
      if (isa<triton::gpu::MemDescIndexOp>(candidate.getDefiningOp()))
        return candidate;
      if (memDescType.getShape().size() == 2)
        return candidate;
    }
    return Value();
  };

  for (triton::DotOp dot : dotsInFor) {
    auto aType = dot.getA().getType();
    auto bType = dot.getB().getType();
    auto aEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(bType.getEncoding());
    int aKWidth = aEnc.getKWidth();
    int bKWidth = bEnc.getKWidth();
    assert(aKWidth == bKWidth);

    auto kSize = aType.getShape().back();

    // works better with nvidia tensor cores
    unsigned elementWidth = aType.getElementTypeBitWidth();
    if (aKWidth == 0)
      prefetchWidth = 256 / elementWidth;
    else
      prefetchWidth = 8 * aKWidth;

    // Skip prefetching if kSize is less than prefetchWidth
    if (kSize < prefetchWidth)
      continue;
    auto aVals = getPrefetchSrc(dot.getA(), forOp);
    auto bVals = getPrefetchSrc(dot.getB(), forOp);

    if (aVals.size() && bVals.size()) {
      LDBG("A chain length: " << aVals.size());
      LDBG("B chain length: " << bVals.size());
      Value aSmem = selectMemDesc(aVals);
      Value bSmem = selectMemDesc(bVals);
      if (!aSmem || !bSmem)
        continue;
      LDBG("Selected A memdesc type: " << aSmem.getType());
      LDBG("Selected B memdesc type: " << bSmem.getType());
      Value aHeaderDef = getIncomingOp(aSmem);
      Value bHeaderDef = getIncomingOp(bSmem);
      if (!aHeaderDef)
        if (auto barg = dyn_cast<BlockArgument>(dot.getA()))
          aHeaderDef = resolveMemDescFromInit(barg);
      if (!bHeaderDef)
        if (auto barg = dyn_cast<BlockArgument>(dot.getB()))
          bHeaderDef = resolveMemDescFromInit(barg);
      // Only prefetch loop arg
      if (aHeaderDef && bHeaderDef) {
        dots.insert(dot);
        dot2aVals[dot] = aVals;
        dot2bVals[dot] = bVals;
        dot2aHeaderDef[dot] = aHeaderDef;
        dot2bHeaderDef[dot] = bHeaderDef;
        dot2aLoopArg[dot] = aSmem;
        dot2bLoopArg[dot] = bSmem;
        if (auto barg = dyn_cast<BlockArgument>(dot.getA()))
          dot2aYield[dot] = resolveMemDescFromYield(barg);
        else if (auto barg = dyn_cast<BlockArgument>(aSmem))
          dot2aYield[dot] = getYieldOperand(barg);
        else
          dot2aYield[dot] = aSmem;
        if (auto barg = dyn_cast<BlockArgument>(dot.getB()))
          dot2bYield[dot] = resolveMemDescFromYield(barg);
        else if (auto barg = dyn_cast<BlockArgument>(bSmem))
          dot2bYield[dot] = getYieldOperand(barg);
        else
          dot2bYield[dot] = bSmem;
      }
    }
  }

  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (triton::DotOp dot : dots) {
    Attribute dotEncoding = dot.getType().getEncoding();
    Value aPrefetched =
        generatePrefetch(dot2aHeaderDef[dot], 0, true, dotEncoding, builder);
    cloneElementwiseOps(aPrefetched, dot2aVals[dot], builder);
    Value bPrefetched =
        generatePrefetch(dot2bHeaderDef[dot], 1, true, dotEncoding, builder);
    cloneElementwiseOps(bPrefetched, dot2bVals[dot], builder);

    operand2headPrefetch[dot.getA()] = aPrefetched;
    operand2headPrefetch[dot.getB()] = bPrefetched;
  }
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);
  for (triton::DotOp dot : dots) {
    loopArgs.push_back(operand2headPrefetch[dot.getA()]);
    loopArgs.push_back(operand2headPrefetch[dot.getB()]);
  }

  auto newForOp =
      scf::ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  // The insertion point should be placed before the yield op
  auto setInsertionPointBeforeYield = [](OpBuilder &builder,
                                         scf::ForOp newForOp) {
    if (newForOp.getBody()->mightHaveTerminator()) {
      builder.setInsertionPoint(newForOp.getBody()->getTerminator());
    } else {
      builder.setInsertionPointToEnd(newForOp.getBody());
    }
  };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    // If we're currently trying to sink a prefetched dot, we need to stop
    // sinking it (by resetting the insertion point to the end) if we find
    // control flow, or anything that depends on the dot op.
    if (op.getNumRegions() > 0) {
      setInsertionPointBeforeYield(builder, newForOp);
    }
    for (auto operand : op.getOperands()) {
      if (auto def = operand.getDefiningOp()) {
        auto dot = dyn_cast<triton::DotOp>(def);
        if (dot && dots.contains(dot)) {
          setInsertionPointBeforeYield(builder, newForOp);
        }
      }
    }
    Operation *newOp = builder.clone(op, mapping);
    auto dot = dyn_cast<triton::DotOp>(&op);
    if (dot && dots.contains(dot)) {
      Attribute dotEncoding = dot.getType().getEncoding();
      // prefetched dot
      Operation *firstDot = builder.clone(*dot, mapping);
      if (Value a = operand2headPrefetch.lookup(dot.getA()))
        firstDot->setOperand(
            0, newForOp.getTiedLoopRegionIterArg(&*a.use_begin()));
      if (Value b = operand2headPrefetch.lookup(dot.getB()))
        firstDot->setOperand(
            1, newForOp.getTiedLoopRegionIterArg(&*b.use_begin()));

      // remaining part
      int64_t kOff = prefetchWidth;
      int64_t kRem = dot.getA().getType().getShape().back() - prefetchWidth;
      Operation *prevDot = firstDot;
      if (kRem == 0) {
        // There is only one dot while prefetchWidth == kSize so delay issuing
        // it. Meanwhile, newOp should be set to firstDot to make sure the dot
        // result is updated to yield.
        builder.setInsertionPoint(prevDot);
        newOp = firstDot;
      }

      while (kRem != 0) {
        // int64_t kShape = largestPow2(kRem);
        int64_t kShape = prefetchWidth;
        auto insertionPoint = builder.saveInsertionPoint();
        builder.setInsertionPoint(prevDot);
        Value mappedALoop = mapping.lookupOrDefault(dot2aLoopArg[dot]);
        if (!mappedALoop)
          break;
        Value aRem = generatePrefetch(mappedALoop, 0, false, dotEncoding,
                                      builder, kOff, kShape);
        cloneElementwiseOps(aRem, dot2aVals[dot], builder);
        Value mappedBLoop = mapping.lookupOrDefault(dot2bLoopArg[dot]);
        if (!mappedBLoop)
          break;
        Value bRem = generatePrefetch(mappedBLoop, 1, false, dotEncoding,
                                      builder, kOff, kShape);
        cloneElementwiseOps(bRem, dot2bVals[dot], builder);
        builder.restoreInsertionPoint(insertionPoint);
        newOp = builder.clone(*dot, mapping);
        newOp->setOperand(0, aRem);
        newOp->setOperand(1, bRem);
        newOp->setOperand(2, prevDot->getResult(0));
        prevDot = newOp;
        kOff += kShape;
        kRem -= kShape;
        if (kRem == 0) {
          // We want to delay issuing the last dot as long as possible, ideally
          // until after the prefetch.  To accomplish this, set the insertion
          // point above the dot.  If we find anything dependent on the dot (at
          // the top of this loop), we resume inserting after it.
          builder.setInsertionPoint(prevDot);
        }
      }
    }
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // prefetch next iteration
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));
  for (triton::DotOp dot : dots) {
    Attribute dotEncoding = dot.getType().getEncoding();
    Value mappedAYield = mapping.lookupOrDefault(dot2aYield[dot]);
    if (!mappedAYield)
      mappedAYield = mapping.lookupOrDefault(dot2aLoopArg[dot]);
    if (!mappedAYield)
      continue;
    Value aToYield =
        generatePrefetch(mappedAYield, 0, true, dotEncoding, builder);
    cloneElementwiseOps(aToYield, dot2aVals[dot], builder);
    yieldValues.push_back(aToYield);
    // bToYield
    Value mappedBYield = mapping.lookupOrDefault(dot2bYield[dot]);
    if (!mappedBYield)
      mappedBYield = mapping.lookupOrDefault(dot2bLoopArg[dot]);
    if (!mappedBYield)
      continue;
    Value bToYield =
        generatePrefetch(mappedBYield, 1, true, dotEncoding, builder);
    cloneElementwiseOps(bToYield, dot2bVals[dot], builder);
    yieldValues.push_back(bToYield);
  }
  // Update ops of yield
  builder.setInsertionPointToEnd(newForOp.getBody());
  if (!yieldValues.empty())
    scf::YieldOp::create(builder, yieldOp.getLoc(), yieldValues);
  return newForOp;
}

} // anonymous namespace

struct PrefetchPass : public impl::TritonGPUPrefetchBase<PrefetchPass> {
  void runOnOperation() override {

    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    for (scf::ForOp forOp : loops) {
      scf::ForOp rotated = rotateLocalLoadChains(forOp);
      Prefetcher prefetcher(rotated);

      if (prefetcher.initialize().failed())
        continue;

      prefetcher.emitPrologue();

      scf::ForOp newForOp = prefetcher.createNewForOp();

      for (unsigned i = 0; i < rotated->getNumResults(); ++i)
        rotated->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      rotated->erase();
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
