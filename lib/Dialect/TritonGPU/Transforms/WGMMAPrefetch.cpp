#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tritongpu-wgmma-prefetch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
using namespace mlir;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUWGMMAPREFETCH
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// RSGEMM means the lhs is in register and the rhs is in
// shared memory
LogicalResult isSupportedRSGEMM(ttng::WarpGroupDotOp dotOp) {
  Value operandA = dotOp.getA();
  Value operandB = dotOp.getB();

  // operandB is in shared memoryassert
  auto encA = dyn_cast<TensorOrMemDesc>(operandA.getType()).getEncoding();
  auto encB = dyn_cast<TensorOrMemDesc>(operandB.getType()).getEncoding();

  if (isa<DotOperandEncodingAttr>(encA) && isa<NVMMASharedEncodingAttr>(encB)){
    auto bShape = cast<MemDescType>(operandB.getType()).getShape();
    auto rank = bShape.size();
    auto EncB = cast<NVMMASharedEncodingAttr>(encB);
    auto SwizzleByteWidth = EncB.getSwizzlingByteWidth();
    auto ElementBitWidth = EncB.getElementBitWidth();
    auto TransB = EncB.getTransposed();
    int64_t SwizzleDimSize = TransB ? bShape[rank - 2] : bShape[rank - 1];
    // Currently, memory subview does not calculate correct base
    // address for subtile when SwizzleByteWidth is larger than
    // the matrix size in the swizzle dim. It can be fixed through
    // calculating correct base address in subview op for subtile
    if(SwizzleDimSize * ElementBitWidth == SwizzleByteWidth * 8)
      return llvm::success();
  }

  return llvm::failure();
}

class WGMMAPrefetcher {
  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;

  unsigned prefetchWidth;

  /// dots to be prefetched
  SetVector<ttng::WarpGroupDotOp> dots;
  /// dot => dot operand
  DenseMap<Value, Value> dot2aLoopArg;
  DenseMap<Value, Value> dot2aHeaderDef;
  DenseMap<Value, Value> dot2bLoopArg;
  DenseMap<Value, Value> dot2bHeaderDef;
  DenseMap<Value, Value> dot2aSrcMemDesc;
  DenseMap<Value, Value> dot2bSrcMemDesc;
  DenseMap<Value, Value> dot2aYield;
  DenseMap<Value, Value> dot2bYield;
  DenseMap<Value, SmallVector<Value>> dot2aVals;
  DenseMap<Value, SmallVector<Value>> dot2aValsLocalLoad;
  DenseMap<Value, SmallVector<Value>> dot2aValsElementWise;
  DenseMap<Value, SmallVector<Value>> dot2bVals;
  DenseMap<Value, ttng::WarpGroupDotWaitOp> dot2Wait;

  Value generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                         bool loodToReg, Attribute dotEncoding,
                         OpBuilder &builder,
                         std::optional<int64_t> offsetK = std::nullopt,
                         std::optional<int64_t> shapeK = std::nullopt,
                         std::optional<int64_t> kWidth = std::nullopt);

  void cloneElementwiseOps(Value &bRem, const SmallVector<Value> &vals,
                           Value source, OpBuilder &builder);

public:
  WGMMAPrefetcher() = delete;

  WGMMAPrefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();
};

void WGMMAPrefetcher::cloneElementwiseOps(Value &ret,
                                          const SmallVector<Value> &vals,
                                          Value source, OpBuilder &builder) {
  IRMapping mapping;
  mapping.map(source, ret);
  for (int i = 0; i < vals.size(); i++) {
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
  ret = mapping.lookup(vals.back());
}

Value WGMMAPrefetcher::generatePrefetch(
    Value v, unsigned opIdx, bool isPrologue, bool loadToReg,
    Attribute dotEncoding, OpBuilder &builder, std::optional<int64_t> offsetK,
    std::optional<int64_t> shapeK, std::optional<int64_t> kWidth) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<triton::gpu::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  auto rank = shape.size();
  SmallVector<int64_t> offset(rank, 0);
  Type elementType = type.getElementType();

  int64_t kIdx = opIdx == 0 ? rank - 1 : rank - 2;

  offset[kIdx] = isPrologue ? 0 : prefetchWidth;
  shape[kIdx] = isPrologue ? prefetchWidth : (shape[kIdx] - prefetchWidth);

  if (shapeK)
    shape[kIdx] = *shapeK;
  if (offsetK)
    offset[kIdx] = *offsetK;

  SmallVector<Value> offsetsVal;
  for (int64_t off : offset)
    offsetsVal.push_back(
        builder.create<arith::ConstantIntOp>(v.getLoc(), off, 32));
  Value newSmem = builder.create<triton::gpu::MemDescSubviewOp>(
      v.getLoc(),
      triton::gpu::MemDescType::get(
          shape, elementType, type.getEncoding(), type.getMemorySpace(),
          type.getMutableMemory(), type.getAllocShape()),
      v, offsetsVal);

  if (loadToReg) {
    int64_t newKWidth = kWidth ? *kWidth : prefetchWidth / 8;

    auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
        builder.getContext(), opIdx, dotEncoding, newKWidth);
    Value prefetchSlice = builder.create<triton::gpu::LocalLoadOp>(
        v.getLoc(), RankedTensorType::get(shape, elementType, dotOperandEnc),
        newSmem);
    return prefetchSlice;
  }

  return newSmem;
}

LogicalResult WGMMAPrefetcher::initialize() {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<TensorOrMemDesc>(v.getType()).getEncoding();
  };

  SmallVector<nvidia_gpu::WarpGroupDotOp> dotsInFor;

  // Step 1: check the condition if the forloop can be prefetched
  for (Operation &op : *loop) {
    if (auto dotOp = dyn_cast<ttng::WarpGroupDotOp>(op)) {
      auto dstMmaEnc =
          dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(dotOp.getResult()));
      dotsInFor.push_back(dotOp);
    } else if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // To be conservative, we only allow WarpGroupDotOp in the foor loop;
      return llvm::failure();
    }
  }

  if (dotsInFor.empty()) {
    return llvm::failure();
  }

  if (dotsInFor.size() > 1) {
    return llvm::failure();
  }

  for (ttng::WarpGroupDotOp dotOp : dotsInFor) {
    if (isSupportedRSGEMM(dotOp).failed())
      return llvm::failure();
  }

  auto getPrefetchSrc = [](Value v) -> SmallVector<Value> {
    // walk back to conversion
    Operation *op = v.getDefiningOp();
    bool foundConvertFromShared = false;
    SmallVector<Value> rets;
    if (isa<LocalAllocOp>(op)) {
      return rets;
    }

    rets.push_back(op->getResult(0));
    if (dyn_cast<MemDescSubviewOp>(op))
      return rets;
    LDBG("Prefetch src: " << *op);
    while (op) {
      if (op->getNumOperands() != 1)
        break;
      if (!op->getResult(0).hasOneUse())
        break;
      rets.push_back(op->getOperand(0));
      if (auto cvt = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        // NYI for other encodings, for example if we have transpose
        // in the chain
        if (isa<DotOperandEncodingAttr>(cvt.getType().getEncoding()))
          foundConvertFromShared = true;
        break;
      }
      op = op->getOperand(0).getDefiningOp();
      if (op)
        LDBG("op: " << *op);
    }
    std::reverse(rets.begin(), rets.end());

    if (foundConvertFromShared)
      return rets;
    return {};
  };

  for (ttng::WarpGroupDotOp dotOp : dotsInFor) {
    // If getMaxNumImpreciseAcc > 0, WGMMA.cpp will have
    // extra treatment for dotOp (e.g., add accumulator).
    // Therefore, we disable the optimization here
    // when getMaxNumImpreciseAcc > 0;
    if(dotOp.getMaxNumImpreciseAcc() > 0){
      return failure();
    }

    auto aType = dotOp.getA().getType();
    auto bType = dotOp.getB().getType();

    auto aEnc = mlir::cast<DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc = mlir::cast<NVMMASharedEncodingAttr>(bType.getEncoding());

    if (!aEnc || !bEnc) {
      return failure();
    }
    auto aElementBitWidth = aType.getElementTypeBitWidth();
    auto bElementBitWidth = bType.getElementTypeBitWidth();
    assert((aElementBitWidth == bElementBitWidth) &&
           "BitWidth of a and b for dot does not match");

    // Get Prefetchwidth based on the instruction shape in K dim
    auto dType = dotOp.getType();
    auto mmaEnncoding = cast<NvidiaMmaEncodingAttr>(dType.getEncoding());
    auto instrShape = mmaEnncoding.getInstrShape();
    prefetchWidth = instrShape[2];

    auto kSize = aType.getShape().back();
    if (kSize < prefetchWidth)
      return failure();

    auto aVals = getPrefetchSrc(dotOp.getA());
    auto bVals = getPrefetchSrc(dotOp.getB());

    if (aVals.size() && bVals.size()) {
      Value aSmem = aVals.front();
      Value bSmem = bVals.front();
      if (!dyn_cast<MemDescSubviewOp>(aSmem.getDefiningOp()) ||
          !dyn_cast<MemDescSubviewOp>(bSmem.getDefiningOp())) {
        return failure();
      }
      auto dotOpResult = dotOp.getResult();
      if (!dotOpResult.hasOneUse())
        return failure();

      auto dotOpUser = *(dotOpResult.getUsers().begin());
      auto dotWait = dyn_cast<nvidia_gpu::WarpGroupDotWaitOp>(dotOpUser);
      if (!dotWait)
        return failure();

      dots.insert(dotOp);
      dot2aVals[dotOp] = aVals;
      for (auto op : aVals) {
        if (isa<MemDescSubviewOp, LocalLoadOp>(op.getDefiningOp())) {
          dot2aValsLocalLoad[dotOp].push_back(op);
        } else {
          auto curOp = op.getDefiningOp();
          if(curOp->hasTrait<mlir::OpTrait::Elementwise>() && isMemoryEffectFree(curOp))
            dot2aValsElementWise[dotOp].push_back(op);
          else
           return failure();
        }
      }

      dot2bVals[dotOp] = bVals;
      dot2aSrcMemDesc[dotOp] = aSmem;
      dot2bSrcMemDesc[dotOp] = bSmem;
      dot2Wait[dotOp] = dotWait;
    }
  }

  return llvm::success();
}

scf::ForOp WGMMAPrefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);

  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  auto setInsertionPointBeforeYield = [](OpBuilder &builder,
                                         scf::ForOp newForOp) {
    if (newForOp.getBody()->mightHaveTerminator()) {
      builder.setInsertionPoint(newForOp.getBody()->getTerminator());
    } else {
      builder.setInsertionPointToEnd(newForOp.getBody());
    }
  };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (op.getNumRegions() > 0) {
      setInsertionPointBeforeYield(builder, newForOp);
    }
    for (auto operand : op.getOperands()) {
      if (auto def = operand.getDefiningOp()) {
        auto dot = dyn_cast<nvidia_gpu::WarpGroupDotOp>(def);
        if (dot && dots.contains(dot)) {
          setInsertionPointBeforeYield(builder, newForOp);
        }
      }
    }
    Operation *newOp = builder.clone(op, mapping);
    auto dot = dyn_cast<nvidia_gpu::WarpGroupDotOp>(&op);
    if (dot && dots.contains(dot)) {
      Attribute dotEncoding = dot.getType().getEncoding();

      int64_t kShape = dot.getA().getType().getShape().back();
      auto aEnc =
          dyn_cast<DotOperandEncodingAttr>(dot.getA().getType().getEncoding());
      int64_t kWidth = aEnc.getKWidth();
      int64_t subTileCnt = ceil<int64_t>(kShape, prefetchWidth);

      SmallVector<Value> PrefetchedA;

      // Prefetching
      for (int i = 0; i < subTileCnt; i++) {
        Value aRem = generatePrefetch(mapping.lookup(dot2aSrcMemDesc[dot]), 0,
                                      false, true, dotEncoding, builder,
                                      prefetchWidth * i, prefetchWidth, kWidth);
        PrefetchedA.push_back(aRem);
      }

      // Interleave elementwise with WGMMA
      nvidia_gpu::WarpGroupDotOp prevDot;
      Value UseC = nullptr;
      if(dot.getUseC()){
        UseC = mapping.lookup(dot.getUseC());
      }
      Value OpC = mapping.lookup(dot.getC());

      for (int i = 0; i < subTileCnt; i++) {
        cloneElementwiseOps(PrefetchedA[i], dot2aValsElementWise[dot],
                            dot2aValsLocalLoad[dot].back(), builder);
        Value bSubtile = generatePrefetch(mapping.lookup(dot2bSrcMemDesc[dot]),
                                          1, false, false, dotEncoding, builder,
                                          prefetchWidth * i, prefetchWidth);
        if(i > 0){
          OpC = prevDot.getResult();
          // for subtiles with index > 0, useC is set to 1 for
          // using the result of prevDot
          UseC = builder.create<mlir::arith::ConstantIntOp>(newOp->getLoc(), 1, 1);
        }
        auto newDotOp = builder.create<nvidia_gpu::WarpGroupDotOp>(
            newOp->getLoc(), dot.getType(),
            PrefetchedA[i], bSubtile, OpC, UseC, dot.getInputPrecision(),
            dot.getMaxNumImpreciseAcc(), dot.getIsAsync()
            );
        prevDot = newDotOp;
      }
      newOp = (Operation *) prevDot;
    }
    auto dotWait = dyn_cast<nvidia_gpu::WarpGroupDotWaitOp>(newOp);
    if (dotWait) {
      builder.setInsertionPoint(dotWait);
    }

    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));

  builder.setInsertionPointToEnd(newForOp.getBody());
  if (!yieldValues.empty())
    builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);

  return newForOp;
}

} // namespace

struct WGMMAPrefetchPass
    : public impl::TritonGPUWGMMAPrefetchBase<WGMMAPrefetchPass> {
  void runOnOperation() override {
    // The detailed explanation of this pass can be found in
    // https://github.com/triton-lang/triton/pull/6196
    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }

    getOperation()->walk([&](scf::ForOp forOp) {
      WGMMAPrefetcher prefetcher(forOp);

      if (prefetcher.initialize().failed())
        return;

      scf::ForOp newForOp = prefetcher.createNewForOp();

      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
