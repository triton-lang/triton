#include "MaskedOpsToLLVM.h"

#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.h"
#include <tuple>

using namespace mlir;
using namespace mlir::triton::gpu;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUMASKEDOPSTOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace mlir::triton::AMD {

Value createRegularLoadFromMaskedOp(RewriterBase &rewriter, Location loc,
                                    amdgpu::MaskedLoadOp loadOp) {
  Type elemTy = loadOp.getResult().getType();
  Value ptr = loadOp.getPtr();
  triton::CacheModifier cacheMod = loadOp.getCache();

  bool volatileFlag, nonTmpFlag;
  std::tie(volatileFlag, nonTmpFlag) =
      mlir::LLVM::AMD::getCacheModifierFlagsForLoadStore(
          cacheMod, mlir::LLVM::AMD::MemoryOp::Load);

  //              | volatile | non-tmp | gcn instr gfx94
  // LLVM::LoadOp | 0        | 0       | (ca) global load
  //              | 0/1      | 1       | (cg) global load nt
  //              | 1        | 0       | (cv) flat load sc0 sc1
  auto load = LLVM::LoadOp::create(rewriter, loc, elemTy, ptr, /*alignment=*/0,
                                   volatileFlag, nonTmpFlag);
  if (loadOp.getForceNoAlias())
    AMD::addLocalLoadNoAliasScope(load);
  return load;
}

LLVM::StoreOp createUnmaskedStoreFromMaskedOp(RewriterBase &rewriter,
                                              Location loc,
                                              amdgpu::MaskedStoreOp storeOp) {
  Value val = storeOp.getValue();
  Type elemTy = val.getType();
  Value ptr = storeOp.getPtr();

  bool volatileFlag, nonTmpFlag;
  std::tie(volatileFlag, nonTmpFlag) =
      mlir::LLVM::AMD::getCacheModifierFlagsForLoadStore(
          storeOp.getCache(), mlir::LLVM::AMD::MemoryOp::Store);

  int alignment = 0;
  if (auto vecTy = dyn_cast<VectorType>(elemTy)) {
    Type vecElemTy = vecTy.getElementType();
    int elemSizeInBytes = vecElemTy.getIntOrFloatBitWidth() / 8;
    alignment = elemSizeInBytes * vecTy.getNumElements();
  }

  //               | volatile | non-tmp | gcn instr gfx94
  // LLVM::StoreOp | 0        | 0       | (cg) global store
  //               | 0        | 1       | (cs) global store nt
  //               | 1        | 0/1     | (wt) global store sc0 sc1
  auto store = LLVM::StoreOp::create(rewriter, loc, val, ptr, alignment,
                                     volatileFlag, nonTmpFlag);
  if (storeOp.getForceNoAlias())
    AMD::addLocalLoadNoAliasScope(store);
  return store;
}

} // namespace mlir::triton::AMD

namespace {

class ConvertMaskedRegionOp final
    : public OpRewritePattern<triton::amdgpu::MaskedRegionOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::amdgpu::MaskedRegionOp regionOp,
                                PatternRewriter &rewriter) const override {
    Location loc = regionOp.getLoc();
    Region &region = regionOp.getBody();
    if (!region.hasOneBlock())
      return regionOp.emitOpError("expected one body block");

    Block *body = &region.front();
    auto yield = dyn_cast<triton::amdgpu::MaskedYieldOp>(body->getTerminator());
    if (!yield)
      return regionOp.emitOpError("expected `amdg.masked_yield` terminator");

    rewriter.setInsertionPoint(regionOp);
    if (mlir::matchPattern(regionOp.getMask(), mlir::m_One())) {
      ValueRange results = yield.getValues();
      rewriter.inlineBlockBefore(body, regionOp, {});
      rewriter.replaceOp(regionOp, results);
      rewriter.eraseOp(yield);
      return success();
    }

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterRegion =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    auto joinArgs = afterRegion->addArguments(
        regionOp.getResultTypes(),
        SmallVector<Location>(regionOp.getNumResults(), loc));
    SmallVector<Value> replacementValues(joinArgs.begin(), joinArgs.end());

    Block *trueBlock = rewriter.createBlock(afterRegion);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, regionOp.getMask(), trueBlock,
                           ValueRange{}, afterRegion,
                           regionOp.getFalseValues());

    rewriter.inlineBlockBefore(body, trueBlock, trueBlock->end(), {});

    rewriter.setInsertionPoint(yield);
    LLVM::BrOp::create(rewriter, loc, yield.getValues(), afterRegion);
    rewriter.eraseOp(yield);

    rewriter.replaceOp(regionOp, replacementValues);
    return success();
  }
};

class ConvertMaskedLoadOp final
    : public OpRewritePattern<triton::amdgpu::MaskedLoadOp> {
public:
  ConvertMaskedLoadOp(MLIRContext *context, const AMD::TargetInfo &targetInfo)
      : OpRewritePattern(context), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(triton::amdgpu::MaskedLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = loadOp.getLoc();
    Type elemTy = loadOp.getResult().getType();
    Value mask = loadOp.getMask();
    Value falseVal = loadOp.getFalseVal();
    Value multicastMask = loadOp.getMulticastMask();
    Value ptr = loadOp.getPtr();
    triton::CacheModifier cacheMod = loadOp.getCache();

    auto createLoad = [&](Location loadLoc) -> Value {
      if (!multicastMask)
        return AMD::createRegularLoadFromMaskedOp(rewriter, loadLoc, loadOp);

      int vecBits = 0;
      if (auto vecTy = dyn_cast<VectorType>(elemTy)) {
        vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
      } else {
        vecBits = elemTy.getIntOrFloatBitWidth();
      }
      assert(vecBits != 0);

      // We can only multicast for 32, 64, 128 bit load size (hw limitation).
      if (targetInfo.supportsClusterLoadBitWidth(vecBits)) {
        std::string intrinsic =
            "llvm.amdgcn.cluster.load.b" + std::to_string(vecBits);
        auto cacheModBits = LLVM::AMD::getCtrlBitsForCacheModifierOnTarget(
            cacheMod, true, targetInfo);
        // The intrinsics only work with int32 or vec of int32 for >32bit.
        Type resTy = i32_ty;
        if (vecBits > 32)
          resTy = vec_ty(i32_ty, vecBits / 32);
        TritonLLVMOpBuilder b(loadLoc, rewriter);
        auto clusterLoadOp = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loadLoc, intrinsic, {resTy},
            {ptr, b.i32_val(cacheModBits), multicastMask});
        return b.bitcast(clusterLoadOp->getResult(0), elemTy);
      }

      loadOp.emitRemark() << "Multicast with bit width " << vecBits
                          << " is not supported on " << targetInfo.getArch()
                          << " falling back to regular load";
      return AMD::createRegularLoadFromMaskedOp(rewriter, loadLoc, loadOp);
    };

    rewriter.setInsertionPoint(loadOp);
    if (mlir::matchPattern(mask, mlir::m_One())) {
      Value loadResult = createLoad(loc);
      rewriter.replaceOp(loadOp, loadResult);
      return success();
    }

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument(elemTy, loc);

    Block *trueBlock = rewriter.createBlock(afterLoad);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, mask, trueBlock, ValueRange{},
                           afterLoad, ValueRange{falseVal});

    rewriter.setInsertionPointToStart(trueBlock);
    Value loadResult = createLoad(loc);
    LLVM::BrOp::create(rewriter, loc, ValueRange{loadResult}, afterLoad);

    rewriter.replaceOp(loadOp, afterLoad->getArgument(0));
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

class ConvertMaskedStoreOp final
    : public OpRewritePattern<triton::amdgpu::MaskedStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::amdgpu::MaskedStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Location loc = storeOp.getLoc();
    Value mask = storeOp.getMask();

    rewriter.setInsertionPoint(storeOp);
    if (mlir::matchPattern(mask, mlir::m_One())) {
      AMD::createUnmaskedStoreFromMaskedOp(rewriter, loc, storeOp);
      rewriter.eraseOp(storeOp);
      return success();
    }

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterStore =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterStore);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, mask, trueBlock, afterStore);

    rewriter.setInsertionPointToStart(trueBlock);
    AMD::createUnmaskedStoreFromMaskedOp(rewriter, loc, storeOp);
    LLVM::BrOp::create(rewriter, loc, afterStore);

    rewriter.eraseOp(storeOp);
    return success();
  }
};

struct TritonAMDGPUMaskedOpsToLLVMPass final
    : public triton::impl::TritonAMDGPUMaskedOpsToLLVMBase<
          TritonAMDGPUMaskedOpsToLLVMPass> {
  explicit TritonAMDGPUMaskedOpsToLLVMPass(StringRef gfxArch) {
    this->gfxArch = gfxArch.str();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    AMD::TargetInfo targetInfo(this->gfxArch.getValue());
    if (failed(AMD::lowerMaskedOpsToLLVM(module, targetInfo)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::AMD {

void populateMaskedOpsToLLVMPatterns(RewritePatternSet &patterns,
                                     const TargetInfo &targetInfo) {
  patterns.add<ConvertMaskedRegionOp>(patterns.getContext());
  patterns.add<ConvertMaskedLoadOp>(patterns.getContext(), targetInfo);
  patterns.add<ConvertMaskedStoreOp>(patterns.getContext());
}

LogicalResult lowerMaskedOpsToLLVM(ModuleOp module,
                                   const TargetInfo &targetInfo) {
  RewritePatternSet patterns(module.getContext());
  populateMaskedOpsToLLVMPatterns(patterns, targetInfo);
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return failure();

  WalkResult remainingMaskedOps = module.walk([](Operation *op) {
    if (isa<triton::amdgpu::MaskedRegionOp, triton::amdgpu::MaskedLoadOp,
            triton::amdgpu::MaskedStoreOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (remainingMaskedOps.wasInterrupted())
    return failure();
  return success();
}

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUMaskedOpsToLLVMPass(StringRef gfxArch) {
  return std::make_unique<TritonAMDGPUMaskedOpsToLLVMPass>(gfxArch);
}

} // namespace mlir::triton::AMD
