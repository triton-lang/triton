#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonAMDGPUToLLVM/Passes.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <tuple>

using namespace mlir;
using namespace mlir::triton::gpu;

namespace {

class ConvertMaskedLoadOp
    : public OpRewritePattern<triton::amdgpu::MaskedLoadOp> {
public:
  ConvertMaskedLoadOp(MLIRContext *context, const AMD::TargetInfo &targetInfo)
      : OpRewritePattern(context), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(triton::amdgpu::MaskedLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto elemTy = loadOp.getResult().getType();
    auto ptr = loadOp.getPtr();
    auto mask = loadOp.getMask();
    auto falseVal = loadOp.getFalseVal();
    auto multicastMask = loadOp.getMulticastMask();
    auto cacheMod = loadOp.getCache();

    bool volatileFlag, nonTmpFlag;
    std::tie(volatileFlag, nonTmpFlag) =
        mlir::LLVM::AMD::getCacheModifierFlagsForLoadStore(
            cacheMod, mlir::LLVM::AMD::MemoryOp::Load);

    auto createLoadWithAttrs = [&](Location loadLoc) -> Value {
      int vecBits = 0;
      if (auto vecTy = dyn_cast<VectorType>(elemTy)) {
        vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
      } else {
        vecBits = elemTy.getIntOrFloatBitWidth();
      }
      assert(vecBits != 0);
      // We can only multicast for 32, 64, 128 bit load size (hw limitation)
      if (multicastMask && targetInfo.supportsClusterLoadBitWidth(vecBits)) {
        std::string intrinsic =
            "llvm.amdgcn.cluster.load.b" + std::to_string(vecBits);
        auto cacheModBits = LLVM::AMD::getCtrlBitsForCacheModifierOnTarget(
            cacheMod, true, targetInfo);
        // The intrinsics only works with int32 or vec of int32 for >32bit
        Type resTy = i32_ty;
        if (vecBits > 32) {
          resTy = vec_ty(i32_ty, vecBits / 32);
        }
        auto clusterLoadOp = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, intrinsic, {resTy},
            {ptr, b.i32_val(cacheModBits), multicastMask});
        return b.bitcast(clusterLoadOp->getResult(0), elemTy);
      } else if (multicastMask) {
        loadOp.emitRemark()
            << "Multicast with bit width " << vecBits << " is not supported on "
            << targetInfo.getArch() << " falling back to regular load";
      }
      // Emit a regular load
      auto load =
          LLVM::LoadOp::create(rewriter, loadLoc, elemTy, ptr, /*alignment*/ 0,
                               volatileFlag, nonTmpFlag);
      if (loadOp.getForceNoAlias()) {
        AMD::addLocalLoadNoAliasScope(load);
      }
      return load;
    };

    bool useDirectLoad = mlir::matchPattern(mask, mlir::m_One());

    if (useDirectLoad) {
      auto loadResult = createLoadWithAttrs(loc);
      rewriter.replaceOp(loadOp, loadResult);
      return success();
    }

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument({elemTy}, {loc});

    Block *trueBlock = rewriter.createBlock(afterLoad);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, mask, trueBlock, ValueRange{},
                           afterLoad, ValueRange{falseVal});
    rewriter.setInsertionPointToStart(trueBlock);
    //              | vialatile | non-tmp | gcn instr gfx94
    // LLVM::LoadOp | 0         | 0       | (ca) global load
    //              | 0/1       | 1       | (cg) global load nt
    //              | 1         | 0       | (cv) flat load sc0 sc1
    auto loadResult = createLoadWithAttrs(loc);
    LLVM::BrOp::create(rewriter, loc, ValueRange{loadResult}, afterLoad);

    rewriter.replaceOp(loadOp, afterLoad->getArgument(0));

    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

class ConvertMaskedStoreOp
    : public OpRewritePattern<triton::amdgpu::MaskedStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::amdgpu::MaskedStoreOp storeOp,
                                PatternRewriter &rewriter) const override {

    auto loc = storeOp.getLoc();
    auto val = storeOp.getValue();
    auto elemTy = storeOp.getValue().getType();
    auto ptr = storeOp.getPtr();
    auto mask = storeOp.getMask();

    bool volatileFlag, nonTmpFlag;
    std::tie(volatileFlag, nonTmpFlag) =
        mlir::LLVM::AMD::getCacheModifierFlagsForLoadStore(
            storeOp.getCache(), mlir::LLVM::AMD::MemoryOp::Store);

    int alignment = 0;
    if (auto vecTy = dyn_cast<VectorType>(elemTy)) {
      auto vecElemTy = vecTy.getElementType();
      auto elemSizeInBytes = vecElemTy.getIntOrFloatBitWidth() / 8;
      alignment = elemSizeInBytes * vecTy.getNumElements();
    }

    auto createStoreWithAttrs = [&](Location storeLoc) -> LLVM::StoreOp {
      auto store = LLVM::StoreOp::create(rewriter, storeLoc, val, ptr,
                                         alignment, volatileFlag, nonTmpFlag);
      if (storeOp.getForceNoAlias()) {
        AMD::addLocalLoadNoAliasScope(store);
      }
      return store;
    };

    bool useDirectStore = mlir::matchPattern(mask, mlir::m_One());

    if (useDirectStore) {
      auto llvmStoreOp = createStoreWithAttrs(loc);
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
    //               | vialatile | non-tmp | gcn instr gfx94
    // LLVM::StoreOp | 0         | 0       | (cg) global store
    //               | 0         | 1       | (cs) global store nt
    //               | 1         | 0/1     | (wt) global store sc0 sc1
    auto llvmStoreOp = createStoreWithAttrs(loc);
    LLVM::BrOp::create(rewriter, loc, afterStore);
    rewriter.setInsertionPointToStart(afterStore);
    rewriter.eraseOp(storeOp);
    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {

void populateMaskedOpsToLLVMPatterns(RewritePatternSet &patterns,
                                     const TargetInfo &targetInfo) {
  patterns.add<ConvertMaskedLoadOp>(patterns.getContext(), targetInfo);
  patterns.add<ConvertMaskedStoreOp>(patterns.getContext());
}
} // namespace mlir::triton::AMD

// namespace mlir::triton
