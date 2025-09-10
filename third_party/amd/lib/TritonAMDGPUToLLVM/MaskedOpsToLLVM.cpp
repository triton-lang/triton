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
#include "triton/Tools/Sys/GetEnv.hpp"
#include <tuple>

using namespace mlir;
using namespace mlir::triton::gpu;

namespace {

class ConvertMaskedLoadOp
    : public OpRewritePattern<triton::amdgpu::MaskedLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::amdgpu::MaskedLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto elemTy = loadOp.getResult().getType();
    auto ptr = loadOp.getPtr();
    auto mask = loadOp.getMask();
    auto falseVal = loadOp.getFalseVal();

    bool volatileFlag, nonTmpFlag;
    std::tie(volatileFlag, nonTmpFlag) =
        mlir::LLVM::AMD::getCacheModifierFlagsForLoadStore(
            loadOp.getCache(), mlir::LLVM::AMD::MemoryOp::Load);

    auto createLoadWithAttrs = [&](Location loadLoc) -> LLVM::LoadOp {
      auto load = rewriter.create<LLVM::LoadOp>(
          loadLoc, elemTy, ptr, /*alignment*/ 0, volatileFlag, nonTmpFlag);
      if (loadOp.getForceNoAlias()) {
        AMD::addLocalLoadNoAliasScope(load);
      }
      return load;
    };

    bool useDirectLoad = mlir::matchPattern(mask, mlir::m_One());

    if (useDirectLoad) {
      auto llvmLoadOp = createLoadWithAttrs(loc);
      rewriter.replaceOp(loadOp, llvmLoadOp.getResult());
      return success();
    }

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument({elemTy}, {loc});

    Block *trueBlock = rewriter.createBlock(afterLoad);

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, mask, trueBlock, ValueRange{},
                                    afterLoad, ValueRange{falseVal});
    rewriter.setInsertionPointToStart(trueBlock);
    //              | vialatile | non-tmp | gcn instr gfx94
    // LLVM::LoadOp | 0         | 0       | (ca) global load
    //              | 0/1       | 1       | (cg) global load nt
    //              | 1         | 0       | (cv) flat load sc0 sc1
    auto llvmLoadOp = createLoadWithAttrs(loc);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{llvmLoadOp->getResult(0)},
                                afterLoad);

    rewriter.replaceOp(loadOp, afterLoad->getArgument(0));

    return success();
  }
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
      auto store = rewriter.create<LLVM::StoreOp>(storeLoc, val, ptr, alignment,
                                                  volatileFlag, nonTmpFlag);
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
    rewriter.create<LLVM::CondBrOp>(loc, mask, trueBlock, afterStore);
    rewriter.setInsertionPointToStart(trueBlock);
    //               | vialatile | non-tmp | gcn instr gfx94
    // LLVM::StoreOp | 0         | 0       | (cg) global store
    //               | 0         | 1       | (cs) global store nt
    //               | 1         | 0/1     | (wt) global store sc0 sc1
    auto llvmStoreOp = createStoreWithAttrs(loc);
    rewriter.create<LLVM::BrOp>(loc, afterStore);
    rewriter.setInsertionPointToStart(afterStore);
    rewriter.eraseOp(storeOp);
    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {

void populateMaskedOpsToLLVMPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertMaskedLoadOp>(patterns.getContext());
  patterns.add<ConvertMaskedStoreOp>(patterns.getContext());
}
} // namespace mlir::triton::AMD

// namespace mlir::triton
