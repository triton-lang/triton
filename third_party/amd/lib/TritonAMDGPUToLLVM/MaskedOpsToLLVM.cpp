#include "TritonAMDGPUToLLVM/Passes.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// #include "triton/Dialect/Triton/IR/Dialect.h"
// #include "triton/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "AsyncUtility.h" // is this okay?
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <optional>

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTMASKEDOPSTOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
}

using namespace mlir;
using namespace mlir::triton::gpu;

namespace {

static std::pair<bool, bool>
getCacheModifierFlagsForLoad(triton::amdgpu::MaskedLoadOp loadOp) {
  auto cm = loadOp.getCache();
  bool isVolatile = false;
  bool isNonTemporal = false;

  switch (cm) {
  case triton::CacheModifier::CA:
    // ca: volatile=false, nontemporal=false
    break;
  case triton::CacheModifier::CG:
    // cg: volatile=false, nontemporal=true
    isNonTemporal = true;
    break;
  case triton::CacheModifier::CV:
    // cv: volatile=true, nontemporal=X
    isVolatile = true;
    break;
  default:
    // Default: no special flags
    break;
  }

  return std::make_pair(isVolatile, isNonTemporal);
}

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

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument({elemTy}, {loc});

    Block *trueBlock = rewriter.createBlock(afterLoad);

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, mask, trueBlock, ValueRange{},
                                    afterLoad, ValueRange{falseVal});
    rewriter.setInsertionPointToStart(trueBlock);
    auto [volatileFlag, nonTmpFlag] = getCacheModifierFlagsForLoad(loadOp);

    auto llvmLoadOp = rewriter.create<LLVM::LoadOp>(loc, elemTy, ptr, /*alignment*/0,
                                                    volatileFlag, nonTmpFlag);

    if (loadOp.getForceNoAlias()) {
      AMD::addLocalLoadNoAliasScope(llvmLoadOp);
    }

    rewriter.create<LLVM::BrOp>(loc, ValueRange{llvmLoadOp->getResult(0)},
                                afterLoad);

    rewriter.replaceOp(loadOp, afterLoad->getArgument(0));

    return success();
  }
};

struct ConvertMaskedOpsToLLVM
    : public triton::impl::ConvertMaskedOpsToLLVMBase<ConvertMaskedOpsToLLVM> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive);

    RewritePatternSet patterns(context);
    patterns.add<ConvertMaskedLoadOp>(context);

    if (applyPatternsGreedily(mod, std::move(patterns), config)
            .failed()) { // is it okay to fail?
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertMaskedOpsToLLVMPass() {
  return std::make_unique<ConvertMaskedOpsToLLVM>();
}

} // namespace mlir::triton