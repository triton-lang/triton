#include "TritonAMDGPUToLLVM/Passes.h"

#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTBUILTINFUNCTOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

class CallOpConversion : public mlir::RewritePattern {
public:
  CallOpConversion(mlir::MLIRContext *context)
      : mlir::RewritePattern(LLVM::CallOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto callOp = cast<LLVM::CallOp>(op);
    if (isPredicatedLoad(callOp)) {
      return convertPredicatedLoad(callOp, rewriter);
    } else if (isPredicatedStore(callOp)) {
      return convertPredicatedStore(callOp, rewriter);
    } else if (isWrappedLLVMIntrinsic(callOp)) {
      return convertToLLVMIntrinsic(callOp, rewriter);
    } else {
      return failure();
    }
  }

private:
  bool isPredicatedLoad(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(mlir::LLVM::AMD::predicatedLoad);
  }

  bool isPredicatedLoadCA(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(
        mlir::LLVM::AMD::predicatedLoadCA);
  }

  bool isPredicatedLoadCG(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(
        mlir::LLVM::AMD::predicatedLoadCG);
  }

  bool isPredicatedLoadCV(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(
        mlir::LLVM::AMD::predicatedLoadCV);
  }

  bool isPredicatedStore(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(
        mlir::LLVM::AMD::predicatedStore);
  }

  bool isPredicatedStoreCS(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(
        mlir::LLVM::AMD::predicatedStoreCS);
  }

  bool isPredicatedStoreCG(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(
        mlir::LLVM::AMD::predicatedStoreCG);
  }

  bool isPredicatedStoreWT(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().contains(
        mlir::LLVM::AMD::predicatedStoreWT);
  }

  bool isWrappedLLVMIntrinsic(LLVM::CallOp callOp) const {
    if (std::optional<StringRef> callee = callOp.getCallee()) {
      if (callee.value().starts_with("__triton_hip_")) {
        return true;
      }
    }
    return false;
  }

  LogicalResult convertPredicatedStore(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    auto operands = callOp.getOperands();

    auto loc = callOp.getLoc();
    auto ptr = operands[0];
    auto val = operands[1];
    auto pred = operands[2];

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterStore =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterStore);
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, afterStore);
    rewriter.setInsertionPointToStart(trueBlock);
    /*
                  | vialatile | non-tmp | gcn instr gfx94
    LLVM::StoreOp | 0         | 0       | (cg) global store
                  | 0         | 1       | (cs) global store nt
                  | 1         | 0/1     | (wt) global store sc0 sc1
    */
    bool vialatileFlag = isPredicatedStoreWT(callOp);
    bool nonTmpFlag = isPredicatedStoreCS(callOp);
    auto storeOp = rewriter.create<LLVM::StoreOp>(
        loc, val, ptr, /*alignment=*/0, vialatileFlag, nonTmpFlag);
    rewriter.create<LLVM::BrOp>(loc, afterStore);
    rewriter.setInsertionPointToStart(afterStore);
    rewriter.eraseOp(callOp);
    return mlir::success();
  }

  LogicalResult convertPredicatedLoad(LLVM::CallOp callOp,
                                      mlir::PatternRewriter &rewriter) const {
    auto operands = callOp.getOperands();
    auto result = callOp.getResult();

    auto loc = callOp.getLoc();
    auto elemTy = result.getType();
    auto ptr = operands[0];
    auto pred = operands[1];
    auto falseVal = operands[2];

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument({elemTy}, {loc});
    Block *trueBlock = rewriter.createBlock(afterLoad);
    Block *falseBlock =
        rewriter.splitBlock(trueBlock, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, trueBlock, falseBlock);
    rewriter.setInsertionPointToStart(trueBlock);
    /*
                 | vialatile | non-tmp | gcn instr gfx94
    LLVM::LoadOp | 0         | 0       | (ca) global load
                 | 0/1       | 1       | (cg) global load nt
                 | 1         | 0       | (cv) flat load sc0 sc1
    */
    bool vialatileFlag = isPredicatedLoadCV(callOp);
    bool nonTmpFlag = isPredicatedLoadCG(callOp);
    auto loadOp = rewriter.create<LLVM::LoadOp>(
        loc, elemTy, ptr, /*alignment=*/0, vialatileFlag, nonTmpFlag);
    rewriter.create<LLVM::BrOp>(loc, loadOp->getResult(0), afterLoad);
    rewriter.setInsertionPointToStart(falseBlock);
    rewriter.create<LLVM::BrOp>(loc, falseVal, afterLoad);
    rewriter.setInsertionPointToStart(afterLoad);
    Value loadVal = afterLoad->getArgument(0);
    rewriter.replaceOp(callOp, loadVal);
    return mlir::success();
  }

  LogicalResult convertToLLVMIntrinsic(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    StringRef calleeName = callOp.getCallee().value();

    auto operands = callOp.getOperands();
    auto result = callOp.getResult();

    LLVM::LLVMFunctionType calleeType = callOp.getCalleeFunctionType();
    Type returnType = calleeType.getReturnType();

    auto loc = callOp.getLoc();

    Operation *replacementOp = nullptr;
    if (calleeName == "__triton_hip_iabs") {
      assert(operands.size() == 1);
      replacementOp = rewriter.create<LLVM::AbsOp>(loc, returnType, operands[0],
                                                   /*is_int_min_poison=*/false);
    } else if (calleeName == "__triton_hip_fabs") {
      assert(operands.size() == 1);
      replacementOp =
          rewriter.create<LLVM::FAbsOp>(loc, returnType, operands[0]);
    } else if (calleeName == "__triton_hip_llrint") {
      assert(operands.size() == 1);
      // Note, LrintOp and LlrintOp result in a code-gen error
      Operation *op = rewriter.create<LLVM::RintOp>(loc, operands[0].getType(),
                                                    operands[0]);
      replacementOp =
          rewriter.create<LLVM::FPToSIOp>(loc, returnType, op->getResult(0));
    } else if (calleeName == "__triton_hip_fast_fdividef") {
      assert(operands.size() == 2);
      auto name = StringAttr::get(callOp.getContext(), "llvm.amdgcn.rcp.f32");
      LLVM::FastmathFlagsAttr defaultFlags{};
      auto rcpOp = rewriter.create<LLVM::CallIntrinsicOp>(
          loc, returnType, name, operands[1], defaultFlags);

      replacementOp = rewriter.create<LLVM::FMulOp>(
          loc, returnType, operands[0], rcpOp->getResult(0), defaultFlags);
    }

    if (replacementOp) {
      rewriter.replaceOp(callOp, replacementOp);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct ConvertBuiltinFuncToLLVM
    : public triton::impl::ConvertBuiltinFuncToLLVMBase<
          ConvertBuiltinFuncToLLVM> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    GreedyRewriteConfig config;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Aggressive;

    RewritePatternSet patterns(context);
    patterns.add<CallOpConversion>(context);

    if (mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns), config)
            .failed()) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertBuiltinFuncToLLVMPass() {
  return std::make_unique<ConvertBuiltinFuncToLLVM>();
}

} // namespace triton
} // namespace mlir
