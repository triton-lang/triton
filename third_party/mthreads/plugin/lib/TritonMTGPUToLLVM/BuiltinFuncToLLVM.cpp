#include "TritonMTGPUToLLVM/Passes.h"

#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTMTGPUBUILTINFUNCTOLLVM
#include "TritonMTGPUToLLVM/Passes.h.inc"
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
    } else {
      return failure();
    }
  }

private:
  bool isPredicatedLoad(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().find(mlir::LLVM::MUSA::Predicated_Load) !=
           llvm::StringRef::npos;
  }

  bool isPredicatedStore(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().find(
               mlir::LLVM::MUSA::Predicated_Store) != llvm::StringRef::npos;
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
    auto storeOp = rewriter.create<LLVM::StoreOp>(loc, val, ptr);
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
    auto loadOp = rewriter.create<LLVM::LoadOp>(loc, elemTy, ptr);
    rewriter.create<LLVM::BrOp>(loc, loadOp->getResult(0), afterLoad);
    rewriter.setInsertionPointToStart(falseBlock);
    rewriter.create<LLVM::BrOp>(loc, falseVal, afterLoad);
    rewriter.setInsertionPointToStart(afterLoad);
    Value loadVal = afterLoad->getArgument(0);
    rewriter.replaceOp(callOp, loadVal);
    return mlir::success();
  }
};

struct ConvertBuiltinFuncToLLVM
    : public triton::impl::ConvertMTGPUBuiltinFuncToLLVMBase<
          ConvertBuiltinFuncToLLVM> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // Disable block merging because of:
    // https://github.com/llvm/llvm-project/issues/63230
    // TODO(giuseros): enable block merging once the above ticket is completed
    GreedyRewriteConfig config;
    config.enableRegionSimplification = false;

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

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMTGPUBuiltinFuncToLLVMPass() {
  return std::make_unique<ConvertBuiltinFuncToLLVM>();
}

} // namespace triton
} // namespace mlir
