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
    if (isPredicatedLoadNT(callOp)) {
      return convertPredicatedLoad(callOp, rewriter, /*nt=*/true);
    } else if (isPredicatedLoad(callOp)) {
      return convertPredicatedLoad(callOp, rewriter, /*nt=*/false);
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
    return callOp.getCallee().value().find(mlir::LLVM::AMD::Predicated_Load) !=
           llvm::StringRef::npos;
  }

  bool isPredicatedLoadNT(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().find(
               mlir::LLVM::AMD::Predicated_Load_NT) != llvm::StringRef::npos;
  }

  bool isPredicatedStore(LLVM::CallOp callOp) const {
    return callOp.getCallee().value().find(mlir::LLVM::AMD::Predicated_Store) !=
           llvm::StringRef::npos;
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
    auto storeOp = rewriter.create<LLVM::StoreOp>(loc, val, ptr);
    rewriter.create<LLVM::BrOp>(loc, afterStore);
    rewriter.setInsertionPointToStart(afterStore);
    rewriter.eraseOp(callOp);
    return mlir::success();
  }

  LogicalResult convertPredicatedLoad(LLVM::CallOp callOp,
                                      mlir::PatternRewriter &rewriter,
                                      bool nt) const {
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
    auto loadOp = nt ? rewriter.create<LLVM::LoadOp>(
                           loc, elemTy, ptr, /*alignment=*/0,
                           /*isVolatile=*/false, /*isNonTemporal=*/true)
                     : rewriter.create<LLVM::LoadOp>(loc, elemTy, ptr);
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

    LLVM::LLVMFunctionType calleeType = callOp.getCalleeType().value();
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

    // Disable block merging because of:
    // https://github.com/llvm/llvm-project/issues/63230
    // TODO(giuseros): enable block merging once the above ticket is completed
    GreedyRewriteConfig config;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Normal;

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
