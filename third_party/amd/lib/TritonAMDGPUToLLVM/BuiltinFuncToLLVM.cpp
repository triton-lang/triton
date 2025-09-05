#include "TritonAMDGPUToLLVM/Passes.h"

#include "AsyncUtility.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTBUILTINFUNCTOLLVM
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

class CallOpConversion : public OpRewritePattern<LLVM::CallOp> {
public:
  CallOpConversion(mlir::MLIRContext *context, bool ftz)
      : OpRewritePattern(context, 1), ftz(ftz) {}

  LogicalResult
  matchAndRewrite(LLVM::CallOp callOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (isWrappedLLVMIntrinsic(callOp)) {
      return convertToLLVMIntrinsic(callOp, rewriter);
    } else {
      return failure();
    }
  }

private:
  bool isWrappedLLVMIntrinsic(LLVM::CallOp callOp) const {
    if (std::optional<StringRef> callee = callOp.getCallee()) {
      if (callee.value().starts_with("__triton_hip_")) {
        return true;
      }
    }
    return false;
  }

  // Utility function to create fast exponential operation
  Operation *createFastExpf(mlir::PatternRewriter &rewriter, Location loc,
                            Value input, Type returnType, bool ftz) const {
    assert(input.getType().getIntOrFloatBitWidth() == 32);
    const double log2e = 1.4426950408889634;
    LLVM::FastmathFlagsAttr defaultFlags{};

    auto mulOp = rewriter.create<LLVM::FMulOp>(
        loc, rewriter.getF32Type(), input,
        LLVM::createConstantF32(loc, rewriter, log2e), defaultFlags);

    const char *intrinsic = ftz ? "llvm.amdgcn.exp2.f32" : "llvm.exp2.f32";
    return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, returnType,
                                           mulOp->getResult(0));
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
      const char *intrinsic = "llvm.amdgcn.rcp.f32";
      auto rcpOp = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic,
                                                   returnType, operands[1]);

      LLVM::FastmathFlagsAttr defaultFlags{};
      replacementOp = rewriter.create<LLVM::FMulOp>(
          loc, returnType, operands[0], rcpOp->getResult(0), defaultFlags);
    } else if (calleeName == "__triton_hip_fast_expf") {
      assert(operands.size() == 1);
      assert(operands[0].getType().getIntOrFloatBitWidth() == 32);
      replacementOp =
          createFastExpf(rewriter, loc, operands[0], returnType, ftz);
    } else if (calleeName == "__triton_hip_fast_tanhf") {
      assert(operands.size() == 1);
      assert(operands[0].getType().getIntOrFloatBitWidth() == 32);
      LLVM::FastmathFlagsAttr defaultFlags{};

      // Calculate 2*x
      auto twoX = rewriter.create<LLVM::FMulOp>(
          loc, rewriter.getF32Type(), operands[0],
          LLVM::createConstantF32(loc, rewriter, 2.0), defaultFlags);

      // Calculate fast_expf(2*x) using the utility function
      auto exp2X = createFastExpf(rewriter, loc, twoX->getResult(0),
                                  rewriter.getF32Type(), ftz);

      // Calculate exp2X - 1
      auto exp2XMinus1 = rewriter.create<LLVM::FSubOp>(
          loc, rewriter.getF32Type(), exp2X->getResult(0),
          LLVM::createConstantF32(loc, rewriter, 1.0), defaultFlags);

      // Calculate exp2X + 1
      auto exp2XPlus1 = rewriter.create<LLVM::FAddOp>(
          loc, rewriter.getF32Type(), exp2X->getResult(0),
          LLVM::createConstantF32(loc, rewriter, 1.0), defaultFlags);

      // Calculate tanh(X) = (exp2X - 1) / (exp2X + 1)
      replacementOp = rewriter.create<LLVM::FDivOp>(
          loc, returnType, exp2XMinus1->getResult(0), exp2XPlus1->getResult(0),
          defaultFlags);
    }

    if (replacementOp) {
      rewriter.replaceOp(callOp, replacementOp);
      return mlir::success();
    }

    return mlir::failure();
  }

private:
  bool ftz;
};

struct ConvertBuiltinFuncToLLVM
    : public triton::impl::ConvertBuiltinFuncToLLVMBase<
          ConvertBuiltinFuncToLLVM> {
  explicit ConvertBuiltinFuncToLLVM(bool ftz) { this->ftz = ftz; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive);

    RewritePatternSet patterns(context);
    patterns.add<CallOpConversion>(context, this->ftz);
    if (mlir::applyPatternsGreedily(mod, std::move(patterns), config)
            .failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertBuiltinFuncToLLVMPass(bool ftz) {
  return std::make_unique<ConvertBuiltinFuncToLLVM>(ftz);
}

} // namespace mlir::triton
