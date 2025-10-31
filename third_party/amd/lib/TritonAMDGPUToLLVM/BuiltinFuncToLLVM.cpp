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

    auto mulOp = LLVM::FMulOp::create(
        rewriter, loc, rewriter.getF32Type(), input,
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
      replacementOp =
          LLVM::AbsOp::create(rewriter, loc, returnType, operands[0],
                              /*is_int_min_poison=*/false);
    } else if (calleeName == "__triton_hip_fabs") {
      assert(operands.size() == 1);
      replacementOp =
          LLVM::FAbsOp::create(rewriter, loc, returnType, operands[0]);
    } else if (calleeName == "__triton_hip_llrint") {
      assert(operands.size() == 1);
      // Note, LrintOp and LlrintOp result in a code-gen error
      Operation *op = LLVM::RintOp::create(rewriter, loc, operands[0].getType(),
                                           operands[0]);
      replacementOp =
          LLVM::FPToSIOp::create(rewriter, loc, returnType, op->getResult(0));
    } else if (calleeName == "__triton_hip_fast_fdividef") {
      assert(operands.size() == 2);
      const char *intrinsic = "llvm.amdgcn.rcp.f32";
      auto rcpOp = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic,
                                                   returnType, operands[1]);

      LLVM::FastmathFlagsAttr defaultFlags{};
      replacementOp =
          LLVM::FMulOp::create(rewriter, loc, returnType, operands[0],
                               rcpOp->getResult(0), defaultFlags);
    } else if (calleeName == "__triton_hip_fast_expf") {
      assert(operands.size() == 1);
      assert(operands[0].getType().getIntOrFloatBitWidth() == 32);
      replacementOp =
          createFastExpf(rewriter, loc, operands[0], returnType, ftz);
    } else if (calleeName == "__triton_hip_fast_tanhf") {
      assert(operands.size() == 1);
      assert(operands[0].getType().getIntOrFloatBitWidth() == 32);
      LLVM::FastmathFlagsAttr defaultFlags{};

      // Numerically stable tanh implementation:
      // For positive x: tanh(x) = 1 - 2/(e^(2x) + 1)
      // For negative x: tanh(x) = -tanh(-x) = -(1 - 2/(e^(-2x) + 1))
      //                         = 2/(e^(-2x) + 1) - 1
      // This avoids overflow when e^(2x) becomes infinity for large x

      // Get absolute value of x
      auto absX = LLVM::FAbsOp::create(rewriter, loc, rewriter.getF32Type(),
                                       operands[0]);

      // Calculate 2*|x|
      auto twoAbsX = LLVM::FMulOp::create(
          rewriter, loc, rewriter.getF32Type(), absX,
          LLVM::createConstantF32(loc, rewriter, 2.0), defaultFlags);

      // Calculate e^(2*|x|)
      auto exp2AbsX = createFastExpf(rewriter, loc, twoAbsX->getResult(0),
                                     rewriter.getF32Type(), ftz);

      // Calculate e^(2*|x|) + 1
      auto exp2AbsXPlus1 = LLVM::FAddOp::create(
          rewriter, loc, rewriter.getF32Type(), exp2AbsX->getResult(0),
          LLVM::createConstantF32(loc, rewriter, 1.0), defaultFlags);

      // Calculate 2 / (e^(2*|x|) + 1)
      auto two = LLVM::createConstantF32(loc, rewriter, 2.0);
      auto ratio =
          LLVM::FDivOp::create(rewriter, loc, rewriter.getF32Type(), two,
                               exp2AbsXPlus1->getResult(0), defaultFlags);

      // Calculate 1 - 2/(e^(2*|x|) + 1)
      auto one = LLVM::createConstantF32(loc, rewriter, 1.0);
      auto posResult =
          LLVM::FSubOp::create(rewriter, loc, rewriter.getF32Type(), one,
                               ratio->getResult(0), defaultFlags);

      // Apply the sign of the original input using copysign
      // tanh(x) = sign(x) * (1 - 2/(e^(2*|x|) + 1))
      const char *intrinsic = "llvm.copysign.f32";
      auto args =
          llvm::SmallVector<Value>{posResult->getResult(0), operands[0]};
      replacementOp = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic,
                                                      returnType, args);
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
