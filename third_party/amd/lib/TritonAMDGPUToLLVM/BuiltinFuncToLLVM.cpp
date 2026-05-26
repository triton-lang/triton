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
  CallOpConversion(mlir::MLIRContext *context,
                   const AMD::TargetInfo &targetInfo, bool ftz)
      : OpRewritePattern(context, 1), targetInfo(targetInfo), ftz(ftz) {}

  LogicalResult
  matchAndRewrite(LLVM::CallOp callOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (isWrappedLLVMIntrinsic(callOp) || isOcmlCall(callOp)) {
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

  bool isOcmlCall(LLVM::CallOp callOp) const {
    if (std::optional<StringRef> callee = callOp.getCallee()) {
      if (callee.value().starts_with("__ocml_")) {
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

    Value arg = mulOp->getResult(0);
    if (ftz)
      return ROCDL::ROCDLExp2::create(rewriter, loc, returnType, arg);

    return LLVM::Exp2Op::create(rewriter, loc, returnType, arg);
  }

  LogicalResult convertToLLVMIntrinsic(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    StringRef calleeName = callOp.getCallee().value();

    auto operands = callOp.getOperands();

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
    } else if (calleeName == "__triton_hip_rint") {
      assert(operands.size() == 1);
      replacementOp =
          LLVM::RintOp::create(rewriter, loc, returnType, operands[0]);
    } else if (calleeName == "__triton_hip_fast_fdividef") {
      assert(operands.size() == 2);
      auto rcpOp =
          ROCDL::ROCDLRcp::create(rewriter, loc, returnType, operands[1]);

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

      // Apply the sign of the original input without using copysign intrinsic
      // tanh(x) = sign(x) * (1 - 2/(e^(2*|x|) + 1))
      // Use FCmp + Select + FMul instead of copysign to avoid potential LLVM
      // optimization side effects that may affect other operations
      auto zero = LLVM::createConstantF32(loc, rewriter, 0.0);
      auto negOne = LLVM::createConstantF32(loc, rewriter, -1.0);
      auto isNegative = LLVM::FCmpOp::create(
          rewriter, loc, LLVM::FCmpPredicate::olt, operands[0], zero);
      auto sign = LLVM::SelectOp::create(rewriter, loc, rewriter.getF32Type(),
                                         isNegative, negOne, one);
      replacementOp = LLVM::FMulOp::create(rewriter, loc, returnType,
                                           posResult->getResult(0),
                                           sign->getResult(0), defaultFlags);
    } else if (calleeName == "__triton_hip_clz_i32") {
      assert(operands.size() == 1);
      replacementOp =
          LLVM::CountLeadingZerosOp::create(rewriter, loc, returnType, operands[0]);
    } else if (calleeName == "__triton_hip_clz_i64") {
      assert(operands.size() == 1);
      // clz on i64 input, result is i32
      auto clzOp = LLVM::CountLeadingZerosOp::create(
          rewriter, loc, operands[0].getType(), operands[0]);
      replacementOp =
          LLVM::TruncOp::create(rewriter, loc, returnType, clzOp->getResult(0));
    } else if (calleeName == "__triton_hip_popc_i32") {
      assert(operands.size() == 1);
      replacementOp =
          LLVM::CtPopOp::create(rewriter, loc, returnType, operands[0]);
    } else if (calleeName == "__triton_hip_popc_i64") {
      assert(operands.size() == 1);
      // popc on i64 input, result is i32
      auto ctpopOp = LLVM::CtPopOp::create(
          rewriter, loc, operands[0].getType(), operands[0]);
      replacementOp =
          LLVM::TruncOp::create(rewriter, loc, returnType, ctpopOp->getResult(0));
    } else if (calleeName == "__triton_hip_ffs_i32") {
      assert(operands.size() == 1);
      // ffs(x) = x == 0 ? 0 : ctz(x) + 1
      // Use LLVM cttz with zero_is_poison=false, then add 1, select on zero
      auto zero = rewriter.create<LLVM::ConstantOp>(
          loc, returnType, rewriter.getIntegerAttr(returnType, 0));
      auto one = rewriter.create<LLVM::ConstantOp>(
          loc, returnType, rewriter.getIntegerAttr(returnType, 1));
      auto ctzOp = LLVM::CountTrailingZerosOp::create(
          rewriter, loc, returnType, operands[0]);
      auto addOne = LLVM::AddOp::create(
          rewriter, loc, returnType, ctzOp->getResult(0), one);
      auto isZero = LLVM::ICmpOp::create(
          rewriter, loc, LLVM::ICmpPredicate::eq, operands[0], zero);
      replacementOp = LLVM::SelectOp::create(
          rewriter, loc, returnType, isZero, zero, addOne->getResult(0));
    } else if (calleeName == "__triton_hip_brev_i32") {
      assert(operands.size() == 1);
      replacementOp =
          LLVM::BitReverseOp::create(rewriter, loc, returnType, operands[0]);
    } else if (calleeName == "__triton_hip_brev_i64") {
      assert(operands.size() == 1);
      replacementOp =
          LLVM::BitReverseOp::create(rewriter, loc, returnType, operands[0]);
    } else if (calleeName == "__triton_hip_mul24_i32") {
      assert(operands.size() == 2);
      // Truncate to 24-bit signed, multiply, return lower 32 bits
      auto mask = rewriter.create<LLVM::ConstantOp>(
          loc, returnType, rewriter.getIntegerAttr(returnType, 0x00FFFFFF));
      auto a = LLVM::AndOp::create(rewriter, loc, returnType, operands[0], mask);
      auto b = LLVM::AndOp::create(rewriter, loc, returnType, operands[1], mask);
      replacementOp =
          LLVM::MulOp::create(rewriter, loc, returnType, a->getResult(0), b->getResult(0));
    } else if (calleeName == "__triton_hip_mul24_u32") {
      assert(operands.size() == 2);
      auto mask = rewriter.create<LLVM::ConstantOp>(
          loc, returnType, rewriter.getIntegerAttr(returnType, 0x00FFFFFF));
      auto a = LLVM::AndOp::create(rewriter, loc, returnType, operands[0], mask);
      auto b = LLVM::AndOp::create(rewriter, loc, returnType, operands[1], mask);
      replacementOp =
          LLVM::MulOp::create(rewriter, loc, returnType, a->getResult(0), b->getResult(0));
    } else if (calleeName == "__triton_hip_mulhi_i32") {
      assert(operands.size() == 2);
      // Sign-extend to i64, multiply, take upper 32 bits
      auto i64Type = rewriter.getIntegerType(64);
      auto a = LLVM::SExtOp::create(rewriter, loc, i64Type, operands[0]);
      auto b = LLVM::SExtOp::create(rewriter, loc, i64Type, operands[1]);
      auto mul = LLVM::MulOp::create(rewriter, loc, i64Type,
                                     a->getResult(0), b->getResult(0));
      auto shift = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getIntegerAttr(i64Type, 32));
      auto shifted = LLVM::AShrOp::create(rewriter, loc, i64Type,
                                          mul->getResult(0), shift);
      replacementOp =
          LLVM::TruncOp::create(rewriter, loc, returnType, shifted->getResult(0));
    } else if (calleeName == "__triton_hip_mulhi_u32") {
      assert(operands.size() == 2);
      // Zero-extend to i64, multiply, take upper 32 bits
      auto i64Type = rewriter.getIntegerType(64);
      auto a = LLVM::ZExtOp::create(rewriter, loc, i64Type, operands[0]);
      auto b = LLVM::ZExtOp::create(rewriter, loc, i64Type, operands[1]);
      auto mul = LLVM::MulOp::create(rewriter, loc, i64Type,
                                     a->getResult(0), b->getResult(0));
      auto shift = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getIntegerAttr(i64Type, 32));
      auto shifted = LLVM::LShrOp::create(rewriter, loc, i64Type,
                                          mul->getResult(0), shift);
      replacementOp =
          LLVM::TruncOp::create(rewriter, loc, returnType, shifted->getResult(0));
    } else if (calleeName == "__triton_hip_hadd_i32") {
      assert(operands.size() == 2);
      // hadd(a,b) = (a + b) >> 1 signed, without overflow
      auto i64Type = rewriter.getIntegerType(64);
      auto a = LLVM::SExtOp::create(rewriter, loc, i64Type, operands[0]);
      auto b = LLVM::SExtOp::create(rewriter, loc, i64Type, operands[1]);
      auto sum = LLVM::AddOp::create(rewriter, loc, i64Type,
                                     a->getResult(0), b->getResult(0));
      auto one = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getIntegerAttr(i64Type, 1));
      auto shifted = LLVM::AShrOp::create(rewriter, loc, i64Type,
                                          sum->getResult(0), one);
      replacementOp =
          LLVM::TruncOp::create(rewriter, loc, returnType, shifted->getResult(0));
    } else if (calleeName == "__triton_hip_hadd_u32") {
      assert(operands.size() == 2);
      // hadd(a,b) = (a + b) >> 1 unsigned, without overflow
      auto i64Type = rewriter.getIntegerType(64);
      auto a = LLVM::ZExtOp::create(rewriter, loc, i64Type, operands[0]);
      auto b = LLVM::ZExtOp::create(rewriter, loc, i64Type, operands[1]);
      auto sum = LLVM::AddOp::create(rewriter, loc, i64Type,
                                     a->getResult(0), b->getResult(0));
      auto one = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getIntegerAttr(i64Type, 1));
      auto shifted = LLVM::LShrOp::create(rewriter, loc, i64Type,
                                          sum->getResult(0), one);
      replacementOp =
          LLVM::TruncOp::create(rewriter, loc, returnType, shifted->getResult(0));
    } else if (calleeName == "__triton_hip_rhadd_i32") {
      assert(operands.size() == 2);
      // rhadd(a,b) = (a + b + 1) >> 1 signed
      auto i64Type = rewriter.getIntegerType(64);
      auto a = LLVM::SExtOp::create(rewriter, loc, i64Type, operands[0]);
      auto b = LLVM::SExtOp::create(rewriter, loc, i64Type, operands[1]);
      auto sum = LLVM::AddOp::create(rewriter, loc, i64Type,
                                     a->getResult(0), b->getResult(0));
      auto one64 = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getIntegerAttr(i64Type, 1));
      auto sumPlus1 = LLVM::AddOp::create(rewriter, loc, i64Type,
                                          sum->getResult(0), one64);
      auto shifted = LLVM::AShrOp::create(rewriter, loc, i64Type,
                                          sumPlus1->getResult(0), one64);
      replacementOp =
          LLVM::TruncOp::create(rewriter, loc, returnType, shifted->getResult(0));
    } else if (calleeName == "__triton_hip_rhadd_u32") {
      assert(operands.size() == 2);
      // rhadd(a,b) = (a + b + 1) >> 1 unsigned
      auto i64Type = rewriter.getIntegerType(64);
      auto a = LLVM::ZExtOp::create(rewriter, loc, i64Type, operands[0]);
      auto b = LLVM::ZExtOp::create(rewriter, loc, i64Type, operands[1]);
      auto sum = LLVM::AddOp::create(rewriter, loc, i64Type,
                                     a->getResult(0), b->getResult(0));
      auto one64 = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getIntegerAttr(i64Type, 1));
      auto sumPlus1 = LLVM::AddOp::create(rewriter, loc, i64Type,
                                          sum->getResult(0), one64);
      auto shifted = LLVM::LShrOp::create(rewriter, loc, i64Type,
                                          sumPlus1->getResult(0), one64);
      replacementOp =
          LLVM::TruncOp::create(rewriter, loc, returnType, shifted->getResult(0));
    } else if (calleeName == "__triton_hip_sad_i32") {
      assert(operands.size() == 3);
      // sad(a,b,c) = |a - b| + c
      auto diff = LLVM::SubOp::create(rewriter, loc, operands[0].getType(),
                                      operands[0], operands[1]);
      auto absDiff = LLVM::AbsOp::create(rewriter, loc, operands[0].getType(),
                                         diff->getResult(0),
                                         /*is_int_min_poison=*/false);
      auto absDiffU = LLVM::BitcastOp::create(rewriter, loc, returnType,
                                              absDiff->getResult(0));
      replacementOp = LLVM::AddOp::create(rewriter, loc, returnType,
                                          absDiffU->getResult(0), operands[2]);
    } else if (calleeName == "__triton_hip_sad_u32") {
      assert(operands.size() == 3);
      // sad(a,b,c) = |a - b| + c (unsigned)
      auto diff = LLVM::SubOp::create(rewriter, loc, returnType,
                                      operands[0], operands[1]);
      // For unsigned: |a-b| = max(a,b) - min(a,b)
      auto aGtB = LLVM::ICmpOp::create(rewriter, loc,
                                       LLVM::ICmpPredicate::ugt,
                                       operands[0], operands[1]);
      auto maxAB = LLVM::SelectOp::create(rewriter, loc, returnType,
                                          aGtB, operands[0], operands[1]);
      auto minAB = LLVM::SelectOp::create(rewriter, loc, returnType,
                                          aGtB, operands[1], operands[0]);
      auto absDiff = LLVM::SubOp::create(rewriter, loc, returnType,
                                         maxAB->getResult(0), minAB->getResult(0));
      replacementOp = LLVM::AddOp::create(rewriter, loc, returnType,
                                          absDiff->getResult(0), operands[2]);
    } else if (calleeName == "__ocml_tanh_f32") {
      if (targetInfo.getISAFamily() == triton::amdgpu::ISAFamily::GFX1250) {
        const char *intrinsic = "llvm.amdgcn.tanh.f32";
        replacementOp = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, intrinsic, returnType, operands[0]);
      }
    }

    if (replacementOp) {
      rewriter.replaceOp(callOp, replacementOp);
      return mlir::success();
    }

    return mlir::failure();
  }

private:
  const AMD::TargetInfo &targetInfo;
  bool ftz;
};

struct ConvertBuiltinFuncToLLVM
    : public triton::impl::ConvertBuiltinFuncToLLVMBase<
          ConvertBuiltinFuncToLLVM> {
  ConvertBuiltinFuncToLLVM(StringRef gfxArch, bool ftz) {
    this->gfxArch = gfxArch.str();
    this->ftz = ftz;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive);

    AMD::TargetInfo targetInfo(this->gfxArch.getValue());
    RewritePatternSet patterns(context);
    patterns.add<CallOpConversion>(context, targetInfo, this->ftz);

    if (failed(applyPatternsGreedily(mod, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertBuiltinFuncToLLVMPass(StringRef gfxArch, bool ftz) {
  return std::make_unique<ConvertBuiltinFuncToLLVM>(gfxArch, ftz);
}

} // namespace mlir::triton
