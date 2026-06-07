#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

using mlir::getElementTypeOrSelf;
using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::ElementwiseOpConversion;
using mlir::triton::gpu::ElementwiseOpConversionBase;
using mlir::triton::gpu::getFunctionType;
using mlir::triton::gpu::MultipleOperandsRange;
using triton::amdgpu::ISAFamily;

namespace {

template <typename OP>
Value EmitDualBF16ElementwiseOp(Location loc,
                                ConversionPatternRewriter &rewriter,
                                MultipleOperandsRange operands) {
  auto v0 = AMD::convertBf16ToFp32(loc, rewriter, operands[0][0]);
  auto v1 = AMD::convertBf16ToFp32(loc, rewriter, operands[0][1]);
  auto result = OP::create(rewriter, loc, f32_ty, v0, v1);
  return AMD::convertFp32ToBf16(loc, rewriter, result, RoundingMode::RTNE);
}

// Override pattern that packs adjacent elementwise ops into vector LLVM ops.
// The default elementwise patterns remain target-agnostic.
template <typename SourceOp, typename LLVMOp>
struct PackedArithOpConversion
    : ElementwiseOpConversionBase<SourceOp,
                                  PackedArithOpConversion<SourceOp, LLVMOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp,
                                  PackedArithOpConversion<SourceOp, LLVMOp>>;
  using OpAdaptor = typename Base::OpAdaptor;

  using Base::Base;

  SmallVector<Value> createDestOps(SourceOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (operands.size() < 2 || !(elemTy.isF32() || elemTy.isBF16()))
      return {};

    Value va = packLLVector(loc, {operands[0][0], operands[1][0]}, rewriter);
    Value vb = packLLVector(loc, {operands[0][1], operands[1][1]}, rewriter);
    Value vr = LLVMOp::create(rewriter, loc, va.getType(), va, vb);
    return unpackLLVector(loc, vr, rewriter);
  }
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {

    return {LLVM::FDivOp::create(rewriter, loc, elemTy, operands[0][0],
                                 operands[0][1])};
  }
};

struct FMulOpConversion
    : ElementwiseOpConversionBase<arith::MulFOp, FMulOpConversion> {

  explicit FMulOpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            ISAFamily isaFamily,
                            PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        isaFamily(isaFamily) {}

  SmallVector<Value> createDestOps(arith::MulFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementTypeOrSelf(op.getLhs());
    auto rhsElemTy = getElementTypeOrSelf(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      if (triton::amdgpu::isRDNA(isaFamily)) {
        // To avoid casting to/from fp32, we compute a dot product with one
        // element of each vector set to zero.
        auto b = TritonLLVMOpBuilder(loc, rewriter);
        Value aVal = packLLVector(
            loc, ValueRange{operands[0][0], b.bf16_val(0.0)}, rewriter);
        Value bVal = packLLVector(
            loc, ValueRange{operands[0][1], b.bf16_val(0.0)}, rewriter);
        return {LLVM::createLLVMIntrinsicCallOp(
                    rewriter, loc, "llvm.amdgcn.fdot2.bf16.bf16", bf16_ty,
                    ValueRange{aVal, bVal, b.bf16_val(0.0)})
                    ->getResult(0)};
      } else {
        return {
            EmitDualBF16ElementwiseOp<LLVM::FMulOp>(loc, rewriter, operands)};
      }
    } else {
      return {LLVM::FMulOp::create(rewriter, loc, elemTy, operands[0][0],
                                   operands[0][1])};
    }
  }

private:
  ISAFamily isaFamily;
};

struct FAddOpConversion
    : ElementwiseOpConversionBase<arith::AddFOp, FAddOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::AddFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementTypeOrSelf(op.getLhs());
    auto rhsElemTy = getElementTypeOrSelf(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FAddOp>(loc, rewriter, operands)};
    } else {
      return {LLVM::FAddOp::create(rewriter, loc, elemTy, operands[0][0],
                                   operands[0][1])};
    }
  }
};

struct FSubOpConversion
    : ElementwiseOpConversionBase<arith::SubFOp, FSubOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::SubFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementTypeOrSelf(op.getLhs());
    auto rhsElemTy = getElementTypeOrSelf(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FSubOp>(loc, rewriter, operands)};
    } else {
      return {LLVM::FSubOp::create(rewriter, loc, elemTy, operands[0][0],
                                   operands[0][1])};
    }
  }
};

static SmallVector<Value> S8_to_Bf16(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> inValues = {v[0], v[1], v[2], v[3]};
  SmallVector<Value> outValues = {};
  for (Value inVal : inValues) {
    Value bf16Val = LLVM::SIToFPOp::create(rewriter, loc, bf16_ty, inVal);
    outValues.push_back(bf16Val);
  }
  return outValues;
}

struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementTypeOrSelf(op.getIn());
    Type outElemTy = getElementTypeOrSelf(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                   operands[2][0], operands[3][0]};
      auto outVals = S8_to_Bf16(loc, rewriter, inVals);
      assert(outVals.size() == 4);
      return outVals;
    } else if (outElemTy.isBF16()) {
      auto value =
          LLVM::SIToFPOp::create(rewriter, loc, f32_ty, operands[0][0]);
      return {AMD::convertFp32ToBf16(loc, rewriter, value, RoundingMode::RTNE)};
    } else {
      return {LLVM::SIToFPOp::create(rewriter, loc, elemTy, operands[0][0])};
    }
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementTypeOrSelf(op.getIn());
    if (inElemTy.isBF16()) {
      auto value = AMD::convertBf16ToFp32(loc, rewriter, operands[0][0]);
      return {LLVM::FPToSIOp::create(rewriter, loc, elemTy, value)};
    } else {
      return {LLVM::FPToSIOp::create(rewriter, loc, elemTy, operands[0][0])};
    }
  }
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementTypeOrSelf(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementTypeOrSelf(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return {AMD::convertBf16ToFp32(loc, rewriter, operands[0][0])};
    } else {
      return {LLVM::FPExtOp::create(rewriter, loc, elemTy, operands[0][0])};
    }
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  explicit TruncFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              ISAFamily isaFamily,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        isaFamily(isaFamily) {}

  SmallVector<Value> createDestOps(arith::TruncFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto outElemTy = getElementTypeOrSelf(op.getOut());
    auto inElemTy = getElementTypeOrSelf(op.getIn());
    if (inElemTy.isF32() && (outElemTy.isBF16() || outElemTy.isF16())) {
      return AMD::convertFp32ToF16rtne(loc, rewriter, inElemTy, outElemTy,
                                       operands, isaFamily);
    }
    return {LLVM::FPTruncOp::create(rewriter, loc, elemTy, operands[0][0])};
  }

private:
  ISAFamily isaFamily;
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // For non-FP32 input, call __ocml_exp_f64 for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = b.fmul(f32_ty, operands[0][0], b.f32_val(log2e));

    // Here we use llvm.exp2.f32 instead of math::Exp2Op. The latter
    // flushes denorms by default, but we want to preserve denorms by default
    // for expOp.
    StringRef funcName = "llvm.exp2.f32";
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    return {LLVM::createLLVMCallOp(rewriter, loc, funcOp, prod).getResult()};
  }
};

struct Exp2OpConversion
    : ElementwiseOpConversionBase<math::Exp2Op, Exp2OpConversion> {
  explicit Exp2OpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                            PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::Exp2Op op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // For non-FP32 input, call __ocml_exp2_f64 for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    // On AMD backend, both intrinsics are lowered to v_exp_f32 instruction,
    // which flushes input and output denorms. `llvm.amdgcn.exp2.f32` provides
    // direct access to v_exp_f32. For `llvm.exp2.f32`, the LLVM backend inserts
    // instructions to handle denorms iff `allow_flush_denorm` is False.
    if (ftz)
      return {ROCDL::ROCDLExp2::create(rewriter, loc, elemTy, operands[0])};

    return {LLVM::Exp2Op::create(rewriter, loc, elemTy, operands[0])};
  }

private:
  bool ftz;
};

struct RsqrtOpConversion
    : ElementwiseOpConversionBase<math::RsqrtOp, RsqrtOpConversion> {
  explicit RsqrtOpConversion(LLVMTypeConverter &typeConverter,
                             ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                             PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::RsqrtOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // This pass only deals with FP32 input with ftz configuration. Other cases
    // are delegate to MLIR.
    //
    // For FP16/FP64 input, it's lowered to __ocml_rsqrt_f16/__ocml_rsqrt_f64.
    //
    // For FP32 input with non-ftz configuration, it's lowered to
    // __ocml_rsqrt_f32, which will check the ftz/daz settings in the backend
    // dynamically to decide to preserve/flush denorms.
    if (elemTy.getIntOrFloatBitWidth() != 32 || !ftz)
      return {};

    // `llvm.amdgcn.rsq.f32` provides direct access to v_rsq_f32_e32.
    return {ROCDL::ROCDLRsq::create(rewriter, loc, elemTy, operands[0])};
  }

private:
  bool ftz;
};

static inline std::pair<Value, Value>
scaleUpIfDenorm(ConversionPatternRewriter &rewriter, Location loc,
                const Value &src, float scaleThreshold, float scaleFactor) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value needScale = b.fcmp_ogt(b.f32_val(scaleThreshold), src);
  Value scaledSrc = b.fmul(f32_ty, src, b.f32_val(scaleFactor));
  Value selectedSrc = b.select(needScale, scaledSrc, src);
  return {needScale, selectedSrc};
}

static inline Value scaleDownIfDenorm(ConversionPatternRewriter &rewriter,
                                      Location loc, const Value &src,
                                      Value needScale, float scaleFactor) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value scaledSrc = b.fmul(f32_ty, src, b.f32_val(scaleFactor));
  return b.select(needScale, scaledSrc, src);
}

struct SqrtOpConversion
    : ElementwiseOpConversionBase<math::SqrtOp, SqrtOpConversion> {
  explicit SqrtOpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                            PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::SqrtOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // This function only handles FP32 inputs. Other data types are lowered to
    // LLVM::SqrtOp by MLIR.
    //
    // On the AMDGPU backend, instructions legalized from LLVM::SqrtOp are
    // designed to produce IEEE-compliant results and always preserve denorms.
    // But what we actually need is an approximated SQRT. So we need to manually
    // lower the op.
    //
    // Differences in this approach are
    // 1. Refinement iterations following llvm.amdgcn.sqrt.f32 are removed to
    // improve performance.
    // 2. With ftz enabled, the scaling-up-and-down process is bypassed to
    // ensure denorms are flushed to zero.
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    Value needScale = b.false_val();
    Value scaledSrc = operands[0][0];
    if (!ftz) {
      // For non-ftz cases, if the input value is below 2^{-96}, it needs to be
      // scaled up by a factor of 2^{32}, to prevent it from being flushed by
      // llvm.amdgcn.sqrt.f32.
      //
      // The result is then scaled down afterward to get the correct result.
      // Reference:
      // https://github.com/llvm/llvm-project/blob/0876c11c/llvm/lib/Target/AMDGPU/AMDGPULegalizerInfo.cpp#L5235-L5314.
      std::tie(needScale, scaledSrc) = scaleUpIfDenorm(
          rewriter, loc, operands[0][0], 0x1.0p-96f, 0x1.0p+32f);
    }

    // llvm.amdgcn.sqrt.f32 provides direct access to v_sqrt_f32, which provides
    // 1ULP accuracy and flushs denorms.
    Value intrinsicsOutput =
        ROCDL::ROCDLSqrt::create(rewriter, loc, elemTy, scaledSrc);

    if (!ftz) {
      // In case of non-ftz, we need to calibrate the results by scaling down by
      // a factor of 2^{-16}.
      return {scaleDownIfDenorm(rewriter, loc, intrinsicsOutput, needScale,
                                0x1.0p-16f)};
    } else {
      return {intrinsicsOutput};
    }
  }

private:
  bool ftz;
};

struct ClampFOpConversion
    : ElementwiseOpConversionBase<triton::ClampFOp, ClampFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<triton::ClampFOp, ClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(triton::ClampFOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (!(elemTy.isF16() || elemTy.isF32()))
      return {};

    Value x = operands[0][0];
    Value lo = operands[0][1];
    Value hi = operands[0][2];

    Value med = ROCDL::FMed3Op::create(rewriter, loc, elemTy, x, lo, hi);

    // `PropagateNaN::ALL` requires us to return NaN if x is NaN. Since `v_med3`
    // returns the min if any operand is NaN, we must explicitly check NaN.
    if (op.getPropagateNan() == PropagateNan::ALL) {
      Value isNan =
          LLVM::FCmpOp::create(rewriter, loc, LLVM::FCmpPredicate::une, x, x);
      Value res = LLVM::SelectOp::create(rewriter, loc, isNan, x, med);
      return {res};
    }

    return {med};
  }
};
} // namespace

namespace mlir::triton::AMD {
void adjustModeRegister(ModuleOp mod, const TargetInfo &targetInfo) {
  MLIRContext *ctx = mod->getContext();
  Location loc = mod->getLoc();
  mlir::OpBuilder builder(ctx);
  auto auxBuilder = TritonLLVMOpBuilder(loc, builder);

  mod->walk([&](LLVM::LLVMFuncOp func) {
    using attrType = triton::amdgpu::SetFP8ClampingAttr;
    auto attrName = attrType::getMnemonic();
    if (!func->hasAttrOfType<attrType>(attrName))
      return;
    else
      func->removeAttr(attrName);

    if (func.getBody().empty())
      return;
    auto &body = func.getBody().front();
    builder.setInsertionPoint(&body.front());

    // This is the location of the fp16_ovfl flag in the Mode register. It's
    // calculated following this formula:
    //     (mode register ID = 1) | (Offset << 6) | ((Width - 1) << 11)
    // In this case, Offset = 23 and Width = 1.
    // When the bit is 0/1, the conversion from fp32/fp16/bf16 to fp8/bf8 is
    // in non-saturation/saturation mode.
    Value fp16OVFLModeRegLoc = auxBuilder.i32_val(1473);
    LLVM::createLLVMIntrinsicCallOp(
        builder, loc, "llvm.amdgcn.s.setreg", {},
        {fp16OVFLModeRegLoc, auxBuilder.i32_val(1)});
  });
}

void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, bool ftz,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    const TargetInfo &targetInfo, PatternBenefit benefit) {

  // fmin (return NaN if either op is NaN)
  patterns.add<ElementwiseOpConversion<arith::MinimumFOp, LLVM::MinimumOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  // fmax (return NaN if either op is NaN)
  patterns.add<ElementwiseOpConversion<arith::MaximumFOp, LLVM::MaximumOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<triton::PreciseDivFOp, LLVM::FDivOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<triton::PreciseSqrtOp, LLVM::SqrtOp>>(
      typeConverter, axisInfoAnalysis, benefit);

  if (targetInfo.getISAFamily() == ISAFamily::GFX1250) {
    auto gfx1250Benefit = benefit.getBenefit() + 1;
    patterns.add<PackedArithOpConversion<arith::SubFOp, LLVM::FSubOp>>(
        typeConverter, axisInfoAnalysis, gfx1250Benefit);
    patterns.add<PackedArithOpConversion<arith::AddFOp, LLVM::FAddOp>>(
        typeConverter, axisInfoAnalysis, gfx1250Benefit);
    patterns.add<PackedArithOpConversion<arith::MulFOp, LLVM::FMulOp>>(
        typeConverter, axisInfoAnalysis, gfx1250Benefit);
  }

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FMulOpConversion>(typeConverter, axisInfoAnalysis,
                                 targetInfo.getISAFamily(), benefit);

  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis,
                                   targetInfo.getISAFamily(), benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  // ExpOpConversionApprox will try using __ocml_exp2_f32 if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // later pass will call __ocml_exp_f64 for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  // Exp2OpConversion will use llvm.exp2.f32 or llvm.amdgcn.exp2.f32
  // based on the ftz flag if the input type is FP32. For FP64 input,
  // Exp2OpConversion will return failure and later pass will call
  // __ocml_exp2_f64 for higher-precision calculation
  patterns.add<Exp2OpConversion>(typeConverter, axisInfoAnalysis, ftz, benefit);
  patterns.add<RsqrtOpConversion>(typeConverter, axisInfoAnalysis, ftz,
                                  benefit);
  patterns.add<SqrtOpConversion>(typeConverter, axisInfoAnalysis, ftz, benefit);
  patterns.add<ClampFOpConversion>(typeConverter, axisInfoAnalysis,
                                   benefit.getBenefit() + 1);
  triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
  bool hwNanPropagationSupported = targetInfo.supportMaximumMinimum();
  triton::populateMinMaxFOpToLLVMPattern(typeConverter, patterns,
                                         axisInfoAnalysis,
                                         hwNanPropagationSupported, benefit);
  triton::populateClampFOpToLLVMPattern(typeConverter, patterns,
                                        axisInfoAnalysis, targetInfo, benefit);
}
} // namespace mlir::triton::AMD
