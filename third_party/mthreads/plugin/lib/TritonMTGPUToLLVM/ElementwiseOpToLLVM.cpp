#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using namespace mlir::triton::gpu;
using namespace mlir::triton::MUSA;

namespace mlir::triton {

namespace gpu {
namespace {

struct Fp8ConversionDesc {
  std::string funcName;
  size_t numElements;
};

static SmallVector<Value> convertFp8(const LLVMTypeConverter *typeConverter,
                                     Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     const SmallVector<Value> &v,
                                     Type &srcElementType, Type &dstElementType,
                                     const std::string funcName) {
  size_t numElements = v.size();
  Type inpType;
  Type outType;
  Value inVals;

  if (numElements == 1) {
    inpType = typeConverter->convertType(srcElementType);
    outType = typeConverter->convertType(dstElementType);
    inVals = v[0];
  } else {
    inpType = vec_ty(typeConverter->convertType(srcElementType), numElements);
    outType = vec_ty(typeConverter->convertType(dstElementType), numElements);
    inVals = undef(inpType);
    for (size_t i = 0; i < numElements; i++) {
      inVals = insert_element(inpType, inVals, v[i], i32_val(i));
    }
  }

  Type funcType = LLVM::LLVMFunctionType::get(outType, inpType);

  std::string libName = "";
  std::string libPath = "";

  // Call libdevice
  LLVM::LLVMFuncOp funcOp = appendOrGetExternFuncOp(
      rewriter, v[0].getDefiningOp(), funcName, funcType, libName, libPath);
  auto outVals = rewriter.create<LLVM::CallOp>(loc, funcOp, inVals).getResult();

  SmallVector<Value> ret;
  for (size_t i = 0; i < numElements; i++) {
    ret.push_back(numElements == 1 ? outVals
                                   : extract_element(typeConverter->convertType(
                                                         dstElementType),
                                                     outVals, i32_val(i)));
  }
  return ret;
}

// Attempts to use vectorized conversions via inline PTX when possible.
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<FpToFpOp, FpToFpOpConversion> {
  using ElementwiseOpConversionBase<
      FpToFpOp, FpToFpOpConversion>::ElementwiseOpConversionBase;

  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  static Value convertBf16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    auto as_int16 = bitcast(v, i16_ty);
    auto as_int32 = zext(i32_ty, as_int16);
    auto shifted = shl(i32_ty, as_int32, i32_val(16));
    return bitcast(shifted, f32_ty);
  }

  static Value convertFp32ToBf16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, const RoundingMode rounding) {
    if (rounding == RoundingMode::RTZ) {
      auto as_int32 = bitcast(v, i32_ty);
      auto shifted = lshr(i32_ty, as_int32, i32_val(16));
      auto truncated = trunc(i16_ty, shifted);
      return bitcast(truncated, bf16_ty);
    }
    // Otherwise it is (rounding == RoundingMode::RTNE)
    auto as_uint32 = bitcast(v, i32_ty);
    auto check_exponent =
        and_(i32_ty, xor_(i32_ty, as_uint32, i32_val(0xffffffff)),
             i32_val(0x7f800000));
    auto exponent_not_all1s = icmp_ne(check_exponent, i32_val(0));
    auto exponent_all1s = icmp_eq(check_exponent, i32_val(0));
    auto rounded =
        add(i32_ty, i32_val(0x7fff),
            and_(i32_ty, lshr(i32_ty, as_uint32, i32_val(16)), i32_val(1)));
    rounded = add(i32_ty, rounded, as_uint32);
    auto res = select(exponent_not_all1s, rounded, as_uint32);

    auto preserve_nan =
        and_(i1_ty, exponent_all1s,
             icmp_ne(and_(i32_ty, as_uint32, i32_val(0xffff)), i32_val(0)));
    auto nan = or_(i32_ty, as_uint32, i32_val(0x10000));
    res = select(preserve_nan, nan, res);

    auto shifted = lshr(i32_ty, res, i32_val(16));
    auto truncated = trunc(i16_ty, shifted);
    return bitcast(truncated, bf16_ty);
  }

  std::pair<std::string, size_t>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode,
                    bool enableFp8Burst2) const {
    auto F8E4M3TyID = TypeID::get<Float8E4M3FNUZType>();
    auto F8E5M2TyID = TypeID::get<Float8E5M2Type>();
    auto F16TyID = TypeID::get<Float16Type>();
    auto BF16TyID = TypeID::get<BFloat16Type>();
    auto F32TyID = TypeID::get<Float32Type>();
    auto F64TyID = TypeID::get<Float64Type>();

    auto undefRounding = static_cast<RoundingMode>(-1);

    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>,
                    SmallVector<Fp8ConversionDesc>>
        srcMap = {
            // F8 -> F32
            {{F8E4M3TyID, F32TyID, undefRounding},
             {{"__mt_tt_v2e4m3_to_v2f32", 2}, {"__mt_tt_e4m3_to_f32", 1}}},
            {{F8E5M2TyID, F32TyID, undefRounding},
             {{"__mt_tt_v2e5m2_to_v2f32", 2}, {"__mt_tt_e5m2_to_f32", 1}}},
            // F8 -> F16
            {{F8E4M3TyID, F16TyID, undefRounding},
             {{"__mt_tt_v2e4m3_to_v2f16", 2}, {"__mt_tt_e4m3_to_f16", 1}}},
            {{F8E5M2TyID, F16TyID, undefRounding},
             {{"__mt_tt_v2e5m2_to_v2f16", 2}, {"__mt_tt_e5m2_to_f16", 1}}},
            // F8 -> BF16
            {{F8E4M3TyID, BF16TyID, undefRounding},
             {{"__mt_tt_v2e4m3_to_v2bf16", 2}, {"__mt_tt_e4m3_to_bf16", 1}}},
            {{F8E5M2TyID, BF16TyID, undefRounding},
             {{"__mt_tt_v2e5m2_to_v2bf16", 2}, {"__mt_tt_e5m2_to_bf16", 1}}},
            // F32 -> F8
            {{F32TyID, F8E4M3TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2f32_to_v2e4m3", 2}, {"__mt_tt_f32_to_e4m3", 1}}},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2f32_to_v2e5m2", 2}, {"__mt_tt_f32_to_e5m2", 1}}},
            // F16 -> F8
            {{F16TyID, F8E4M3TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2f16_to_v2e4m3", 2}, {"__mt_tt_f16_to_e4m3", 1}}},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2f16_to_v2e5m2", 2}, {"__mt_tt_f16_to_e5m2", 1}}},
            // BF16 -> F8
            {{BF16TyID, F8E4M3TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2bf16_to_v2e4m3", 2}, {"__mt_tt_bf16_to_e4m3", 1}}},
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTNE},
             {{"__mt_tt_v2bf16_to_v2e5m2", 2}, {"__mt_tt_bf16_to_e5m2", 1}}},
        };
    std::tuple<TypeID, TypeID, RoundingMode> key = {
        srcTy.getTypeID(), dstTy.getTypeID(),
        roundingMode.value_or(undefRounding)};
    if (srcMap.count(key) == 0) {
      llvm::errs() << "Unsupported conversion from " << srcTy << " to "
                   << dstTy;
      if (roundingMode.has_value())
        llvm::errs() << " with rounding mode "
                     << stringifyRoundingMode(roundingMode.value());
      llvm::errs() << "\n";
      llvm::report_fatal_error("Unsupported rounding mode for conversion.");
    }
    auto convDesc =
        enableFp8Burst2 ? srcMap.lookup(key)[0] : srcMap.lookup(key)[1];

    return {convDesc.funcName, convDesc.numElements};
  }

  SmallVector<Value> createDestOps(FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto srcElementType = getElementType(op.getSrc());
    auto dstElementType = getElementType(op.getResult());
    auto roundingMode = op.getRounding();

    bool isFp8Converion =
        srcElementType.isFloat8E4M3FNUZ() || srcElementType.isFloat8E5M2() ||
        dstElementType.isFloat8E4M3FNUZ() || dstElementType.isFloat8E5M2();
    assert(isFp8Converion &&
           "For now only Fp8 conversions are supported for the op FpToFp.");

    if (dstElementType.isFloat8E5M2() || dstElementType.isFloat8E4M3FNUZ()) {
      assert(roundingMode.has_value() &&
             "Rounding mode must be specified for convertsions to fp8");

      // For now only RTNE is supported for all conversions
      if (roundingMode.value() != RoundingMode::RTNE) {
        llvm::errs() << "Unsupported rounding mode for conversion to fp8: "
                     << stringifyRoundingMode(roundingMode.value()) << "\n";
        llvm_unreachable("");
      }
    }

    // Default disable fp8 burst2
    bool enableFp8Burst2 = false;
    std::string envValue =
        mlir::triton::tools::getStrEnv("MUSA_ENABLE_FP8_BURST2");
    if (!envValue.empty() &&
        (envValue == "true" || envValue == "TRUE" || envValue == "1")) {
      enableFp8Burst2 = true;
    }

    auto [funcName, numElements] = getConversionFunc(
        srcElementType, dstElementType, roundingMode, enableFp8Burst2);

    // FP8 conversions
    if (isFp8Converion) {
      SmallVector<Value> inVals;
      for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
        inVals.push_back(operands[i][0]);
      }
      inVals.resize(numElements,
                    undef(typeConverter->convertType(srcElementType)));
      SmallVector<Value> outVals =
          convertFp8(getTypeConverter(), loc, rewriter, inVals, srcElementType,
                     dstElementType, funcName);
      return outVals;
    }
    llvm_unreachable("Unsupported conversion");
    return {};
  }

private:
  int computeCapability;
};

template <typename OP>
Value EmitDualBF16ElementwiseOp(Location loc,
                                ConversionPatternRewriter &rewriter,
                                MultipleOperandsRange operands) {
  auto v0 =
      FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0]);
  auto v1 =
      FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][1]);
  auto result = rewriter.create<OP>(loc, f32_ty, v0, v1);
  return FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, result,
                                               RoundingMode::RTNE);
}

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    return {rewriter.create<LLVM::FDivOp>(loc, elemTy, operands[0][0],
                                          operands[0][1])};
  }
};

struct FMulOpConversion
    : ElementwiseOpConversionBase<arith::MulFOp, FMulOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::MulFOp, FMulOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::MulFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FMulOp>(loc, rewriter, operands)};
    } else {
      return {rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

struct FAddOpConversion
    : ElementwiseOpConversionBase<arith::AddFOp, FAddOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::AddFOp, FAddOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::AddFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FAddOp>(loc, rewriter, operands)};
    } else {
      return {rewriter.create<LLVM::FAddOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

struct FSubOpConversion
    : ElementwiseOpConversionBase<arith::SubFOp, FSubOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SubFOp, FSubOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::SubFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FSubOp>(loc, rewriter, operands)};
    } else {
      return {rewriter.create<LLVM::FSubOp>(loc, elemTy, operands[0][0],
                                            operands[0][1])};
    }
  }
};

// Uses inline ptx to convert s8/u8 to bf16, since the
struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8)) {
      // TODO(lingfeng.qiu): use inline asm to vectorize 4*s8.
      // s8 -> s32 -> fp32 -> bf16
      Value i32Val = sext(i32_ty, operands[0][0]);
      Value f32Val = inttofloat(f32_ty, i32Val);
      f32Val = bitcast(f32Val, i32_ty);
      auto shifted = lshr(i32_ty, f32Val, i32_val(16));
      auto truncated = trunc(i16_ty, shifted);
      auto outVal = bitcast(truncated, bf16_ty);
      return {outVal};
    } else if (outElemTy.isBF16()) {
      auto value = rewriter.create<LLVM::SIToFPOp>(loc, f32_ty, operands[0][0]);
      return {FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, value,
                                                    RoundingMode::RTNE)};
    } else {
      return {rewriter.create<LLVM::SIToFPOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, operands[0][0])};
  }
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return {
          FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0])};
    } else {
      return {rewriter.create<LLVM::FPExtOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::TruncFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto inElemTy = getElementType(op.getIn());
      assert(inElemTy.isF32() && "unsupported conversion");
      return {// Trunc uses the default rounding mode: RTNE
              FpToFpOpConversion::convertFp32ToBf16(
                  loc, rewriter, operands[0][0], RoundingMode::RTNE)};
    } else {
      return {rewriter.create<LLVM::FPTruncOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox> {
  using Base = ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // For non-FP32 input, call __nv_expf for higher-precision calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = fmul(f32_ty, operands[0][0], f32_val(log2e));

    return {rewriter.create<math::Exp2Op>(loc, f32_ty, prod,
                                          adaptor.getAttributes().getValue())};
  }
};

struct ClampFOpConversion
    : ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion> {
  using Base = ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit ClampFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  bool isClipPattern(ClampFOp op) const {
    bool xorsignAbsAvailable = (computeCapability >= 90);
    // Pattern matching the sequence of clamp(x, -limit, limit) to generate
    // more efficient PTX code. NOTE: This pattern matching is not general
    // enough, but it is sufficient. We detect only two cases here:
    // 1. where the "-limit" is computed as 0 - limit:
    //   %cst = arith.constant dense<0.000000e+00>
    //   %8 = tt.load %7, %2
    //   %11 = arith.subf %cst, %8
    //   %12 = tt.clamp %5, %11, %8
    // 2. where "-limit" and "limit" are constants.
    //   %cst_6 = arith.constant dense<-6.0000e+00>
    //   %cst_7 = arith.constant dense<6.0000e+00>
    //   %160 = tt.clamp %158, %cst_6, %cst_7
    bool patternFound = false;

    auto getSplatInitializer = [](Value v) -> std::optional<double> {
      if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto attr = mlir::dyn_cast<DenseIntOrFPElementsAttr>(
                constOp.getValueAttr())) {
          if (attr.isSplat()) {
            return attr.getSplatValue<APFloat>().convertToDouble();
          }
        }
      }
      return std::nullopt;
    };

    if (xorsignAbsAvailable) {
      if (auto subOp = op.getOperand(1).getDefiningOp<arith::SubFOp>()) {
        if (subOp.getOperand(1) == op.getOperand(2)) {
          auto initializer = getSplatInitializer(subOp.getOperand(0));
          if (initializer.has_value() && initializer.value() == 0.0) {
            patternFound = true;
          }
        }
      } else {
        auto initializer1 = getSplatInitializer(op.getOperand(1));
        auto initializer2 = getSplatInitializer(op.getOperand(2));
        if (initializer1.has_value() && initializer2.has_value() &&
            initializer1.value() == -initializer2.value()) {
          patternFound = true;
        }
      }
    }
    return patternFound;
  }

  SmallVector<Value> emitOptimization(ClampFOp op,
                                      ConversionPatternRewriter &rewriter,
                                      Type elemTy,
                                      MultipleOperandsRange operands,
                                      Location loc) const {
    llvm_unreachable("This function is not implemented yet.");
    return {};
  }

  SmallVector<Value> createDestOps(ClampFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (isClipPattern(op)) {
      return emitOptimization(op, rewriter, elemTy, operands, loc);
    }
    return {};
  }

private:
  int computeCapability;
};

template <typename TritonOp>
struct OpToExternCallConversion
    : public ElementwiseOpConversionBase<TritonOp,
                                         OpToExternCallConversion<TritonOp>> {
  using Base =
      ElementwiseOpConversionBase<TritonOp, OpToExternCallConversion<TritonOp>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit OpToExternCallConversion(LLVMTypeConverter &typeConverter,
                                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                                    StringRef externFuncName,
                                    PatternBenefit benefit)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit),
        funcName(externFuncName) {}

  SmallVector<Value> createDestOps(TritonOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    return {
        rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]).getResult()};
  }

private:
  StringRef funcName;
};
} // namespace
} // namespace gpu

} // namespace mlir::triton

void mlir::triton::MUSA::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  patterns.add<OpToExternCallConversion<triton::PreciseSqrtOp>>(
      typeConverter, axisInfoAnalysis, "__nv_fsqrt_rn", benefit);
  patterns.add<OpToExternCallConversion<triton::PreciseDivFOp>>(
      typeConverter, axisInfoAnalysis, "__nv_fdiv_rn", benefit);

  mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FMulOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);

  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  bool hwNanPropagationSupported = targetInfo.supportMaximumMinimum();
  mlir::triton::populateMinMaxFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, hwNanPropagationSupported,
      benefit);
  mlir::triton::populateClampFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
}

void mlir::triton::MUSA::populateClampFOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  patterns.add<ClampFOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);
}
