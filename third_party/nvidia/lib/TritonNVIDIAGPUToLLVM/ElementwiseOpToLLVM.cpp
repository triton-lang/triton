#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/AtomicPTXBuilder.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <type_traits>

using namespace mlir::triton::gpu;

namespace mlir::triton {

namespace gpu {
namespace {

// The caller has proved that every FP32 value is exactly representable in BF16.
static SmallVector<Value>
truncateExactFp32ToBf16(Location loc, ConversionPatternRewriter &rewriter,
                        Value input) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (input.getType().isF32()) {
    Value bits = b.bitcast(input, i32_ty);
    Value highHalf = b.trunc(i16_ty, b.lshr(bits, b.i32_val(16)));
    return {b.bitcast(highHalf, bf16_ty)};
  }
  Value halves = b.bitcast(input, vec_ty(i16_ty, 8));
  Value highHalves = LLVM::ShuffleVectorOp::create(
      rewriter, loc, halves, b.undef(halves.getType()),
      ArrayRef<int32_t>{1, 3, 5, 7});
  return unpackLLVector(loc, b.bitcast(highHalves, vec_ty(bf16_ty, 4)),
                        rewriter);
}

// Keep the bit-trick conversion on SM90+, where cvt has low throughput, while
// exposing its arithmetic to LLVM's known-bits optimizations.
static SmallVector<Value> convertS8ToBf16(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          ArrayRef<Value> values) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Unpack to shifted bf16.
  Value input = packLLVector(loc, values, rewriter);
  Value exponent = b.i8_val(0x43);
  Value exponents =
      packLLVector(loc, {exponent, exponent, exponent, exponent}, rewriter);
  Value bytes =
      LLVM::ShuffleVectorOp::create(rewriter, loc, input, exponents,
                                    ArrayRef<int32_t>{0, 4, 1, 4, 2, 4, 3, 4});
  Value bits = b.bitcast(bytes, vec_ty(i16_ty, 4));
  // Zero the least exp bit.
  Value mantissaMask = b.i16_val(0xff7f);
  Value lhsMask = packLLVector(
      loc, {mantissaMask, mantissaMask, mantissaMask, mantissaMask}, rewriter);
  Value lhs = b.bitcast(b.and_(bits, lhsMask), vec_ty(bf16_ty, 4));
  // Zero the mantissa.
  Value exponentMask = b.i16_val(0xff80);
  Value rhsMask = packLLVector(
      loc, {exponentMask, exponentMask, exponentMask, exponentMask}, rewriter);
  Value rhs = b.bitcast(b.and_(bits, rhsMask), vec_ty(bf16_ty, 4));
  // Subtract the offset.
  Value converted = LLVM::FSubOp::create(rewriter, loc, lhs, rhs);
  return unpackLLVector(loc, converted, rewriter);
}

typedef std::function<SmallVector<Value>(Location, ConversionPatternRewriter &,
                                         const SmallVector<Value> &)>
    ConverterT;

static ConverterT makeE5M2F16Converter(bool toFp16) {
  return [toFp16](Location loc, ConversionPatternRewriter &rewriter,
                  const SmallVector<Value> &values) -> SmallVector<Value> {
    assert(values.size() == 4 && "expected four packed elements");
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value input = packLLVector(loc, values, rewriter);
    Value result;
    if (toFp16) {
      // E5M2 occupies the high byte of the corresponding FP16 value.
      Value zero = b.bitcast(b.i32_val(0), vec_ty(i8_ty, 4));
      result = LLVM::ShuffleVectorOp::create(
          rewriter, loc, input, zero,
          ArrayRef<int32_t>{4, 0, 4, 1, 4, 2, 4, 3});
      result = b.bitcast(result, vec_ty(f16_ty, 4));
    } else {
      Value bytes = b.bitcast(input, vec_ty(i8_ty, 8));
      result = LLVM::ShuffleVectorOp::create(rewriter, loc, bytes,
                                             b.undef(bytes.getType()),
                                             ArrayRef<int32_t>{1, 3, 5, 7});
    }
    return unpackLLVector(loc, result, rewriter);
  };
}

static SmallVector<Value>
convertF16ToE5M2RtzPtx(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &values) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value input =
      b.bitcast(packLLVector(loc, values, rewriter), vec_ty(i32_ty, 2));
  // Keep the packed sequence on SM90+, where LLVM uses more registers for
  // strided byte outputs.
  PTXBuilder builder;
  auto &convert = *builder.create(R"({
.reg .b32 a<2>;
and.b32 a0, $1, 0xfffefffe;
and.b32 a1, $2, 0xfffefffe;
prmt.b32 $0, a0, a1, 0x7531;
})");
  convert({builder.newOperand("=r"),
           builder.newOperand(b.extract_element(input, b.i32_val(0)), "r"),
           builder.newOperand(b.extract_element(input, b.i32_val(1)), "r")},
          /*onlyAttachMLIRArgs=*/true);
  Value result = builder.launch(rewriter, loc, i32_ty, false);
  return unpackLLVector(loc, b.bitcast(result, vec_ty(i8_ty, 4)), rewriter);
}

static ConverterT makeSoftwareE5M2Converter(bool fromFp32) {
  return [fromFp32](Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &values) -> SmallVector<Value> {
    assert(values.size() == 4 && "expected four packed elements");
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> rounded;
    for (int i = 0; i < 2; ++i) {
      Value halves;
      if (fromFp32) {
        SmallVector<Value> inputs;
        for (Value value : ArrayRef(values).slice(2 * i, 2)) {
          // Preserve which side of an E5M2 midpoint the FP32 input lies on
          // before truncating to FP16. Every midpoint has 20 low zero bits,
          // and FP16 truncation discards at most 16 bits at these midpoints.
          Value bits = b.bitcast(value, i32_ty);
          Value sticky =
              b.and_(b.add(bits, b.i32_val(0xffff)), b.i32_val(0x10000));
          inputs.push_back(b.bitcast(b.or_(bits, sticky), f32_ty));
        }
        halves = NVVM::ConvertF32x2ToF16x2Op::create(
            rewriter, loc, vec_ty(f16_ty, 2), inputs[1], inputs[0], Value(),
            NVVM::FPRoundingMode::RZ, NVVM::SaturationMode::NONE, false);
      } else {
        halves = packLLVector(loc, ArrayRef(values).slice(2 * i, 2), rewriter);
      }
      Value bits = b.bitcast(halves, i32_ty);
      Value magnitude = b.and_(bits, b.i32_val(0x7fff7fff));

      // LLVM comparisons need separate predicate-to-mask conversions. Packed
      // set.nan returns 1.0 per NaN lane, restoring NaNs after the numeric
      // clamp.
      PTXBuilder builder;
      auto *nan = builder.newOperand("=r");
      auto *input = builder.newOperand(magnitude, "r");
      (*builder.create("set.nan.f16x2.f16x2"))(nan, input, input);
      Value nanBits = builder.launch(rewriter, loc, i32_ty, false);
      Value maximum = b.bitcast(b.i32_val(0x7b007b00), vec_ty(f16_ty, 2));
      Value clamped = LLVM::createLLVMIntrinsicCallOp(
                          rewriter, loc, "llvm.minimumnum", vec_ty(f16_ty, 2),
                          {b.bitcast(magnitude, vec_ty(f16_ty, 2)), maximum})
                          .getResult(0);
      magnitude = b.bitcast(clamped, i32_ty);
      // Saturation prevents either rounding addition from carrying across
      // lanes.
      Value lsb =
          b.and_(b.lshr(magnitude, b.i32_val(8)), b.i32_val(0x00010001));
      magnitude = b.add(b.add(magnitude, b.i32_val(0x007f007f)), lsb);
      magnitude = b.or_(magnitude, nanBits);
      Value signedHalves = LLVM::CopySignOp::create(
          rewriter, loc, vec_ty(f16_ty, 2),
          b.bitcast(magnitude, vec_ty(f16_ty, 2)), halves);
      rounded.push_back(b.bitcast(signedHalves, i32_ty));
    }
    Value bytes =
        b.bitcast(packLLVector(loc, rounded, rewriter), vec_ty(i8_ty, 8));
    Value result = LLVM::ShuffleVectorOp::create(rewriter, loc, bytes,
                                                 b.undef(bytes.getType()),
                                                 ArrayRef<int32_t>{1, 3, 5, 7});
    return unpackLLVector(loc, result, rewriter);
  };
}

static ConverterT makeNativeFp8Converter(Type srcTy, Type dstTy, bool useFp32) {
  return [srcTy, dstTy,
          useFp32](Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &values) -> SmallVector<Value> {
    assert(values.size() == 2 && "expected a packed conversion");
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value result;
    if (useFp32) {
      Value lo = values[0];
      Value hi = values[1];
      if (!srcTy.isF32()) {
        lo = b.fpext(f32_ty, lo);
        hi = b.fpext(f32_ty, hi);
      }
      // The first scalar operand fills the high half of the packed result.
      result = NVVM::ConvertF32x2ToF8x2Op::create(
          rewriter, loc, vec_ty(i8_ty, 2), hi, lo, NVVM::FPRoundingMode::RN,
          NVVM::SaturationMode::SATFINITE, false, dstTy);
    } else {
      Value packed = packLLVector(loc, values, rewriter);
      if (srcTy.isF16()) {
        result = NVVM::ConvertF16x2ToF8x2Op::create(
            rewriter, loc, vec_ty(i8_ty, 2), packed, false, dstTy);
      } else if (srcTy.isBF16()) {
        result = NVVM::ConvertBF16x2ToF8x2Op::create(
            rewriter, loc, vec_ty(i8_ty, 2), packed, NVVM::FPRoundingMode::RN,
            NVVM::SaturationMode::SATFINITE, false, dstTy);
      } else {
        result = NVVM::ConvertF8x2ToF16x2Op::create(
            rewriter, loc, vec_ty(f16_ty, 2), packed, srcTy, false);
      }
    }
    return unpackLLVector(loc, result, rewriter);
  };
}

// Use packed conversions where the target supports them.
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<FpToFpOp, FpToFpOpConversion> {
  using ElementwiseOpConversionBase<
      FpToFpOp, FpToFpOpConversion>::ElementwiseOpConversionBase;

  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability, int ptxVersion,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability), ptxVersion(ptxVersion) {}

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return LLVM::FPExtOp::create(rewriter, loc, f32_ty, v);
  }

  static Value convertFp32To16BitFloat(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Value v, Type dstTy,
                                       RoundingMode rounding) {
    assert((dstTy.isF16() || dstTy.isBF16()) && "expected a 16-bit float");
    StringRef dstName = dstTy.isBF16() ? "bf16" : "f16";
    switch (rounding) {
    case RoundingMode::RTNE:
      return LLVM::FPTruncOp::create(rewriter, loc, dstTy, v);
    case RoundingMode::RTZ:
      break;
    default:
      emitError(loc) << "unsupported rounding mode for f32->" << dstName
                     << " conversion: " << stringifyRoundingMode(rounding)
                     << "\n";
      llvm::report_fatal_error(
          "unsupported rounding mode for f32->" + dstName +
          " conversion: " + stringifyRoundingMode(rounding) + "\n");
    }
    // The pinned NVVM dialect has no scalar directed-rounding conversion op.
    auto name = ("llvm.nvvm.f2" + dstName + ".rz").str();
    return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, dstTy, {v})
        .getResult(0);
  }

  static Value convert16BitFloat(Location loc,
                                 ConversionPatternRewriter &rewriter, Value v,
                                 Type dstTy, RoundingMode rounding,
                                 int computeCapability) {
    if (computeCapability < 90) {
      Value extended = LLVM::FPExtOp::create(rewriter, loc, f32_ty, v);
      return convertFp32To16BitFloat(loc, rewriter, extended, dstTy, rounding);
    }
    // LLVM rejects equal-width casts, and NVPTX does not lower bf2h.rn.
    PTXBuilder builder;
    auto *result = builder.newOperand("=h");
    auto *input = builder.newOperand(v, "h");
    auto &cvt = *builder.create("cvt");
    cvt.o(rounding == RoundingMode::RTZ ? "rz" : "rn")
        .o(dstTy.isBF16() ? "bf16" : "f16")
        .o(v.getType().isBF16() ? "bf16" : "f16")(result, input);
    return builder.launch(rewriter, loc, dstTy, false);
  }

  std::pair<ConverterT, size_t>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode) const {
    // Blackwell and newer can convert packed BF16/FP8 values directly.
    // Require PTX 9.2 to support both directions.
    bool hasPackedBf16 = computeCapability >= 100 && ptxVersion >= 92;

    if (computeCapability < 89 &&
        (isa<Float8E4M3FNType>(srcTy) || isa<Float8E4M3FNType>(dstTy))) {
      llvm::report_fatal_error("Conversion from/to f8e4m3nv is only supported "
                               "on compute capability >= 89\n");
    }
    if (computeCapability >= 89) {
      if (isa<Float8E4M3FNType, Float8E5M2Type>(dstTy) &&
          (srcTy.isF16() || srcTy.isBF16() || srcTy.isF32()) &&
          roundingMode == RoundingMode::RTNE) {
        bool useFp32 = srcTy.isF32() || (srcTy.isBF16() && !hasPackedBf16);
        return {makeNativeFp8Converter(srcTy, dstTy, useFp32), 2};
      }
      if (isa<Float8E4M3FNType, Float8E5M2Type>(srcTy) && dstTy.isF16() &&
          !roundingMode.has_value())
        return {makeNativeFp8Converter(srcTy, dstTy, false), 2};
    }
    if (isa<Float8E5M2Type>(srcTy) && dstTy.isF16() && !roundingMode)
      return {makeE5M2F16Converter(/*toFp16=*/true), 4};
    if (srcTy.isF16() && isa<Float8E5M2Type>(dstTy) &&
        roundingMode == RoundingMode::RTZ) {
      if (computeCapability >= 90)
        return {convertF16ToE5M2RtzPtx, 4};
      return {makeE5M2F16Converter(/*toFp16=*/false), 4};
    }

    if (isa<Float8E5M2Type>(dstTy) && (srcTy.isF16() || srcTy.isF32()) &&
        roundingMode == RoundingMode::RTNE)
      return {makeSoftwareE5M2Converter(srcTy.isF32()), 4};
    if (computeCapability < 89 && isa<Float8E5M2Type>(srcTy) &&
        dstTy.isBF16() && !roundingMode) {
      return {[](Location loc, ConversionPatternRewriter &rewriter,
                 const SmallVector<Value> &values) {
                auto halves = makeE5M2F16Converter(true)(loc, rewriter, values);
                Value widened =
                    LLVM::FPExtOp::create(rewriter, loc, vec_ty(f32_ty, 4),
                                          packLLVector(loc, halves, rewriter));
                return truncateExactFp32ToBf16(loc, rewriter, widened);
              },
              4};
    }

    if (isa<Float8E4M3FNType, Float8E5M2Type>(srcTy) && dstTy.isBF16() &&
        !roundingMode) {
      return {[srcTy, hasPackedBf16, computeCapability = computeCapability](
                  Location loc, ConversionPatternRewriter &rewriter,
                  const SmallVector<Value> &values) -> SmallVector<Value> {
                if (hasPackedBf16) {
                  // An omitted scale factor supplies 1.0 for both elements.
                  Value packed = NVVM::ConvertF8x2ToBF16x2Op::create(
                      rewriter, loc, vec_ty(bf16_ty, 2),
                      packLLVector(loc, values, rewriter), Value(), srcTy,
                      NVVM::SaturationMode::NONE, false);
                  return unpackLLVector(loc, packed, rewriter);
                }
                auto halves = makeNativeFp8Converter(srcTy, f16_ty, false)(
                    loc, rewriter, values);
                for (Value &value : halves)
                  value =
                      convert16BitFloat(loc, rewriter, value, bf16_ty,
                                        RoundingMode::RTNE, computeCapability);
                return halves;
              },
              2};
    }
    llvm::errs() << "Unsupported conversion from " << srcTy << " to " << dstTy;
    if (roundingMode.has_value())
      llvm::errs() << " with rounding mode "
                   << stringifyRoundingMode(roundingMode.value());
    llvm::errs() << "\n";
    llvm::report_fatal_error("Unsupported rounding mode for conversion.");
  }

  SmallVector<Value> createDestOps(FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcElementType = getElementTypeOrSelf(op.getSrc());
    auto dstElementType = getElementTypeOrSelf(op.getResult());
    auto roundingMode = op.getRounding();

    if (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(dstElementType)) {
      assert(roundingMode.has_value() &&
             "Rounding mode must be specified for convertsions to fp8");

      // For now only RTNE is supported for conversions from fp16 to fp8
      if (!srcElementType.isF32() &&
          roundingMode.value() != RoundingMode::RTNE) {
        llvm::report_fatal_error(
            "Unsupported rounding mode for conversion to fp8: " +
            stringifyRoundingMode(roundingMode.value()) + "\n");
      }
    }

    if (srcElementType.isF16() && dstElementType.isF32()) {
      return llvm::to_vector(llvm::map_range(operands[0], [&](Value v) {
        return convertFp16ToFp32(loc, rewriter, v);
      }));
    }

    if ((srcElementType.isBF16() && dstElementType.isF16()) ||
        (srcElementType.isF16() && dstElementType.isBF16())) {
      return {convert16BitFloat(loc, rewriter, operands[0][0], dstElementType,
                                roundingMode.value_or(RoundingMode::RTNE),
                                computeCapability)};
    }

    if (srcElementType.isF32() &&
        (dstElementType.isF16() || dstElementType.isBF16())) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->fp16/bf16 conversion");
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(convertFp32To16BitFloat(
            loc, rewriter, v, dstElementType, roundingMode.value()));
      }
      return outVals;
    }

    bool useSoftwareFp8 = computeCapability < 89 &&
                          isa<Float8E5M2Type>(dstElementType) &&
                          roundingMode == RoundingMode::RTNE;
    bool useFP16IntermediateSrc =
        (srcElementType.isF32() &&
         (!(llvm::isa<Float8E5M2Type>(dstElementType) ||
            (computeCapability >= 89 &&
             llvm::isa<Float8E4M3FNType>(dstElementType))) ||
          roundingMode.value() == RoundingMode::RTZ)) ||
        (useSoftwareFp8 && srcElementType.isBF16());
    bool isDstFP32 = dstElementType.isF32();
    Type srcType = useFP16IntermediateSrc ? f16_ty : srcElementType;
    Type dstType = isDstFP32 ? f16_ty : dstElementType;
    auto [cvtFunc, numElements] =
        getConversionFunc(srcType, dstType, roundingMode);
    SmallVector<Value> inVals;
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    if (useFP16IntermediateSrc) {
      for (Value &v : inVals) {
        if (srcElementType.isBF16()) {
          // BF16 is exact in FP16 throughout E5M2's nonzero rounding range.
          v = convertFp32To16BitFloat(loc, rewriter, b.fpext(f32_ty, v), f16_ty,
                                      RoundingMode::RTNE);
        } else {
          v = convertFp32To16BitFloat(loc, rewriter, v, f16_ty,
                                      RoundingMode::RTZ);
        }
      }
    }
    inVals.resize(numElements, b.undef(typeConverter->convertType(srcType)));
    SmallVector<Value> outVals = cvtFunc(loc, rewriter, inVals);
    assert(outVals.size() == inVals.size());
    outVals.resize(std::min(numElements, operands.size()));
    if (isDstFP32)
      for (Value &v : outVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    // Pack values
    return outVals;
  }

private:
  int computeCapability;
  int ptxVersion;
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion>;
  using OpAdaptor = typename Base::OpAdaptor;

  explicit ExtFOpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            int computeCapability,
                            PatternBenefit benefit = patternBenefitDefault)
      : Base(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  SmallVector<Value> createDestOps(arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // Keep FPExt on Blackwell so LLVM can select mixed BF16 arithmetic.
    if (computeCapability < 100 && elemTy.isF32() &&
        getElementTypeOrSelf(op.getIn()).isBF16()) {
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      Value bits = b.zext(i32_ty, b.bitcast(operands[0][0], i16_ty));
      return {b.bitcast(b.shl(bits, b.i32_val(16)), f32_ty)};
    }
    return {LLVM::FPExtOp::create(rewriter, loc, elemTy, operands[0],
                                  adaptor.getAttributes().getValue())};
  }

private:
  int computeCapability;
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (!elemTy.isF32() && !elemTy.isF64())
      llvm::report_fatal_error("Unsupported bitwidth");
    return {NVVM::DivFOp::create(
        rewriter, loc, elemTy, operands[0][0], operands[0][1],
        elemTy.isF32() ? NVVM::FPRoundingMode::NONE : NVVM::FPRoundingMode::RN,
        /*ftz=*/false, /*approx=*/false, /*full=*/elemTy.isF32())};
  }
};

template <typename SourceOp>
struct IntToFPOpConversion
    : ElementwiseOpConversionBase<SourceOp, IntToFPOpConversion<SourceOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp, IntToFPOpConversion<SourceOp>>;
  using OpAdaptor = typename Base::OpAdaptor;
  static constexpr bool isSigned = std::is_same_v<SourceOp, arith::SIToFPOp>;
  using DestOp = std::conditional_t<isSigned, LLVM::SIToFPOp, LLVM::UIToFPOp>;

  explicit IntToFPOpConversion(LLVMTypeConverter &typeConverter,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               int computeCapability,
                               PatternBenefit benefit = patternBenefitDefault)
      : Base(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  SmallVector<Value> createDestOps(SourceOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementTypeOrSelf(op.getIn());
    Type outElemTy = getElementTypeOrSelf(op.getOut());
    // BF16 selects increase register use on SM8x.
    bool useBf16Select = outElemTy.isBF16() &&
                         (computeCapability < 80 || computeCapability >= 90);
    if (inElemTy.isInteger(1) && (outElemTy.isF16() || useBf16Select)) {
      // Signed i1 represents 0 or -1; select the exact floating-point values.
      Value one = LLVM::ConstantOp::create(
          rewriter, loc, elemTy,
          rewriter.getFloatAttr(elemTy, isSigned ? -1.0 : 1.0));
      Value zero = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                            rewriter.getFloatAttr(elemTy, 0.0));
      return {LLVM::SelectOp::create(rewriter, loc, operands[0][0], one, zero)};
    }
    if (isSigned && outElemTy.isBF16() && inElemTy.isInteger(8) &&
        (operands.size() >= 4 || computeCapability >= 90)) {
      // Every signed byte is exact in BF16, so no rounding is needed.
      if (operands.size() < 4) {
        Value converted =
            LLVM::SIToFPOp::create(rewriter, loc, f32_ty, operands[0][0]);
        return truncateExactFp32ToBf16(loc, rewriter, converted);
      }
      SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                   operands[2][0], operands[3][0]};
      if (computeCapability >= 90)
        return convertS8ToBf16(loc, rewriter, inVals);
      Value converted =
          LLVM::SIToFPOp::create(rewriter, loc, vec_ty(f32_ty, 4),
                                 packLLVector(loc, inVals, rewriter));
      return truncateExactFp32ToBf16(loc, rewriter, converted);
    }
    auto result = DestOp::create(rewriter, loc, elemTy, operands[0],
                                 adaptor.getAttributes().getValue());
    if constexpr (!isSigned)
      result.setNonNeg(adaptor.getNonNeg());
    return {result};
  }

private:
  int computeCapability;
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // For non-FP32 input, call __nv_expf for higher-precision calculation
    if (getIntOrFloatOrPtrBitWidth(elemTy) != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = b.fmul(f32_ty, operands[0][0], b.f32_val(log2e));

    return {NVVM::Ex2Op::create(rewriter, loc, f32_ty, prod, /*ftz=*/false)};
  }
};

struct PackedArithOpConversion
    : ConvertOpToLLVMPattern<nvidia_gpu::PackedArithOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(nvidia_gpu::PackedArithOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto tensorType = op.getResult().getType();
    auto spec = nvidia_gpu::getPackedArithInstructionSpec(op);
    const auto &resultInfo = *spec.result;
    unsigned packWidth = resultInfo.lanes;
    bool isHomogeneousFloat =
        (resultInfo.suffix == "f32x2" || resultInfo.suffix == "f16x2" ||
         resultInfo.suffix == "bf16x2") &&
        llvm::all_of(spec.operands,
                     [&](const auto *info) { return info == spec.result; });
    SmallVector<SmallVector<Value>> operandValues;
    for (Value operand : adaptor.getOperands())
      operandValues.push_back(
          unpackUniqueTensorElements(loc, operand, rewriter));
    unsigned packCount =
        operandValues.front().size() / spec.operands.front()->storageLanes();

    Type resultRegisterType = int_ty(resultInfo.registerBits);
    Type resultVectorType =
        vec_ty(getTypeConverter()->convertType(tensorType.getElementType()),
               packWidth);
    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> packedResults;
    packedResults.reserve(packCount * packWidth);
    for (unsigned packIndex = 0; packIndex < packCount; ++packIndex) {
      SmallVector<Value> packedOperands;
      for (auto [index, values] : llvm::enumerate(operandValues)) {
        const auto &info = *spec.operands[index];
        unsigned width = info.storageLanes();
        auto lanes = ArrayRef(values).slice(packIndex * width, width);
        packedOperands.push_back(packLLVector(loc, lanes, rewriter));
      }
      Value resultVector;
      if (isHomogeneousFloat) {
        switch (op.getOpKind()) {
        case nvidia_gpu::PackedArithOpKind::ADD:
          resultVector =
              LLVM::FAddOp::create(rewriter, loc, resultVectorType,
                                   packedOperands[0], packedOperands[1]);
          break;
        case nvidia_gpu::PackedArithOpKind::SUB:
          resultVector =
              LLVM::FSubOp::create(rewriter, loc, resultVectorType,
                                   packedOperands[0], packedOperands[1]);
          break;
        case nvidia_gpu::PackedArithOpKind::MUL:
          resultVector =
              LLVM::FMulOp::create(rewriter, loc, resultVectorType,
                                   packedOperands[0], packedOperands[1]);
          break;
        case nvidia_gpu::PackedArithOpKind::FMA:
          resultVector = LLVM::FMAOp::create(
              rewriter, loc, resultVectorType, packedOperands[0],
              packedOperands[1], packedOperands[2]);
          break;
        case nvidia_gpu::PackedArithOpKind::MIN:
        case nvidia_gpu::PackedArithOpKind::MAX: {
          StringRef name = op.getOpKind() == nvidia_gpu::PackedArithOpKind::MIN
                               ? "llvm.minimumnum"
                               : "llvm.maximumnum";
          // Preserve numeric results for sNaNs and the ordering of signed
          // zeros.
          resultVector =
              LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name,
                                              resultVectorType, packedOperands)
                  .getResult(0);
          break;
        }
        default:
          break;
        }
      }
      if (!resultVector) {
        PTXBuilder ptxBuilder;
        SmallVector<PTXBuilder::Operand *> asmOperands;
        asmOperands.push_back(ptxBuilder.newOperand(
            "=" +
            NVIDIA::getPtxRegisterSizeCode(resultInfo.registerBits, false)));
        for (auto [index, packed] : llvm::enumerate(packedOperands)) {
          unsigned bits = spec.operands[index]->registerBits;
          asmOperands.push_back(ptxBuilder.newOperand(
              b.bitcast(packed, int_ty(bits)),
              NVIDIA::getPtxRegisterSizeCode(bits, false)));
        }
        auto &instruction = *ptxBuilder.create(
            stringifyPackedArithOpKind(op.getOpKind()).str());
        instruction.o(spec.modifiers.str(), !spec.modifiers.empty())
            .o(resultInfo.suffix.str());
        for (const auto *info :
             ArrayRef(spec.operands).take_front(spec.operandSuffixes))
          instruction.o(info->suffix.str());
        instruction(asmOperands);

        Value packedResult =
            ptxBuilder.launch(rewriter, loc, resultRegisterType,
                              /*hasSideEffect=*/false);
        resultVector = b.bitcast(packedResult, resultVectorType);
      }
      llvm::append_range(packedResults,
                         unpackLLVector(loc, resultVector, rewriter));
    }

    rewriter.replaceOp(op, packUniqueTensorElements(loc, getTypeConverter(),
                                                    packedResults, rewriter,
                                                    tensorType));
    return success();
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
    // min.xorsign.abs requires hopper or newer
    if (computeCapability < 90) {
      return false;
    }

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

    auto getSplatInitializer = [](Value v) -> std::optional<double> {
      DenseTypedElementsAttr denseAttr;
      if (matchPattern(v, m_Constant(&denseAttr))) {
        if (denseAttr.isSplat()) {
          return denseAttr.getSplatValue<APFloat>().convertToDouble();
        }
        return std::nullopt;
      }
      FloatAttr floatAttr;
      if (matchPattern(v, m_Constant(&floatAttr))) {
        return floatAttr.getValue().convertToDouble();
      }
      return std::nullopt;
    };

    // clampf %x (negf %max) %max
    if (auto negOp = op.getOperand(1).getDefiningOp<arith::NegFOp>()) {
      if (negOp.getOperand() == op.getOperand(2)) {
        return true;
      }
    }
    auto lowerSplat = op.getOperand(1).getDefiningOp<SplatOp>();
    auto upperSplat = op.getOperand(2).getDefiningOp<SplatOp>();
    if (lowerSplat && upperSplat) {
      auto negOp = lowerSplat.getSrc().getDefiningOp<arith::NegFOp>();
      if (negOp && negOp.getOperand() == upperSplat.getSrc())
        return true;
    }

    // clampf %x (sub 0.0 %max) %max
    if (auto subOp = op.getOperand(1).getDefiningOp<arith::SubFOp>()) {
      if (subOp.getOperand(1) == op.getOperand(2)) {
        auto initializer = getSplatInitializer(subOp.getOperand(0));
        if (initializer.has_value() && initializer.value() == 0.0) {
          return true;
        }
      }
    }

    // clampf %x, %min, %max (where min = -max = constant)
    auto initializer1 = getSplatInitializer(op.getOperand(1));
    auto initializer2 = getSplatInitializer(op.getOperand(2));
    if (initializer1.has_value() && initializer2.has_value() &&
        initializer1.value() == -initializer2.value()) {
      return true;
    }
    return false;
  }

  SmallVector<Value> emitOptimization(ClampFOp op,
                                      ConversionPatternRewriter &rewriter,
                                      Type elemTy,
                                      MultipleOperandsRange operands,
                                      Location loc) const {
    std::string name = "llvm.nvvm.fmin";
    if (op.getPropagateNan() == PropagateNan::ALL) {
      name += ".nan";
    }
    name += ".xorsign.abs";
    if (elemTy.isF32()) {
      name += ".f";
    } else if (elemTy.isF16()) {
      name += ".f16";
    } else if (elemTy.isBF16()) {
      name += ".bf16";
    }

    Type resultTy = operands[0][0].getType();
    Value args[] = {operands[0][0], operands[0][2]};
    auto callOp =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, resultTy, args);
    return {callOp.getResult(0)};
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
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult()};
  }

private:
  StringRef funcName;
};

struct MulhiUIOpConversion
    : public ElementwiseOpConversionBase<MulhiUIOp, MulhiUIOpConversion> {
  using Base = ElementwiseOpConversionBase<MulhiUIOp, MulhiUIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(MulhiUIOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
    assert(bitWidth == 32 || bitWidth == 64);
    StringRef intrinsic =
        bitWidth == 32 ? "llvm.nvvm.mulhi.ui" : "llvm.nvvm.mulhi.ull";
    // A widened multiply/shift relies on LLVM recognizing explicit extensions.
    // InstCombine can replace those extensions with masks, losing mul.hi.
    return {LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, elemTy,
                                            operands[0])
                .getResult(0)};
  }
};
} // namespace
} // namespace gpu

} // namespace mlir::triton

void mlir::triton::NVIDIA::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  mlir::triton::populateElementwiseOpToLLVMPatterns(typeConverter, patterns,
                                                    axisInfoAnalysis, benefit);

  patterns.add<MulhiUIOpConversion>(typeConverter, axisInfoAnalysis,
                                    benefit.getBenefit() + 1);

#define POPULATE_OP(SRC_OP, DST_OP, ...)                                       \
  patterns.add<                                                                \
      ElementwiseOpConversion<SRC_OP, DST_OP __VA_OPT__(, ) __VA_ARGS__>>(     \
      typeConverter, axisInfoAnalysis, benefit)

  POPULATE_OP(arith::SubFOp, LLVM::FSubOp);
  POPULATE_OP(arith::AddFOp, LLVM::FAddOp);
  POPULATE_OP(arith::MulFOp, LLVM::FMulOp);
  POPULATE_OP(triton::PreciseDivFOp, LLVM::FDivOp);
  POPULATE_OP(triton::PreciseSqrtOp, LLVM::SqrtOp);
  POPULATE_OP(math::SqrtOp, LLVM::SqrtOp, [](LLVM::SqrtOp op) {
    op.setFastmathFlags(LLVM::FastmathFlags::afn);
  });
  POPULATE_OP(math::RsqrtOp, NVVM::RsqrtOp);
  POPULATE_OP(triton::ApproxDivFOp, LLVM::FDivOp, [](LLVM::FDivOp op) {
    op.setFastmathFlags(LLVM::FastmathFlags::afn);
  });

  POPULATE_OP(arith::TruncFOp, LLVM::FPTruncOp);
  POPULATE_OP(arith::FPToSIOp, LLVM::FPToSIOp);

#undef POPULATE_OP

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis,
                                 computeCapability, benefit);
  patterns.add<IntToFPOpConversion<arith::SIToFPOp>,
               IntToFPOpConversion<arith::UIToFPOp>>(
      typeConverter, axisInfoAnalysis, computeCapability, benefit);
  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability,
                                   targetInfo.getPtxVersion(), benefit);
  patterns.add<PackedArithOpConversion>(typeConverter, benefit);

  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __nv_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  bool hwNanPropagationSupported = computeCapability >= 80;
  mlir::triton::populateMinMaxFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, hwNanPropagationSupported,
      benefit);
  mlir::triton::populateClampFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
}

void mlir::triton::NVIDIA::populateClampFOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  patterns.add<ClampFOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);
}
