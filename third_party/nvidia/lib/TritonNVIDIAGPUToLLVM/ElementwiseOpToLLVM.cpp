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

using namespace mlir::triton::gpu;

namespace mlir::triton {

namespace gpu {
namespace {

/* ----- FP8E5M2 ------ */
// This data-type is the standard FP8E5M2 format

struct Fp8ConversionDesc {
  std::string ptx;
  int inVecWidthBits;
  size_t numElements;
};

static Fp8ConversionDesc Fp_to_Fp8E5M2_RTNE(bool fromFp32) {
  std::string ptx = R"({
.reg .b32 h<2>, a<2>, n<2>, t<2>, maxval;
)";
  if (fromFp32) {
    // Preserve which side of an E5M2 midpoint each input lies on before
    // truncating to FP16. Every midpoint has its low 20 FP32 bits clear, and
    // FP16 truncation discards at most 16 bits there (the smallest midpoint
    // is 2^-17). Jam the low 16 bits into bit 16: u |= (u + 0xffff) & 0x10000.
    // This preserves final E5M2 rounding, not general FP16 round-to-odd.
    ptx += R"(.reg .b32 x<4>;
add.u32 x0, $1, 0xffff;
add.u32 x1, $2, 0xffff;
add.u32 x2, $3, 0xffff;
add.u32 x3, $4, 0xffff;
lop3.b32 x0, $1, x0, 0x10000, 0xf8;
lop3.b32 x1, $2, x1, 0x10000, 0xf8;
lop3.b32 x2, $3, x2, 0x10000, 0xf8;
lop3.b32 x3, $4, x3, 0x10000, 0xf8;
cvt.rz.f16x2.f32 h0, x1, x0;
cvt.rz.f16x2.f32 h1, x3, x2;
)";
  } else {
    ptx += R"(mov.b32 h0, $1;
mov.b32 h1, $2;
)";
  }

  // Work on two FP16 magnitudes per register. set.nan produces 1.0 (0x3c00)
  // for NaNs and zero otherwise. ORing this into a saturated 0x7b00 produces
  // NaN 0x7f00, without changing the other half.
  // min without .NaN maps NaNs to the finite operand too; n restores them.
  //
  // Their shared exponent bias makes this work for subnormals as well.
  // Add 0x7f + retained LSB per half to round ties to even. Clamping first
  // prevents overflow, so neither addition can carry between halves.
  ptx += R"(and.b32 a0, h0, 0x7fff7fff;
and.b32 a1, h1, 0x7fff7fff;
set.nan.f16x2.f16x2 n0, a0, a0;
set.nan.f16x2.f16x2 n1, a1, a1;
mov.b32 maxval, 0x7b007b00;
min.f16x2 a0, a0, maxval;
min.f16x2 a1, a1, maxval;
shr.b32 t0, a0, 8;
shr.b32 t1, a1, 8;
and.b32 t0, t0, 0x00010001;
and.b32 t1, t1, 0x00010001;
add.u32 a0, a0, 0x007f007f;
add.u32 a1, a1, 0x007f007f;
add.u32 a0, a0, t0;
add.u32 a1, a1, t1;
or.b32 a0, a0, n0;
or.b32 a1, a1, n1;
lop3.b32 a0, a0, h0, 0x80008000, 0xf8;
lop3.b32 a1, a1, h1, 0x80008000, 0xf8;
prmt.b32 $0, a0, a1, 0x7531;
})";
  return {ptx, 32, 4};
}

const Fp8ConversionDesc Fp16_to_Fp8E5M2_RTZ = {
    "{                            \n"
    ".reg .b32 a<2>;              \n"
    "and.b32 a0, $1, 0xfffefffe;  \n"   // a0 &= 0xfffefffe
    "and.b32 a1, $2, 0xfffefffe;  \n"   // (strip lowest bit)
    "prmt.b32 $0, a0, a1, 0x7531; \n\t" // output = a1a0
    "}",
    32, 4};

static const Fp8ConversionDesc Fp8E5M2_to_Fp16 = {
    "{\n"
    "prmt.b32 $0, 0, $2, 0x5140;\n"
    "prmt.b32 $1, 0, $2, 0x7362;\n"
    "}",
    32, 4};

static Fp8ConversionDesc Fp8_to_Bf16(int computeCapability, bool isE5M2) {
  if (isE5M2 && computeCapability < 89) {
    // E5M2 embeds exactly in FP16. Widening preserves infinities and quiets
    // NaNs; every finite result is exactly representable in BF16.
    return {R"({
.reg .b32 a<2>;
.reg .b16 h<4>;
.reg .f32 f<4>;
prmt.b32 a0, 0, $2, 0x5140;
prmt.b32 a1, 0, $2, 0x7362;
mov.b32 {h0, h1}, a0;
mov.b32 {h2, h3}, a1;
cvt.f32.f16 f0, h0;
cvt.f32.f16 f1, h1;
cvt.f32.f16 f2, h2;
cvt.f32.f16 f3, h3;
prmt.b32 $0, f0, f1, 0x7632;
prmt.b32 $1, f2, f3, 0x7632;
})",
            32, 4};
  }

  std::string ptx = R"({
.reg .b32 a;
.reg .f16 a<2>;
.reg .b16 b<2>;
)";
  ptx +=
      isE5M2 ? "cvt.rn.f16x2.e5m2x2 a, $1;\n" : "cvt.rn.f16x2.e4m3x2 a, $1;\n";
  ptx += "mov.b32 {a0, a1}, a;\n";
  if (computeCapability >= 90) {
    ptx += R"(cvt.bf16.f16 b0, a0;
cvt.bf16.f16 b1, a1;
)";
  } else {
    ptx += R"(.reg .f32 c<2>;
cvt.f32.f16 c0, a0;
cvt.f32.f16 c1, a1;
cvt.rn.bf16.f32 b0, c0;
cvt.rn.bf16.f32 b1, c1;
)";
  }
  ptx += R"(mov.b32 $0, {b0, b1};
})";
  return {ptx, 16, 2};
}

/* ----- Packed integer to BF16 ------ */
static const std::string S8_to_Bf16 =
    "{                                           \n"
    ".reg .s8 s<4>;                              \n"
    ".reg .f32 f<4>;                             \n"
    "mov.b32 {s0, s1, s2, s3}, $2;               \n" // unpack
    "cvt.rn.f32.s8 f0, s0;                       \n" // no s8->bf16 pre-Hopper
    "cvt.rn.f32.s8 f1, s1;                       \n" // fi[0:15] is always 0
    "cvt.rn.f32.s8 f2, s2;                       \n" //
    "cvt.rn.f32.s8 f3, s3;                       \n" //
    "prmt.b32 $0, f0, f1, 0x7632;                \n" // f32->bf16 + pack
    "prmt.b32 $1, f2, f3, 0x7632;                \n" //
    "}";
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

static ConverterT makeConverterFromPtx(const std::string &ptxAsm, Type inType,
                                       Type outType,
                                       const int inVecWidthBits = 32) {
  ConverterT converter =
      [ptxAsm, inType, outType,
       inVecWidthBits](Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) -> SmallVector<Value> {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int numElements = v.size();
    assert(numElements == 4 || numElements == 2 && "invalid vector size");

    auto ctx = rewriter.getContext();
    int inBitwidth = inType.getIntOrFloatBitWidth();
    int outBitwidth = outType.getIntOrFloatBitWidth();
    // first, we pack `v` into 32-bit ints
    int inVecWidth = inVecWidthBits / inBitwidth;
    auto inVecTy = vec_ty(inType, inVecWidth);
    SmallVector<Value> inPacked(numElements / inVecWidth, b.undef(inVecTy));
    for (size_t i = 0; i < numElements; i++)
      inPacked[i / inVecWidth] = b.insert_element(
          inVecTy, inPacked[i / inVecWidth], v[i], b.i32_val(i % inVecWidth));
    for (size_t i = 0; i < inPacked.size(); i++)
      inPacked[i] = b.bitcast(inPacked[i], int_ty(inVecWidthBits));

    // then, we run the provided inline PTX
    int outVecWidth = 32 / outBitwidth;
    int outNums = numElements / outVecWidth;
    PTXBuilder builder;
    SmallVector<PTXBuilder::Operand *> operands;
    auto inConstraint = inVecWidthBits == 16 ? "h" : "r";
    for (int i = 0; i < outNums; i++) {
      operands.push_back(builder.newOperand("=r"));
    }

    for (Value inVal : inPacked) {
      operands.push_back(builder.newOperand(inVal, inConstraint));
    }

    auto &ptxOp = *builder.create(ptxAsm);
    ptxOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto outVecTy = vec_ty(outType, outVecWidth);
    SmallVector<Value> outPacked;
    if (outNums == 1)
      outPacked.push_back(builder.launch(rewriter, loc, outVecTy, false));
    else {
      auto outStructTy = struct_ty(SmallVector<Type>(outNums, outVecTy));
      auto outStruct = builder.launch(rewriter, loc, outStructTy, false);
      for (int i = 0; i < outNums; i++)
        outPacked.push_back(b.extract_val(outVecTy, outStruct, i));
    }
    // unpack the output
    SmallVector<Value> ret;
    for (size_t i = 0; i < numElements; i++)
      ret.push_back(b.extract_element(outType, outPacked[i / outVecWidth],
                                      b.i32_val(i % outVecWidth)));
    return ret;
  };
  return converter;
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
    StringRef suffix;
    switch (rounding) {
    case RoundingMode::RTNE:
      suffix = "rn";
      break;
    case RoundingMode::RTZ:
      suffix = "rz";
      break;
    default:
      emitError(loc) << "unsupported rounding mode for f32->" << dstName
                     << " conversion: " << stringifyRoundingMode(rounding)
                     << "\n";
      llvm::report_fatal_error(
          "unsupported rounding mode for f32->" + dstName +
          " conversion: " + stringifyRoundingMode(rounding) + "\n");
    }
    auto name = ("llvm.nvvm.f2" + dstName + "." + suffix).str();
    return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, name, dstTy, {v})
        .getResult(0);
  }

  std::pair<ConverterT, size_t>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode) const {
    auto F8E4M3TyID = TypeID::get<Float8E4M3FNType>();
    auto F8E5M2TyID = TypeID::get<Float8E5M2Type>();
    auto F16TyID = TypeID::get<Float16Type>();
    auto BF16TyID = TypeID::get<BFloat16Type>();
    auto F32TyID = TypeID::get<Float32Type>();

    auto undefRounding = static_cast<RoundingMode>(-1);

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

    DenseMap<std::tuple<TypeID, TypeID, RoundingMode>, Fp8ConversionDesc>
        srcMap = {
            // F8 -> F16
            {{F8E5M2TyID, F16TyID, undefRounding}, Fp8E5M2_to_Fp16},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp_to_Fp8E5M2_RTNE(/*fromFp32=*/false)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTZ}, Fp16_to_Fp8E5M2_RTZ},
            // NVVM's FP8-to-BF16 op emits a scaled instruction.
            // Keep the unscaled conversion here.
            {{F8E5M2TyID, BF16TyID, undefRounding},
             hasPackedBf16
                 ? Fp8ConversionDesc{"cvt.rn.bf16x2.e5m2x2 $0, $1;", 16, 2}
                 : Fp8_to_Bf16(computeCapability, /*isE5M2=*/true)},
            {{F8E4M3TyID, BF16TyID, undefRounding},
             hasPackedBf16
                 ? Fp8ConversionDesc{"cvt.rn.bf16x2.e4m3x2 $0, $1;", 16, 2}
                 : Fp8_to_Bf16(computeCapability, /*isE5M2=*/false)},
            // F32 -> F8
            {{F32TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp_to_Fp8E5M2_RTNE(/*fromFp32=*/true)},
        };
    std::tuple<TypeID, TypeID, RoundingMode> key = {
        srcTy.getTypeID(), dstTy.getTypeID(),
        roundingMode.value_or(undefRounding)};
    if (!srcMap.contains(key)) {
      llvm::errs() << "Unsupported conversion from " << srcTy << " to "
                   << dstTy;
      if (roundingMode.has_value())
        llvm::errs() << " with rounding mode "
                     << stringifyRoundingMode(roundingMode.value());
      llvm::errs() << "\n";
      llvm::report_fatal_error("Unsupported rounding mode for conversion.");
    }
    auto convDesc = srcMap.lookup(key);
    return {makeConverterFromPtx(convDesc.ptx,
                                 getTypeConverter()->convertType(srcTy),
                                 getTypeConverter()->convertType(dstTy),
                                 convDesc.inVecWidthBits),
            convDesc.numElements};
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
      Value v = operands[0][0];
      if (computeCapability < 90) {
        v = LLVM::FPExtOp::create(rewriter, loc, f32_ty, v);
        auto rounding = roundingMode.value_or(RoundingMode::RTNE);
        return {convertFp32To16BitFloat(loc, rewriter, v, dstElementType,
                                        rounding)};
      }
      PTXBuilder builder;
      auto *result = builder.newOperand("=h");
      auto *input = builder.newOperand(v, "h");
      auto &cvt = *builder.create("cvt");
      cvt.o(roundingMode == RoundingMode::RTZ ? "rz" : "rn")
          .o(dstElementType.isBF16() ? "bf16" : "f16")
          .o(srcElementType.isBF16() ? "bf16" : "f16")(result, input);
      return {builder.launch(rewriter, loc, dstElementType, false)};
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

struct RemFOpConversion
    : ElementwiseOpConversionBase<arith::RemFOp, RemFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::RemFOp, RemFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::RemFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // NVPTX expands llvm.frem inline as a - trunc(a/b) * b, which drops the
    // dividend's sign on an exact-zero remainder and mishandles inf/NaN
    // (#11986). libdevice fmod covers the whole domain and costs about the
    // same (same div/multiply sequence inside fmodf).
    StringRef funcName;
    if (elemTy.isF32())
      funcName = "__nv_fmodf";
    else if (elemTy.isF64())
      funcName = "__nv_fmod";
    else
      return {}; // let the generic llvm.frem lowering take it
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    return {
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult()};
  }
};

struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion>;
  using Adaptor = typename Base::OpAdaptor;

  explicit SIToFPOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementTypeOrSelf(op.getIn());
    Type outElemTy = getElementTypeOrSelf(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                   operands[2][0], operands[3][0]};
      if (computeCapability >= 90)
        return convertS8ToBf16(loc, rewriter, inVals);
      auto cvtFunc = makeConverterFromPtx(
          S8_to_Bf16, getTypeConverter()->convertType(inElemTy),
          getTypeConverter()->convertType(outElemTy));
      auto outVals = cvtFunc(loc, rewriter, inVals);
      assert(outVals.size() == 4);
      return outVals;
    } else {
      return {LLVM::SIToFPOp::create(rewriter, loc, elemTy, operands[0][0])};
    }
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
    bool isHomogeneousHalf =
        (resultInfo.suffix == "f16x2" || resultInfo.suffix == "bf16x2") &&
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
      if (isHomogeneousHalf) {
        switch (op.getOpKind()) {
        case nvidia_gpu::PackedArithOpKind::ADD:
          resultVector =
              NVVM::AddFOp::create(rewriter, loc, resultVectorType,
                                   packedOperands[0], packedOperands[1]);
          break;
        case nvidia_gpu::PackedArithOpKind::SUB:
          resultVector =
              NVVM::SubFOp::create(rewriter, loc, resultVectorType,
                                   packedOperands[0], packedOperands[1]);
          break;
        case nvidia_gpu::PackedArithOpKind::FMA:
          resultVector = NVVM::FmaOp::create(
              rewriter, loc, resultVectorType, packedOperands[0],
              packedOperands[1], packedOperands[2], NVVM::FPRoundingMode::RN);
          break;
        case nvidia_gpu::PackedArithOpKind::MIN:
        case nvidia_gpu::PackedArithOpKind::MAX: {
          StringRef name = op.getOpKind() == nvidia_gpu::PackedArithOpKind::MIN
                               ? "llvm.nvvm.fmin."
                               : "llvm.nvvm.fmax.";
          resultVector = LLVM::createLLVMIntrinsicCallOp(
                             rewriter, loc, (name + resultInfo.suffix).str(),
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
  POPULATE_OP(triton::ApproxDivFOp, LLVM::FDivOp, [](LLVM::FDivOp op) {
    op.setFastmathFlags(LLVM::FastmathFlags::afn);
  });

  POPULATE_OP(arith::ExtFOp, LLVM::FPExtOp);
  POPULATE_OP(arith::TruncFOp, LLVM::FPTruncOp);
  POPULATE_OP(arith::FPToSIOp, LLVM::FPToSIOp);

#undef POPULATE_OP

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  // One above the default so we take frem from the generic LLVM::FRemOp
  // mapping registered by the shared populate.
  patterns.add<RemFOpConversion>(typeConverter, axisInfoAnalysis,
                                 PatternBenefit(benefit.getBenefit() + 1));
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
