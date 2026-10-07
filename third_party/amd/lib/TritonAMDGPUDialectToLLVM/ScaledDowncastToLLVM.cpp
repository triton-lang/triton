#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

// Convert a raw E8M0 scale byte to an f32 scale. E8M0 stores a biased f32
// exponent, so we can shift it into the f32 exponent field.
Value scaleToF32(TritonLLVMOpBuilder &b, Value scaleByte) {
  MLIRContext *ctx = scaleByte.getContext();
  return b.bitcast(
      b.shl(b.zext(IntegerType::get(ctx, 32), scaleByte), b.i32_val(23)),
      Float32Type::get(ctx));
}

// Split `count` little-endian bytes out of a packed i32 `word` and append them
// (as i8) to `results`.
void appendBytesFromWord(TritonLLVMOpBuilder &b, Value word, int count,
                         SmallVectorImpl<Value> &results) {
  Type i8 = IntegerType::get(word.getContext(), 8);
  for (int i = 0; i < count; ++i) {
    Value shifted =
        i == 0 ? word : b.lshr(word, b.i32_val(static_cast<int64_t>(8 * i)));
    results.push_back(b.trunc(i8, shifted));
  }
}

// Map each output byte (register index) to the index of the scale register
// that applies to it. Compact scales are shared by several output bytes, so the
// returned vector may repeat scale indices.
template <typename OpT>
SmallVector<int> computeScaleRegisters(OpT op, int64_t numOutputValues) {
  MLIRContext *ctx = op.getContext();
  auto outputTy = op.getOutput().getType();
  auto scaleTy = op.getScale().getType();
  int64_t axis = op.getAxis();
  int64_t elementsPerScale =
      outputTy.getShape()[axis] / scaleTy.getShape()[axis];

  LinearLayout outputLL = triton::gpu::toLinearLayout(outputTy);
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");
  // unpackUniqueTensorElements drops registers that only broadcast an already
  // held scale value. Use the same compact register space for scale indexing.
  LinearLayout scaleLL =
      triton::gpu::toLinearLayout(scaleTy).removeZeroBasesAlongDim(kReg);

  auto outputToScaleLL =
      OpT::computeScaleLayout(outputLL, axis, elementsPerScale);
  assert(outputToScaleLL && "expected valid scale layout after verifier");
  LinearLayout outputRegToScaleReg = outputToScaleLL->invertAndCompose(scaleLL);

  SmallVector<int> scaleRegisters;
  scaleRegisters.reserve(numOutputValues);
  for (int i = 0; i < numOutputValues; ++i) {
    auto scaleCoord = outputRegToScaleReg.apply(
        {{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    auto regIt = llvm::find_if(
        scaleCoord, [&](const auto &dimVal) { return dimVal.first == kReg; });
    assert(regIt != scaleCoord.end() && "scale register mapping missing");
    scaleRegisters.push_back(regIt->second);
  }
  return scaleRegisters;
}

// Check that all values in [groupStart, groupStart + groupSize) map to the same
// scale register, which is required for each hardware conversion group.
LogicalResult verifyGroupUsesSingleScale(ConversionPatternRewriter &rewriter,
                                         Operation *op,
                                         ArrayRef<int> scaleRegisters,
                                         int64_t groupStart, int groupSize,
                                         StringRef failureMessage) {
  int scaleReg = scaleRegisters[groupStart];
  for (int j = 1; j < groupSize; ++j) {
    if (scaleRegisters[groupStart + j] != scaleReg)
      return rewriter.notifyMatchFailure(op, failureMessage);
  }
  return success();
}

struct ScaledDowncastFp4OpPattern
    : ConvertOpToLLVMPattern<amdgpu::ScaledDowncastFp4Op> {
  ScaledDowncastFp4OpPattern(const LLVMTypeConverter &converter,
                             const AMD::TargetInfo &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(amdgpu::ScaledDowncastFp4Op downcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!targetInfo.supportsHwScaledDowncast())
      return rewriter.notifyMatchFailure(
          downcastOp, "scaled FP4 downcast requires CDNA4 or CDNA5");

    auto loc = downcastOp.getLoc();
    // `input` is the unpacked tensor: two consecutive register elements along
    // `axis` (positions 2*byte and 2*byte+1) provide the low/high nibble of one
    // output byte, so there are twice as many input registers as output bytes.
    auto inputVals =
        unpackUniqueTensorElements(loc, adaptor.getInput(), rewriter);
    auto scaleVals =
        unpackUniqueTensorElements(loc, adaptor.getScale(), rewriter);
    assert(inputVals.size() % 2 == 0);
    int64_t numOutputBytes = inputVals.size() / 2;

    auto scaleRegisters = computeScaleRegisters(downcastOp, numOutputBytes);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> results;
    results.reserve(numOutputBytes);
    Type inputElemTy = inputVals.front().getType();

    if (targetInfo.supportsCvtPkScalePk8()) {
      // Work on groups of 4 bytes since each intrinsic packs 8 fp4 values.
      Value zero = isa<Float32Type>(inputElemTy)
                       ? b.f32_val(0.0)
                       : b.bitcast(b.i16_val(0), inputElemTy);
      for (int i = 0; i < numOutputBytes; i += 4) {
        // Note that groupSize is in bytes not in fp4 values.
        int groupSize = std::min<int>(4, numOutputBytes - i);
        if (failed(verifyGroupUsesSingleScale(
                rewriter, downcastOp, scaleRegisters, i, groupSize,
                "each gfx1250 pk8 conversion group must share one scale")))
          return failure();
        int scaleReg = scaleRegisters[i];

        // Gather the group's 8 unpacked values (low/high nibble per byte),
        // zero-padding the tail of an incomplete group.
        Value src = b.undef(vec_ty(inputElemTy, 8));
        for (int byte = 0; byte < 4; ++byte) {
          Value low = byte < groupSize ? inputVals[2 * (i + byte)] : zero;
          Value high = byte < groupSize ? inputVals[2 * (i + byte) + 1] : zero;
          src = b.insert_element(src, low, b.i32_val(2 * byte));
          src = b.insert_element(src, high, b.i32_val(2 * byte + 1));
        }
        Value scaleF32 = scaleToF32(b, scaleVals[scaleReg]);
        Value packed;
        if (isa<Float32Type>(inputElemTy))
          packed = ROCDL::CvtScaleF32Pk8Fp4F32Op::create(rewriter, loc, i32_ty,
                                                         src, scaleF32);
        else if (isa<Float16Type>(inputElemTy))
          packed = ROCDL::CvtScaleF32Pk8Fp4F16Op::create(rewriter, loc, i32_ty,
                                                         src, scaleF32);
        else
          packed = ROCDL::CvtScaleF32Pk8Fp4Bf16Op::create(rewriter, loc, i32_ty,
                                                          src, scaleF32);
        // Split the packed i32 back into individual output bytes.
        appendBytesFromWord(b, packed, groupSize, results);
      }
    } else {
      // The pk(2) intrinsic scales and packs one value pair (2 fp inputs) into
      // one byte of an i32 result. `byteSel` selects the byte to store into.
      for (int i = 0; i < numOutputBytes; i += 4) {
        int groupSize = std::min<int>(4, numOutputBytes - i);
        Value packed = b.i32_val(0);
        for (int byteSel = 0; byteSel < groupSize; ++byteSel) {
          int idx = i + byteSel;
          Value low = inputVals[2 * idx];
          Value high = inputVals[2 * idx + 1];
          Value scaleF32 = scaleToF32(b, scaleVals[scaleRegisters[idx]]);
          if (isa<Float32Type>(inputElemTy)) {
            packed = ROCDL::CvtScaleF32PkFp4F32Op::create(
                rewriter, loc, i32_ty, packed, low, high, scaleF32, byteSel);
          } else {
            Value src = b.undef(vec_ty(inputElemTy, 2));
            src = b.insert_element(src, low, b.i32_val(0));
            src = b.insert_element(src, high, b.i32_val(1));
            if (isa<Float16Type>(inputElemTy))
              packed = ROCDL::CvtScaleF32PkFp4F16Op::create(
                  rewriter, loc, i32_ty, packed, src, scaleF32, byteSel);
            else
              packed = ROCDL::CvtScaleF32PkFp4Bf16Op::create(
                  rewriter, loc, i32_ty, packed, src, scaleF32, byteSel);
          }
        }
        appendBytesFromWord(b, packed, groupSize, results);
      }
    }

    Value result = packUniqueTensorElements(loc, getTypeConverter(), results,
                                            rewriter, downcastOp.getType());
    rewriter.replaceOp(downcastOp, result);
    return success();
  }

  const AMD::TargetInfo &targetInfo;
};

struct ScaledDowncastFp8OpPattern
    : ConvertOpToLLVMPattern<amdgpu::ScaledDowncastFp8Op> {
  ScaledDowncastFp8OpPattern(const LLVMTypeConverter &converter,
                             const AMD::TargetInfo &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(amdgpu::ScaledDowncastFp8Op downcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!targetInfo.supportsHwScaledDowncast())
      return rewriter.notifyMatchFailure(
          downcastOp, "scaled FP8 downcast requires CDNA4 or CDNA5");

    auto loc = downcastOp.getLoc();
    // fp8 downcast is elementwise: input and output share the same shape and
    // encoding, so the number of output fp8 bytes equals the number of input
    // values. The scale may be compact (one raw E8M0 byte shared by several
    // consecutive values along `axis`), so map each output value to its scale
    // register.
    auto inputVals =
        unpackUniqueTensorElements(loc, adaptor.getInput(), rewriter);
    auto scaleVals =
        unpackUniqueTensorElements(loc, adaptor.getScale(), rewriter);
    int64_t numOutputBytes = inputVals.size();
    auto scaleRegisters = computeScaleRegisters(downcastOp, numOutputBytes);

    bool isE4M3 = isa<Float8E4M3FNType>(
        downcastOp.getOutput().getType().getElementType());
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> results;
    results.reserve(numOutputBytes);
    Type inputElemTy = inputVals.front().getType();
    Value zero = isa<Float32Type>(inputElemTy)
                     ? b.f32_val(0.0)
                     : b.bitcast(b.i16_val(0), inputElemTy);

    if (targetInfo.supportsCvtPkScalePk8()) {
      // Work on groups of 8 bytes since each intrinsic packs 8 fp8 values.
      Type v2i32Ty = vec_ty(i32_ty, 2);
      for (int i = 0; i < numOutputBytes; i += 8) {
        int groupSize = std::min<int>(8, numOutputBytes - i);
        if (failed(verifyGroupUsesSingleScale(
                rewriter, downcastOp, scaleRegisters, i, groupSize,
                "each gfx1250 pk8 conversion group must share one scale")))
          return failure();
        int scaleReg = scaleRegisters[i];
        // Gather the group's 8 values, zero-padding an incomplete tail.
        Value src = b.undef(vec_ty(inputElemTy, 8));
        for (int j = 0; j < 8; ++j) {
          Value v = j < groupSize ? inputVals[i + j] : zero;
          src = b.insert_element(src, v, b.i32_val(j));
        }
        Value scaleF32 = scaleToF32(b, scaleVals[scaleReg]);
        Value packed;
        if (isa<Float32Type>(inputElemTy)) {
          if (isE4M3)
            packed = ROCDL::CvtScaleF32Pk8Fp8F32Op::create(
                rewriter, loc, v2i32Ty, src, scaleF32);
          else
            packed = ROCDL::CvtScaleF32Pk8Bf8F32Op::create(
                rewriter, loc, v2i32Ty, src, scaleF32);
        } else if (isa<Float16Type>(inputElemTy)) {
          if (isE4M3)
            packed = ROCDL::CvtScaleF32Pk8Fp8F16Op::create(
                rewriter, loc, v2i32Ty, src, scaleF32);
          else
            packed = ROCDL::CvtScaleF32Pk8Bf8F16Op::create(
                rewriter, loc, v2i32Ty, src, scaleF32);
        } else {
          if (isE4M3)
            packed = ROCDL::CvtScaleF32Pk8Fp8Bf16Op::create(
                rewriter, loc, v2i32Ty, src, scaleF32);
          else
            packed = ROCDL::CvtScaleF32Pk8Bf8Bf16Op::create(
                rewriter, loc, v2i32Ty, src, scaleF32);
        }
        // Extract each fp8 byte from the two packed i32 words (4 bytes each).
        Value lo = b.extract_element(packed, b.i32_val(0));
        Value hi = b.extract_element(packed, b.i32_val(1));
        appendBytesFromWord(b, lo, std::min(groupSize, 4), results);
        if (groupSize > 4)
          appendBytesFromWord(b, hi, groupSize - 4, results);
      }
    } else {
      // The pk intrinsic scales one value pair and writes the resulting 2 fp8
      // bytes into one 16-bit lane (`pair`) of a 2xi16 accumulator.
      Type v2i16Ty = vec_ty(i16_ty, 2);
      for (int i = 0; i < numOutputBytes; i += 4) {
        int groupSize = std::min<int>(4, numOutputBytes - i);
        Value acc = b.undef(v2i16Ty);
        int numPairs = (groupSize + 1) / 2;
        for (int pair = 0; pair < numPairs; ++pair) {
          int base = i + 2 * pair;
          // The intrinsic applies a single scale to the pair, so both elements
          // must map to one scale register.
          int scaleReg = scaleRegisters[base];
          if (base + 1 < i + groupSize && scaleRegisters[base + 1] != scaleReg)
            return rewriter.notifyMatchFailure(
                downcastOp,
                "each CDNA4 pk conversion pair must share one scale");
          Value scaleF32 = scaleToF32(b, scaleVals[scaleReg]);
          Value e0 = inputVals[base];
          Value e1 = base + 1 < i + groupSize ? inputVals[base + 1] : zero;
          if (isa<Float32Type>(inputElemTy)) {
            if (isE4M3)
              acc = ROCDL::CvtScaleF32PkFp8F32Op::create(
                  rewriter, loc, v2i16Ty, acc, e0, e1, scaleF32, pair);
            else
              acc = ROCDL::CvtScaleF32PkBf8F32Op::create(
                  rewriter, loc, v2i16Ty, acc, e0, e1, scaleF32, pair);
          } else {
            Value s = b.undef(vec_ty(inputElemTy, 2));
            s = b.insert_element(s, e0, b.i32_val(0));
            s = b.insert_element(s, e1, b.i32_val(1));
            if (isa<Float16Type>(inputElemTy)) {
              if (isE4M3)
                acc = ROCDL::CvtScaleF32PkFp8F16Op::create(
                    rewriter, loc, v2i16Ty, acc, s, scaleF32, pair);
              else
                acc = ROCDL::CvtScaleF32PkBf8F16Op::create(
                    rewriter, loc, v2i16Ty, acc, s, scaleF32, pair);
            } else {
              if (isE4M3)
                acc = ROCDL::CvtScaleF32PkFp8Bf16Op::create(
                    rewriter, loc, v2i16Ty, acc, s, scaleF32, pair);
              else
                acc = ROCDL::CvtScaleF32PkBf8Bf16Op::create(
                    rewriter, loc, v2i16Ty, acc, s, scaleF32, pair);
            }
          }
        }
        // Extract each fp8 byte from the packed accumulator.
        appendBytesFromWord(b, b.bitcast(acc, i32_ty), groupSize, results);
      }
    }

    Value result = packUniqueTensorElements(loc, getTypeConverter(), results,
                                            rewriter, downcastOp.getType());
    rewriter.replaceOp(downcastOp, result);
    return success();
  }

  const AMD::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::AMD::populateScaledDowncastOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const AMD::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<ScaledDowncastFp4OpPattern>(typeConverter, targetInfo, benefit);
  patterns.add<ScaledDowncastFp8OpPattern>(typeConverter, targetInfo, benefit);
}
