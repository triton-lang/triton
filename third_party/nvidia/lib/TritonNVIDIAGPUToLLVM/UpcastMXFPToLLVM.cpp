#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"

#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <array>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Convert 8 fp4 elements packed into a 32bit reg into 8 bf16 elements packed
// into 4 32bits regs.
static constexpr const char *FP4ToBF16Ptx =
    "{\n"
    ".reg .b32 a<14>;\n"
    "and.b32  	a0, $4, -2004318072;\n\t"
    "shr.u32 	a1, a0, 3;\n\t"
    "and.b32  	a2, $4, 2004318071;\n\t"
    "shr.u32 	a3, a2, 16;\n\t"
    "shr.u32 	a4, a0, 19;\n\t"
    "prmt.b32 a5, -1065353216, -1065336832, a2;\n\t"
    "prmt.b32 a6, -1065353216, -1065336832, a3;\n\t"
    "prmt.b32 a7, 1061109504, 1077952576, a2;\n\t"
    "prmt.b32 a8, 1061109504, 1077952576, a3;\n\t"
    "prmt.b32 a9, 32768, 0, a1;\n\t"
    "prmt.b32 a10, 32768, 0, a4;\n\t"
    "or.b32  	a11, a7, a9;\n\t"
    "or.b32  	a12, a8, a10;\n\t"
    "prmt.b32 $0, a5, a11, 20800;\n\t"
    "prmt.b32 $1, a5, a11, 29538;\n\t"
    "prmt.b32 $2, a6, a12, 20800;\n\t"
    "prmt.b32 $3, a6, a12, 29538;\n\t"
    "}";

static constexpr const char *FP4ToFP16Ptx =
    "{\n"
    ".reg .b32           a<11>;\n"
    ".reg .b16           t<4>;\n"
    "and.b32             a0, $4, 0x77777777;\n\t"
    "and.b32             a1, $4, 0x88888888;\n\t"
    "shr.u32             a2, a1, 3;\n\t"
    "shr.u32             a3, a0, 16;\n\t"
    "shr.u32             a4, a2, 16;\n\t"
    "prmt.b32            a5, 0x3C383000, 0x4C484440, a0;\n"
    "prmt.b32            a6, 0x3C383000, 0x4C484440, a3;\n"
    "prmt.b32            a7, 0x00008000, 0x0, a2;\n"
    "prmt.b32            a8, 0x00008000, 0x0, a4;\n"
    "or.b32              a9, a5, a7;\n\t"
    "or.b32              a10, a6, a8;\n\t"
    "mov.b32             {t0, t1}, a9;\n"
    "mov.b32             {t2, t3}, a10;\n"
    "cvt.rn.f16x2.e4m3x2 $0, t0;\n"
    "cvt.rn.f16x2.e4m3x2 $1, t1;\n"
    "cvt.rn.f16x2.e4m3x2 $2, t2;\n"
    "cvt.rn.f16x2.e4m3x2 $3, t3;\n"
    "}";

static Value createInlineAsmUpcast(Location loc, RewriterBase &rewriter,
                                   Type retType, Value packedVec,
                                   const char *ptxAsm) {
  PTXBuilder builder;
  SmallVector<PTXBuilder::Operand *> operands;
  for (int i = 0; i < 4; i++) {
    operands.push_back(builder.newOperand("=r"));
  }
  operands.push_back(builder.newOperand(packedVec, "r"));
  auto &ptxOp = *builder.create(ptxAsm);
  ptxOp(operands, /*onlyAttachMLIRArgs=*/true);
  Value result = builder.launch(rewriter, loc, retType, false);
  return result;
}

static SmallVector<Value> convertFP4x2To16x2(RewriterBase &rewriter,
                                             Location loc, Type targetTy,
                                             ArrayRef<Value> values) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> results;
  MLIRContext *ctx = rewriter.getContext();
  bool isFP16 = targetTy == f16_ty;
  bool isBF16 = targetTy == bf16_ty;
  assert(isFP16 || isBF16);
  assert(values.size() % 4 == 0);
  for (int i = 0; i < values.size(); i += 4) {
    Value v0 = values[i];
    Value v1 = values[i + 1];
    Value v2 = values[i + 2];
    Value v3 = values[i + 3];
    Value packedVec = b.undef(vec_ty(i8_ty, 4));
    packedVec = b.insert_element(packedVec, v0, b.i32_val(0));
    packedVec = b.insert_element(packedVec, v1, b.i32_val(1));
    packedVec = b.insert_element(packedVec, v2, b.i32_val(2));
    packedVec = b.insert_element(packedVec, v3, b.i32_val(3));
    SmallVector<Type> rets(4, i32_ty);
    Type retType = struct_ty(rets);
    const char *upcastPtx = isFP16 ? FP4ToFP16Ptx : FP4ToBF16Ptx;
    Value ret =
        createInlineAsmUpcast(loc, rewriter, retType, packedVec, upcastPtx);
    for (int i = 0; i < 4; i++) {
      Value extractI32 = b.extract_val(ret, i);
      Value vecbf16 = b.bitcast(extractI32, vec_ty(targetTy, 2));
      results.push_back(b.extract_element(vecbf16, b.i32_val(0)));
      results.push_back(b.extract_element(vecbf16, b.i32_val(1)));
    }
  }
  return results;
}

Value mxfpScale(RewriterBase &rewriter, Location loc, Value v, Value scale,
                Type fp_ty, bool fastMath) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value scaleFP;
  if (fp_ty == bf16_ty) {
    scaleFP = b.bitcast(b.shl(b.zext(i16_ty, scale), b.i16_val(7)), fp_ty);
  } else {
    assert(fp_ty == f16_ty);
    scaleFP = b.bitcast(b.shl(b.zext(i32_ty, scale), b.i32_val(23)),
                        rewriter.getF32Type());
    scaleFP = b.fptrunc(fp_ty, scaleFP);
  }
  Value scaledV = b.fmul(b.bitcast(v, fp_ty), scaleFP);
  if (fastMath)
    return scaledV;
  // Account for NaN in the scale as per the mxfp specification.
  Value scaleIsNan = b.icmp_eq(scale, b.i8_val(0xff));
  return b.select(scaleIsNan, b.bitcast(b.i16_val(0x7fff), fp_ty), scaledV);
};

namespace {
class UpcastMXFPOpPattern : public ConvertOpToLLVMPattern<UpcastMXFPOp> {
private:
  const TargetInfoBase &targetInfo;

public:
  UpcastMXFPOpPattern(LLVMTypeConverter &typeConverter,
                      const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<UpcastMXFPOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(UpcastMXFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto tyX = cast<RankedTensorType>(op->getOperandTypes()[0]);
    auto operands = adaptor.getOperands();

    auto xVals = unpackLLElements(loc, operands[0], rewriter);
    auto scaleVals = unpackLLElements(loc, operands[1], rewriter);
    auto fpType = op.getFpType();
    auto outType = op.getType().getElementType();

    Value tid = b.tid_val();
    auto mod = op->getParentOfType<ModuleOp>();
    Value warpSize =
        b.i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpId = b.udiv(tid, warpSize);
    Value laneId = b.urem(tid, warpSize);

    auto kWidth =
        cast<DotOperandEncodingAttr>(op.getType().getEncoding()).getKWidth();

    if (fpType == ScaleDotElemType::E2M1)
      xVals = convertFP4x2To16x2(rewriter, loc, outType, xVals);

    // Each thread owns elements of 4 mxfp vectors so we need 4 scales
    // Since we go from a threadShape of 8x4 to 16x2, we let c = tid / 4 * 2
    // Then, we need elements c and c + 16 for the first two mxfp vectors
    // and elements c + 1 and c + 17 for the last two mxfp vectors
    auto c = b.mul(b.udiv(laneId, b.i32_val(4)), b.i32_val(2));
    std::array<Value, 4> ci = {c, b.add(c, b.i32_val(16)),
                               b.add(c, b.i32_val(1)), b.add(c, b.i32_val(17))};

    // TODO Move this logic to using LinearLayouts
    // Each scale in a warp has to be replicated to cover a tile of shape mxk =
    // 16x64 This 16x64 tile is split into 4 subtiles of shape 8x32, each of
    // which will have to gather a scale and multiply its relevant part of the
    // mxfp vector This tile of 8x32 is split in to 8x4 vectors, leaving each
    // vector with 1x8 mxfp elements as long as kWidth * 4 <= 32
    assert(kWidth <= 8 &&
           "NYI for larger kWidth (but we could do it with less shuffles!)");
    for (auto [i, scaleVal] : llvm::enumerate(scaleVals)) {
      for (int mxfp = 0; mxfp < 2; ++mxfp) {
        auto si = std::array<Value, 2>{
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[mxfp * 2 + 0]),
            targetInfo.shuffleIdx(rewriter, loc, scaleVal, ci[mxfp * 2 + 1])};
        for (int rep = 0; rep < 8 / kWidth; ++rep) {
          for (int subTile = 0; subTile < 2; ++subTile) {
            for (int k = 0; k < kWidth; ++k) {
              auto idx =
                  32 * i + 16 * mxfp + rep * 2 * kWidth + subTile * kWidth + k;
              xVals[idx] = mxfpScale(rewriter, loc, xVals[idx], si[subTile],
                                     outType, op.getFastMath());
            }
          }
        }
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::NVIDIA::populateUpcastMXFPToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<UpcastMXFPOpPattern>(typeConverter, targetInfo, benefit);
}
