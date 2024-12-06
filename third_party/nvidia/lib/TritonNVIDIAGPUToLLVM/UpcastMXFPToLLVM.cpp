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
static constexpr const char *ptxAsm =
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

static Value createInlineAsmUpcast(Location loc, RewriterBase &rewriter,
                                   Type retType, Value packedVec) {
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

static SmallVector<Value> convertMxfp4x2ToBf16x2PTX(RewriterBase &rewriter,
                                                    Location loc,
                                                    ArrayRef<Value> values) {
  SmallVector<Value> results;
  MLIRContext *ctx = rewriter.getContext();
  assert(values.size() % 4 == 0);
  for (int i = 0; i < values.size(); i += 4) {
    Value v0 = values[i];
    Value v1 = values[i + 1];
    Value v2 = values[i + 2];
    Value v3 = values[i + 3];
    Value packedVec = undef(vec_ty(i8_ty, 4));
    packedVec = insert_element(packedVec, v0, i32_val(0));
    packedVec = insert_element(packedVec, v1, i32_val(1));
    packedVec = insert_element(packedVec, v2, i32_val(2));
    packedVec = insert_element(packedVec, v3, i32_val(3));
    SmallVector<Type> rets(4, i32_ty);
    Type retType = struct_ty(rets);
    Value ret = createInlineAsmUpcast(loc, rewriter, retType, packedVec);
    for (int i = 0; i < 4; i++) {
      Value extractI32 = extract_val(ret, i);
      Value vecbf16 = bitcast(extractI32, vec_ty(bf16_ty, 2));
      results.push_back(extract_element(vecbf16, i32_val(0)));
      results.push_back(extract_element(vecbf16, i32_val(1)));
    }
  }
  return results;
}

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
    auto tyX = cast<RankedTensorType>(op->getOperandTypes()[0]);
    auto operands = adaptor.getOperands();

    auto xVals = unpackLLElements(loc, operands[0], rewriter);
    auto scaleVals = unpackLLElements(loc, operands[1], rewriter);
    auto fpType = op.getFpType();

    Value tid = tid_val();
    auto mod = op->getParentOfType<ModuleOp>();
    Value warpSize =
        i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpId = udiv(tid, warpSize);
    Value laneId = urem(tid, warpSize);

    auto kWidth =
        cast<DotOperandEncodingAttr>(op.getType().getEncoding()).getKWidth();

    if (fpType == ScaleDotElemType::E2M1)
      xVals = convertMxfp4x2ToBf16x2PTX(rewriter, loc, xVals);

    // Each thread owns elements of 4 mxfp vectors so we need 4 scales
    // Since we go from a threadShape of 8x4 to 16x2, we let c = tid / 4 * 2
    // Then, we need elements c and c + 16 for the first two mxfp vectors
    // and elements c + 1 and c + 17 for the last two mxfp vectors
    auto c = mul(udiv(laneId, i32_val(4)), i32_val(2));
    std::array<Value, 4> ci = {c, add(c, i32_val(16)), add(c, i32_val(1)),
                               add(c, i32_val(17))};

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
              xVals[idx] =
                  LLVM::mxfpScaleBf16(rewriter, loc, xVals[idx], si[subTile]);
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
