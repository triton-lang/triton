#include "PatternTritonGPUOpToLLVM.h"

#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Fp4ToFpOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Convert 8 fp4 elements packed into a 32bit reg into 8 bf16 elements packed
// into 4 32bits regs.
static constexpr const char *FP4ToBP16Ptx =
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
                                   bool toFp16, Type retType, Value packedVec) {
  PTXBuilder builder;
  SmallVector<PTXBuilder::Operand *> operands;
  for (int i = 0; i < 4; i++) {
    operands.push_back(builder.newOperand("=r"));
  }
  operands.push_back(builder.newOperand(packedVec, "r"));
  auto &ptxOp = *builder.create(toFp16 ? FP4ToFP16Ptx : FP4ToBP16Ptx);
  ptxOp(operands, /*onlyAttachMLIRArgs=*/true);
  Value result = builder.launch(rewriter, loc, retType, false);
  return result;
}

namespace {
class Fp4ToFpOpPattern : public Fp4ToFpOpConversionBase {
public:
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : Fp4ToFpOpConversionBase(typeConverter, benefit) {}

protected:
  std::array<Value, 8> upcastPackedFp4(Fp4ToFpOp op,
                                       ConversionPatternRewriter &rewriter,
                                       Value packedVec,
                                       Type elemType) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    bool toFp16 = elemType == f16_ty;
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Type> rets(4, i32_ty);
    Type retType = struct_ty(rets);
    Value ret =
        createInlineAsmUpcast(loc, rewriter, toFp16, retType, packedVec);
    std::array<Value, 8> results;
    for (int i = 0; i < 4; i++) {
      Value extractI32 = b.extract_val(ret, i);
      Value elements = b.bitcast(extractI32, vec_ty(elemType, 2));
      results[2 * i] = b.extract_element(elements, b.i32_val(0));
      results[2 * i + 1] = b.extract_element(elements, b.i32_val(1));
    }
    return results;
  }
};
} // anonymous namespace

void mlir::triton::NVIDIA::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
