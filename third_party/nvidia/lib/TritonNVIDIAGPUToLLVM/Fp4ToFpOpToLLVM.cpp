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

namespace {
class Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
public:
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto tyX = cast<RankedTensorType>(op->getOperandTypes()[0]);
    auto operands = adaptor.getOperands();

    auto xVals = unpackLLElements(loc, operands[0], rewriter);

    SmallVector<Value> results;
    MLIRContext *ctx = rewriter.getContext();
    assert(xVals.size() % 4 == 0);
    for (int i = 0; i < xVals.size(); i += 4) {
      Value v0 = xVals[i];
      Value v1 = xVals[i + 1];
      Value v2 = xVals[i + 2];
      Value v3 = xVals[i + 3];
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

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::NVIDIA::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
