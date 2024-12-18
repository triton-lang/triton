#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <array>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
SmallVector<Value, 4> upcast8xMxfp4(RewriterBase &rewriter, Fp4ToFpOp op,
                                    bool tofp16, Value packedVec) {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // MXFP4 has 4 bits, S.EE.M, for Sign, Exponent, and Mantissa respectively.
  // For a specific S, we have a total of 8 bit patterns. We can encode all
  // these 8 resultant bf16/fp16 bit patterns in a lookup table (LUT). It
  // happens that llvm.amdgcn.perm supports selecting 4 bytes from 8 input bytes
  // using a 4-byte selector. So the overall idea is to use llvm.amdgcn.perm to
  // implement such a LUT; though we need to select the two bytes for the
  // resultant bf16/fp16 bit patterns separately. For the byte containing S, we
  // also need to handle the S and E bits separately.

  // FP4 has 4 bits: S.EE.M. Bf16/fp16 bit patterns for positive values:
  //
  // FP4    | BF16   | FP16   | Value
  // ------ | ------ | ------ | -----
  // 0.00.0 | 0x0000 | 0x0000 | + 0.0
  // 0.00.1 | 0x3f00 | 0x3800 | + 0.5
  // 0.01.0 | 0x3f80 | 0x3c00 | + 1.0
  // 0.01.1 | 0x3fc0 | 0x3e00 | + 1.5
  // 0.10.0 | 0x4000 | 0x4000 | + 2.0
  // 0.10.1 | 0x4040 | 0x4200 | + 3.0
  // 0.11.0 | 0x4080 | 0x4400 | + 4.0
  // 0.11.1 | 0x40c0 | 0x4600 | + 6.0
  //
  // Encode Byte #0 (M) for BF16/FP16 in a LUT.
  Value resB0LutLo = tofp16 ? b.i32_val(0) : b.i32_val(0xc0800000);
  Value resB0LutHi = tofp16 ? b.i32_val(0) : b.i32_val(0xc0804000);
  // Encode Byte #1 (EM, non-S part) for BF16/FP16 in a LUT.
  Value resB1LutLoNoS = tofp16 ? b.i32_val(0x3e3c3800) : b.i32_val(0x3f3f3f00);
  Value resB1LutHiNoS = tofp16 ? b.i32_val(0x46444240) : b.i32_val(0x40404040);

  Type i32Ty = rewriter.getI32Type();
  auto permU32FnTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty, i32Ty});
  LLVM::LLVMFuncOp funcOp =
      appendOrGetExternFuncOp(rewriter, op, "llvm.amdgcn.perm", permU32FnTy);

  // Start with 8 mxfp4 elements in a single i32 register
  // | e7e6 | e5e4 | e3e2 | e1e0 |
  Value input = b.bitcast(packedVec, i32Ty);

  // Step 1: extract EM bits for elements 0,2,4,6 and 1,3,5,7 respectively.
  // e2m1_6420_idx = | 0[0e6EM] | 0[0e4EM] | 0[0e2EM] | 0[0e0EM] |
  Value e2m1_6420_idx = b.and_(input, b.i32_val(0x07070707));
  // e2m1_7531_idx = | [0e7EM]0 | [0e5EM]0 | [0e3EM]0 | [0e1EM]0 |
  Value e2m1_7531_idx = b.and_(input, b.i32_val(0x70707070));
  // e2m1_7531_idx = | 0[0e7EM] | 0[0e5EM] | 0[0e3EM] | 0[0e1EM] |
  e2m1_7531_idx = b.lshr(e2m1_7531_idx, b.i32_val(4));

  // Step 2: extract S bit for elements 0,2,4,6 and 1,3,5,7
  // s_6420 = | 0[e6S000] | 0[e4S000] | 0[e2S000] | 0[e0S000] |
  Value s_6420 = b.and_(input, b.i32_val(0x08080808));
  // s_6420 = | [e6S000]0 | [e4S000]0 | [e2S000]0 | [e0S000]0 |
  s_6420 = b.shl(s_6420, b.i32_val(4));
  // s_7531 = | [e7S000]0 | [e5S000]0 | [e3S000]0 | [e1S000]0 |
  Value s_7531 = b.and_(input, b.i32_val(0x80808080));

  // Step 3: Upcast elements 0,2,4,6 to 4 16-bit elements
  // Select Byte #0. It's always 0 if upcasting to fp16.
  // resB0_6420 = | e6B0 | e4B0 | e2B0 | e0B0 |
  Value resB0_6420 = b.i32_val(0);
  if (!tofp16) {
    resB0_6420 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {resB0LutHi, resB0LutLo, e2m1_6420_idx})
                     .getResult();
  }
  // Select Byte #1
  Value resB1NoS_6420 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1LutHiNoS, resB1LutLoNoS, e2m1_6420_idx})
          .getResult();
  // resB1_6420 = | e6B1 | e4B1 | e2B1 | e0B1 |
  Value resB1_6420 = b.or_(resB1NoS_6420, s_6420);
  // Construct 16-bit values of e0 and e2
  // res_20 = | e2B1 | e2B0 | e0B1 | e0B0 | = | e2_f16 | e0_f16 |
  Value res_20 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_6420, resB0_6420, b.i32_val(0x05010400)})
          .getResult();
  // Construct 16-bit values of e4 and e6
  // res_64 = | e6B1 | e6B0 | e4B1 | e4B0 | = | e6_f16 | e4_f16 |
  Value res_64 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_6420, resB0_6420, b.i32_val(0x07030602)})
          .getResult();

  // Step 4: Upcast elements 1,3,5,7 to 4 16-bit elements
  // This is a copy of step 3 on different group of elements
  // Select Byte #0. It's always 0 if upcasting to fp16.
  // resB0_7531 = | e7B0 | e5B0 | e3B0 | e1B0 |
  Value resB0_7531 = b.i32_val(0);
  if (!tofp16) {
    resB0_7531 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {resB0LutHi, resB0LutLo, e2m1_7531_idx})
                     .getResult();
  }
  // Select Byte #1
  Value resB1NoS_7531 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1LutHiNoS, resB1LutLoNoS, e2m1_7531_idx})
          .getResult();
  // resB1_7531 = | e7B1 | e5B1 | e3B1 | e1B1 |
  Value resB1_7531 = b.or_(resB1NoS_7531, s_7531);
  // Construct 16-bit values of e1 and e3
  // res_31 = | e3B1 | e3B0 | e1B1 | e1B0 | = | e3_f16 | e1_f16 |
  Value res_31 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_7531, resB0_7531, b.i32_val(0x05010400)})
          .getResult();
  // Construct 16-bit values of e5 and e7
  // res_75 = | e7B1 | e7B0 | e5B1 | e5B0 | = | e7_f16 | e5_f16 |
  Value res_75 =
      LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                             {resB1_7531, resB0_7531, b.i32_val(0x07030602)})
          .getResult();

  // Step 5: Reorder 16-bit elements to be 0,1,2,3,4,5,6,7
  // res_10 = | e1_f16 | e0_f16 |
  Value res_10 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_31, res_20, b.i32_val(0x05040100)})
                     .getResult();
  // res_32 = | e3_f16 | e2_f16 |
  Value res_32 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_31, res_20, b.i32_val(0x07060302)})
                     .getResult();
  // res_54 = | e5_f16 | e4_f16 |
  Value res_54 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_75, res_64, b.i32_val(0x05040100)})
                     .getResult();
  // res_76 = | e7_f16 | e6_f16 |
  Value res_76 = LLVM::createLLVMCallOp(rewriter, loc, funcOp,
                                        {res_75, res_64, b.i32_val(0x07060302)})
                     .getResult();

  return {res_10, res_32, res_54, res_76};
}
class Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
public:
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto elemType = op.getType().getElementType();
    assert(elemType == f16_ty || elemType == bf16_ty);
    bool toFp16 = elemType == f16_ty;

    auto xVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    SmallVector<Value> results;
    results.reserve(xVals.size() * 2);
    assert(xVals.size() % 4 == 0);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (int i = 0; i < xVals.size(); i += 4) {
      Value v0 = xVals[i];
      Value v1 = xVals[i + 1];
      Value v2 = xVals[i + 2];
      Value v3 = xVals[i + 3];
      Value packedVec = b.undef(vec_ty(i8_ty, 4));
      packedVec = b.insert_element(packedVec, v0, b.i32_val(0));
      packedVec = b.insert_element(packedVec, v1, b.i32_val(1));
      packedVec = b.insert_element(packedVec, v2, b.i32_val(2));
      packedVec = b.insert_element(packedVec, v3, b.i32_val(3));
      SmallVector<Value, 4> v4i32 =
          upcast8xMxfp4(rewriter, op, toFp16, packedVec);
      for (int j = 0; j < 4; j++) {
        Value elements = b.bitcast(v4i32[j], vec_ty(elemType, 2));
        results.push_back(b.extract_element(elements, b.i32_val(0)));
        results.push_back(b.extract_element(elements, b.i32_val(1)));
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), xVals, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::AMD::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
