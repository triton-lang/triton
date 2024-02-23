#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir::triton::gpu;
namespace {

struct AddPtrOpConversion : public ConvertOpToLLVMPattern<AddPtrOp> {
  using ConvertOpToLLVMPattern<AddPtrOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto typeConverter = getTypeConverter();
    auto resultTensorTy = resultTy.dyn_cast<RankedTensorType>();
    if (resultTensorTy) {
      unsigned elems = getTotalElemsPerThread(resultTy);
      Type elemTy = typeConverter->convertType(
          resultTensorTy.getElementType().cast<PointerType>().getPointeeType());
      Type ptrTy = typeConverter->convertType(resultTensorTy.getElementType());
      auto ptrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
      auto offsets = unpackLLElements(loc, adaptor.getOffset(), rewriter);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(ptrTy, elemTy, ptrs[i], offsets[i]);
      }
      Value view =
          packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(resultTy.isa<PointerType>());
      auto resultPtrTy = typeConverter->convertType(resultTy);
      auto resultElemTy = typeConverter->convertType(
          resultTy.cast<PointerType>().getPointeeType());
      Value result =
          gep(resultPtrTy, resultElemTy, adaptor.getPtr(), adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

} // namespace
void mlir::triton::populateAddPtrOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
}
