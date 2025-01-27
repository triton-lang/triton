#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

inline size_t getSourceSize(Value &source) {
  ArrayRef<Type> types = cast<LLVM::LLVMStructType>(source.getType()).getBody();
  return types.size();
}

struct ConcatOpConversion : public ConvertOpToLLVMPattern<amdgpu::ConcatOp> {
  explicit ConcatOpConversion(LLVMTypeConverter &typeConverter,
                              PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<amdgpu::ConcatOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(amdgpu::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());

    auto sources = adaptor.getSources();

    size_t totalNumElements = 0;
    for (auto source : sources) {
      totalNumElements += getSourceSize(source);
    }

    size_t currNumElements = 0;
    llvm::SmallVector<Value> resultVals(totalNumElements);
    for (auto source : sources) {
      auto elements = unpackLLElements(loc, source, rewriter);
      for (auto [idx, element] : llvm::enumerate(elements)) {
        resultVals[currNumElements + idx] = element;
      }
      currNumElements += getSourceSize(source);
    }

    Value ret = packLLElements(loc, this->getTypeConverter(), resultVals,
                               rewriter, resultTy);

    rewriter.replaceOp(op, ret);

    return llvm::success();
  }
};
} // namespace

namespace mlir::triton::AMD {
void populateConcatOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns,
                                    mlir::PatternBenefit benefit) {
  patterns.add<ConcatOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
