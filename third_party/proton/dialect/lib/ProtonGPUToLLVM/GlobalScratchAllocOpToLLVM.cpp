#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/PatternProtonOpToLLVM.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "third_party/proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

namespace mlir {
FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &converter);
}

namespace {

static void filterFuncAttributes(LLVM::LLVMFuncOp op, bool filterArgAttrs,
                                 SmallVectorImpl<NamedAttribute> &result) {

  for (const auto &attr : op->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == op.getFunctionTypeAttrName() ||
        attr.getName() == "std.varargs" ||
        (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
      continue;
    result.push_back(attr);
  }
}

struct GlobalScratchAllocOpConversion
   : public ConvertOpToLLVMPattern<proton::gpu::GlobalScratchAllocOp> {
	   explicit GlobalScratchAllocOpConversion(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<proton::gpu::GlobalScratchAllocOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(proton::gpu::GlobalScratchAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto loc = func.getLoc();
    auto ctx = func->getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto &region = func.getBody();
    auto gmemBase = region.addArgument(globalPtrTy, loc);    
//    auto gmemBase = func.getArgument(func.getNumArguments() - 1);
    //TODO: make this offset more meaningful
    Value offset = b.i32_val(0);
    Value bufferStart =
      b.gep(mlir::LLVM::LLVMPointerType::get(ctx, 1), i8_ty, gmemBase, offset);    
    rewriter.replaceOp(op, bufferStart);
    return success();
  }
protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::proton::populateGlobalScratchAllocOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, targetInfo, benefit);
}
