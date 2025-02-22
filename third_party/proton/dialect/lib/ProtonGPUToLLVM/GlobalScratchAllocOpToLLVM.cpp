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

triton::FuncOp amendFuncOp(LLVM::LLVMFuncOp func,
                           ConversionPatternRewriter &rewriter,
                           const TargetInfoBase &targetInfo) {
    auto loc = func.getLoc();
    auto ctx = func->getContext();
    auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);
    auto funcTy = func.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(func.getArgumentTypes());
    amendedInputTy.push_back(globalPtrTy);
    auto amendedFuncTy =
        FunctionType::get(ctx, amendedInputTy, func.getResultTypes());

    SmallVector<NamedAttribute> amendedAttrs;
    
//    filterFuncAttributes(func, /*filterArgAttrs=*/true, amendedAttrs);
//    if (auto argAttrs = func.getAllArgAttrs()) {
//      llvm::SmallVector<mlir::Attribute> amendedArgAttrs(argAttrs.begin(),
//                                                         argAttrs.end());
//      while (amendedArgAttrs.size() < amendedInputTy.size()) {
//        amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
//      }
//      amendedAttrs.push_back(
//          rewriter.getNamedAttr(func.getArgAttrsAttrName(),
//                                rewriter.getArrayAttr(amendedArgAttrs)));
//    }
    auto amendedFuncOp = rewriter.create<triton::FuncOp >(
        func.getLoc(), func.getName(), amendedFuncTy);
    auto &region = func.getBody();
    region.addArgument(globalPtrTy, loc);
    rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                                amendedFuncOp.end());
    return amendedFuncOp;
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
    auto m =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto func = *m.getOps<LLVM::LLVMFuncOp>().begin();    
    auto funcTy = func.getFunctionType();
    auto amendedFuncOp = amendFuncOp(func, rewriter, targetInfo);
    FailureOr<LLVM::LLVMFuncOp> maybeNewFuncOp =
        mlir::convertFuncOpToLLVMFuncOp(amendedFuncOp, rewriter,
                                        *getTypeConverter());    
    if (failed(maybeNewFuncOp)) {
      return failure();
    }

    LLVM::LLVMFuncOp newFuncOp = *maybeNewFuncOp;    
    auto amendedFuncTy = newFuncOp.getFunctionType();
    llvm::errs() << funcTy << "\n";
    llvm::errs() << amendedFuncTy << "\n";
//    llvm::errs() << m << "\n";

    rewriter.eraseOp(func);
//    rewriter.eraseOp(amendedFuncOp);
    rewriter.eraseOp(op);
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
