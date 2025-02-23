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
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/Utility.h"

namespace mlir {
FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &converter);
}

namespace {
using namespace mlir;
using namespace mlir::triton;

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

triton::FuncOp amendFuncOp(LLVM::LLVMFuncOp funcOp,
                           ConversionPatternRewriter &rewriter,
                           const TargetInfoBase &targetInfo) {
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);    
    auto funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector(funcOp.getArgumentTypes());
//    amendedInputTy.push_back(globalPtrTy);
    auto amendedFuncTy =
        FunctionType::get(ctx, amendedInputTy, funcOp.getResultTypes());
    SmallVector<NamedAttribute> amendedAttrs;    
//    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
//    if (auto argAttrs = funcOp.getAllArgAttrs()) {
//      llvm::SmallVector<mlir::Attribute> amendedArgAttrs(argAttrs.begin(),
//                                                         argAttrs.end());
//      while (amendedArgAttrs.size() < amendedInputTy.size()) {
//        amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
//      }
//      amendedAttrs.push_back(
//          rewriter.getNamedAttr(funcOp.getArgAttrsAttrName(),
//                                rewriter.getArrayAttr(amendedArgAttrs)));
//    }
    auto amendedFuncOp = rewriter.create<triton::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();
//    region.addArgument(globalPtrTy, loc);
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
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto amendedFuncOp = amendFuncOp(func, rewriter, targetInfo);
    FailureOr<LLVM::LLVMFuncOp> maybeNewFuncOp =
        mlir::convertFuncOpToLLVMFuncOp(amendedFuncOp, rewriter,
                                        *getTypeConverter());
    if (failed(maybeNewFuncOp)) {
      return failure();
    }

    LLVM::LLVMFuncOp newFuncOp = *maybeNewFuncOp;
    auto ctx = func->getContext();    
    newFuncOp->setAttr("nvvm.kernel",
                       rewriter.getIntegerAttr(type::u1Ty(ctx), 1));
    newFuncOp.setLinkage(LLVM::Linkage::External);    

    rewriter.eraseOp(func);
    IRRewriter newRewriter(newFuncOp->getContext());    
    newRewriter.eraseOp(op);
//    llvm::errs() << funcTy << "\n";
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
