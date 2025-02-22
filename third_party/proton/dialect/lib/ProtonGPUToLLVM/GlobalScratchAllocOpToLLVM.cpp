#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/PatternProtonOpToLLVM.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/Utility.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "third_party/proton/dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

namespace mlir {
FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &converter);
}

namespace {
using namespace mlir;
using namespace mlir::triton;

triton::FuncOp amendFuncOp(LLVM::LLVMFuncOp funcOp,
                           ConversionPatternRewriter &rewriter,
                           const TargetInfoBase &targetInfo) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  Location loc = moduleOp->getLoc();
  auto ctx = funcOp->getContext();
  auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, 1);
  auto funcTy = funcOp.getFunctionType();
  auto amendedInputTy = llvm::to_vector(funcOp.getArgumentTypes());
  unsigned oldNumArgs = amendedInputTy.size();
  amendedInputTy.push_back(globalPtrTy);
  auto amendedFuncTy =
      FunctionType::get(ctx, amendedInputTy, funcOp.getResultTypes());
  auto amendedFuncOp = rewriter.create<triton::FuncOp>(
      funcOp.getLoc(), funcOp.getName(), amendedFuncTy);
  auto &region = funcOp.getBody();
  region.addArgument(globalPtrTy, amendedFuncOp.getLoc());
  rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                              amendedFuncOp.end());
  IRMapping mapper;
  if (auto argAttrs = funcOp.getAllArgAttrs()) {
    SmallVector<Attribute> newArgAttrs;
    newArgAttrs.reserve(amendedInputTy.size());
    for (unsigned i = 0; i != oldNumArgs; ++i)
      if (!mapper.contains(funcOp.getArgument(i)))
        newArgAttrs.push_back(argAttrs[i]);
    amendedFuncOp.setAllArgAttrs(newArgAttrs);
  }
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

    rewriter.eraseOp(op);
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto amendedFuncOp = amendFuncOp(funcOp, rewriter, targetInfo);
    FailureOr<LLVM::LLVMFuncOp> maybeNewFuncOp =
        mlir::convertFuncOpToLLVMFuncOp(amendedFuncOp, rewriter,
                                        *getTypeConverter());
    if (failed(maybeNewFuncOp)) {
      return failure();
    }

    LLVM::LLVMFuncOp newFuncOp = *maybeNewFuncOp;
    auto ctx = funcOp->getContext();
    newFuncOp->setAttr("nvvm.kernel",
                       rewriter.getIntegerAttr(type::u1Ty(ctx), 1));
    newFuncOp.setLinkage(LLVM::Linkage::External);
    rewriter.eraseOp(funcOp);
    rewriter.eraseOp(amendedFuncOp);
    rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
    auto loc = amendedFuncOp.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto gmemBase = newFuncOp.getArgument(newFuncOp.getNumArguments() - 1);
    // TODO: implement this offset value
    auto opOffset = 0;
    auto llvmPointerType = LLVM::LLVMPointerType::get(ctx);
    rewriter.create<LLVM::GEPOp>(loc, llvmPointerType, llvmPointerType,
                                 gmemBase, b.i32_val(opOffset));
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::proton::populateGlobalScratchAllocOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, targetInfo,
                                               benefit);
}
