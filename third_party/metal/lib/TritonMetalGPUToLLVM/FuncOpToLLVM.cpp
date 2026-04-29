

#include "PatternTritonGPUOpToLLVM.h"
#include "TritonMetalGPUToLLVM/MetalKernelArgs.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {
struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter,
                   const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = amendFuncOp(funcOp, rewriter, targetInfo);
    FailureOr<LLVM::LLVMFuncOp> maybeNewFuncOp =
        mlir::convertFuncOpToLLVMFuncOp(amendedFuncOp, rewriter,
                                        *getTypeConverter());
    if (failed(maybeNewFuncOp)) {
      return failure();
    }

    auto ctx = funcOp->getContext();
    LLVM::LLVMFuncOp newFuncOp = *maybeNewFuncOp;
    handleArgPtrDatatype(funcOp, newFuncOp);

    if (triton::isKernel(funcOp)) {
      newFuncOp.setLinkage(LLVM::Linkage::External);
      auto funcType = newFuncOp.getFunctionType();
      SmallVector<Type> params(funcType.getParams());
      SmallVector<Type> origParams(funcType.getParams());

      // convert scalar user args to ptr addrspace(1)
      {
        auto numUserArgs = newFuncOp.getNumArguments();
        auto globalPtrType = LLVM::LLVMPointerType::get(ctx, 1);
        for (unsigned i = 0; i < numUserArgs; i++) {
          if (!mlir::isa<LLVM::LLVMPointerType>(params[i])) {
            params[i] = globalPtrType; // scalar -> ptr
            newFuncOp.setArgAttr(i, "llvm.noundef", rewriter.getUnitAttr());
            newFuncOp.setArgAttr(i, "llvm.nocapture", rewriter.getUnitAttr());
            newFuncOp.setArgAttr(i, "llvm.readonly", rewriter.getUnitAttr());
            newFuncOp.setArgAttr(
                i, "llvm.dereferenceable",
                rewriter.getIntegerAttr(IntegerType::get(ctx, 64), 4));
          }
        }

        newFuncOp.setFunctionType(
            LLVM::LLVMFunctionType::get(funcType.getReturnType(), params));

        // modify first block
        auto &firstBlock = newFuncOp.getBody().front();
        rewriter.setInsertionPointToStart(&firstBlock);
        for (unsigned i = 0; i < numUserArgs; i++) {
          auto arg = firstBlock.getArgument(i);
          if (!mlir::isa<LLVM::LLVMPointerType>(origParams[i])) {
            arg.setType(globalPtrType);
          }
        }
      }

      // set func attributes
      {
        SmallVector<Attribute> funcAttrs;
        auto addStr = [&](StringRef s) {
          funcAttrs.push_back(rewriter.getStringAttr(s));
        };
        addStr("nounwind");
        addStr("no-builtins");
        newFuncOp.setPassthroughAttr(ArrayAttr::get(ctx, funcAttrs));

        newFuncOp.setUnnamedAddr(LLVM::UnnamedAddr::Local);
      }

      // pass thread/simd/group idxs as extra i32 params to kernel
      // TODO set metadata and handle multiple dims
      auto i32Type = IntegerType::get(ctx, 32);

      SmallVector<DictionaryAttr> argAttrs;
      newFuncOp.getAllArgAttrs(argAttrs);

      // add kNumI32ExtraArgs i32 args (num_programs, thread_idx, simdgroup_idx,
      // threadgroup_idx)
      // see MetalKernelArgs.h for layout
      for (int i = 0; i < mlir::triton::metal::kNumI32ExtraArgs; ++i) {
        params.push_back(i32Type);
      }
      newFuncOp.setFunctionType(
          LLVM::LLVMFunctionType::get(funcType.getReturnType(), params));

      // first entry block receives args from function params
      // so need to add additional params to first entry block
      auto &region = newFuncOp.getBody();
      auto loc = funcOp.getLoc();
      auto noundef =
          rewriter.getNamedAttr("llvm.noundef", rewriter.getUnitAttr());
      auto argAttr = DictionaryAttr::get(ctx, {noundef});
      for (int i = 0; i < mlir::triton::metal::kNumI32ExtraArgs; ++i) {
        region.addArgument(i32Type, loc);
        argAttrs.push_back(argAttr);
      }
      newFuncOp.setAllArgAttrs(argAttrs);
    } else {
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
    }

    rewriter.eraseOp(funcOp);
    rewriter.eraseOp(amendedFuncOp);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::metal::populateFuncOpConversionPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<FuncOpConversion>(typeConverter, targetInfo, benefit);
}