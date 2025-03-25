//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"

namespace mlir {
FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &converter);
}

namespace {

using namespace mlir;
using namespace mlir::triton;

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
  /// attributes.
  static void filterFuncAttributes(triton::FuncOp op, bool filterArgAttrs,
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

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = funcOp;
    if (!LLVM::isKernel(funcOp))
      llvm_unreachable("Unsuppprted funcOp is not Kernel");

    LLVM::LLVMFuncOp newFuncOp = *mlir::convertFuncOpToLLVMFuncOp(
        amendedFuncOp, rewriter, *getTypeConverter());
    if (!newFuncOp) {
      return failure();
    }

    auto ctx = funcOp->getContext();

    if (LLVM::isKernel(funcOp)) {
      // Set an attribute to indicate this function is a kernel entry.
      newFuncOp->setAttr("xpu.kernel",
                         rewriter.getIntegerAttr(type::u1Ty(ctx), 1));
      newFuncOp.setLinkage(LLVM::Linkage::External);
    } else {
      // The noinline attribute will be used by the LLVM codegen to prevent
      // inlining.
      // https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/LLVMIR/IR/LLVMInlining.cpp#L267
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      rewriter.eraseOp(amendedFuncOp);
      newFuncOp.setLinkage(LLVM::Linkage::Internal);
    }
    rewriter.eraseOp(funcOp);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateFuncOpConversionPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FuncOpConversion>(typeConverter, benefit);
}
