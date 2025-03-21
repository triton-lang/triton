#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/GlobalScratchAllocOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct GlobalScratchAllocOpConversion
    : public ConvertOpToLLVMPattern<proton::gpu::GlobalScratchAllocOp> {
  GlobalScratchAllocOpConversion(LLVMTypeConverter &converter,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  LogicalResult
  matchAndRewrite(proton::gpu::GlobalScratchAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto ctx = funcOp->getContext();
    auto loc = funcOp.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() - 1);
    auto llvmPointerType = LLVM::LLVMPointerType::get(ctx);
    funcOp->setAttr("ttg.profile_scratch_memory_size",
                    rewriter.getI32IntegerAttr(1));
    funcOp->setAttr("ttg.profile_scratch_memory_alignment",
                    rewriter.getI32IntegerAttr(1));
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, llvmPointerType, llvmPointerType, gmemBase, b.i32_val(0));

    return success();
  }
};

} // namespace

void mlir::triton::proton::gpu::populateGlobalScratchAllocOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, benefit);
}
