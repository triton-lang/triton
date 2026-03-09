// ConvertTritonAppleGPUToLLVM pass
// Lowers all AppleMmaEncoding ops to LLVM IR with simdgroup intrinsics.

#include "TritonAppleGPUToLLVM/Passes.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::applegpu {

namespace {

struct ConvertTritonAppleGPUToLLVMPass
    : public PassWrapper<ConvertTritonAppleGPUToLLVMPass, OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTritonAppleGPUToLLVMPass)

    StringRef getArgument() const override {
        return "convert-triton-apple-gpu-to-llvm";
    }

    StringRef getDescription() const override {
        return "Lower TritonGPU ops with AppleMmaEncoding to LLVM IR";
    }

    void runOnOperation() override {
        auto mod = getOperation();
        auto *ctx = &getContext();

        LLVMTypeConverter typeConverter(ctx);
        RewritePatternSet patterns(ctx);
        ConversionTarget target(*ctx);

        // Register Apple dot op patterns
        populateDotOpToLLVMPatterns(typeConverter, patterns);

        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addIllegalOp<triton::DotOp>();  // must be lowered

        if (failed(applyPartialConversion(mod, target, std::move(patterns))))
            signalPassFailure();
    }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createConvertTritonAppleGPUToLLVMPass() {
    return std::make_unique<ConvertTritonAppleGPUToLLVMPass>();
}

} // namespace mlir::triton::applegpu
