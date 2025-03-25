//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "triton/Conversion/TritonXPUToLLVM/Passes.h"
// clang-format off
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"

// #include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

// clang-format on

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_CONVERTTRITONXPUTOLLVM
#include "triton/Conversion/TritonXPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<LLVM::XPU::LLVMXPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::xpu::TritonXPUDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    // addLegalDialect<index::IndexDialect>(); // TODO[dyq]: necessary?
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<LLVM::XPU::LLVMXPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonXPUToLLVM
    : public triton::impl::ConvertTritonXPUToLLVMBase<ConvertTritonXPUToLLVM> {
  using ConvertTritonXPUToLLVMBase::ConvertTritonXPUToLLVMBase;

  ConvertTritonXPUToLLVM(uint32_t xpu_arch, uint32_t buffer_size)
      : ConvertTritonXPUToLLVMBase({xpu_arch, buffer_size}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonXPUToLLVMTypeConverter typeConverter(context,
                                               option); // we can reuse it
    TritonLLVMConversionTarget convTarget(*context);

    // Allocate shared memory and set barrier
    // TODO[dyq]: necessary to open?
    // ModuleAllocation allocation(mod);
    // ModuleMembarAnalysis membarPass(&allocation);
    // membarPass.run();

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      option.overrideIndexBitwidth(32);
      TritonXPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      // TODO[dyq]: add [nvvm.maxntid, nvvm.kernel] attr
      mlir::triton::xpu::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // LLVM_DEBUG(llvm::dbgs() << "\nAfter Lower Functions:\n" << mod << "\n");

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of
    // each function
    initSharedMemory(typeConverter);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    triton::xpu::TargetInfo targetInfo(xpu_arch, buffer_size);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    // Make benefit for XPU specific patterns higher so they apply before common
    // patterns
    int xpuBenefit = benefit + 1;

    // TODO[dyq]: Open allToLLVMPatterns
    mlir::triton::xpu::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    // TODO[dyq]: XPUSDNN-CHECK add DotOp Lowering Pattern
    // mlir::triton::xpu::populateDotOpToLLVMPatterns(typeConverter, patterns,
    //                                                benefit);
    mlir::triton::xpu::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

    mlir::triton::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

    mlir::triton::xpu::populateTTXPUVectorizedOpToLLVMConversionPatterns(
        typeConverter, targetInfo, patterns, benefit);

    // TODO[dyq]:
    mlir::triton::xpu::populateTTXPUUtilityOpToLLVMConversionPatterns(
        typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit);

    mlir::triton::xpu::populateLoadStoreOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit);

    mlir::triton::xpu::populateGPUToXPUConversionPatterns(
        typeConverter, patterns, targetInfo, benefit);

    mlir::triton::xpu::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                      targetInfo, benefit);
    // mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
    //                                            targetInfo, benefit);

    // mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
    //                                                 targetInfo, benefit);
    // mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
    //                                            targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     benefit);
    mlir::triton::xpu::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                   benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);

    mlir::triton::xpu::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                                    xpuBenefit);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    // mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
    //                                             targetInfo, benefit);
    // mlir::triton::populateMemoryOpToLLVMPattern(typeConverter, targetInfo,
    //                                             patterns, benefit);
    mlir::triton::xpu::populateMakeRangeOpToLLVMPattern(
        typeConverter, targetInfo, patterns, benefit);

    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }

private:
  Value smem;

  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.

    // XPU: the shared space used by Triton will be put at the end of
    // SHARED_MEMORY section since only pointer is used without the real
    // allocation
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::Internal,
        "global_smem", /*value=*/Attribute(), /*alignment=*/8, 2);

    // TODO[dyq]: llvm-18 don't need to set pointer type, will this logic be
    // changed?
    SmallVector<LLVM::LLVMFuncOp> funcs;
    mod.walk([&](LLVM::LLVMFuncOp func) { funcs.push_back(func); });
    assert(funcs.size() == 1 &&
           "Inliner pass is expected before TritonXPUToLLVM");
    b.setInsertionPointToStart(&funcs[0].getBody().front());
    smem = b.create<LLVM::AddressOfOp>(loc, global);

    // TODO[dyq]: llvm-18 don't need to set pointer type, this type maybe cause
    // error
    // auto ptrTy =
    // LLVM::LLVMPointerType::get(typeConverter.convertType(b.getI8Type()), 2);
    auto ptrTy = LLVM::LLVMPointerType::get(mod.getContext(), 2);
    smem = b.create<LLVM::BitcastOp>(loc, ptrTy, smem);
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonXPUToLLVMPass() {
  return std::make_unique<ConvertTritonXPUToLLVM>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonXPUToLLVMPass(uint32_t xpu_arch, uint32_t buffer_size) {
  return std::make_unique<ConvertTritonXPUToLLVM>(xpu_arch, buffer_size);
}

} // namespace triton
} // namespace mlir
