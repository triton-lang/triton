#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "third_party/proton/dialect/include/TritonProtonToLLVM/PatternTritonProtonOpToLLVM.h"

namespace {

struct InitLocalBufferOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::InitLocalBufferOp> {
  explicit InitLocalBufferOpConversion(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::InitLocalBufferOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::InitLocalBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Location loc = UnknownLoc::get(rewriter.getContext());
  auto ctx = moduleOp.getContext();
  size_t contentSize = op.getBufferSize();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  LLVM::GlobalOp global;
  {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
   global = rewriter.create<LLVM::GlobalOp>(
        UnknownLoc::get(ctx), globalType,
        /*isConstant=*/false, LLVM::Linkage::External,
        "proton_smem", /*value=*/Attribute(), /*alignment=*/16, 3);
        
  }
  Value zero = i32_val(0);
  Type globalPtrType = LLVM::LLVMPointerType::get(ctx, global.getAddrSpace());
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      UnknownLoc::get(ctx), globalPtrType, global.getSymName());
  Value stringStart =
      gep(ptr_ty(ctx), i8_ty, globalPtr, SmallVector<Value>({zero}));  
  store(i32_val(42), stringStart);  
  rewriter.replaceOp(op, stringStart);

    
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::proton::populateInitLocalBufferOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<InitLocalBufferOpConversion>(typeConverter, targetInfo, benefit);
}
