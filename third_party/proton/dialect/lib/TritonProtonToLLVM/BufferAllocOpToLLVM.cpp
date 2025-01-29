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

struct BufferAllocOpConversion
    : public ConvertOpToLLVMPattern<mlir::triton::proton::BufferAllocOp> {
  explicit BufferAllocOpConversion(LLVMTypeConverter &typeConverter,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<mlir::triton::proton::BufferAllocOp>(
            typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::BufferAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Location loc = UnknownLoc::get(rewriter.getContext());
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ctx = moduleOp.getContext();
    size_t bufferSize = op.getBufferSize();
    auto globalType = LLVM::LLVMArrayType::get(i8_ty, bufferSize);
    SmallVector<uint8_t> buffer(bufferSize);
    buffer.push_back(0); // zero_initalizer
    auto dataAttrType =
        RankedTensorType::get({static_cast<int64_t>(buffer.size())}, i8_ty);
    auto dataAttr =
        DenseElementsAttr::get(dataAttrType, llvm::ArrayRef(buffer));
    auto arrayTy =
        LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), buffer.size());
    LLVM::GlobalOp global;
    {
      RewriterBase::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      global = rewriter.create<LLVM::GlobalOp>(UnknownLoc::get(ctx), globalType,
                                               /*isConstant=*/false,
                                               LLVM::Linkage::External,
                                               "proton_buffer", dataAttr);
    }
    global.setAddrSpace(1);
    global.setExternallyInitialized(true);
    Value zero = b.i32_val(0);
    Type globalPtrType = LLVM::LLVMPointerType::get(ctx, global.getAddrSpace());
    Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
        UnknownLoc::get(rewriter.getContext()), globalPtrType,
        global.getSymName());
    Value bufferStart =
        b.gep(ptr_ty(ctx), i8_ty, globalPtr, SmallVector<Value>({zero}));
    rewriter.replaceOp(op, bufferStart);
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::proton::populateBufferAllocOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<BufferAllocOpConversion>(typeConverter, targetInfo, benefit);
}
