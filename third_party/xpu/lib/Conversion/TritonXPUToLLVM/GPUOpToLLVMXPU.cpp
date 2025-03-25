//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "xpu/lib/Conversion/TritonXPUToLLVM/PatternTritonXPUOpToLLVM.h"

namespace mlir {

struct XPUIntrinsicOpConversionBase {
  explicit XPUIntrinsicOpConversionBase(LLVMTypeConverter &typeConverter,
                                        const xpu::TargetInfo &targetInfo)
      : targetInfo(targetInfo),
        indexBitwidth(typeConverter.getIndexTypeBitwidth()) {}

protected:
  const xpu::TargetInfo &targetInfo;
  unsigned indexBitwidth;
};

// for physical id
template <typename Op, typename XPUOp>
struct XPUIndexIntrinsicOpLowering : public ConvertOpToLLVMPattern<Op>,
                                     public XPUIntrinsicOpConversionBase {

public:
  explicit XPUIndexIntrinsicOpLowering(LLVMTypeConverter &typeConverter,
                                       const xpu::TargetInfo &targetInfo,
                                       PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Op>(typeConverter, benefit),
        XPUIntrinsicOpConversionBase(typeConverter, targetInfo) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    Value newOp = rewriter.create<XPUOp>(loc, type::i32Ty(context));

    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp);
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp);
    }

    rewriter.replaceOp(op, {newOp});
    return success();
  }
};

// for logical id: refer xtdk_sys.h
template <typename Op, unsigned Num>
struct XPULoadParamIntrinsicOpLowering : public ConvertOpToLLVMPattern<Op>,
                                         public XPUIntrinsicOpConversionBase {

public:
  explicit XPULoadParamIntrinsicOpLowering(LLVMTypeConverter &typeConverter,
                                           const xpu::TargetInfo &targetInfo,
                                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Op>(typeConverter, benefit),
        XPUIntrinsicOpConversionBase(typeConverter, targetInfo) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    Value newOp = rewriter.create<mlir::LLVM::XPU::LoadParamOp>(
        loc, type::i32Ty(context), i32_val(Num));

    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp);
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, IntegerType::get(context, indexBitwidth), newOp);
    }

    rewriter.replaceOp(op, {newOp});
    return success();
  }
};

} // namespace mlir

void mlir::triton::xpu::populateGPUToXPUConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<XPUIndexIntrinsicOpLowering<mlir::gpu::ThreadIdOp,
                                           mlir::LLVM::XPU::CoreIdOp>,
               XPULoadParamIntrinsicOpLowering<mlir::gpu::BlockIdOp, 0>,
               XPULoadParamIntrinsicOpLowering<mlir::gpu::GridDimOp, 1>,
               XPULoadParamIntrinsicOpLowering<mlir::gpu::BlockDimOp, 2>>(
      typeConverter, targetInfo, benefit);
}
