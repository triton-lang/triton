/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct MakeTensorPtrOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::MakeTensorPtrOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::MakeTensorPtrOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto offsets = adaptor.getOffsets();
    auto shapes = adaptor.getShape();
    auto strides = adaptor.getStrides();
    auto base = adaptor.getBase();
    auto result = op.getResult();

    SmallVector<Value> elems;
    for (auto offset : offsets)
      elems.push_back(offset);
    for (auto shape : shapes)
      elems.push_back(shape);
    for (auto stride : strides)
      elems.push_back(stride);

    elems.push_back(base);

    auto newValue = getTypeConverter()->packLLElements(
        op.getLoc(), elems, rewriter, result.getType());
    rewriter.replaceOp(op, newValue);
    return success();
  }
};

struct AdvanceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AdvanceOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AdvanceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto loc = op.getLoc();
    auto ptrType = op.getPtr().getType();
    auto tensorPtr = adaptor.getPtr();

    auto offsets = adaptor.getOffsets();
    auto elems = getTypeConverter()->unpackLLElements(loc, tensorPtr, rewriter);

    SmallVector<Value, 2> newOffsets;

    for (auto [offset, oldOffset] : llvm::zip_first(offsets, elems)) {
      newOffsets.push_back((add(offset, oldOffset)));
    }

    for (size_t i = 0; i < newOffsets.size(); ++i) {
      elems[i] = newOffsets[i];
    }

    auto newValue = getTypeConverter()->packLLElements(op.getLoc(), elems,
                                                       rewriter, ptrType);
    rewriter.replaceOp(op, newValue);
    return success();
  }
};
} // namespace

void mlir::triton::populateTensorPtrOpsToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation, PatternBenefit benefit) {
  patterns.add<MakeTensorPtrOpConversion>(typeConverter, benefit);
  patterns.add<AdvanceOpConversion>(typeConverter, benefit);
  return;
}
