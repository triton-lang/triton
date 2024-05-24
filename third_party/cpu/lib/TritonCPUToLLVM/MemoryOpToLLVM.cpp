#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_MEMORYOPTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

// TODO: use enums to access struct fields.
struct ExtractMemRefOpConversion : public OpConversionPattern<ExtractMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value tensorPtrStruct = rewriter.getRemappedValue(op.getSrc());
    auto memRefTy = cast<MemRefType>(op.getType());
    auto rank = memRefTy.getRank();
    auto memRefStructTy = getTypeConverter()->convertType(op.getType());
    auto memRefStructFields =
        cast<LLVM::LLVMStructType>(memRefStructTy).getBody();
    auto i64Ty = IntegerType::get(getContext(), 64);

    auto copyValue = [&](Value to, int64_t idxFrom, int64_t idxTo) {
      auto valueTy = memRefStructFields[idxTo];
      Value val = rewriter.create<LLVM::ExtractValueOp>(
          loc, valueTy, tensorPtrStruct, idxFrom);
      return rewriter.create<LLVM::InsertValueOp>(loc, memRefStructTy, to, val,
                                                  idxTo);
    };

    Value res = undef(memRefStructTy);
    // Copy base.
    res = copyValue(res, 0, 1);
    // Use 0 offset.
    res = rewriter.create<LLVM::InsertValueOp>(loc, memRefStructTy, res,
                                               i64_val(0), 2);
    // Copy shape.
    res = copyValue(res, 2, 3);
    // Copy strides.
    res = copyValue(res, 3, 4);

    rewriter.replaceOp(op, res);

    return success();
  }
};

struct ExtractIndicesOpConversion
    : public OpConversionPattern<ExtractIndicesOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractIndicesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    Value tensorPtrStruct = rewriter.getRemappedValue(op.getSrc());
    auto rank = op.getNumResults();
    auto i64Ty = IntegerType::get(getContext(), 64);
    SmallVector<Value> indices;

    for (int64_t i = 0; i < rank; i++) {
      Value offs = rewriter.create<LLVM::ExtractValueOp>(
          loc, i64Ty, tensorPtrStruct, SmallVector<int64_t, 2>{1, i});
      Value stride = rewriter.create<LLVM::ExtractValueOp>(
          loc, i64Ty, tensorPtrStruct, SmallVector<int64_t, 2>{3, i});
      indices.push_back(rewriter.create<LLVM::MulOp>(loc, offs, stride));
    }

    rewriter.replaceOp(op, indices);

    return success();
  }
};

struct PtrToMemRefOpConversion : public OpConversionPattern<PtrToMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PtrToMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value ptr = rewriter.getRemappedValue(op.getSrc());
    auto memRefStructTy = getTypeConverter()->convertType(op.getType());

    Value res = undef(memRefStructTy);
    res =
        rewriter.create<LLVM::InsertValueOp>(loc, memRefStructTy, res, ptr, 1);
    rewriter.replaceOp(op, res);

    return success();
  }
};

struct MakeTensorPtrOpConversion : public OpConversionPattern<MakeTensorPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto structTy = getTypeConverter()->convertType(op.getType());
    auto i64Ty = IntegerType::get(getContext(), 64);

    auto insertArray = [&](Value structVal, auto values, int64_t idx,
                           Type zextTo = nullptr) {
      for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        Value val = values[i];
        if (zextTo)
          val = rewriter.create<LLVM::ZExtOp>(loc, zextTo, val);
        structVal = rewriter.create<LLVM::InsertValueOp>(
            loc, structTy, structVal, val, SmallVector<int64_t, 2>{idx, i});
      }
      return structVal;
    };

    Value res = undef(structTy);
    // 0 - base pointer.
    auto base = rewriter.getRemappedValue(op.getBase());
    res = rewriter.create<LLVM::InsertValueOp>(loc, structTy, res, base, 0);
    // 1 - array<rank> for offsets. Promote values to i64.
    res = insertArray(res, op.getOffsets(), 1, i64Ty);
    // 2 - array<rank> for shape.
    res = insertArray(res, op.getShape(), 2);
    // 3 - array<rank> for strides.
    res = insertArray(res, op.getStrides(), 3);

    rewriter.replaceOp(op, res);

    return success();
  }
};

struct AdvanceOpConversion : public OpConversionPattern<AdvanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto i64Ty = IntegerType::get(getContext(), 64);
    Value res = rewriter.getRemappedValue(op.getPtr());
    Type structTy = res.getType();
    auto offsets = op.getOffsets();

    for (int64_t i = 0; i < offsets.size(); ++i) {
      auto oldOffset = rewriter.create<LLVM::ExtractValueOp>(
          loc, i64Ty, res, SmallVector<int64_t, 2>{1, i});
      auto step = rewriter.create<LLVM::SExtOp>(loc, i64Ty, offsets[i]);
      auto newOffset = rewriter.create<LLVM::AddOp>(loc, oldOffset, step);
      res = rewriter.create<LLVM::InsertValueOp>(loc, structTy, res, newOffset,
                                                 SmallVector<int64_t, 2>{1, i});
    }

    rewriter.replaceOp(op, res);

    return success();
  }
};

struct LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type ptrTy = LLVM::LLVMPointerType::get(getContext());
    Value ptr = rewriter.getRemappedValue(op.getPtr());
    Type resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resTy, ptr, 0,
                                              op.getIsVolatile());
    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value ptr = rewriter.getRemappedValue(op.getPtr());
    Value val = rewriter.getRemappedValue(op.getValue());
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, val, ptr);
    return success();
  }
};

struct PtrToIntOpConversion : public OpConversionPattern<triton::PtrToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = rewriter.getRemappedValue(op.getSrc());
    Type resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resTy, src);
    return success();
  }
};

struct IntToPtrOpConversion : public OpConversionPattern<triton::IntToPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = rewriter.getRemappedValue(op.getSrc());
    Type resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, resTy, src);
    return success();
  }
};

struct AddPtrOpConversion : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expect only scalar pointers here.
    assert(isa<PointerType>(op.getType()));
    auto ptrTy = cast<PointerType>(op.getPtr().getType());
    Type elemTy = getTypeConverter()->convertType(ptrTy.getPointeeType());
    Type resTy = getTypeConverter()->convertType(ptrTy);
    Value ptr = rewriter.getRemappedValue(op.getPtr());
    Value offset = rewriter.getRemappedValue(op.getOffset());
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, resTy, elemTy, ptr, offset);
    return success();
  }
};

struct PtrBitcastConversion : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // By this moment we expect tt.bitcast used only for scalar pointer casts.
    // This cast becomes NOP for LLVM dialect, so simply return the source arg.
    assert(isa<PointerType>(op.getType()));
    assert(isa<PointerType>(op.getSrc().getType()));
    Value src = rewriter.getRemappedValue(op.getSrc());
    rewriter.replaceOp(op, src);
    return success();
  }
};

struct PtrSelectConversion : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // By this moment we expect tt.bitcast used only for scalar pointer casts.
    // This cast becomes NOP for LLVM dialect, so simply return the source arg.
    if (!isa<PointerType>(op.getType()))
      return failure();

    Value trueVal = rewriter.getRemappedValue(op.getTrueValue());
    Value falseVal = rewriter.getRemappedValue(op.getFalseValue());
    Value cond = rewriter.getRemappedValue(op.getCondition());
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, cond, trueVal, falseVal);
    return success();
  }
};

struct MemoryOpToLLVM
    : public triton::impl::MemoryOpToLLVMBase<MemoryOpToLLVM> {
  using MemoryOpToLLVMBase::MemoryOpToLLVMBase;

  MemoryOpToLLVM() : MemoryOpToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    RewritePatternSet patterns(context);
    patterns.add<ExtractMemRefOpConversion>(typeConverter, context);
    patterns.add<MakeTensorPtrOpConversion>(typeConverter, context);
    patterns.add<AdvanceOpConversion>(typeConverter, context);
    patterns.add<ExtractIndicesOpConversion>(typeConverter, context);
    patterns.add<LoadOpConversion>(typeConverter, context);
    patterns.add<StoreOpConversion>(typeConverter, context);
    patterns.add<PtrToIntOpConversion>(typeConverter, context);
    patterns.add<IntToPtrOpConversion>(typeConverter, context);
    patterns.add<PtrToMemRefOpConversion>(typeConverter, context);
    patterns.add<AddPtrOpConversion>(typeConverter, context);
    patterns.add<PtrBitcastConversion>(typeConverter, context);
    patterns.add<PtrSelectConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createMemoryOpToLLVMPass() {
  return std::make_unique<MemoryOpToLLVM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
