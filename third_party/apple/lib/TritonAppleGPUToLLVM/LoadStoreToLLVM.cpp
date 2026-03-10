// LoadStoreToLLVM.cpp — Lower tt.load/tt.store/tt.addptr to LLVM IR
//
// Handles both scalar and blocked-tensor (struct-of-pointers) paths.

#include "TritonAppleGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::triton;

namespace mlir::triton::applegpu {

namespace {

// Unpack an LLVM struct into individual values, or return {v} if scalar.
static SmallVector<Value> unpackElems(Value v, OpBuilder &b, Location loc) {
    if (!v) return {};
    if (auto sTy = dyn_cast<LLVMStructType>(v.getType())) {
        SmallVector<Value> elems(sTy.getBody().size());
        for (size_t i = 0; i < elems.size(); ++i)
            elems[i] = ExtractValueOp::create(b, loc, sTy.getBody()[i], v,
                                               ArrayRef<int64_t>{(int64_t)i});
        return elems;
    }
    return {v};
}

// Pack a list of values into an LLVM struct of the same types.
// Always packs, even for a single element, to match the type converter's
// struct<(T)> expectation for 1-element tensors.
static Value packElems(ArrayRef<Value> elems, OpBuilder &b, Location loc) {
    SmallVector<Type> tys;
    for (auto v : elems) tys.push_back(v.getType());
    auto sTy = LLVMStructType::getLiteral(b.getContext(), tys);
    Value result = UndefOp::create(b, loc, sTy);
    for (size_t i = 0; i < elems.size(); ++i)
        result = InsertValueOp::create(b, loc, sTy, result, elems[i],
                                        ArrayRef<int64_t>{(int64_t)i});
    return result;
}

// tt.addptr %base, %offset → GEP per element (scalar or struct path)
struct AddPtrOpConversion : public ConvertOpToLLVMPattern<triton::AddPtrOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value base   = adaptor.getPtr();
        Value offset = adaptor.getOffset();

        // Determine element type from the original Triton pointer type
        auto srcTy = op.getPtr().getType();
        Type elemTy;
        if (auto ptrTy = dyn_cast<triton::PointerType>(srcTy)) {
            elemTy = getTypeConverter()->convertType(ptrTy.getPointeeType());
        } else {
            // tensor<Nx!tt.ptr<T>> — element type is T
            auto tensorTy = cast<RankedTensorType>(srcTy);
            auto tPtrTy = cast<triton::PointerType>(tensorTy.getElementType());
            elemTy = getTypeConverter()->convertType(tPtrTy.getPointeeType());
        }
        if (!elemTy) return failure();

        // Scalar addptr: bare pointer + scalar offset → single GEP.
        if (!isa<LLVMStructType>(base.getType())) {
            Value gep = LLVM::GEPOp::create(rewriter, loc,
                base.getType(), elemTy, base,
                ArrayRef<LLVM::GEPArg>{offset});
            rewriter.replaceOp(op, gep);
            return success();
        }

        auto basePtrs   = unpackElems(base,   rewriter, loc);
        auto offsets    = unpackElems(offset, rewriter, loc);

        if (basePtrs.size() != offsets.size())
            return failure();

        SmallVector<Value> results;
        for (size_t i = 0; i < basePtrs.size(); ++i) {
            results.push_back(LLVM::GEPOp::create(rewriter, loc,
                basePtrs[i].getType(), elemTy, basePtrs[i],
                ArrayRef<LLVM::GEPArg>{offsets[i]}));
        }

        rewriter.replaceOp(op, packElems(results, rewriter, loc));
        return success();
    }
};

// tt.load %ptr → load per element (scalar or struct-of-pointers)
struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value ptr = adaptor.getPtr();
        Type resultTy = getTypeConverter()->convertType(op.getType());
        if (!resultTy) return failure();

        // Scalar load: ptr is a bare LLVM pointer, result is a scalar type.
        if (!isa<LLVMStructType>(ptr.getType())) {
            Value val = LLVM::LoadOp::create(rewriter, loc, resultTy, ptr);
            Value maskOperand  = adaptor.getMask();
            Value otherOperand = adaptor.getOther();
            if (maskOperand) {
                Value other = otherOperand
                    ? otherOperand
                    : LLVM::UndefOp::create(rewriter, loc, resultTy);
                val = LLVM::SelectOp::create(rewriter, loc, maskOperand, val, other);
            }
            rewriter.replaceOp(op, val);
            return success();
        }

        auto ptrs = unpackElems(ptr, rewriter, loc);

        // Tensor load: resultTy is a struct (even 1-element tensors → struct<(f32)>).
        auto sTy = dyn_cast<LLVMStructType>(resultTy);
        if (!sTy || sTy.getBody().size() != ptrs.size())
            return failure();
        Value maskOperand  = adaptor.getMask();
        Value otherOperand = adaptor.getOther();
        auto masks  = maskOperand  ? unpackElems(maskOperand,  rewriter, loc) : SmallVector<Value>{};
        auto others = otherOperand ? unpackElems(otherOperand, rewriter, loc) : SmallVector<Value>{};

        SmallVector<Value> loaded;
        for (size_t i = 0; i < ptrs.size(); ++i) {
            Value val = LLVM::LoadOp::create(rewriter, loc, sTy.getBody()[i], ptrs[i]);
            if (!masks.empty()) {
                Value other = others.empty()
                    ? LLVM::UndefOp::create(rewriter, loc, sTy.getBody()[i])
                    : others[i];
                val = LLVM::SelectOp::create(rewriter, loc, masks[i], val, other);
            }
            loaded.push_back(val);
        }
        rewriter.replaceOp(op, packElems(loaded, rewriter, loc));
        return success();
    }
};

// tt.store %ptr, %val → store per element
struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();

        Value ptr = adaptor.getPtr();
        Value val = adaptor.getValue();

        auto ptrs = unpackElems(ptr, rewriter, loc);
        auto vals = unpackElems(val, rewriter, loc);

        if (ptrs.size() != vals.size())
            return failure();

        Value maskOperand = adaptor.getMask();
        auto masks = maskOperand ? unpackElems(maskOperand, rewriter, loc) : SmallVector<Value>{};

        for (size_t i = 0; i < ptrs.size(); ++i) {
            Value toStore = vals[i];
            if (!masks.empty() && masks[i]) {
                // Masked store: load existing value, select(mask, new_val, existing)
                // so unmasked lanes write back what was already there (no-op).
                Value existing = LLVM::LoadOp::create(rewriter, loc, vals[i].getType(), ptrs[i]);
                toStore = LLVM::SelectOp::create(rewriter, loc, masks[i], vals[i], existing);
            }
            LLVM::StoreOp::create(rewriter, loc, toStore, ptrs[i]);
        }
        rewriter.eraseOp(op);
        return success();
    }
};

} // anonymous namespace

void populateLoadStoreToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit) {
    patterns.add<AddPtrOpConversion, LoadOpConversion, StoreOpConversion>(
        typeConverter, benefit);
}

} // namespace mlir::triton::applegpu
