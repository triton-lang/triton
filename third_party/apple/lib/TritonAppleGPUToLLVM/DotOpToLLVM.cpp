// DotOpToLLVM: lower tt.dot with AppleMmaEncoding → air simdgroup intrinsics
//
// VERIFIED intrinsic names (macOS 26, M1, disassembled from real Metal shader):
//
//   load:  air.simdgroup_matrix_8x8_load.v64f32.p3f32(
//              ptr addrspace(3),        ← threadgroup memory pointer
//              <2 x i64> shape,         ← {rows, cols} = {8, 8}
//              <2 x i64> strides,       ← {row_stride=1, col_stride=8} (col-major)
//                                          OR {1, 8} for row-major
//              <2 x i64> offset)        ← {0, 0}
//          → <64 x float>
//
//   mma:   air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(
//              <64 x float> A,
//              <64 x float> B,
//              <64 x float> C)
//          → <64 x float>
//
//   store: air.simdgroup_matrix_8x8_store.v64f32.p3f32(
//              <64 x float> C,
//              ptr addrspace(3),
//              <2 x i64> shape,
//              <2 x i64> strides,
//              <2 x i64> offset)
//
// KEY INSIGHT: Metal IR uses <64 x float> for the whole 8x8 tile — NOT per-thread
// fragments. The Metal compiler distributes across 32 threads internally.
// This means at the LLVM IR level each "thread" logically holds the full tile,
// and the backend splits it. This is different from CUDA/PTX where you explicitly
// manage per-thread fragments.
//
// This simplifies our lowering significantly — no per-thread fragment bookkeeping.

#include "TritonAppleGPUToLLVM/Passes.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallVector.h"

namespace tt  = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
using namespace mlir::triton::applegpu;

namespace {

// <64 x float> — the simdgroup matrix type at Metal IR level
static Type getSimdgroupMatrixType(MLIRContext *ctx) {
    return LLVM::getVectorType(Float32Type::get(ctx), 64);
}

// <2 x i64> constant for shape/stride/offset args
static Value makeI64Vec2(OpBuilder &b, Location loc, int64_t a, int64_t b_val) {
    auto ty = LLVM::getVectorType(IntegerType::get(b.getContext(), 64), 2);
    Value vec = UndefOp::create(b, loc, ty);
    Value va  = arith::ConstantIntOp::create(b, loc, a,     64);
    Value vb  = arith::ConstantIntOp::create(b, loc, b_val, 64);
    Value i0  = arith::ConstantIntOp::create(b, loc, 0, 32);
    Value i1  = arith::ConstantIntOp::create(b, loc, 1, 32);
    vec = InsertElementOp::create(b, loc, ty, vec, va, i0);
    vec = InsertElementOp::create(b, loc, ty, vec, vb, i1);
    return vec;
}

// Declare or get an air intrinsic function in the module
static LLVMFuncOp getOrInsertIntrinsic(OpBuilder &b, ModuleOp mod,
                                        StringRef name, LLVMFunctionType fnTy) {
    if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
        return fn;
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(mod.getBody());
    return LLVMFuncOp::create(b, mod.getLoc(), name, fnTy,
                               Linkage::External);
}

// Lower tt.dot (with AppleMmaEncoding) to air simdgroup intrinsics.
//
// Strategy:
//   1. Store A, B, C operands to threadgroup memory
//   2. Load into <64 x float> simdgroup matrix registers
//   3. Call air.simdgroup_matrix_8x8_multiply_accumulate
//   4. Store result back
//   5. Load from threadgroup memory into output tensor
//
// This is the "correct but not optimized" path. Optimization (avoiding
// TG roundtrip) requires operands already in simdgroup format via
// DotOperandEncoding — that's a later optimization pass.
struct DotOpAppleMmaConversion : public ConvertOpToLLVMPattern<tt::DotOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        tt::DotOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {

        auto loc = op.getLoc();
        auto ctx = op.getContext();
        auto mod = op->getParentOfType<ModuleOp>();

        auto cType = cast<RankedTensorType>(op.getC().getType());
        if (!isa<AppleMmaEncodingAttr>(cType.getEncoding()))
            return failure();

        auto matTy = getSimdgroupMatrixType(ctx);

        // Threadgroup memory for A, B, C (8x8 float each)
        // In real Metal IR these are module-level addrspace(3) globals
        // For now: alloca in threadgroup addrspace (simplified)
        auto tgPtrTy = LLVMPointerType::get(ctx, /*addrspace=*/3);
        auto f32Ty   = Float32Type::get(ctx);
        auto arrTy   = LLVMArrayType::get(f32Ty, 64);

        // TODO: proper threadgroup alloca — for now use stack as placeholder
        // Real impl needs addrspace(3) global or threadgroup alloca support
        Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
        Value tgA = LLVM::AllocaOp::create(rewriter, loc, tgPtrTy, arrTy, one);
        Value tgB = LLVM::AllocaOp::create(rewriter, loc, tgPtrTy, arrTy, one);
        Value tgC = LLVM::AllocaOp::create(rewriter, loc, tgPtrTy, arrTy, one);

        // Store operands to TG memory (simplified — real impl stores per-thread)
        LLVM::StoreOp::create(rewriter, loc, adaptor.getA(), tgA);
        LLVM::StoreOp::create(rewriter, loc, adaptor.getB(), tgB);
        LLVM::StoreOp::create(rewriter, loc, adaptor.getC(), tgC);

        // Shape/stride/offset args for load/store
        Value shape   = makeI64Vec2(rewriter, loc, 8, 8);
        Value strideA = makeI64Vec2(rewriter, loc, 1, 8);  // row-major
        Value strideB = makeI64Vec2(rewriter, loc, 1, 8);
        Value strideC = makeI64Vec2(rewriter, loc, 1, 8);
        Value offset  = makeI64Vec2(rewriter, loc, 0, 0);

        auto vec2i64Ty = LLVM::getVectorType(IntegerType::get(ctx, 64), 2);

        // Load intrinsic: air.simdgroup_matrix_8x8_load.v64f32.p3f32
        auto loadFnTy = LLVMFunctionType::get(matTy,
            {tgPtrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty}, false);
        auto loadFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_load.v64f32.p3f32", loadFnTy);

        Value matA = LLVM::CallOp::create(rewriter, loc, loadFn,
            ValueRange{tgA, shape, strideA, offset}).getResult();
        Value matB = LLVM::CallOp::create(rewriter, loc, loadFn,
            ValueRange{tgB, shape, strideB, offset}).getResult();
        Value matC = LLVM::CallOp::create(rewriter, loc, loadFn,
            ValueRange{tgC, shape, strideC, offset}).getResult();

        // MMA intrinsic: air.simdgroup_matrix_8x8_multiply_accumulate.v64f32×4
        auto mmaFnTy = LLVMFunctionType::get(matTy,
            {matTy, matTy, matTy}, false);
        auto mmaFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
            mmaFnTy);

        Value result = LLVM::CallOp::create(rewriter, loc, mmaFn,
            ValueRange{matA, matB, matC}).getResult();

        // Store result back to TG memory
        auto storeFnTy = LLVMFunctionType::get(LLVMVoidType::get(ctx),
            {matTy, tgPtrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty}, false);
        auto storeFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_store.v64f32.p3f32", storeFnTy);

        LLVM::CallOp::create(rewriter, loc, storeFn,
            ValueRange{result, tgC, shape, strideC, offset});

        // Load result from TG memory into output
        Value out = LLVM::LoadOp::create(rewriter, loc, arrTy, tgC);
        rewriter.replaceOp(op, out);
        return success();
    }
};

} // anonymous namespace

namespace mlir::triton::applegpu {

void populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    PatternBenefit benefit) {
    patterns.add<DotOpAppleMmaConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::applegpu
