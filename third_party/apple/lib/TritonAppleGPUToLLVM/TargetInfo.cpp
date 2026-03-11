// TargetInfo.cpp — Apple MPS TargetInfoBase implementation
//
// Metal/AIR programId: air.threadgroup_position_in_grid = CTA/block index (Triton program_id)
// NOT air.thread_position_in_grid which is the flat global thread ID.

#include "TritonAppleGPUToLLVM/TargetInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::triton;

namespace mlir::triton::applegpu {

static LLVMFuncOp getOrInsertAirIntrinsic3xi32(RewriterBase &rewriter,
                                                ModuleOp mod, StringRef name) {
    if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
        return fn;
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto *ctx = mod.getContext();
    auto retTy = LLVMArrayType::get(IntegerType::get(ctx, 32), 3);
    auto fnTy = LLVMFunctionType::get(retTy, {}, false);
    return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                               Linkage::External);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                             ModuleOp moduleOp, ProgramIDDim axis) const {
    auto *ctx = moduleOp.getContext();
    auto fn = getOrInsertAirIntrinsic3xi32(rewriter, moduleOp,
                                            "air.threadgroup_position_in_grid");
    auto retTy = LLVMArrayType::get(IntegerType::get(ctx, 32), 3);
    Value gridVec = LLVM::CallOp::create(rewriter, loc, fn, ValueRange{}).getResult();
    int idx = static_cast<int>(axis);
    return LLVM::ExtractValueOp::create(rewriter, loc, IntegerType::get(ctx, 32),
                                         gridVec, ArrayRef<int64_t>{idx});
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
    return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                          Value cmp) const {
    return arith::ConstantIntOp::create(rewriter, loc, 0, 64);
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                          triton::gpu::AddrSpace targets) const {
    // air.threadgroup.barrier(i32 flags, i32 mem_scope)
    // flags=1 (memory), mem_scope=1 (threadgroup)
    auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto voidTy = LLVMVoidType::get(rewriter.getContext());
    auto fnTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
    LLVMFuncOp fn;
    if (auto existing = mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier"))
        fn = existing;
    else {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        fn = LLVMFuncOp::create(rewriter, mod.getLoc(),
            "air.wg.barrier", fnTy, Linkage::External);
    }
    Value flags = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value scope = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    LLVM::CallOp::create(rewriter, loc, fn, ValueRange{flags, scope});
}

void TargetInfo::clusterBarrier(Location loc, RewriterBase &rewriter) const {
    barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                               std::optional<Value> ctaId, Value val,
                               Value pred) const {
    assert(!ctaId && "Apple does not support cross-CTA transfers");
    LLVM::StoreOp::create(rewriter, loc, val, ptr);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                               std::optional<Value> ctaId, Type elemTy,
                               Value pred, Operation *localLoadOp) const {
    assert(!ctaId && "Apple does not support cross-CTA transfers");
    return LLVM::LoadOp::create(rewriter, loc, elemTy, ptr).getResult();
}

// Helper: get or insert an AIR simd_shuffle intrinsic declaration.
// Metal AIR shuffle intrinsics:
//   float: air.simd_shuffle_xor.f32(float, i16)
//   i32:   air.simd_shuffle_xor.s.i32(i32, i16)
//   f16:   air.simd_shuffle_xor.f16(half, i16)
//   bf16:  air.simd_shuffle_xor.bf16(bfloat, i16)
static LLVMFuncOp getOrInsertShuffleIntrinsic(RewriterBase &rewriter,
                                               ModuleOp mod, StringRef kind,
                                               Type valTy) {
    // Build intrinsic name based on type
    // Build base: "air.simd_shuffle" or "air.simd_shuffle_xor" etc.
    std::string base = "air.simd_shuffle";
    if (!kind.empty())
        base += "_" + kind.str();
    std::string name = base + ".";
    if (valTy.isF32())
        name += "f32";
    else if (valTy.isF16())
        name += "f16";
    else if (valTy.isBF16())
        name += "bf16";
    else if (valTy.isF64())
        name += "f64";
    else if (valTy.isInteger(32))
        name = base + ".s.i32";
    else if (valTy.isInteger(16))
        name = base + ".s.i16";
    else if (valTy.isInteger(64))
        name = base + ".s.i64";
    else
        llvm_unreachable("unsupported shuffle type");

    if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
        return fn;
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto i16Ty = IntegerType::get(mod.getContext(), 16);
    auto fnTy = LLVMFunctionType::get(valTy, {valTy, i16Ty}, false);
    return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                               Linkage::External);
}

// Emit a shuffle call, handling bitcast for types that need i32 shuffle
static Value emitShuffle(RewriterBase &rewriter, Location loc, Value val,
                          Value offset, StringRef kind) {
    auto mod = val.getDefiningOp()
                   ? val.getDefiningOp()->getParentOfType<ModuleOp>()
                   : cast<BlockArgument>(val)
                         .getOwner()
                         ->getParentOp()
                         ->getParentOfType<ModuleOp>();
    Type valTy = val.getType();

    // For types narrower than 32 bits, bitcast to i32, shuffle, bitcast back
    if (valTy.isF16() || valTy.isBF16() || valTy.isInteger(16) ||
        valTy.isInteger(8) || valTy.isInteger(1)) {
        auto i32Ty = IntegerType::get(mod.getContext(), 32);
        Value extended;
        if (valTy.isInteger())
            extended = LLVM::ZExtOp::create(rewriter, loc, i32Ty, val);
        else {
            auto i16Ty = IntegerType::get(mod.getContext(), 16);
            Value asInt = LLVM::BitcastOp::create(rewriter, loc, i16Ty, val);
            extended = LLVM::ZExtOp::create(rewriter, loc, i32Ty, asInt);
        }
        auto fn = getOrInsertShuffleIntrinsic(rewriter, mod, kind, i32Ty);
        Value result =
            LLVM::CallOp::create(rewriter, loc, fn, ValueRange{extended, offset})
                .getResult();
        Value truncated =
            LLVM::TruncOp::create(rewriter, loc, IntegerType::get(mod.getContext(), 16), result);
        if (valTy.isInteger())
            return LLVM::TruncOp::create(rewriter, loc, valTy, truncated);
        return LLVM::BitcastOp::create(rewriter, loc, valTy, truncated);
    }

    auto fn = getOrInsertShuffleIntrinsic(rewriter, mod, kind, valTy);
    return LLVM::CallOp::create(rewriter, loc, fn, ValueRange{val, offset})
        .getResult();
}

static Value makeI16Const(RewriterBase &rewriter, Location loc, int val) {
    auto i16Ty = IntegerType::get(rewriter.getContext(), 16);
    return LLVM::ConstantOp::create(rewriter, loc, i16Ty,
                                     rewriter.getI16IntegerAttr(val));
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                              int i) const {
    return emitShuffle(rewriter, loc, val, makeI16Const(rewriter, loc, i), "xor");
}
Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
    return emitShuffle(rewriter, loc, val, makeI16Const(rewriter, loc, i), "up");
}
Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                              int i) const {
    return emitShuffle(rewriter, loc, val, makeI16Const(rewriter, loc, i), "");
}
Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                              Value i) const {
    auto i16Ty = IntegerType::get(rewriter.getContext(), 16);
    Value offset = LLVM::TruncOp::create(rewriter, loc, i16Ty, i);
    return emitShuffle(rewriter, loc, val, offset, "");
}
Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                           Value b, Value selector) const { return a; }

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                             SmallVector<Value> &acc, triton::ReduceOp op,
                             unsigned reduceLaneIdMask) const {
    return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
    if (resultElementTy.isInteger(32)) return "__mulhi";
    if (resultElementTy.isInteger(64)) return "__mul64hi";
    llvm_unreachable("unsupported mulhi type");
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                         int formatStrByteCount, ValueRange args,
                         ArrayRef<bool> isSigned) const {}
void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                         ArrayRef<bool> isSigned) const {}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                             StringRef message, StringRef file, StringRef func,
                             int line) const {
    LLVM::Trap::create(rewriter, loc);
}

} // namespace mlir::triton::applegpu
