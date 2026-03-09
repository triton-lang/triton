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
    // threadgroup_barrier — use LLVM fence as placeholder
    // TODO: emit air.wg.barrier intrinsic
    LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::seq_cst,
                           StringRef("threadgroup"));
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

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                              int i) const { return val; }
Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                             int i) const { return val; }
Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                              int i) const { return val; }
Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                              Value i) const { return val; }
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
