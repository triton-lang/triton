// ConvertTritonAppleGPUToLLVM pass
//
// Lowers TritonGPU IR → LLVM IR for Apple MPS using shared Triton patterns
// and an Apple-specific TargetInfo.

#include "TritonAppleGPUToLLVM/Passes.h"
#include "TritonAppleGPUToLLVM/TargetInfo.h"
#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::applegpu {

namespace {

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::arith;
namespace ttg = mlir::triton::gpu;

// ConvertLayoutOp for DotOperandEncoding or blocked→blocked:
//
// - DotOperandEncoding target: identity pass-through (elements same per thread)
// - blocked→blocked: TG scatter/gather redistribution
struct ConvertLayoutOpAppleConversion
    : public mlir::ConvertOpToLLVMPattern<ttg::ConvertLayoutOp> {
    using mlir::ConvertOpToLLVMPattern<ttg::ConvertLayoutOp>::ConvertOpToLLVMPattern;

    // Per-context counter for unique TG global names
    static unsigned &getCounter(MLIRContext *ctx) {
        static llvm::DenseMap<MLIRContext *, unsigned> counters;
        return counters[ctx];
    }

    LogicalResult matchAndRewrite(
        ttg::ConvertLayoutOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {

        auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
        auto dstTy = cast<RankedTensorType>(op.getResult().getType());

        // Case 1: DotOperandEncoding target — identity pass-through
        if (isa<ttg::DotOperandEncodingAttr>(dstTy.getEncoding())) {
            rewriter.replaceOp(op, adaptor.getSrc());
            return success();
        }

        // Case 2: blocked→blocked redistribution via TG scatter/gather
        auto srcEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
        auto dstEnc = dyn_cast<ttg::BlockedEncodingAttr>(dstTy.getEncoding());
        if (!srcEnc || !dstEnc)
            return failure();

        // Same encoding — identity
        if (srcEnc == dstEnc) {
            rewriter.replaceOp(op, adaptor.getSrc());
            return success();
        }

        auto loc = op.getLoc();
        auto *ctx = op.getContext();
        auto mod = op->getParentOfType<ModuleOp>();
        auto shape = srcTy.getShape();
        if (shape.size() != 2) return failure();

        int64_t rows = shape[0], cols = shape[1];
        auto elemTy  = getTypeConverter()->convertType(srcTy.getElementType());
        auto i32Ty   = IntegerType::get(ctx, 32);
        auto i64Ty   = IntegerType::get(ctx, 64);
        auto tgPtrTy = LLVMPointerType::get(ctx, 3);

        // Get lane/warp IDs (same helpers as DotOp)
        auto laneIdFnTy = LLVMFunctionType::get(i32Ty, {}, false);
        LLVMFuncOp laneIdFn;
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mod.getBody());
            if (auto fn = mod.lookupSymbol<LLVMFuncOp>("air.thread_index_in_simdgroup"))
                laneIdFn = fn;
            else
                laneIdFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                    "air.thread_index_in_simdgroup", laneIdFnTy, Linkage::External);
        }

        auto arrI32x3Ty = LLVMArrayType::get(i32Ty, 3);
        auto tidFnTy = LLVMFunctionType::get(arrI32x3Ty, {}, false);
        LLVMFuncOp tidFn;
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mod.getBody());
            if (auto fn = mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
                tidFn = fn;
            else
                tidFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                    "air.thread_position_in_threadgroup", tidFnTy, Linkage::External);
        }

        auto barrFnTy = LLVMFunctionType::get(LLVMVoidType::get(ctx), {i32Ty, i32Ty}, false);
        LLVMFuncOp tgBarrFn;
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mod.getBody());
            if (auto fn = mod.lookupSymbol<LLVMFuncOp>("air.threadgroup.barrier"))
                tgBarrFn = fn;
            else
                tgBarrFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                    "air.threadgroup.barrier", barrFnTy, Linkage::External);
        }

        Value laneId = LLVM::CallOp::create(rewriter, loc, laneIdFn,
                                              ValueRange{}).getResult();
        Value tidStruct = LLVM::CallOp::create(rewriter, loc, tidFn,
                                                ValueRange{}).getResult();
        Value tid32 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty,
                          tidStruct, ArrayRef<int64_t>{0});
        Value c32    = arith::ConstantIntOp::create(rewriter, loc, 32, 32);
        Value warpId = arith::DivUIOp::create(rewriter, loc, tid32, c32);

        // Create TG global for scatter/gather
        unsigned id = getCounter(ctx)++;
        std::string tgName = ("__tg_cvt_" + llvm::Twine(id)).str();
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mod.getBody());
            auto arrTy = LLVMArrayType::get(elemTy, rows * cols);
            LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                                    Linkage::Internal, tgName, Attribute(), 4, 3u);
        }
        auto tgGlobal = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
        Value tgPtr = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgGlobal.getName());

        // Helper: compute base (row, col) with wrap + in-bounds predicate
        // Returns {bR, bC, pred} where pred is true if this thread owns valid data
        auto makeBase = [&](ttg::BlockedEncodingAttr enc)
            -> std::tuple<Value, Value, Value> {
            auto spt = enc.getSizePerThread();
            auto tpw = enc.getThreadsPerWarp();
            auto wpc = enc.getWarpsPerCTA();
            int64_t sM = spt[0], sN = spt[1];
            int64_t tM = tpw[0], tN = tpw[1];
            int64_t wM = wpc[0], wN = wpc[1];
            int64_t tileM = wM * tM * sM;
            int64_t tileN = wN * tN * sN;

            Value wN_v = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
            Value tN_v = arith::ConstantIntOp::create(rewriter, loc, tN, 32);
            Value tMsM = arith::ConstantIntOp::create(rewriter, loc, tM * sM, 32);
            Value sM_v = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
            Value tNsN = arith::ConstantIntOp::create(rewriter, loc, tN * sN, 32);
            Value sN_v = arith::ConstantIntOp::create(rewriter, loc, sN, 32);

            // Respect layout order: order[0] is the fastest-changing dimension
            auto order = enc.getOrder();
            bool colFastest = (order[0] == 1); // order=[1,0] => col fastest (default)

            // Warp decomposition: faster dim uses mod, slower uses div
            Value wR, wC;
            if (colFastest) {
                wR = arith::DivUIOp::create(rewriter, loc, warpId, wN_v);
                wC = arith::RemUIOp::create(rewriter, loc, warpId, wN_v);
            } else {
                Value wM_v = arith::ConstantIntOp::create(rewriter, loc, wM, 32);
                wR = arith::RemUIOp::create(rewriter, loc, warpId, wM_v);
                wC = arith::DivUIOp::create(rewriter, loc, warpId, wM_v);
            }
            // Lane decomposition: faster dim uses mod, slower uses div
            Value lR, lC;
            if (colFastest) {
                lR = arith::DivUIOp::create(rewriter, loc, laneId, tN_v);
                lC = arith::RemUIOp::create(rewriter, loc, laneId, tN_v);
            } else {
                Value tM_v = arith::ConstantIntOp::create(rewriter, loc, tM, 32);
                lR = arith::RemUIOp::create(rewriter, loc, laneId, tM_v);
                lC = arith::DivUIOp::create(rewriter, loc, laneId, tM_v);
            }

            Value bR = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, wR, tMsM),
                arith::MulIOp::create(rewriter, loc, lR, sM_v));
            Value bC = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, wC, tNsN),
                arith::MulIOp::create(rewriter, loc, lC, sN_v));

            // Compute in-bounds predicate before wrapping
            Value pred;
            auto i1Ty = IntegerType::get(ctx, 1);
            if (tileM > rows || tileN > cols) {
                Value trueVal = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
                pred = trueVal;
                if (tileM > rows) {
                    // Check: bR + max_sM_offset < rows (i.e. bR < rows since sM offsets are 0-based)
                    Value rowsV = arith::ConstantIntOp::create(rewriter, loc, rows, 32);
                    Value inR = arith::CmpIOp::create(rewriter, loc,
                        arith::CmpIPredicate::ult, bR, rowsV);
                    pred = arith::AndIOp::create(rewriter, loc, pred, inR);
                    bR = arith::RemUIOp::create(rewriter, loc, bR, rowsV);
                }
                if (tileN > cols) {
                    Value colsV = arith::ConstantIntOp::create(rewriter, loc, cols, 32);
                    Value inC = arith::CmpIOp::create(rewriter, loc,
                        arith::CmpIPredicate::ult, bC, colsV);
                    pred = arith::AndIOp::create(rewriter, loc, pred, inC);
                    bC = arith::RemUIOp::create(rewriter, loc, bC, colsV);
                }
            } else {
                pred = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
            }
            return {bR, bC, pred};
        };

        // Use LinearLayout-based offsets (matches upstream element ordering)
        auto srcOffsets = emitOffsetForLayout(srcEnc, srcTy);
        auto dstOffsets = emitOffsetForLayout(dstEnc, dstTy);

        // Convert to (row, col) pairs
        SmallVector<std::pair<int64_t, int64_t>> srcCoords, dstCoords;
        for (auto &off : srcOffsets)
            srcCoords.push_back({off[0], off[1]});
        for (auto &off : dstOffsets)
            dstCoords.push_back({off[0], off[1]});

        // Unpack source elements
        Value src = adaptor.getSrc();
        SmallVector<Value> srcElems;
        if (auto sTy = dyn_cast<LLVMStructType>(src.getType())) {
            for (unsigned i = 0; i < sTy.getBody().size(); ++i)
                srcElems.push_back(ExtractValueOp::create(rewriter, loc,
                    sTy.getBody()[i], src, ArrayRef<int64_t>{(int64_t)i}));
        } else {
            srcElems = {src};
        }

        if (srcElems.size() != srcCoords.size())
            return failure();

        auto [srcBaseRow, srcBaseCol, srcPred] = makeBase(srcEnc);
        auto [dstBaseRow, dstBaseCol, dstPred] = makeBase(dstEnc);

        // Flat index helper
        auto flatIdx = [&](Value bR, Value bC, int64_t rOff, int64_t cOff) -> Value {
            Value r = arith::AddIOp::create(rewriter, loc, bR,
                arith::ConstantIntOp::create(rewriter, loc, rOff, 32));
            Value c = arith::AddIOp::create(rewriter, loc, bC,
                arith::ConstantIntOp::create(rewriter, loc, cOff, 32));
            Value f = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, r,
                    arith::ConstantIntOp::create(rewriter, loc, cols, 32)), c);
            return arith::ExtUIOp::create(rewriter, loc, i64Ty, f);
        };

        // Scatter source elements (guarded by in-bounds predicate)
        {
            auto [prevBlock, ifBlock, thenBlock] =
                createIfBlock(rewriter, loc, srcPred);
            (void)prevBlock;
            rewriter.setInsertionPointToStart(ifBlock);
            for (size_t i = 0; i < srcElems.size(); ++i) {
                auto [rOff, cOff] = srcCoords[i];
                Value idx = flatIdx(srcBaseRow, srcBaseCol, rOff, cOff);
                Value gep = LLVM::GEPOp::create(rewriter, loc,
                    tgPtrTy, elemTy, tgPtr, ArrayRef<LLVM::GEPArg>{idx});
                LLVM::StoreOp::create(rewriter, loc, srcElems[i], gep);
            }
            rewriter.setInsertionPointToStart(thenBlock);
        }

        // Barrier
        Value fenceTG = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
        Value execMod = arith::ConstantIntOp::create(rewriter, loc, 4, 32);
        LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

        // Gather destination elements
        SmallVector<Value> dstElems;
        for (size_t i = 0; i < dstCoords.size(); ++i) {
            auto [rOff, cOff] = dstCoords[i];
            Value idx = flatIdx(dstBaseRow, dstBaseCol, rOff, cOff);
            Value gep = LLVM::GEPOp::create(rewriter, loc,
                tgPtrTy, elemTy, tgPtr, ArrayRef<LLVM::GEPArg>{idx});
            dstElems.push_back(LLVM::LoadOp::create(rewriter, loc, elemTy, gep).getResult());
        }

        // Pack result
        auto outTy = getTypeConverter()->convertType(dstTy);
        if (!outTy) return failure();

        if (auto outSt = dyn_cast<LLVMStructType>(outTy)) {
            if (outSt.getBody().size() != dstElems.size())
                return failure();
            Value result = UndefOp::create(rewriter, loc, outSt);
            for (size_t i = 0; i < dstElems.size(); ++i)
                result = InsertValueOp::create(rewriter, loc, outSt,
                             result, dstElems[i], ArrayRef<int64_t>{(int64_t)i});
            rewriter.replaceOp(op, result);
        } else {
            rewriter.replaceOp(op, dstElems[0]);
        }
        return success();
    }
};

// Lower triton::AtomicRMWOp → air.atomic.global.{op}.{type}
//
// Metal uses explicit AIR intrinsics for atomics:
//   float @air.atomic.global.add.f32(float addrspace(1)*, float, i32 order, i32 scope, i1 volatile)
//   i32   @air.atomic.global.add.s.i32(i32 addrspace(1)*, i32, i32 order, i32 scope, i1 volatile)
//   i32   @air.atomic.global.max.s.i32(...)
//   i32   @air.atomic.global.min.s.i32(...)
//   i32   @air.atomic.global.xchg.s.i32(...)
//
// For unsupported native atomics (f32 max/min, f16/bf16 add), we emit a CAS loop:
//   air.atomic.global.cmpxchg.weak.i32(ptr, expected_ptr, desired, succ_order, fail_order, scope, vol)
//   returns old i32. Expected is passed by pointer and updated on failure.
struct AtomicRMWOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    // Declare an AIR intrinsic function if not already declared
    LLVMFuncOp declareAIR(ConversionPatternRewriter &rewriter, ModuleOp mod,
                          StringRef name, Type retTy, ArrayRef<Type> argTys) const {
        if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
            return fn;
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        auto fnTy = LLVMFunctionType::get(retTy, argTys, false);
        return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy, Linkage::External);
    }

    // Emit a direct AIR atomic intrinsic call (no CAS loop)
    Value emitDirectAtomic(ConversionPatternRewriter &rewriter, Location loc,
                           ModuleOp mod, StringRef airName, Type valueTy,
                           Value ptr) const {
        auto *ctx = rewriter.getContext();
        auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
        auto i32Ty = IntegerType::get(ctx, 32);
        auto i1Ty  = IntegerType::get(ctx, 1);
        auto fn = declareAIR(rewriter, mod, airName, valueTy,
                             {ptrTy, valueTy, i32Ty, i32Ty, i1Ty});
        // Unused return needed: the call still needs a value operand.
        // Actually this helper is called from emitDirectAtomicCall below.
        (void)fn;
        return {};
    }

    // Emit CAS loop for f32 max/min:
    //   alloca expected
    //   load old from *ptr (via xchg 0 trick or just initial load)
    //   loop:
    //     store old → expected
    //     new_f = max/min(old_f, val_f)
    //     new_i = bitcast new_f → i32
    //     old_i = bitcast old_f → i32
    //     store old_i → expected
    //     old_ret = cmpxchg(ptr_i32, &expected, new_i, ...)
    //     expected_after = load expected
    //     cmp = icmp eq old_ret, old_i (success if unchanged)
    //     br cmp → done, loop
    //   done:
    //     result = bitcast old_ret → float
    Value emitF32CASLoop(ConversionPatternRewriter &rewriter, Location loc,
                         ModuleOp mod, Value ptr, Value val, RMWOp rmwOp) const {
        auto *ctx = rewriter.getContext();
        auto f32Ty = Float32Type::get(ctx);
        auto i32Ty = IntegerType::get(ctx, 32);
        auto i1Ty  = IntegerType::get(ctx, 1);
        auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);  // device
        auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0); // private (alloca)

        // Declare cmpxchg intrinsic
        auto cmpxchgFn = declareAIR(rewriter, mod,
            "air.atomic.global.cmpxchg.weak.i32", i32Ty,
            {ptrTy, ptrTy0, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty});

        // Alloca for expected value (i32)
        Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
        Value expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, i32Ty, one, /*alignment=*/4);

        // Initial load: use xchg to atomically read current value
        // Actually, a simple non-atomic load is fine for the initial guess —
        // the CAS loop will retry if it's stale.
        Value oldI32 = LLVM::LoadOp::create(rewriter, loc, i32Ty, ptr);
        Value oldF32 = LLVM::BitcastOp::create(rewriter, loc, f32Ty, oldI32);

        // Create loop and exit blocks
        Block *currentBlock = rewriter.getInsertionBlock();
        Block *afterBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        Block *loopBlock = rewriter.createBlock(afterBlock);

        // Branch from current block to loop
        rewriter.setInsertionPointToEnd(currentBlock);
        LLVM::BrOp::create(rewriter, loc, ValueRange{oldF32, oldI32}, loopBlock);

        // Loop block: phi for old_f32, old_i32
        loopBlock->addArgument(f32Ty, loc);
        loopBlock->addArgument(i32Ty, loc);
        Value phiOldF32 = loopBlock->getArgument(0);
        Value phiOldI32 = loopBlock->getArgument(1);

        rewriter.setInsertionPointToStart(loopBlock);

        // Compute new value
        Value newF32;
        if (rmwOp == RMWOp::MAX)
            newF32 = LLVM::MaximumOp::create(rewriter, loc, phiOldF32, val);
        else
            newF32 = LLVM::MinimumOp::create(rewriter, loc, phiOldF32, val);

        Value newI32 = LLVM::BitcastOp::create(rewriter, loc, i32Ty, newF32);

        // Store expected (old) into alloca
        LLVM::StoreOp::create(rewriter, loc, phiOldI32, expectedAlloca);

        // CAS: cmpxchg(ptr, &expected, desired, succ_order=0, fail_order=0, scope=2, vol=true)
        Value order0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
        Value scope2 = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
        Value volT   = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

        Value oldRet = LLVM::CallOp::create(rewriter, loc, cmpxchgFn,
            ValueRange{ptr, expectedAlloca, newI32, order0, order0, scope2, volT}).getResult();

        // Check success: old_ret == old_i32 means no other thread changed it
        Value success = LLVM::ICmpOp::create(rewriter, loc,
            LLVM::ICmpPredicate::eq, oldRet, phiOldI32);

        // On failure, the expected alloca now contains the current value
        Value failedOldF32 = LLVM::BitcastOp::create(rewriter, loc, f32Ty, oldRet);

        // Branch: success → afterBlock, failure → loopBlock with new old values
        LLVM::CondBrOp::create(rewriter, loc, success,
            afterBlock, ValueRange{},
            loopBlock, ValueRange{failedOldF32, oldRet});

        // After block: result is the last successful old value (bitcast old_ret to f32)
        // But we need the result in afterBlock. Add a block arg.
        afterBlock->addArgument(f32Ty, loc);
        // Fix: afterBlock needs args from both paths. Actually we always arrive from
        // the success path of the CondBr above. Let me restructure.

        // Actually, CondBrOp success path goes to afterBlock — we need to pass the result.
        // Let me redo: erase the CondBr and rebuild with the right args.
        rewriter.eraseOp(success.getDefiningOp()->getBlock()->getTerminator());
        LLVM::CondBrOp::create(rewriter, loc, success,
            afterBlock, ValueRange{phiOldF32},
            loopBlock, ValueRange{failedOldF32, oldRet});

        rewriter.setInsertionPointToStart(afterBlock);
        return afterBlock->getArgument(0);
    }

    // Emit CAS loop for f16/bf16 atomic add.
    // Strategy: bitcast ptr to i32*, load i32, extract the target half, compute,
    // pack back, cmpxchg i32.
    // Since Triton scalar atomics always target a single element, and the pointer
    // is already to the specific f16/bf16 element, we need to:
    //   1. Align ptr down to i32 boundary
    //   2. Determine which half (low/high) within the i32
    //   3. CAS loop on the i32
    // But actually, Triton's atomic_rmw on f16 gives us a ptr to a single f16.
    // We need to widen to i32 for the CAS. The element could be at an odd offset.
    //
    // Simpler approach: just use i16 CAS if Metal supports it.
    // Metal does NOT have i16 cmpxchg. So we must use i32.
    //
    // For the i32 widening approach:
    //   - ptr_i32 = ptr & ~3  (align down)
    //   - byte_offset = ptr & 3  → 0 or 2
    //   - shift = byte_offset * 8  → 0 or 16
    //   - mask = 0xFFFF << shift
    Value emitF16BF16CASLoop(ConversionPatternRewriter &rewriter, Location loc,
                              ModuleOp mod, Value ptr, Value val,
                              Type elemTy, RMWOp rmwOp) const {
        auto *ctx = rewriter.getContext();
        auto i16Ty = IntegerType::get(ctx, 16);
        auto i32Ty = IntegerType::get(ctx, 32);
        auto i64Ty = IntegerType::get(ctx, 64);
        auto i1Ty  = IntegerType::get(ctx, 1);
        auto f32Ty = Float32Type::get(ctx);
        auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
        auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0);

        auto cmpxchgFn = declareAIR(rewriter, mod,
            "air.atomic.global.cmpxchg.weak.i32", i32Ty,
            {ptrTy, ptrTy0, i32Ty, i32Ty, i32Ty, i32Ty, i1Ty});

        // Alloca for expected
        Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
        Value expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, i32Ty, one, /*alignment=*/4);

        // Compute aligned i32 pointer and shift amount
        // ptr_as_int = ptrtoint ptr
        Value ptrInt = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, ptr);
        // byte_offset = ptr_as_int & 3
        Value three64 = arith::ConstantIntOp::create(rewriter, loc, 3, 64);
        Value byteOff64 = LLVM::AndOp::create(rewriter, loc, ptrInt, three64);
        // aligned_ptr_int = ptr_as_int & ~3
        Value notThree64 = arith::ConstantIntOp::create(rewriter, loc, ~3LL, 64);
        Value alignedInt = LLVM::AndOp::create(rewriter, loc, ptrInt, notThree64);
        Value alignedPtr = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, alignedInt);
        // shift = byte_offset * 8 (in i32)
        Value byteOff32 = LLVM::TruncOp::create(rewriter, loc, i32Ty, byteOff64);
        Value eight = arith::ConstantIntOp::create(rewriter, loc, 8, 32);
        Value shift = LLVM::MulOp::create(rewriter, loc, byteOff32, eight);
        // mask = 0xFFFF << shift
        Value mask16 = arith::ConstantIntOp::create(rewriter, loc, 0xFFFF, 32);
        Value mask = LLVM::ShlOp::create(rewriter, loc, mask16, shift);
        Value notMask = LLVM::XOrOp::create(rewriter, loc, mask,
            arith::ConstantIntOp::create(rewriter, loc, -1, 32));

        // Convert val to f32 for computation, then back
        // Actually: val is already f16 or bf16. We do the add in f32 for simplicity.
        Value valF32;
        if (elemTy.isF16())
            valF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, val);
        else // bf16
            valF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, val);

        // Initial load
        Value oldI32 = LLVM::LoadOp::create(rewriter, loc, i32Ty, alignedPtr);

        // Create loop and exit blocks
        Block *currentBlock = rewriter.getInsertionBlock();
        Block *afterBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        Block *loopBlock = rewriter.createBlock(afterBlock);

        rewriter.setInsertionPointToEnd(currentBlock);
        LLVM::BrOp::create(rewriter, loc, ValueRange{oldI32}, loopBlock);

        loopBlock->addArgument(i32Ty, loc);
        Value phiOldI32 = loopBlock->getArgument(0);

        rewriter.setInsertionPointToStart(loopBlock);

        // Extract the target i16 from the i32 word
        Value shifted = LLVM::LShrOp::create(rewriter, loc, phiOldI32, shift);
        Value oldI16 = LLVM::TruncOp::create(rewriter, loc, i16Ty, shifted);

        // Convert old i16 to f32
        Value oldF32;
        if (elemTy.isF16()) {
            Value oldF16 = LLVM::BitcastOp::create(rewriter, loc, Float16Type::get(ctx), oldI16);
            oldF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, oldF16);
        } else {
            Value oldBF16 = LLVM::BitcastOp::create(rewriter, loc, BFloat16Type::get(ctx), oldI16);
            oldF32 = arith::ExtFOp::create(rewriter, loc, f32Ty, oldBF16);
        }

        // Compute: add in f32
        Value newF32 = arith::AddFOp::create(rewriter, loc, oldF32, valF32);

        // Convert back to i16
        Value newI16;
        if (elemTy.isF16()) {
            Value newF16 = arith::TruncFOp::create(rewriter, loc, Float16Type::get(ctx), newF32);
            newI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, newF16);
        } else {
            Value newBF16 = arith::TruncFOp::create(rewriter, loc, BFloat16Type::get(ctx), newF32);
            newI16 = LLVM::BitcastOp::create(rewriter, loc, i16Ty, newBF16);
        }

        // Pack back into i32: (old & ~mask) | (new_i16_zext << shift)
        Value newI32Ext = LLVM::ZExtOp::create(rewriter, loc, i32Ty, newI16);
        Value newShifted = LLVM::ShlOp::create(rewriter, loc, newI32Ext, shift);
        Value cleared = LLVM::AndOp::create(rewriter, loc, phiOldI32, notMask);
        Value newI32 = LLVM::OrOp::create(rewriter, loc, cleared, newShifted);

        // Store expected, call cmpxchg
        LLVM::StoreOp::create(rewriter, loc, phiOldI32, expectedAlloca);
        Value order0 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
        Value scope2 = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
        Value volT   = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

        Value oldRet = LLVM::CallOp::create(rewriter, loc, cmpxchgFn,
            ValueRange{alignedPtr, expectedAlloca, newI32, order0, order0, scope2, volT}).getResult();

        Value success = LLVM::ICmpOp::create(rewriter, loc,
            LLVM::ICmpPredicate::eq, oldRet, phiOldI32);

        LLVM::CondBrOp::create(rewriter, loc, success,
            afterBlock, ValueRange{},
            loopBlock, ValueRange{oldRet});

        // After block: return the old element value (before our update)
        rewriter.setInsertionPointToStart(afterBlock);

        // Extract old element from the last successful i32
        // We need the pre-update value. The phiOldI32 at success is the value
        // that matched. Extract the element from it.
        // Actually, we need to pass the extracted old value out. Let me add a block arg.

        // Reconstruct: on success, phiOldI32 was the matched expected.
        // The element we care about is oldI16 (extracted above). But that's in the loop block.
        // Simpler: add afterBlock arg with the old f16/bf16 value.

        // Redo: erase terminator and rebuild
        // The loop block terminator is the CondBrOp we just created.
        // We need to pass oldI16 to afterBlock on success.
        afterBlock->addArgument(elemTy, loc);

        auto *term = loopBlock->getTerminator();
        rewriter.setInsertionPoint(term);

        // Convert oldI16 to the element type
        Value oldElem;
        if (elemTy.isF16())
            oldElem = LLVM::BitcastOp::create(rewriter, loc, Float16Type::get(ctx), oldI16);
        else
            oldElem = LLVM::BitcastOp::create(rewriter, loc, BFloat16Type::get(ctx), oldI16);

        rewriter.eraseOp(term);
        LLVM::CondBrOp::create(rewriter, loc, success,
            afterBlock, ValueRange{oldElem},
            loopBlock, ValueRange{oldRet});

        rewriter.setInsertionPointToStart(afterBlock);
        return afterBlock->getArgument(0);
    }

    LogicalResult matchAndRewrite(
        triton::AtomicRMWOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {

        auto loc = op.getLoc();
        auto *ctx = op.getContext();
        auto mod = op->getParentOfType<ModuleOp>();

        Value llPtr = adaptor.getPtr();
        Value llVal = adaptor.getVal();
        Value llMask = adaptor.getMask();

        // Only handle scalar atomics for now
        if (isa<RankedTensorType>(op.getType()))
            return failure();

        auto rmwOp = op.getAtomicRmwOp();

        // Determine AIR intrinsic name or CAS loop
        Type valueElemTy = getTypeConverter()->convertType(op.getType());
        std::string airName;
        bool needsCAS = false;
        if (valueElemTy.isF32()) {
            switch (rmwOp) {
            case RMWOp::FADD: airName = "air.atomic.global.add.f32"; break;
            case RMWOp::XCHG: airName = "air.atomic.global.xchg.f32"; break;
            // Note: float max/min is lowered by Triton frontend to
            // bitcast(f32→i32) + masked atomic_max.s.i32/atomic_umin.u.i32,
            // so RMWOp::MAX/MIN with f32 type should never reach here.
            default: return failure();
            }
        } else if (valueElemTy.isF16() || valueElemTy.isBF16()) {
            switch (rmwOp) {
            case RMWOp::FADD: needsCAS = true; break;
            default: return failure();
            }
        } else if (valueElemTy.isInteger(32)) {
            switch (rmwOp) {
            case RMWOp::ADD:  airName = "air.atomic.global.add.s.i32"; break;
            case RMWOp::MAX:  airName = "air.atomic.global.max.s.i32"; break;
            case RMWOp::MIN:  airName = "air.atomic.global.min.s.i32"; break;
            case RMWOp::UMAX: airName = "air.atomic.global.max.u.i32"; break;
            case RMWOp::UMIN: airName = "air.atomic.global.min.u.i32"; break;
            case RMWOp::AND:  airName = "air.atomic.global.and.s.i32"; break;
            case RMWOp::OR:   airName = "air.atomic.global.or.s.i32"; break;
            case RMWOp::XOR:  airName = "air.atomic.global.xor.s.i32"; break;
            case RMWOp::XCHG: airName = "air.atomic.global.xchg.i32"; break;
            default: return failure();
            }
        } else {
            return failure();
        }

        // CAS loop path
        if (needsCAS) {
            Value result;
            if (valueElemTy.isF32()) {
                result = emitF32CASLoop(rewriter, loc, mod, llPtr, llVal, rmwOp);
            } else {
                result = emitF16BF16CASLoop(rewriter, loc, mod, llPtr, llVal,
                                            valueElemTy, rmwOp);
            }
            rewriter.replaceOp(op, result);
            return success();
        }

        // Direct AIR intrinsic path
        auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);
        auto i32Ty = IntegerType::get(ctx, 32);
        auto i1Ty  = IntegerType::get(ctx, 1);
        auto fnTy  = LLVMFunctionType::get(valueElemTy,
                         {ptrTy, valueElemTy, i32Ty, i32Ty, i1Ty}, false);
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mod.getBody());
            if (!mod.lookupSymbol<LLVMFuncOp>(airName))
                LLVMFuncOp::create(rewriter, mod.getLoc(),
                    airName, fnTy, Linkage::External);
        }
        auto atomicFn = mod.lookupSymbol<LLVMFuncOp>(airName);

        // Args: ptr, value, memory_order=0 (relaxed), scope=2 (device), volatile=true
        Value order   = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
        Value scope   = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
        Value vol     = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

        if (llMask) {
            // Wrap atomic in conditional: if (mask) { atomic } else { undef }
            auto *currentBlock = rewriter.getInsertionBlock();
            auto *afterBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
            auto *atomicBlock = rewriter.createBlock(afterBlock);

            // Add block arg to afterBlock for the result
            afterBlock->addArgument(valueElemTy, loc);

            // Branch: if mask → atomicBlock, else → afterBlock with zero
            rewriter.setInsertionPointToEnd(currentBlock);
            Value zeroVal = LLVM::ConstantOp::create(rewriter, loc, valueElemTy,
                rewriter.getZeroAttr(valueElemTy));
            LLVM::CondBrOp::create(rewriter, loc, llMask,
                atomicBlock, ValueRange{},
                afterBlock, ValueRange{zeroVal});

            // Atomic block: call intrinsic, branch to afterBlock
            rewriter.setInsertionPointToStart(atomicBlock);
            Value atomicResult = LLVM::CallOp::create(rewriter, loc, atomicFn,
                ValueRange{llPtr, llVal, order, scope, vol}).getResult();
            LLVM::BrOp::create(rewriter, loc, ValueRange{atomicResult}, afterBlock);

            rewriter.setInsertionPointToStart(afterBlock);
            rewriter.replaceOp(op, afterBlock->getArgument(0));
        } else {
            Value result = LLVM::CallOp::create(rewriter, loc, atomicFn,
                               ValueRange{llPtr, llVal, order, scope, vol}).getResult();
            rewriter.replaceOp(op, result);
        }
        return success();
    }
};

// Lower triton::AtomicCASOp → air.atomic.global.cmpxchg.weak.{i32,i64}
//
// Metal CAS signature:
//   i32 @air.atomic.global.cmpxchg.weak.i32(
//       ptr addrspace(1) ptr, ptr addrspace(0) expected,
//       i32 desired, i32 succ_order, i32 fail_order, i32 scope, i1 volatile)
// expected is passed by pointer and updated on failure.
// Returns old value.
struct AtomicCASOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        triton::AtomicCASOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {

        auto loc = op.getLoc();
        auto *ctx = op.getContext();
        auto mod = op->getParentOfType<ModuleOp>();

        Value llPtr = adaptor.getPtr();
        Value llCmp = adaptor.getCmp();
        Value llVal = adaptor.getVal();

        // Only handle scalar atomics
        if (isa<RankedTensorType>(op.getType()))
            return failure();

        Type valueTy = getTypeConverter()->convertType(op.getType());

        // Determine intrinsic name based on type width
        std::string airName;
        Type casTy; // type for the CAS operation (may bitcast to this)
        if (valueTy.isInteger(32) || valueTy.isF32()) {
            airName = "air.atomic.global.cmpxchg.weak.i32";
            casTy = IntegerType::get(ctx, 32);
        } else if (valueTy.isInteger(64) || valueTy.isF64()) {
            airName = "air.atomic.global.cmpxchg.weak.i64";
            casTy = IntegerType::get(ctx, 64);
        } else {
            return failure();
        }

        auto ptrTy  = LLVM::LLVMPointerType::get(ctx, 1); // device
        auto ptrTy0 = LLVM::LLVMPointerType::get(ctx, 0); // private
        auto i32Ty  = IntegerType::get(ctx, 32);
        auto i1Ty   = IntegerType::get(ctx, 1);

        // Declare intrinsic
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mod.getBody());
            if (!mod.lookupSymbol<LLVMFuncOp>(airName)) {
                auto fnTy = LLVMFunctionType::get(casTy,
                    {ptrTy, ptrTy0, casTy, i32Ty, i32Ty, i32Ty, i1Ty}, false);
                LLVMFuncOp::create(rewriter, mod.getLoc(), airName, fnTy, Linkage::External);
            }
        }
        auto casFn = mod.lookupSymbol<LLVMFuncOp>(airName);

        // Bitcast cmp/val to integer type if needed (e.g. f32 → i32)
        Value cmpI = llCmp, valI = llVal;
        bool needBitcast = (valueTy != casTy);
        if (needBitcast) {
            cmpI = LLVM::BitcastOp::create(rewriter, loc, casTy, llCmp);
            valI = LLVM::BitcastOp::create(rewriter, loc, casTy, llVal);
        }

        // Alloca for expected value — must be in entry block (not inside loops)
        Value one, expectedAlloca;
        {
            OpBuilder::InsertionGuard guard(rewriter);
            auto &entryBlock = op->getParentOfType<LLVM::LLVMFuncOp>().getBody().front();
            rewriter.setInsertionPointToStart(&entryBlock);
            one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
            expectedAlloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy0, casTy, one, /*alignment=*/4);
        }

        // Store cmp into expected alloca
        LLVM::StoreOp::create(rewriter, loc, cmpI, expectedAlloca);

        // Call cmpxchg: order=0 (relaxed), scope=2 (device), volatile=true
        Value order = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
        Value scope = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
        Value vol   = arith::ConstantIntOp::create(rewriter, loc, 1, 1);

        Value oldI = LLVM::CallOp::create(rewriter, loc, casFn,
            ValueRange{llPtr, expectedAlloca, valI, order, order, scope, vol})
            .getResult();

        // Bitcast back if needed
        Value result = oldI;
        if (needBitcast)
            result = LLVM::BitcastOp::create(rewriter, loc, valueTy, oldI);

        rewriter.replaceOp(op, result);
        return success();
    }
};

// Lower ttg::WarpIdOp → air.dispatch_thread_id[0] / threadsPerWarp.
struct WarpIdOpConversion
    : public mlir::ConvertOpToLLVMPattern<triton::gpu::WarpIdOp> {
    using mlir::ConvertOpToLLVMPattern<triton::gpu::WarpIdOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(
        triton::gpu::WarpIdOp op,
        triton::gpu::WarpIdOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto *ctx = op.getContext();
        auto i32Ty = IntegerType::get(ctx, 32);
        auto mod = op->getParentOfType<ModuleOp>();

        // Use air.thread_position_in_threadgroup (returns [3 x i32]) + extractvalue 0.
        // _add_air_metadata() rewrites this call+extractvalue to a function arg.
        auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
        auto tidFnTy    = LLVMFunctionType::get(arrI32x3Ty, {}, false);
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mod.getBody());
            if (!mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup"))
                LLVMFuncOp::create(rewriter, mod.getLoc(),
                    "air.thread_position_in_threadgroup", tidFnTy, Linkage::External);
        }
        auto tidFn     = mod.lookupSymbol<LLVMFuncOp>("air.thread_position_in_threadgroup");
        Value tidStruct = LLVM::CallOp::create(rewriter, loc, tidFn,
                                                ValueRange{}).getResult();
        Value tid       = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty,
                              tidStruct, ArrayRef<int64_t>{0});

        // warpId = tid / threadsPerWarp (32 on Apple simdgroup)
        int tpw = ttg::lookupThreadsPerWarp(rewriter);
        Value warpSize = arith::ConstantIntOp::create(rewriter, loc, tpw, 32);
        Value warpId = arith::DivUIOp::create(rewriter, loc, tid, warpSize);
        rewriter.replaceOp(op, warpId);
        return success();
    }
};

// Lower triton::GetNumProgramsOp → call @air.threadgroups_per_grid() + extractvalue
// Returns the grid dimension (number of threadgroups) for the given axis.
struct GetNumProgramsOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto *ctx = rewriter.getContext();
        auto i32Ty = IntegerType::get(ctx, 32);
        auto arrTy = LLVM::LLVMArrayType::get(i32Ty, 3);
        auto fnTy = LLVMFunctionType::get(arrTy, {}, false);

        auto fnName = StringRef("air.threadgroups_per_grid");
        auto mod = op->getParentOfType<ModuleOp>();
        if (!mod.lookupSymbol<LLVMFuncOp>(fnName)) {
            OpBuilder b(mod.getBodyRegion());
            b.setInsertionPointToStart(mod.getBody());
            LLVMFuncOp::create(b, mod.getLoc(), fnName, fnTy,
                                Linkage::External);
        }
        auto fn = mod.lookupSymbol<LLVMFuncOp>(fnName);

        Value gridStruct = LLVM::CallOp::create(rewriter, loc, fn,
                                                  ValueRange{}).getResult();
        int axis = static_cast<int>(op.getAxis());
        Value result = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty,
                            gridStruct, ArrayRef<int64_t>{(int64_t)axis});
        rewriter.replaceOp(op, result);
        return success();
    }
};

// Lower triton::FuncOp → LLVM::LLVMFuncOp for Apple Metal kernels.
//
// Metal passes scalar kernel args (i32, i64, etc.) via setBytes — a pointer
// to constant address space (addrspace 2). The LLVM IR must reflect this:
// scalar args become `i32 addrspace(2)*` pointers, and we insert explicit
// loads at function entry. This matches what `xcrun metal` emits for
// `constant T&` parameters, and eliminates the Python regex workaround.
//
// Pointer args (addrspace 1 = device) are passed through unchanged.
struct AppleFuncOpConversion
    : public ConvertOpToLLVMPattern<triton::FuncOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const override {
        auto *ctx = funcOp.getContext();
        auto loc  = funcOp.getLoc();
        bool isKernel = triton::isKernel(funcOp);

        // Build new LLVM arg types.
        // Kernel: scalar i32/i64/etc → addrspace(2)* pointer (Metal constant buffer).
        // Device function: convert types directly, no addrspace wrapping.
        SmallVector<Type> newArgTypes;
        SmallVector<bool> isScalar;
        for (auto argTy : funcOp.getFunctionType().getInputs()) {
            Type converted = getTypeConverter()->convertType(argTy);
            if (!converted) return failure();
            if (isKernel && isa<IntegerType>(converted)) {
                auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addrspace=*/2);
                newArgTypes.push_back(ptrTy);
                isScalar.push_back(true);
            } else {
                newArgTypes.push_back(converted);
                isScalar.push_back(false);
            }
        }

        // Build return type: void for kernels, converted type for device functions.
        Type retTy = LLVM::LLVMVoidType::get(ctx);
        if (!isKernel) {
            auto results = funcOp.getFunctionType().getResults();
            if (results.size() == 1) {
                retTy = getTypeConverter()->convertType(results[0]);
                if (!retTy) return failure();
            } else if (results.size() > 1) {
                // Pack multiple return values into a struct
                SmallVector<Type> memberTypes;
                for (auto resTy : results) {
                    Type converted = getTypeConverter()->convertType(resTy);
                    if (!converted) return failure();
                    memberTypes.push_back(converted);
                }
                retTy = LLVM::LLVMStructType::getLiteral(ctx, memberTypes);
            }
        }

        auto llvmFuncTy = LLVM::LLVMFunctionType::get(retTy, newArgTypes);
        auto newFuncOp = LLVM::LLVMFuncOp::create(
            rewriter, loc, funcOp.getName(), llvmFuncTy,
            LLVM::Linkage::External);

        // Move function body into new func
        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                     newFuncOp.end());

        // Fix up block argument types and insert loads for scalar kernel args
        Block &entryBlock = newFuncOp.getBody().front();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&entryBlock);

        for (unsigned i = 0; i < newArgTypes.size(); ++i) {
            BlockArgument oldArg = entryBlock.getArgument(i);
            if (isScalar[i]) {
                oldArg.setType(newArgTypes[i]);
                auto origTy = getTypeConverter()->convertType(
                    funcOp.getFunctionType().getInput(i));
                Value loaded = LLVM::LoadOp::create(rewriter, loc, origTy, oldArg);
                oldArg.replaceAllUsesExcept(loaded, loaded.getDefiningOp());
            } else {
                oldArg.setType(newArgTypes[i]);
            }
        }

        rewriter.eraseOp(funcOp);
        return success();
    }
};

// Lower triton::PrintOp → no-op (Metal has no printf).
// Erase the op so it doesn't block legalization.
struct ApplePrintOpConversion
    : public ConvertOpToLLVMPattern<triton::PrintOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

// Lower triton::CallOp → LLVM::CallOp for Apple device function calls.
// Unlike CUDA, we don't append shared memory stack pointers.
struct AppleCallOpConversion
    : public ConvertOpToLLVMPattern<triton::CallOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(triton::CallOp callOp, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const override {
        auto loc = callOp.getLoc();
        auto promotedOperands = getTypeConverter()->promoteOperands(
            loc, callOp->getOperands(), adaptor.getOperands(), rewriter);

        // Build result type
        SmallVector<Type> resultTypes;
        for (auto resTy : callOp.getResultTypes()) {
            Type converted = getTypeConverter()->convertType(resTy);
            if (!converted) return failure();
            resultTypes.push_back(converted);
        }

        if (resultTypes.size() <= 1) {
            // Single or void return — direct lowering
            auto newCallOp = LLVM::CallOp::create(
                rewriter, loc, resultTypes.empty() ? TypeRange() : TypeRange(resultTypes),
                promotedOperands, callOp->getAttrs());
            newCallOp.getProperties().setOpBundleSizes(
                rewriter.getDenseI32ArrayAttr({}));
            newCallOp.getProperties().setOperandSegmentSizes(
                {static_cast<int>(promotedOperands.size()), 0});
            rewriter.replaceOp(callOp, newCallOp.getResults());
        } else {
            // Multi-return: call returns a struct, extract each field
            auto *ctx = rewriter.getContext();
            auto structTy = LLVM::LLVMStructType::getLiteral(ctx, resultTypes);
            auto newCallOp = LLVM::CallOp::create(
                rewriter, loc, TypeRange(structTy),
                promotedOperands, callOp->getAttrs());
            newCallOp.getProperties().setOpBundleSizes(
                rewriter.getDenseI32ArrayAttr({}));
            newCallOp.getProperties().setOperandSegmentSizes(
                {static_cast<int>(promotedOperands.size()), 0});

            SmallVector<Value> extracted;
            Value structResult = newCallOp.getResult();
            for (unsigned i = 0; i < resultTypes.size(); ++i) {
                extracted.push_back(LLVM::ExtractValueOp::create(
                    rewriter, loc, resultTypes[i], structResult,
                    ArrayRef<int64_t>{static_cast<int64_t>(i)}));
            }
            rewriter.replaceOp(callOp, extracted);
        }
        return success();
    }
};

struct ConvertTritonAppleGPUToLLVMPass
    : public PassWrapper<ConvertTritonAppleGPUToLLVMPass,
                         OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        ConvertTritonAppleGPUToLLVMPass)

    StringRef getArgument() const override {
        return "convert-triton-apple-gpu-to-llvm";
    }
    StringRef getDescription() const override {
        return "Lower TritonGPU ops (Apple MPS) to LLVM IR";
    }

    void runOnOperation() override {
        auto mod = getOperation();
        auto *ctx = &getContext();

        TargetInfo targetInfo;
        TritonGPUToLLVMTypeConverter typeConverter(ctx, targetInfo);

        // Create global_smem for shared memory (threadgroup addrspace 3).
        // Size comes from ttg.shared attribute set by allocate-shared-memory pass.
        {
            int64_t smemSize = 0;
            if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.shared"))
                smemSize = attr.getValue().getZExtValue();
            // Always create global_smem — some ops (histogram) need it
            // even if allocate-shared-memory didn't set ttg.shared.
            if (smemSize == 0) smemSize = 8;  // minimum
            {
                OpBuilder b(mod.getBodyRegion());
                auto loc = mod.getLoc();
                auto elemTy = typeConverter.convertType(b.getIntegerType(8));
                auto arrayTy = LLVM::LLVMArrayType::get(elemTy, smemSize);
                LLVM::GlobalOp::create(
                    b, loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::Internal,
                    "global_smem", /*value=*/Attribute(), /*alignment=*/16,
                    /*addrSpace=*/3u);
            }
        }

        RewritePatternSet patterns(ctx);
        ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

        // Apple func lowering: kernel args → addrspace(2)* + load, device fns direct.
        // Higher priority than shared FuncOpConversion (which is NVIDIA-specific).
        patterns.add<AppleFuncOpConversion>(
            typeConverter, PatternBenefit(patternBenefitDefault + 20));
        // Apple call lowering: no shared memory stack pointer appending.
        patterns.add<AppleCallOpConversion>(
            typeConverter, PatternBenefit(patternBenefitDefault + 20));

        // Shared Triton → LLVM patterns (handles device functions, non-kernel)
        mlir::triton::populateFuncOpConversionPattern(
            typeConverter, patterns, targetInfo, patternBenefitDefault);
        mlir::triton::populateSPMDOpToLLVMPattern(
            typeConverter, patterns, targetInfo, patternBenefitDefault);
        mlir::triton::populateMemoryOpToLLVMPatterns(
            typeConverter, targetInfo, patterns, patternBenefitDefault);
        mlir::triton::populateMakeRangeOpToLLVMPattern(
            typeConverter, targetInfo, patterns, patternBenefitDefault);
        mlir::triton::populateControlFlowOpToLLVMPattern(
            typeConverter, patterns, targetInfo, patternBenefitDefault);
        mlir::triton::populateConvertLayoutOpToLLVMPatterns(
            typeConverter, targetInfo, patterns, patternBenefitDefault);
        mlir::triton::populateReduceOpToLLVMPatterns(
            typeConverter, patterns, targetInfo, patternBenefitDefault);
        mlir::triton::populateScanOpToLLVMPatterns(
            typeConverter, patterns, targetInfo, patternBenefitDefault);

        // Histogram → shared memory atomics + barrier
        mlir::triton::populateHistogramOpToLLVMPatterns(
            typeConverter, patterns, targetInfo, patternBenefitDefault);

        // Apple-specific patterns
        populateDotOpToLLVMPatterns(typeConverter, patterns,
                                     patternBenefitDefault);
        populateLoadStoreToLLVMPatterns(typeConverter, patterns,
                                         patternBenefitDefault);


        // WarpIdOp → tid / 32 (needed by shared range/layout helpers)
        patterns.add<WarpIdOpConversion>(typeConverter, patternBenefitDefault);

        // PrintOp → no-op (Metal has no printf)
        patterns.add<ApplePrintOpConversion>(typeConverter, patternBenefitDefault + 10);

        // GetNumProgramsOp → air.threadgroups_per_grid
        patterns.add<GetNumProgramsOpAppleConversion>(typeConverter,
            PatternBenefit(patternBenefitDefault + 10));

        // AtomicRMWOp → air.atomic.global.{add,max,min}.{f32,s.i32}
        patterns.add<AtomicRMWOpAppleConversion>(typeConverter, patternBenefitDefault + 10);

        // AtomicCASOp → air.atomic.global.cmpxchg.weak.{i32,i64}
        patterns.add<AtomicCASOpAppleConversion>(typeConverter, patternBenefitDefault + 10);

        // Identity convert_layout for DotOperandEncoding (higher priority than
        // shared upstream convert_layout patterns which are NVIDIA-specific).
        patterns.add<ConvertLayoutOpAppleConversion>(
            typeConverter, PatternBenefit(patternBenefitDefault + 10));

        // Standard dialect lowerings — arith first, then Triton view patterns
        // override arith::ConstantOp for tensor splats (higher benefit).
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter,
                                                            patterns);
        // Triton elementwise + view patterns override arith scalar patterns
        // for tensor types (higher benefit wins for same op).
        mlir::triton::populateElementwiseOpToLLVMPatterns(
            typeConverter, patterns, axisInfoAnalysis, targetInfo,
            patternBenefitDefault + 1);
        mlir::triton::populateClampFOpToLLVMPattern(
            typeConverter, patterns, axisInfoAnalysis, targetInfo,
            patternBenefitDefault + 1);
#define POPULATE_FLOAT_OP(SRC_OP, DST_OP)                                      \
        patterns.add<mlir::triton::gpu::ElementwiseOpConversion<SRC_OP, DST_OP>>(\
            typeConverter, axisInfoAnalysis, patternBenefitDefault + 1)
        POPULATE_FLOAT_OP(arith::AddFOp,   LLVM::FAddOp);
        POPULATE_FLOAT_OP(arith::SubFOp,   LLVM::FSubOp);
        POPULATE_FLOAT_OP(arith::MulFOp,   LLVM::FMulOp);
        POPULATE_FLOAT_OP(arith::DivFOp,   LLVM::FDivOp);
        POPULATE_FLOAT_OP(triton::PreciseDivFOp, LLVM::FDivOp);
        POPULATE_FLOAT_OP(arith::ExtFOp,   LLVM::FPExtOp);
        POPULATE_FLOAT_OP(arith::TruncFOp, LLVM::FPTruncOp);
        POPULATE_FLOAT_OP(arith::SIToFPOp, LLVM::SIToFPOp);
        POPULATE_FLOAT_OP(arith::FPToSIOp, LLVM::FPToSIOp);
#undef POPULATE_FLOAT_OP
        mlir::triton::populateViewOpToLLVMPatterns(
            typeConverter, patterns, patternBenefitDefault + 1);
        // Expand math::ErfOp to polynomial approximation before MathToLLVM
        // (there is no llvm.erf intrinsic — NVIDIA uses libdevice, we expand inline)
        mlir::populatePolynomialApproximateErfPattern(patterns);
        mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                               patterns);
        mlir::index::populateIndexToLLVMConversionPatterns(typeConverter,
                                                            patterns);
        mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

        // Conversion target: everything must lower to LLVM dialect
        ConversionTarget target(*ctx);
        target.addIllegalDialect<triton::TritonDialect>();
        target.addIllegalDialect<triton::gpu::TritonGPUDialect>();
        target.addIllegalDialect<applegpu::TritonAppleGPUDialect>();
        target.addIllegalDialect<mlir::arith::ArithDialect>();
        target.addLegalDialect<LLVM::LLVMDialect>();
        // gpu.thread_id is emitted by shared make_range/SPMD patterns;
        // it will be lowered to air intrinsics by a subsequent pass.
        target.addLegalOp<mlir::gpu::ThreadIdOp>();
        target.addLegalOp<mlir::gpu::BarrierOp>();
        target.addLegalOp<mlir::UnrealizedConversionCastOp>();

        if (failed(applyPartialConversion(mod, target, std::move(patterns))))
            signalPassFailure();
    }
};

// ── LowerGPUToAirPass ─────────────────────────────────────────────────────
//
// Converts remaining gpu.thread_id / gpu.block_dim ops (emitted by shared
// Triton patterns like make_range / SPMD) to air intrinsics / constants so
// the MLIR module is pure LLVM dialect before llvm::toModule().
//
//   gpu.thread_id x  →  call @air.dispatch_thread_id[0]() : i32, index_cast
//   gpu.thread_id y/z → arith.constant 0 : index
//   gpu.block_dim x  →  arith.constant <numThreads> : index   (from module attr)
//   gpu.block_dim y/z → arith.constant 1 : index
//
struct LowerGPUToAirPass
    : public PassWrapper<LowerGPUToAirPass, OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGPUToAirPass)

    StringRef getArgument() const override { return "lower-gpu-to-air"; }
    StringRef getDescription() const override {
        return "Lower gpu.thread_id / gpu.block_dim to air intrinsics / constants";
    }

    void runOnOperation() override {
        ModuleOp mod = getOperation();
        auto *ctx = &getContext();
        auto i32Ty  = IntegerType::get(ctx, 32);

        // Declare air.thread_position_in_threadgroup once at module start.
        // Returns [3 x i32]; we extractvalue index 0 for the flat thread ID.
        // _add_air_metadata() rewrites this call+extractvalue pattern to an arg.
        auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
        auto tidFnName  = StringRef("air.thread_position_in_threadgroup");
        auto tidFnTy    = LLVMFunctionType::get(arrI32x3Ty, {}, false);
        if (!mod.lookupSymbol<LLVMFuncOp>(tidFnName)) {
            OpBuilder b(mod.getBodyRegion());
            b.setInsertionPointToStart(mod.getBody());
            LLVMFuncOp::create(b, mod.getLoc(), tidFnName, tidFnTy,
                                Linkage::External);
        }
        auto tidFn = mod.lookupSymbol<LLVMFuncOp>(tidFnName);

        // Read total thread count from module attributes for gpu.block_dim.
        int64_t threadsPerWarp = 32;
        int64_t numWarps       = 4;
        if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.threads-per-warp"))
            threadsPerWarp = a.getInt();
        if (auto a = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
            numWarps = a.getInt();
        int64_t totalThreads = threadsPerWarp * numWarps;

        IRRewriter rewriter(ctx);

        // Walk and replace gpu.thread_id / gpu.block_dim
        // gpu.thread_id/block_dim return `index` type. Downstream users (e.g.
        // make_range) have already been lowered to LLVM i64/i32 ops by this
        // point. We need to produce a value of the same `index` type and let
        // the existing index-to-llvm lowering handle it — but that already
        // ran. So we emit LLVM ops directly:
        //   gpu.thread_id x → llvm.call @air.dispatch_thread_id[0]() → i32
        //                   → llvm.zext i32 → i64  (index = i64 in LLVM)
        //   gpu.thread_id y/z → llvm.mlir.constant(0 : i64)
        //   gpu.block_dim x  → llvm.mlir.constant(totalThreads : i64)
        //   gpu.block_dim y/z → llvm.mlir.constant(1 : i64)
        //
        // The `index` type maps to i64 in the LLVM type system (index-bitwidth=0
        // means native pointer width = 64-bit on Apple Silicon).
        auto i64Ty = IntegerType::get(ctx, 64);

        mod.walk([&](Operation *op) {
            rewriter.setInsertionPoint(op);
            auto loc = op->getLoc();

            if (auto tidOp = dyn_cast<mlir::gpu::ThreadIdOp>(op)) {
                Value replacement;
                if (tidOp.getDimension() == mlir::gpu::Dimension::x) {
                    Value tidStruct = LLVM::CallOp::create(rewriter, loc, tidFn,
                                                            ValueRange{}).getResult();
                    Value i32val    = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty,
                                          tidStruct, ArrayRef<int64_t>{0});
                    // Extend i32 → i64 to match `index` type (pointer width on Apple Silicon).
                    // Use SExt to produce a single instruction without an intermediate SSA:
                    // wrap in a struct-free zext inline by using the i64 directly.
                    // Actually: emit only ExtractValue (i32) then trunc or zext as needed.
                    // The users of gpu.thread_id have already been lowered to expect i64
                    // (via index_to_llvm). Emit zext to match. The extra SSA is OK since
                    // _add_air_metadata's renumbering now correctly handles it.
                    replacement = LLVM::ZExtOp::create(rewriter, loc, i64Ty, i32val);
                } else {
                    replacement = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                        rewriter.getI64IntegerAttr(0));
                }
                rewriter.replaceOp(op, replacement);
                return;
            }

            if (auto bdOp = dyn_cast<mlir::gpu::BlockDimOp>(op)) {
                int64_t val = (bdOp.getDimension() == mlir::gpu::Dimension::x)
                              ? totalThreads : 1;
                Value replacement = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(val));
                rewriter.replaceOp(op, replacement);
                return;
            }

            if (isa<mlir::gpu::BarrierOp>(op)) {
                // gpu.barrier → call @air.wg.barrier(i32 1, i32 1)
                auto voidTy = LLVMVoidType::get(ctx);
                auto barrFnTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
                LLVMFuncOp barrFn;
                if (auto existing = mod.lookupSymbol<LLVMFuncOp>("air.wg.barrier"))
                    barrFn = existing;
                else {
                    OpBuilder::InsertionGuard guard(rewriter);
                    rewriter.setInsertionPointToStart(mod.getBody());
                    barrFn = LLVMFuncOp::create(rewriter, mod.getLoc(),
                        "air.wg.barrier", barrFnTy, Linkage::External);
                }
                Value flags = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                    rewriter.getI32IntegerAttr(1));
                Value scope = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                    rewriter.getI32IntegerAttr(1));
                LLVM::CallOp::create(rewriter, loc, barrFn, ValueRange{flags, scope});
                rewriter.eraseOp(op);
                return;
            }
        });
    }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createConvertTritonAppleGPUToLLVMPass() {
    return std::make_unique<ConvertTritonAppleGPUToLLVMPass>();
}

std::unique_ptr<mlir::Pass> createLowerGPUToAirPass() {
    return std::make_unique<LowerGPUToAirPass>();
}

void registerTritonAppleGPUToLLVMPasses() {
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return std::make_unique<ConvertTritonAppleGPUToLLVMPass>();
    });
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return std::make_unique<LowerGPUToAirPass>();
    });
}

} // namespace mlir::triton::applegpu
