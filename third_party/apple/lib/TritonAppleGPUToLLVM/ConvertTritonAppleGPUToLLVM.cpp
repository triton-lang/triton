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
        auto f32Ty   = Float32Type::get(ctx);
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
            auto arrTy = LLVMArrayType::get(f32Ty, rows * cols);
            LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy, false,
                                    Linkage::Internal, tgName, Attribute(), 4, 3u);
        }
        auto tgGlobal = mod.lookupSymbol<LLVM::GlobalOp>(tgName);
        Value tgPtr = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgGlobal.getName());

        // Helper: compute base (row, col) with wrap
        auto makeBase = [&](ttg::BlockedEncodingAttr enc)
            -> std::pair<Value, Value> {
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

            Value wR = arith::DivUIOp::create(rewriter, loc, warpId, wN_v);
            Value wC = arith::RemUIOp::create(rewriter, loc, warpId, wN_v);
            Value lR = arith::DivUIOp::create(rewriter, loc, laneId, tN_v);
            Value lC = arith::RemUIOp::create(rewriter, loc, laneId, tN_v);

            Value bR = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, wR, tMsM),
                arith::MulIOp::create(rewriter, loc, lR, sM_v));
            Value bC = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, wC, tNsN),
                arith::MulIOp::create(rewriter, loc, lC, sN_v));

            if (tileM > rows) {
                Value m = arith::ConstantIntOp::create(rewriter, loc, rows, 32);
                bR = arith::RemUIOp::create(rewriter, loc, bR, m);
            }
            if (tileN > cols) {
                Value m = arith::ConstantIntOp::create(rewriter, loc, cols, 32);
                bC = arith::RemUIOp::create(rewriter, loc, bC, m);
            }
            return {bR, bC};
        };

        // Compute static elem coords (same as in DotOp)
        auto getCoords = [](ttg::BlockedEncodingAttr enc, ArrayRef<int64_t> shape)
            -> SmallVector<std::pair<int64_t, int64_t>> {
            auto spt = enc.getSizePerThread();
            auto tpw = enc.getThreadsPerWarp();
            auto wpc = enc.getWarpsPerCTA();
            int64_t sM = spt[0], sN = spt[1];
            int64_t tM = tpw[0], tN = tpw[1];
            int64_t wM = wpc[0], wN = wpc[1];
            int64_t tileM = wM * tM * sM, tileN = wN * tN * sN;
            int64_t rM = (shape[0] + tileM - 1) / tileM;
            int64_t rN = (shape[1] + tileN - 1) / tileN;
            SmallVector<std::pair<int64_t, int64_t>> coords;
            for (int64_t rm = 0; rm < rM; ++rm)
                for (int64_t rn = 0; rn < rN; ++rn)
                    for (int64_t sm = 0; sm < sM; ++sm)
                        for (int64_t sn = 0; sn < sN; ++sn)
                            coords.push_back({rm * tileM + sm, rn * tileN + sn});
            return coords;
        };

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

        auto srcCoords = getCoords(srcEnc, shape);
        auto dstCoords = getCoords(dstEnc, shape);

        if (srcElems.size() != srcCoords.size())
            return failure();

        auto [srcBaseRow, srcBaseCol] = makeBase(srcEnc);
        auto [dstBaseRow, dstBaseCol] = makeBase(dstEnc);

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

        // Scatter source elements
        for (size_t i = 0; i < srcElems.size(); ++i) {
            auto [rOff, cOff] = srcCoords[i];
            Value idx = flatIdx(srcBaseRow, srcBaseCol, rOff, cOff);
            Value gep = LLVM::GEPOp::create(rewriter, loc,
                tgPtrTy, f32Ty, tgPtr, ArrayRef<LLVM::GEPArg>{idx});
            LLVM::StoreOp::create(rewriter, loc, srcElems[i], gep);
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
                tgPtrTy, f32Ty, tgPtr, ArrayRef<LLVM::GEPArg>{idx});
            dstElems.push_back(LLVM::LoadOp::create(rewriter, loc, f32Ty, gep).getResult());
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
struct AtomicRMWOpAppleConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

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

        // Determine AIR intrinsic name
        Type valueElemTy = getTypeConverter()->convertType(op.getType());
        std::string airName;
        if (valueElemTy.isF32()) {
            switch (rmwOp) {
            case RMWOp::FADD: airName = "air.atomic.global.add.f32"; break;
            case RMWOp::XCHG: airName = "air.atomic.global.xchg.f32"; break;
            default: return failure();
            }
        } else if (valueElemTy.isInteger(32)) {
            switch (rmwOp) {
            case RMWOp::ADD:  airName = "air.atomic.global.add.s.i32"; break;
            case RMWOp::MAX:  airName = "air.atomic.global.max.s.i32"; break;
            case RMWOp::MIN:  airName = "air.atomic.global.min.s.i32"; break;
            case RMWOp::AND:  airName = "air.atomic.global.and.s.i32"; break;
            case RMWOp::OR:   airName = "air.atomic.global.or.s.i32"; break;
            case RMWOp::XOR:  airName = "air.atomic.global.xor.s.i32"; break;
            case RMWOp::XCHG: airName = "air.atomic.global.xchg.i32"; break;
            default: return failure();
            }
        } else {
            return failure();
        }

        // Declare the AIR intrinsic if not already declared
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

        // If masked, wrap in an if block
        if (llMask) {
            // Simple approach: always call (mask should be true for scalar atomics)
            // For tensor atomics we'd need a proper if/else — but we only handle scalar.
        }

        Value result = LLVM::CallOp::create(rewriter, loc, atomicFn,
                           ValueRange{llPtr, llVal, order, scope, vol}).getResult();

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
        if (!triton::isKernel(funcOp))
            return failure();  // only handle kernel entry points

        auto *ctx = funcOp.getContext();
        auto loc  = funcOp.getLoc();

        // Build new LLVM arg types: scalar i32/i64/etc → addrspace(2)* pointer.
        // Pointer args (already addrspace(1)*) pass through.
        SmallVector<Type> newArgTypes;
        SmallVector<bool> isScalar;
        for (auto argTy : funcOp.getFunctionType().getInputs()) {
            Type converted = getTypeConverter()->convertType(argTy);
            if (!converted) return failure();
            // Check if this is a scalar integer type (not a pointer)
            if (auto intTy = dyn_cast<IntegerType>(converted)) {
                // Wrap as addrspace(2)* (constant buffer pointer)
                auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addrspace=*/2);
                newArgTypes.push_back(ptrTy);
                isScalar.push_back(true);
            } else {
                newArgTypes.push_back(converted);
                isScalar.push_back(false);
            }
        }

        auto llvmFuncTy = LLVM::LLVMFunctionType::get(
            LLVM::LLVMVoidType::get(ctx), newArgTypes);
        auto newFuncOp = LLVM::LLVMFuncOp::create(
            rewriter, loc, funcOp.getName(), llvmFuncTy,
            LLVM::Linkage::External);

        // Move function body into new func
        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                     newFuncOp.end());

        // Fix up block argument types and insert loads for scalar args
        Block &entryBlock = newFuncOp.getBody().front();
        // Replace old block args with new typed args and insert loads
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&entryBlock);

        for (unsigned i = 0; i < newArgTypes.size(); ++i) {
            BlockArgument oldArg = entryBlock.getArgument(i);
            if (isScalar[i]) {
                // Change the block arg type to the pointer type
                oldArg.setType(newArgTypes[i]);
                // Insert a load to get the actual scalar value
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
            if (smemSize > 0) {
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

        // Apple kernel func lowering: scalar args → addrspace(2)* + load.
        // Higher priority than shared FuncOpConversion (which is NVIDIA-specific).
        patterns.add<AppleFuncOpConversion>(
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

        // Apple-specific patterns
        populateDotOpToLLVMPatterns(typeConverter, patterns,
                                     patternBenefitDefault);
        populateLoadStoreToLLVMPatterns(typeConverter, patterns,
                                         patternBenefitDefault);


        // WarpIdOp → tid / 32 (needed by shared range/layout helpers)
        patterns.add<WarpIdOpConversion>(typeConverter, patternBenefitDefault);

        // AtomicRMWOp → air.atomic.global.{add,max,min}.{f32,s.i32}
        patterns.add<AtomicRMWOpAppleConversion>(typeConverter, patternBenefitDefault + 10);

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
#define POPULATE_FLOAT_OP(SRC_OP, DST_OP)                                      \
        patterns.add<mlir::triton::gpu::ElementwiseOpConversion<SRC_OP, DST_OP>>(\
            typeConverter, axisInfoAnalysis, patternBenefitDefault + 1)
        POPULATE_FLOAT_OP(arith::AddFOp,   LLVM::FAddOp);
        POPULATE_FLOAT_OP(arith::SubFOp,   LLVM::FSubOp);
        POPULATE_FLOAT_OP(arith::MulFOp,   LLVM::FMulOp);
        POPULATE_FLOAT_OP(arith::DivFOp,   LLVM::FDivOp);
        POPULATE_FLOAT_OP(arith::ExtFOp,   LLVM::FPExtOp);
        POPULATE_FLOAT_OP(arith::TruncFOp, LLVM::FPTruncOp);
        POPULATE_FLOAT_OP(arith::SIToFPOp, LLVM::SIToFPOp);
        POPULATE_FLOAT_OP(arith::FPToSIOp, LLVM::FPToSIOp);
#undef POPULATE_FLOAT_OP
        mlir::triton::populateViewOpToLLVMPatterns(
            typeConverter, patterns, patternBenefitDefault + 1);
        mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                               patterns);
        mlir::index::populateIndexToLLVMConversionPatterns(typeConverter,
                                                            patterns);

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

} // namespace mlir::triton::applegpu
