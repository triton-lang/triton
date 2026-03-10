// DotOpToLLVM: lower tt.dot to air simdgroup matrix intrinsics via TG memory.
//
// Strategy (TG scatter/gather):
//   1. All threads scatter their A/B/C elements into threadgroup memory.
//   2. Threadgroup barrier to synchronise.
//   3. Each simdgroup loads 8×8 tiles and does MMA with K-accumulation.
//   4. Barrier, then each thread gathers its C elements back.
//
// Supports arbitrary M×K × K×N where M,N,K are multiples of 8.
// Handles any blocked encoding (reads sizePerThread/threadsPerWarp/warpsPerCTA).

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

static Type getSimdgroupMatrixType(MLIRContext *ctx) {
    return LLVM::getVectorType(Float32Type::get(ctx), 64);
}

static Value makeI64Vec2(OpBuilder &b, Location loc, int64_t a, int64_t b_val) {
    auto ty  = LLVM::getVectorType(IntegerType::get(b.getContext(), 64), 2);
    Value vec = UndefOp::create(b, loc, ty);
    Value va  = arith::ConstantIntOp::create(b, loc, a,     64);
    Value vb  = arith::ConstantIntOp::create(b, loc, b_val, 64);
    Value i0  = arith::ConstantIntOp::create(b, loc, 0, 32);
    Value i1  = arith::ConstantIntOp::create(b, loc, 1, 32);
    vec = InsertElementOp::create(b, loc, ty, vec, va, i0);
    vec = InsertElementOp::create(b, loc, ty, vec, vb, i1);
    return vec;
}

static LLVMFuncOp getOrInsertIntrinsic(ConversionPatternRewriter &rewriter,
                                        ModuleOp mod,
                                        StringRef name, LLVMFunctionType fnTy) {
    if (auto fn = mod.lookupSymbol<LLVMFuncOp>(name))
        return fn;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    return LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy,
                               Linkage::External);
}

static LLVM::GlobalOp getOrCreateTGGlobal(ConversionPatternRewriter &rewriter,
                                            ModuleOp mod,
                                            StringRef name, int64_t size) {
    if (auto g = mod.lookupSymbol<LLVM::GlobalOp>(name))
        return g;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    auto f32Ty = Float32Type::get(mod.getContext());
    auto arrTy = LLVMArrayType::get(f32Ty, size);
    return LLVM::GlobalOp::create(rewriter, mod.getLoc(), arrTy,
                                   /*isConstant=*/false,
                                   LLVM::Linkage::Internal,
                                   name,
                                   /*value=*/Attribute(),
                                   /*alignment=*/4,
                                   /*addrspace=*/3u);
}

// Compute the (row, col) coordinates for each element owned by a thread,
// given a blocked encoding and tensor shape.
//
// For blocked encoding with sizePerThread=[sM,sN], threadsPerWarp=[tM,tN],
// warpsPerCTA=[wM,wN], the tile covered by all threads is:
//   tileM = wM * tM * sM,  tileN = wN * tN * sN
//
// Thread (warp w, lane l) owns elements at:
//   For each (sm, sn) in [0,sM) × [0,sN):
//     row = (w / wN) * tM * sM + (l / tN) * sM + sm
//     col = (w % wN) * tN * sN + (l % tN) * sN + sn
//
// If tensor shape exceeds the tile, the pattern repeats (wraps).
// Element index in the struct = linearized over the full shape repetitions.
struct ElemCoord { int64_t row, col, elemIdx; };

static SmallVector<ElemCoord> getBlockedElemCoords(
    ttg::BlockedEncodingAttr enc, ArrayRef<int64_t> shape) {

    auto spt = enc.getSizePerThread();    // [sM, sN]
    auto tpw = enc.getThreadsPerWarp();   // [tM, tN]
    auto wpc = enc.getWarpsPerCTA();      // [wM, wN]

    int64_t sM = spt[0], sN = spt[1];
    int64_t tM = tpw[0], tN = tpw[1];
    int64_t wM = wpc[0], wN = wpc[1];

    int64_t tileM = wM * tM * sM;
    int64_t tileN = wN * tN * sN;

    int64_t M = shape[0], N = shape[1];
    int64_t repsM = (M + tileM - 1) / tileM;
    int64_t repsN = (N + tileN - 1) / tileN;

    int64_t numWarps = wM * wN;
    int64_t numLanes = tM * tN;  // threads per warp
    (void)numWarps;
    (void)numLanes;

    // Total elements per thread
    int64_t elemsPerThread = repsM * repsN * sM * sN;
    SmallVector<ElemCoord> coords(elemsPerThread);

    int64_t idx = 0;
    for (int64_t rm = 0; rm < repsM; ++rm) {
        for (int64_t rn = 0; rn < repsN; ++rn) {
            for (int64_t sm = 0; sm < sM; ++sm) {
                for (int64_t sn = 0; sn < sN; ++sn) {
                    // These are parametric in (w, l) — we store the
                    // offsets that need to be added to the thread's base position.
                    // base_row(w,l) = (w/wN)*tM*sM + (l/tN)*sM
                    // base_col(w,l) = (w%wN)*tN*sN + (l%tN)*sN
                    // full_row = rm*tileM + base_row + sm
                    // full_col = rn*tileN + base_col + sn
                    coords[idx] = {rm * tileM + sm, rn * tileN + sn, idx};
                    idx++;
                }
            }
        }
    }
    return coords;
}

struct DotOpAppleMmaConversion : public ConvertOpToLLVMPattern<tt::DotOp> {
    using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

    static unsigned &getCounter(MLIRContext *ctx) {
        static llvm::DenseMap<MLIRContext *, unsigned> counters;
        return counters[ctx];
    }

    LogicalResult matchAndRewrite(
        tt::DotOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {

        auto loc = op.getLoc();
        auto ctx = op.getContext();
        auto mod = op->getParentOfType<ModuleOp>();

        auto cType = cast<RankedTensorType>(op.getC().getType());
        auto cEnc = dyn_cast<ttg::BlockedEncodingAttr>(cType.getEncoding());
        if (!cEnc && !isa<AppleMmaEncodingAttr>(cType.getEncoding()))
            return failure();
        if (!cEnc)
            return failure();

        auto aType = cast<RankedTensorType>(op.getA().getType());
        auto bType = cast<RankedTensorType>(op.getB().getType());

        int64_t M = cType.getShape()[0];
        int64_t N = cType.getShape()[1];
        int64_t K = aType.getShape()[1];

        auto f32Ty     = Float32Type::get(ctx);
        auto tgPtrTy   = LLVMPointerType::get(ctx, 3);
        auto matTy     = getSimdgroupMatrixType(ctx);
        auto i32Ty     = IntegerType::get(ctx, 32);
        auto i64Ty     = IntegerType::get(ctx, 64);

        // ── Declare air intrinsics ────────────────────────────────────────

        auto laneIdFn = getOrInsertIntrinsic(rewriter, mod,
            "air.thread_index_in_simdgroup",
            LLVMFunctionType::get(i32Ty, {}, false));

        auto voidTy = LLVMVoidType::get(ctx);
        auto barrTy = LLVMFunctionType::get(voidTy, {i32Ty, i32Ty}, false);
        auto tgBarrFn = getOrInsertIntrinsic(rewriter, mod,
            "air.threadgroup.barrier", barrTy);
        auto sgBarrFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup.barrier", barrTy);

        auto vec2i64Ty = LLVM::getVectorType(IntegerType::get(ctx, 64), 2);
        auto loadFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_load.v64f32.p3f32",
            LLVMFunctionType::get(matTy, {tgPtrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty}, false));
        auto mmaFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
            LLVMFunctionType::get(matTy, {matTy, matTy, matTy}, false));
        auto storeFn = getOrInsertIntrinsic(rewriter, mod,
            "air.simdgroup_matrix_8x8_store.v64f32.p3f32",
            LLVMFunctionType::get(voidTy, {matTy, tgPtrTy, vec2i64Ty, vec2i64Ty, vec2i64Ty}, false));

        // ── Constants ────────────────────────────────────────────────────

        Value shape88  = makeI64Vec2(rewriter, loc, 8, 8);
        Value stride18 = makeI64Vec2(rewriter, loc, 1, 8);
        Value zeroOff  = makeI64Vec2(rewriter, loc, 0, 0);
        Value fenceTG  = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
        Value fenceSG  = arith::ConstantIntOp::create(rewriter, loc, 2, 32);
        Value execMod  = arith::ConstantIntOp::create(rewriter, loc, 4, 32);

        // ── Thread identification ─────────────────────────────────────────

        Value laneId = LLVM::CallOp::create(rewriter, loc, laneIdFn,
                                             ValueRange{}).getResult();

        auto arrI32x3Ty = LLVM::LLVMArrayType::get(i32Ty, 3);
        auto tidFn = getOrInsertIntrinsic(rewriter, mod,
            "air.thread_position_in_threadgroup",
            LLVMFunctionType::get(arrI32x3Ty, {}, false));
        Value tidStruct = LLVM::CallOp::create(rewriter, loc, tidFn,
                                                ValueRange{}).getResult();
        Value tid32 = LLVM::ExtractValueOp::create(rewriter, loc, i32Ty,
                          tidStruct, ArrayRef<int64_t>{0});
        Value c32    = arith::ConstantIntOp::create(rewriter, loc, 32, 32);
        Value warpId = arith::DivUIOp::create(rewriter, loc, tid32, c32);

        // ── Get blocked encoding params for A, B, C ───────────────────────

        // A and B may come through convert_layout (blocked→dot_op) or directly.
        auto getBlockedVal = [&](Value tritonVal, Value adaptorVal) -> Value {
            if (auto cvtOp = tritonVal.getDefiningOp<ttg::ConvertLayoutOp>()) {
                Value mapped = rewriter.getRemappedValue(cvtOp.getSrc());
                if (mapped) return mapped;
            }
            return adaptorVal;
        };

        Value llvmA = getBlockedVal(op.getA(), adaptor.getA());
        Value llvmB = getBlockedVal(op.getB(), adaptor.getB());
        Value llvmC = adaptor.getC();
        if (!llvmA || !llvmB || !llvmC)
            return failure();

        // Unpack struct elements
        auto unpack = [&](Value v) -> SmallVector<Value> {
            SmallVector<Value> elems;
            if (auto sTy = dyn_cast<LLVMStructType>(v.getType())) {
                for (unsigned i = 0; i < sTy.getBody().size(); ++i)
                    elems.push_back(ExtractValueOp::create(rewriter, loc,
                        sTy.getBody()[i], v, ArrayRef<int64_t>{(int64_t)i}));
            } else {
                elems = {v};
            }
            return elems;
        };

        auto elemsA = unpack(llvmA);
        auto elemsB = unpack(llvmB);
        auto elemsC = unpack(llvmC);

        // Get the blocked encoding for A and B.
        // If convert_layout exists, use the source encoding; otherwise use directly.
        auto getBlockedEnc = [](Value v) -> ttg::BlockedEncodingAttr {
            if (auto cvt = v.getDefiningOp<ttg::ConvertLayoutOp>()) {
                auto srcTy = dyn_cast<RankedTensorType>(cvt.getSrc().getType());
                if (srcTy)
                    return dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
            }
            auto ty = dyn_cast<RankedTensorType>(v.getType());
            if (ty) return dyn_cast<ttg::BlockedEncodingAttr>(ty.getEncoding());
            return nullptr;
        };

        auto aSrcEnc = getBlockedEnc(op.getA());
        auto bSrcEnc = getBlockedEnc(op.getB());
        if (!aSrcEnc || !bSrcEnc)
            return failure();

        // ── Compute per-thread element coordinates ────────────────────────
        // These give the static offsets; runtime base comes from (warpId, laneId).

        auto aCoordsStatic = getBlockedElemCoords(aSrcEnc, aType.getShape());
        auto bCoordsStatic = getBlockedElemCoords(bSrcEnc, bType.getShape());
        auto cCoordsStatic = getBlockedElemCoords(cEnc, cType.getShape());

        // Verify element counts match
        if ((int64_t)elemsA.size() != (int64_t)aCoordsStatic.size() ||
            (int64_t)elemsB.size() != (int64_t)bCoordsStatic.size() ||
            (int64_t)elemsC.size() != (int64_t)cCoordsStatic.size())
            return failure();

        // ── Compute runtime thread base position ──────────────────────────
        // For encoding with sizePerThread=[sM,sN], threadsPerWarp=[tM,tN],
        // warpsPerCTA=[wM,wN]:
        //   base_row = (warpId / wN) * tM * sM + (laneId / tN) * sM
        //   base_col = (warpId % wN) * tN * sN + (laneId % tN) * sN

        // Compute base (row, col) for this thread within a tensor of shape [rows, cols].
        // Wraps by tileM/tileN to handle redundant threads.
        auto makeBase = [&](ttg::BlockedEncodingAttr enc, int64_t rows, int64_t cols)
            -> std::pair<Value, Value> {
            auto spt = enc.getSizePerThread();
            auto tpw = enc.getThreadsPerWarp();
            auto wpc = enc.getWarpsPerCTA();

            int64_t sM = spt[0], sN = spt[1];
            int64_t tM = tpw[0], tN = tpw[1];
            int64_t wM = wpc[0], wN = wpc[1];
            int64_t tileM = wM * tM * sM;
            int64_t tileN = wN * tN * sN;

            Value wN_val  = arith::ConstantIntOp::create(rewriter, loc, wN, 32);
            Value tN_val  = arith::ConstantIntOp::create(rewriter, loc, tN, 32);
            Value tMsM    = arith::ConstantIntOp::create(rewriter, loc, tM * sM, 32);
            Value sM_val  = arith::ConstantIntOp::create(rewriter, loc, sM, 32);
            Value tNsN    = arith::ConstantIntOp::create(rewriter, loc, tN * sN, 32);
            Value sN_val  = arith::ConstantIntOp::create(rewriter, loc, sN, 32);

            // warpRow = warpId / wN, warpCol = warpId % wN
            Value warpRow = arith::DivUIOp::create(rewriter, loc, warpId, wN_val);
            Value warpCol = arith::RemUIOp::create(rewriter, loc, warpId, wN_val);

            // laneRow = laneId / tN, laneCol = laneId % tN
            Value laneRow = arith::DivUIOp::create(rewriter, loc, laneId, tN_val);
            Value laneCol = arith::RemUIOp::create(rewriter, loc, laneId, tN_val);

            // base_row = warpRow * tM * sM + laneRow * sM
            Value baseRow = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, warpRow, tMsM),
                arith::MulIOp::create(rewriter, loc, laneRow, sM_val));

            // base_col = warpCol * tN * sN + laneCol * sN
            Value baseCol = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, warpCol, tNsN),
                arith::MulIOp::create(rewriter, loc, laneCol, sN_val));

            // Wrap to handle redundant threads (tileM > rows)
            if (tileM > rows) {
                Value modRow = arith::ConstantIntOp::create(rewriter, loc, rows, 32);
                baseRow = arith::RemUIOp::create(rewriter, loc, baseRow, modRow);
            }
            if (tileN > cols) {
                Value modCol = arith::ConstantIntOp::create(rewriter, loc, cols, 32);
                baseCol = arith::RemUIOp::create(rewriter, loc, baseCol, modCol);
            }

            return {baseRow, baseCol};
        };

        auto [aBaseRow, aBaseCol] = makeBase(aSrcEnc, M, K);
        auto [bBaseRow, bBaseCol] = makeBase(bSrcEnc, K, N);
        auto [cBaseRow, cBaseCol] = makeBase(cEnc, M, N);

        // ── Create threadgroup globals ────────────────────────────────────
        // Full M×K for A, K×N for B, M×N for C

        unsigned id = getCounter(ctx)++;
        auto tgA = getOrCreateTGGlobal(rewriter, mod,
            ("__tg_dot_a_" + llvm::Twine(id)).str(), M * K);
        auto tgB = getOrCreateTGGlobal(rewriter, mod,
            ("__tg_dot_b_" + llvm::Twine(id)).str(), K * N);
        auto tgC = getOrCreateTGGlobal(rewriter, mod,
            ("__tg_dot_c_" + llvm::Twine(id)).str(), M * N);

        Value ptrA = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgA.getName());
        Value ptrB = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgB.getName());
        Value ptrC = LLVM::AddressOfOp::create(rewriter, loc, tgPtrTy, tgC.getName());

        // ── GEP helpers ───────────────────────────────────────────────────

        auto scatter1 = [&](Value ptr, Value val, Value flatIdx64) {
            Value gep = LLVM::GEPOp::create(rewriter, loc,
                tgPtrTy, f32Ty, ptr, ArrayRef<LLVM::GEPArg>{flatIdx64});
            LLVM::StoreOp::create(rewriter, loc, val, gep);
        };

        auto gather1 = [&](Value ptr, Value flatIdx64) -> Value {
            Value gep = LLVM::GEPOp::create(rewriter, loc,
                tgPtrTy, f32Ty, ptr, ArrayRef<LLVM::GEPArg>{flatIdx64});
            return LLVM::LoadOp::create(rewriter, loc, f32Ty, gep).getResult();
        };

        // Helper: compute flat index = (baseRow + staticRowOff) * stride + (baseCol + staticColOff)
        auto flatIdx = [&](Value baseRow, Value baseCol,
                           int64_t rowOff, int64_t colOff, int64_t stride) -> Value {
            Value row32 = arith::AddIOp::create(rewriter, loc, baseRow,
                arith::ConstantIntOp::create(rewriter, loc, rowOff, 32));
            Value col32 = arith::AddIOp::create(rewriter, loc, baseCol,
                arith::ConstantIntOp::create(rewriter, loc, colOff, 32));
            Value flat32 = arith::AddIOp::create(rewriter, loc,
                arith::MulIOp::create(rewriter, loc, row32,
                    arith::ConstantIntOp::create(rewriter, loc, stride, 32)),
                col32);
            return arith::ExtUIOp::create(rewriter, loc, i64Ty, flat32);
        };

        // ── Scatter A into TG_A [M×K row-major] ──────────────────────────
        for (size_t i = 0; i < elemsA.size(); ++i) {
            auto &c = aCoordsStatic[i];
            scatter1(ptrA, elemsA[i], flatIdx(aBaseRow, aBaseCol, c.row, c.col, K));
        }

        // ── Scatter B into TG_B [K×N row-major] ──────────────────────────
        for (size_t i = 0; i < elemsB.size(); ++i) {
            auto &c = bCoordsStatic[i];
            scatter1(ptrB, elemsB[i], flatIdx(bBaseRow, bBaseCol, c.row, c.col, N));
        }

        // ── Scatter C into TG_C [M×N row-major] ──────────────────────────
        SmallVector<Value> resultElems(elemsC.begin(), elemsC.end());
        for (size_t i = 0; i < elemsC.size(); ++i) {
            auto &c = cCoordsStatic[i];
            scatter1(ptrC, elemsC[i], flatIdx(cBaseRow, cBaseCol, c.row, c.col, N));
        }

        // ── Threadgroup barrier ───────────────────────────────────────────
        LLVM::CallOp::create(rewriter, loc, tgBarrFn, ValueRange{fenceTG, execMod});

        // ── MMA: for each (tm, tn) output tile, accumulate over K tiles ──
        int64_t tilesM = M / 8;
        int64_t tilesN = N / 8;
        int64_t tilesK = K / 8;

        for (int64_t tm = 0; tm < tilesM; ++tm) {
            for (int64_t tn = 0; tn < tilesN; ++tn) {
                // Load C tile from TG_C at offset (tn*8, tm*8)
                // simdgroup_load args: shape={cols, rows}, stride={col_stride, row_stride}, offset={col, row}
                Value cOff = makeI64Vec2(rewriter, loc, tn * 8, tm * 8);
                Value cStride = makeI64Vec2(rewriter, loc, 1, N);
                Value cShape  = makeI64Vec2(rewriter, loc, N, 8);
                Value matC = LLVM::CallOp::create(rewriter, loc, loadFn,
                    ValueRange{ptrC, cShape, cStride, cOff}).getResult();

                for (int64_t tk = 0; tk < tilesK; ++tk) {
                    Value aOff = makeI64Vec2(rewriter, loc, tk * 8, tm * 8);
                    Value aStride = makeI64Vec2(rewriter, loc, 1, K);
                    Value aShape  = makeI64Vec2(rewriter, loc, K, 8);
                    Value matA = LLVM::CallOp::create(rewriter, loc, loadFn,
                        ValueRange{ptrA, aShape, aStride, aOff}).getResult();

                    Value bOff = makeI64Vec2(rewriter, loc, tn * 8, tk * 8);
                    Value bStride = makeI64Vec2(rewriter, loc, 1, N);
                    Value bShape  = makeI64Vec2(rewriter, loc, N, 8);
                    Value matB = LLVM::CallOp::create(rewriter, loc, loadFn,
                        ValueRange{ptrB, bShape, bStride, bOff}).getResult();

                    // C += A * B
                    matC = LLVM::CallOp::create(rewriter, loc, mmaFn,
                        ValueRange{matA, matB, matC}).getResult();
                }

                // Store result C tile back
                LLVM::CallOp::create(rewriter, loc, storeFn,
                    ValueRange{matC, ptrC, cShape, cStride, cOff});
            }
        }

        // ── Simdgroup barrier before gather ───────────────────────────────
        LLVM::CallOp::create(rewriter, loc, sgBarrFn, ValueRange{fenceSG, execMod});

        // ── Gather C elements back ────────────────────────────────────────
        for (size_t i = 0; i < elemsC.size(); ++i) {
            auto &c = cCoordsStatic[i];
            resultElems[i] = gather1(ptrC, flatIdx(cBaseRow, cBaseCol, c.row, c.col, N));
        }

        // ── Pack result ───────────────────────────────────────────────────
        auto outLLVMTy = getTypeConverter()->convertType(cType);
        if (!outLLVMTy) return failure();

        if (auto outStructTy = dyn_cast<LLVMStructType>(outLLVMTy)) {
            Value result = UndefOp::create(rewriter, loc, outStructTy);
            for (size_t i = 0; i < resultElems.size(); ++i)
                result = InsertValueOp::create(rewriter, loc, outStructTy,
                             result, resultElems[i],
                             ArrayRef<int64_t>{(int64_t)i});
            rewriter.replaceOp(op, result);
        } else {
            rewriter.replaceOp(op, resultElems[0]);
        }
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
