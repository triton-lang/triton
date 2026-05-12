#include "Dialect/TritonMetalGPU/IR/Dialect.h"
#include "TargetInfo.h"
#include "TritonMetalGPUToLLVM/MetalKernelArgs.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::triton::gpu;

namespace ttmetalgpu = mlir::triton::metalgpu;

namespace mlir::triton::metal {

namespace {

static LLVM::LLVMFuncOp
getOrCreateSimdgroupFunc(ConversionPatternRewriter &rewriter, Operation *parent,
                         StringRef name, LLVM::LLVMFunctionType funcType) {
  return appendOrGetExternFuncOp(rewriter, parent, name, funcType);
}

// get <64 x elemTy> simdgroup matrix type (elemTy is f32 or f16)
static VectorType getMatTy(MLIRContext *ctx, Type elemTy) {
  return VectorType::get({64}, elemTy);
}

// returns f32 or f16 for AIR intrinsic names
static std::string getElemSuffix(Type elemTy) {
  if (isa<Float16Type>(elemTy))
    return "f16";
  assert(isa<Float32Type>(elemTy) &&
         "unsupported simdgroup matrix element type");
  return "f32";
}

// get <2 x i64> coordinate type
static VectorType getCoordTy(MLIRContext *ctx) {
  return VectorType::get({2}, IntegerType::get(ctx, 64));
}

struct SimdgroupMMAOpConversion
    : public ConvertOpToLLVMPattern<ttmetalgpu::SimdgroupMMAOp> {
  explicit SimdgroupMMAOpConversion(
      LLVMTypeConverter &typeConverter,
      const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<ttmetalgpu::SimdgroupMMAOp>(typeConverter,
                                                           benefit),
        targetInfo(targetInfo) {}

  Value emitInitFilled(ConversionPatternRewriter &rewriter, Location loc,
                       float fillVal, Type elemTy) const {
    MLIRContext *ctx = rewriter.getContext();
    auto matTy = getMatTy(ctx, elemTy);
    auto funcType = LLVM::LLVMFunctionType::get(matTy, {elemTy});
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    std::string suffix = getElemSuffix(elemTy);
    auto funcOp = getOrCreateSimdgroupFunc(
        rewriter, parentOp,
        "air.simdgroup_matrix_8x8_init_filled.v64" + suffix + "." + suffix,
        funcType);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value fillConst;
    if (isa<Float16Type>(elemTy))
      fillConst = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                           rewriter.getF16FloatAttr(fillVal));
    else
      fillConst = mlir::LLVM::createConstantF32(loc, rewriter, fillVal);
    auto op =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{fillConst});
    return op.getResult();
  }

  static Value emitLoad(ConversionPatternRewriter &rewriter, Location loc,
                        Operation *parentOp, Value ptrBase, Value stride,
                        Value col, Value row, Type elemTy) {
    MLIRContext *ctx = rewriter.getContext();
    auto matTy = getMatTy(ctx, elemTy);
    auto coordTy = getCoordTy(ctx);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx, 3);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i1Ty = IntegerType::get(ctx, 1);
    auto funcType =
        LLVM::LLVMFunctionType::get(matTy, {ptrTy, i64Ty, coordTy, i1Ty});
    std::string suffix = getElemSuffix(elemTy);
    auto funcOp = getOrCreateSimdgroupFunc(rewriter, parentOp,
                                           "air.simdgroup_matrix_8x8_load.v64" +
                                               suffix + ".p3" + suffix,
                                           funcType);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // zext i32 operands to i64 as required by AIR intrinsic
    Value stride64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, stride);
    Value col64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, col);
    Value row64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, row);
    Value coordVec = b.undef(coordTy);
    coordVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, coordVec,
                                             col64, b.i32_val(0));
    coordVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, coordVec,
                                             row64, b.i32_val(1));
    Value falseVal = LLVM::ConstantOp::create(rewriter, loc, i1Ty, (int64_t)0);
    auto op = LLVM::createLLVMCallOp(
                  rewriter, loc, funcOp,
                  ValueRange{ptrBase, stride64, coordVec, falseVal})
                  .getResult();
    return op;
  }

  Value emitMAC(ConversionPatternRewriter &rewriter, Location loc, Value a,
                Value b, Value c, Type inputElemTy, Type outputElemTy) const {
    MLIRContext *ctx = rewriter.getContext();
    auto matInTy = getMatTy(ctx, inputElemTy);
    auto matOutTy = getMatTy(ctx, outputElemTy);
    auto funcType =
        LLVM::LLVMFunctionType::get(matOutTy, {matInTy, matInTy, matOutTy});
    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    std::string inSuffix = getElemSuffix(inputElemTy);
    std::string outSuffix = getElemSuffix(outputElemTy);
    auto funcName = "air.simdgroup_matrix_8x8_multiply_accumulate.v64" +
                    outSuffix + ".v64" + inSuffix + ".v64" + inSuffix + ".v64" +
                    outSuffix;
    auto funcOp =
        getOrCreateSimdgroupFunc(rewriter, parentOp, funcName, funcType);
    return LLVM::createLLVMCallOp(rewriter, loc, funcOp, ValueRange{a, b, c})
        .getResult();
  }

  LogicalResult
  matchAndRewrite(ttmetalgpu::SimdgroupMMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // SimdgroupMMAOp lives in outer loop over inner dim K
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // get base ptrs for smem allocations
    auto aElemTy = cast<MemDescType>(op.getA().getType()).getElementType();
    auto bElemTy = cast<MemDescType>(op.getB().getType()).getElementType();

    auto p3Ty = LLVM::LLVMPointerType::get(ctx, 3);

    auto aSmemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getA(), typeConverter->convertType(aElemTy), rewriter);
    Value aBase = aSmemObj.getBase();
    if (aBase.getType() != p3Ty)
      aBase = LLVM::AddrSpaceCastOp::create(rewriter, loc, p3Ty, aBase);

    auto bSmemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getB(), typeConverter->convertType(bElemTy), rewriter);
    Value bBase = bSmemObj.getBase();
    if (bBase.getType() != p3Ty)
      bBase = LLVM::AddrSpaceCastOp::create(rewriter, loc, p3Ty, bBase);

    // divide tile into smaller 8x8 tiles
    auto outputType = op.getResult().getType();
    auto outputElemTy = outputType.getElementType();
    int64_t outputM = outputType.getDimSize(0);
    int64_t outputN = outputType.getDimSize(1);
    int64_t inputK = cast<MemDescType>(op.getA().getType()).getDimSize(1);

    assert(outputM % 8 == 0 && outputN % 8 == 0 && inputK % 8 == 0 &&
           "block dims (M, N, K) should all be divisible by 8");

    int64_t mTiles = outputM / 8;
    int64_t nTiles = outputN / 8;
    int64_t totalOutputTiles = mTiles * nTiles;
    auto warpsPerCTA = getWarpsPerCTA(outputType);
    auto numWarps = product(warpsPerCTA);
    auto tilesPerSimdgroup = ceil<int64_t>(totalOutputTiles, numWarps);

    auto func = rewriter.getInsertionBlock()
                    ->getParent()
                    ->getParentOfType<LLVM::LLVMFuncOp>();
    unsigned numArgs = func.getNumArguments();

    // build %"struct.metal::simdgroup_matrix" = type { <64 x outputElemTy> }
    Value matAlloca;
    LLVM::LLVMArrayType accumArrTy;
    LLVM::LLVMStructType matStructTy;
    unsigned accumAlign = 64 * (outputElemTy.getIntOrFloatBitWidth() / 8);
    {
      auto vec64OutputElemTy = VectorType::get({64}, outputElemTy);
      matStructTy = LLVM::LLVMStructType::getIdentified(
          ctx, "struct.metal::simdgroup_matrix");
      if (matStructTy.isOpaque())
        (void)matStructTy.setBody({vec64OutputElemTy}, /*isPacked=*/false);

      // alloca [tilesPerSimdgroup x struct] at function entry
      accumArrTy = LLVM::LLVMArrayType::get(matStructTy, tilesPerSimdgroup);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&func.getBody().front());
      matAlloca = LLVM::AllocaOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(ctx, 0), // ptr result
          accumArrTy, b.i32_val(1),
          /*alignment=*/accumAlign);

      // init accumulator to 0 and store into matAlloca
      for (int64_t tileIdx = 0; tileIdx < tilesPerSimdgroup; tileIdx++) {
        Value initVal = emitInitFilled(rewriter, loc, 0.0f, outputElemTy);
        Value accPtr = LLVM::GEPOp::create(
            rewriter, loc, LLVM::LLVMPointerType::get(ctx, 0), accumArrTy,
            matAlloca, ArrayRef<LLVM::GEPArg>{0, (int32_t)tileIdx, 0});

        LLVM::StoreOp::create(rewriter, loc, initVal, accPtr, accumAlign);
      }
    }

    // iterate over 8x8 tiles belonging to this simdgroup
    Value simdgroupIdxInThreadgroupVal =
        func.getArgument(numArgs - mlir::triton::metal::kSimdgroupIdxFromEnd);

    for (int tileIdx = 0; tileIdx < tilesPerSimdgroup; tileIdx++) {
      Value tileIdxVal =
          b.add(simdgroupIdxInThreadgroupVal, b.i32_val(tileIdx * numWarps));

      // guard: only emit if tileIdxVal < totalOutputTiles
      Value inBounds =
          b.icmp_ult(tileIdxVal, b.i32_val((int32_t)totalOutputTiles));
      auto *curBlock = rewriter.getInsertionBlock();
      auto *afterBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
      auto *thenBlock = rewriter.createBlock(afterBlock);
      rewriter.setInsertionPointToEnd(curBlock);
      LLVM::CondBrOp::create(rewriter, loc, inBounds, thenBlock, afterBlock);
      rewriter.setInsertionPointToStart(thenBlock);

      Value nTilesVal = b.i32_val(nTiles);
      Value tileRowVal = b.udiv(tileIdxVal, nTilesVal);
      Value tileColVal = b.urem(tileIdxVal, nTilesVal);

      // iterate over K dimension
      int64_t kTiles = inputK / 8;
      for (int kIdx = 0; kIdx < kTiles; kIdx++) {
        // load from inputs
        Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
        Value inputKVal = b.i32_val(inputK);
        Value outputNVal = b.i32_val(outputN);
        Value kVal = b.mul(b.i32_val(kIdx), b.i32_val(8));
        Value rowVal = b.mul(tileRowVal, b.i32_val(8));
        Value colVal = b.mul(tileColVal, b.i32_val(8));
        Value aLoadResult = emitLoad(rewriter, loc, parentOp, aBase, inputKVal,
                                     kVal, rowVal, aElemTy);
        Value bLoadResult = emitLoad(rewriter, loc, parentOp, bBase, outputNVal,
                                     colVal, kVal, bElemTy);

        Value accPtr = LLVM::GEPOp::create(
            rewriter, loc, LLVM::LLVMPointerType::get(ctx, 0), accumArrTy,
            matAlloca, ArrayRef<LLVM::GEPArg>{0, (int32_t)tileIdx, 0});
        auto vecOutTy = getMatTy(ctx, outputElemTy);
        Value acc =
            LLVM::LoadOp::create(rewriter, loc, vecOutTy, accPtr, accumAlign);

        // call matmul intrinsic
        acc = emitMAC(rewriter, loc, aLoadResult, bLoadResult, acc, aElemTy,
                      outputElemTy);
        LLVM::StoreOp::create(rewriter, loc, acc, accPtr, accumAlign);
      }

      LLVM::BrOp::create(rewriter, loc, afterBlock);
      rewriter.setInsertionPointToStart(afterBlock);
    }

    // real accum data lives in matAlloca and is found by
    // SimdgroupStoreOpToLLVM via function entry block walk
    auto convertedType = typeConverter->convertType(op.getResult().getType());
    Value undefVal = LLVM::UndefOp::create(rewriter, loc, convertedType);
    rewriter.replaceOp(op, undefVal);

    return success();
  }

protected:
  const mlir::triton::metal::TargetInfo &targetInfo;
};

} // namespace

void populateSimdgroupMMAOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<SimdgroupMMAOpConversion>(typeConverter, targetInfo, benefit);
}

} // namespace mlir::triton::metal