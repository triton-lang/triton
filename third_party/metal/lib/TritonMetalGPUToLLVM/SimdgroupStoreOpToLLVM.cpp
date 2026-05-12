#include "Dialect/TritonMetalGPU/IR/Dialect.h"
#include "TargetInfo.h"
#include "TritonMetalGPUToLLVM/MetalKernelArgs.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

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

// get <2 x i64> coordinate type
static VectorType getCoordTy(MLIRContext *ctx) {
  return VectorType::get({2}, IntegerType::get(ctx, 64));
}

// returns f32 or f16 for AIR intrinsic names
static std::string getElemSuffix(Type elemTy) {
  if (isa<Float16Type>(elemTy))
    return "f16";
  assert(isa<Float32Type>(elemTy) &&
         "unsupported simdgroup matrix element type");
  return "f32";
}

static void emitAirSimdgroupStore(ConversionPatternRewriter &rewriter,
                                  Location loc, Operation *parentOp, Value mat,
                                  Value ptrBase, // addrspace(1) ptr
                                  Value stride,  // row stride of dest
                                  Value col, Value row, Type accumElemTy,
                                  Type valueElemTy) {
  MLIRContext *ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto coordTy = getCoordTy(ctx);
  auto ptrTy = LLVM::LLVMPointerType::get(ctx, /*addrspace=*/1);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto i1Ty = IntegerType::get(ctx, 1);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  // cast accum value to valueElemTy if needed (e.g. f32 accum -> f16 store)
  if (accumElemTy != valueElemTy) {
    auto valueMatTy = getMatTy(ctx, valueElemTy);
    mat = LLVM::FPTruncOp::create(rewriter, loc, valueMatTy, mat);
  }

  // use valueElemTy for AIR intrinsic since mat has been cast
  auto matTy = getMatTy(ctx, valueElemTy);
  auto funcType =
      LLVM::LLVMFunctionType::get(voidTy, {matTy, ptrTy, i64Ty, coordTy, i1Ty});
  std::string suffix = getElemSuffix(valueElemTy);
  auto funcOp = getOrCreateSimdgroupFunc(
      rewriter, parentOp,
      "air.simdgroup_matrix_8x8_store.v64" + suffix + ".p3" + suffix, funcType);

  // zext i32 operands to i64 as required by AIR intrinsic
  Value stride64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, stride);
  Value col64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, col);
  Value row64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, row);

  Value coordVec = b.undef(coordTy);
  coordVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, coordVec,
                                           col64, b.i32_val(0));
  coordVec = LLVM::InsertElementOp::create(rewriter, loc, coordTy, coordVec,
                                           row64, b.i32_val(1));
  // TODO in what cases is last param not false?
  Value falseVal = LLVM::ConstantOp::create(rewriter, loc, i1Ty, (int64_t)0);
  LLVM::createLLVMCallOp(
      rewriter, loc, funcOp,
      ValueRange{mat, ptrBase, stride64, coordVec, falseVal});
}

struct SimdgroupStoreOpConversion
    : public ConvertOpToLLVMPattern<ttmetalgpu::SimdgroupStoreOp> {
  explicit SimdgroupStoreOpConversion(
      LLVMTypeConverter &typeConverter,
      const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<ttmetalgpu::SimdgroupStoreOp>(typeConverter,
                                                             benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttmetalgpu::SimdgroupStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // determine which 8x8 tiles this thread's simdgroup are responsible for
    auto valueType = op.getValue().getType();
    auto valueElemTy = valueType.getElementType();
    int64_t valueM = valueType.getDimSize(0);
    int64_t valueN = valueType.getDimSize(1);
    assert(valueM % 8 == 0 && valueN % 8 == 0 &&
           "block dims (M, N) should all be divisible by 8");

    int64_t mTiles = valueM / 8;
    int64_t nTiles = valueN / 8;
    int64_t totalValueTiles = mTiles * nTiles;
    auto warpsPerCTA = getWarpsPerCTA(valueType);
    auto numWarps = product(warpsPerCTA);
    auto tilesPerSimdgroup = ceil<int64_t>(totalValueTiles, numWarps);

    auto func = rewriter.getInsertionBlock()
                    ->getParent()
                    ->getParentOfType<LLVM::LLVMFuncOp>();
    unsigned numArgs = func.getNumArguments();
    Value simdgroupIdxInThreadgroupVal =
        func.getArgument(numArgs - mlir::triton::metal::kSimdgroupIdxFromEnd);

    unsigned accumAlign = 64 * (valueElemTy.getIntOrFloatBitWidth() / 8);

    // adaptor.getPtr() is struct<(ptr<1>)>
    // extract base ptr from struct
    auto p1Ty = LLVM::LLVMPointerType::get(ctx, 1);
    Value ptrBase = LLVM::ExtractValueOp::create(
        rewriter, loc, p1Ty, adaptor.getPtr(), ArrayRef<int64_t>{0});

    // build accum array type
    // store may truncate afterward
    auto accumElemTy = Float32Type::get(ctx);
    auto vec64AccumElemTy = VectorType::get({64}, accumElemTy);
    LLVM::LLVMStructType matStructTy = LLVM::LLVMStructType::getIdentified(
        ctx, "struct.metal::simdgroup_matrix");
    if (matStructTy.isOpaque())
      (void)matStructTy.setBody({vec64AccumElemTy}, /*isPacked=*/false);
    LLVM::LLVMArrayType accumArrTy =
        LLVM::LLVMArrayType::get(matStructTy, tilesPerSimdgroup);

    // find accumulator alloca emitted by SimdgroupMMAOpToLLVM in function entry
    // block
    Value matAlloca;
    for (Operation &entryOp : func.getBody().front()) {
      if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(entryOp)) {
        if (allocaOp.getElemType() == accumArrTy) {
          matAlloca = allocaOp.getResult();
          break;
        }
      }
    }
    assert(matAlloca &&
           "could not find accumulator alloca from SimdgroupMMAOp lowering");
    auto vecOutTy = getMatTy(ctx, accumElemTy);

    // iterate over 8x8 tiles belonging to simdgroup
    for (int tileIdx = 0; tileIdx < tilesPerSimdgroup; tileIdx++) {
      Value tileIdxVal =
          b.add(simdgroupIdxInThreadgroupVal, b.i32_val(tileIdx * numWarps));
      // guard: only emit if tileIdxVal < totalValueTiles
      Value inBounds =
          b.icmp_ult(tileIdxVal, b.i32_val((int32_t)totalValueTiles));
      auto *curBlock = rewriter.getInsertionBlock();
      auto *afterBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
      auto *thenBlock = rewriter.createBlock(afterBlock);
      rewriter.setInsertionPointToEnd(curBlock);
      LLVM::CondBrOp::create(rewriter, loc, inBounds, thenBlock, afterBlock);
      rewriter.setInsertionPointToStart(thenBlock);

      Value nTilesVal = b.i32_val(nTiles);
      Value tileRowVal = b.udiv(tileIdxVal, nTilesVal);
      Value tileColVal = b.urem(tileIdxVal, nTilesVal);

      // convert tile idxs to elem coords
      Value rowVal = b.mul(tileRowVal, b.i32_val(8));
      Value colVal = b.mul(tileColVal, b.i32_val(8));

      Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
      Value accPtr = LLVM::GEPOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(ctx, 0), accumArrTy,
          matAlloca, ArrayRef<LLVM::GEPArg>{0, (int32_t)tileIdx, 0});
      Value acc =
          LLVM::LoadOp::create(rewriter, loc, vecOutTy, accPtr, accumAlign);

      emitAirSimdgroupStore(rewriter, loc, parentOp, acc, ptrBase,
                            adaptor.getStride(), colVal, rowVal, accumElemTy,
                            valueElemTy);

      LLVM::BrOp::create(rewriter, loc, afterBlock);
      rewriter.setInsertionPointToStart(afterBlock);
    }

    rewriter.eraseOp(op);
    return success();
  }

protected:
  const mlir::triton::metal::TargetInfo &targetInfo;
};

} // namespace

void populateSimdgroupStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<SimdgroupStoreOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::metal
