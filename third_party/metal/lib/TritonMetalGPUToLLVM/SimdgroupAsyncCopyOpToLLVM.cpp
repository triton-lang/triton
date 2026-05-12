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

static Value
emitAirSimdgroupAsyncCopy2D(ConversionPatternRewriter &rewriter, Location loc,
                            Operation *parentOp,
                            Value dest,   // addrspace(3) ptr
                            Value src,    // addrspace(1) ptr
                            Value stride, // i32, row stride of src
                            ArrayRef<int64_t> tileShape, // [rows, cols]
                            Type elemTy) {
  MLIRContext *ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto i32Ty = IntegerType::get(ctx, 32);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto long2Ty = VectorType::get({2}, i64Ty);
  auto p0Ty = LLVM::LLVMPointerType::get(ctx, 0); // thread (default)
  auto p1Ty = LLVM::LLVMPointerType::get(ctx, 1); // device
  auto p3Ty = LLVM::LLVMPointerType::get(ctx, 3); // threadgroup

  unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
  int64_t rows = tileShape[0];
  int64_t cols = tileShape[1];

  // returns thread ptr _simdgroup_event_t* (addrspace 0, opaque ptr)
  auto funcType = LLVM::LLVMFunctionType::get(
      p0Ty, {
                i64Ty, i64Ty, // elemSize, elemAlign
                p3Ty,         // dest
                i64Ty, i64Ty, // destElemsPerRow, destElemsPerCol (stride)
                long2Ty,      // destTileSize [cols, rows]
                p1Ty,         // src
                i64Ty, i64Ty, // srcElemsPerRow, srcElemsPerCol (stride)
                long2Ty,      // srcTileSize [cols, rows]
                long2Ty,      // matrixOrigin [col, row]
                i32Ty         // transposeFlag
            });

  auto funcOp = appendOrGetExternFuncOp(
      rewriter, parentOp, "air.simdgroup_async_copy_2d.p3i8.p1i8", funcType);

  // cast ptrs to untyped i8* in the right addr space
  Value destI8 = LLVM::BitcastOp::create(rewriter, loc, p3Ty, dest);
  Value srcI8 = LLVM::BitcastOp::create(rewriter, loc, p1Ty, src);

  // src row stride as i64
  Value strideI64 = LLVM::ZExtOp::create(rewriter, loc, i64Ty, stride);

  // tile size vector
  Value tileVec = b.undef(long2Ty);
  tileVec = LLVM::InsertElementOp::create(rewriter, loc, long2Ty, tileVec,
                                          b.i64_val(cols), b.i32_val(0));
  tileVec = LLVM::InsertElementOp::create(rewriter, loc, long2Ty, tileVec,
                                          b.i64_val(rows), b.i32_val(1));

  // origin = {0, 0}: copy starts at [0, 0] in src
  Value originVec = b.undef(long2Ty);
  originVec = LLVM::InsertElementOp::create(rewriter, loc, long2Ty, originVec,
                                            b.i64_val(0), b.i32_val(0));
  originVec = LLVM::InsertElementOp::create(rewriter, loc, long2Ty, originVec,
                                            b.i64_val(0), b.i32_val(1));

  return LLVM::createLLVMCallOp(
             rewriter, loc, funcOp,
             ValueRange{
                 b.i64_val(elemBytes),
                 b.i64_val(elemBytes), // elemSize, elemAlign
                 destI8, b.i64_val(cols),
                 b.i64_val(1), // dest: elemsPerRow=tile_cols, elemsPerCol=1
                               // (row-major)
                 tileVec, srcI8, strideI64,
                 b.i64_val(1), // src: elemsPerRow=matrix stride, elemsPerCol=1
                               // (row-major)
                 tileVec, originVec,
                 b.i32_val(0) // no transpose
             })
      .getResult();
}

struct SimdgroupAsyncCopyOpConversion
    : public ConvertOpToLLVMPattern<ttmetalgpu::SimdgroupAsyncCopyOp> {
  explicit SimdgroupAsyncCopyOpConversion(
      LLVMTypeConverter &typeConverter,
      const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<ttmetalgpu::SimdgroupAsyncCopyOp>(typeConverter,
                                                                 benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(ttmetalgpu::SimdgroupAsyncCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();

    // src: tensor<MxNx!tt.ptr<elemTy>>
    auto srcTensorTy = cast<RankedTensorType>(op.getSrc().getType());
    auto ptrTy = cast<mlir::triton::PointerType>(srcTensorTy.getElementType());
    Type elemTy = ptrTy.getPointeeType();

    // tile shape from dst MemDesc
    auto dstMemDescTy = cast<MemDescType>(op.getDst().getType());
    ArrayRef<int64_t> tileShape = dstMemDescTy.getShape();

    // recover base ptr of src block
    //
    // srcPtrs[0] is ptr to element owned by this thread in the distributed
    // tensor, simdgroup_async_copy_2d needs top left ptr of tile
    auto srcPtrs = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    // get layout from src tensor encoding
    auto srcEnc =
        cast<triton::gpu::BlockedEncodingAttr>(srcTensorTy.getEncoding());
    SmallVector<unsigned> tpw = triton::gpu::getThreadsPerWarp(
        srcTensorTy.getEncoding(), srcTensorTy.getShape());
    SmallVector<unsigned> order =
        triton::gpu::getOrder(srcEnc, srcTensorTy.getShape());
    auto spt = srcEnc.getSizePerThread(); // elements per thread per dim
    // order[0] is the fastest-varying dim
    unsigned innerDim = order[0];      // fastest dim (col for row-major src)
    unsigned outerDim = order[1];      // slowest dim (row)
    unsigned tpwInner = tpw[innerDim]; // threads along col
    unsigned tpwOuter = tpw[outerDim]; // threads along row
    unsigned sptInner = spt[innerDim]; // elems per thread along col
    unsigned sptOuter = spt[outerDim]; // elems per thread along row

    // get threadId (within entire grid) and simdgroupId (warp id within CTA)
    auto func = rewriter.getInsertionBlock()
                    ->getParent()
                    ->getParentOfType<LLVM::LLVMFuncOp>();
    unsigned numArgs = func.getNumArguments();
    Value threadIdxInGridVal =
        func.getArgument(numArgs - mlir::triton::metal::kThreadIdxFromEnd);
    Value simdgroupIdxInThreadgroupVal =
        func.getArgument(numArgs - mlir::triton::metal::kSimdgroupIdxFromEnd);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto i32Ty = IntegerType::get(ctx, 32);
    Value threadIdInGrid =
        LLVM::TruncOp::create(rewriter, loc, i32Ty, threadIdxInGridVal);
    Value simdgroupIdInThreadgroup = LLVM::TruncOp::create(
        rewriter, loc, i32Ty, simdgroupIdxInThreadgroupVal);

    // threadIdInSimdgroup = threadIdInGrid % (tpwInner * tpwOuter)
    Value threadsPerWarpTotal = b.i32_val(tpwInner * tpwOuter);
    Value threadIdInSimdgroup = b.urem(threadIdInGrid, threadsPerWarpTotal);

    // decompose threadIdInSimdgroup into (row, col) coords in simdgroup tile
    Value laneCol = b.urem(threadIdInSimdgroup, b.i32_val(tpwInner));
    Value laneRow = b.udiv(threadIdInSimdgroup, b.i32_val(tpwInner));

    Value threadRow =
        b.add(b.mul(simdgroupIdInThreadgroup, b.i32_val(tpwOuter * sptOuter)),
              b.mul(laneRow, b.i32_val(sptOuter)));
    Value threadCol = b.mul(laneCol, b.i32_val(sptInner));

    Value stride = adaptor.getStride(); // i32, elements per row of src matrix
    Value threadOffset = b.add(b.mul(threadRow, stride), threadCol);
    Value negOffset = b.sub(b.i32_val(0), threadOffset);

    // walk back to base ptr
    Value srcThreadPtr = srcPtrs[0];
    auto p1Ty = LLVM::LLVMPointerType::get(ctx, 1);
    if (srcThreadPtr.getType() != p1Ty)
      srcThreadPtr =
          LLVM::AddrSpaceCastOp::create(rewriter, loc, p1Ty, srcThreadPtr);
    Value srcBase = LLVM::GEPOp::create(rewriter, loc, p1Ty, elemTy,
                                        srcThreadPtr, ValueRange{negOffset});

    // dst smem base pointer
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getDst(), typeConverter->convertType(elemTy), rewriter);
    Value dstBase = smemObj.getBase();
    auto p3Ty = LLVM::LLVMPointerType::get(ctx, 3);
    if (dstBase.getType() != p3Ty)
      dstBase = LLVM::AddrSpaceCastOp::create(rewriter, loc, p3Ty, dstBase);

    Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    Value event =
        emitAirSimdgroupAsyncCopy2D(rewriter, loc, parentOp, dstBase, srcBase,
                                    adaptor.getStride(), tileShape, elemTy);

    rewriter.replaceOp(op, event);
    return success();
  }

protected:
  const mlir::triton::metal::TargetInfo &targetInfo;
};

} // namespace

void populateSimdgroupAsyncCopyOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const mlir::triton::metal::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<SimdgroupAsyncCopyOpConversion>(typeConverter, targetInfo,
                                               benefit);
}
} // namespace mlir::triton::metal
