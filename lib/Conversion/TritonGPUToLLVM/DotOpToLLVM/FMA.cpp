#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::expandMatrixOrderWithBatch;
using ::mlir::triton::gpu::expandMatrixShapeWithBatch;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;

using ValueTableFMA = std::map<std::tuple<int, int, int>, Value>;

static ValueTableFMA
getValueTableFromStructFMA(Value val, ArrayRef<unsigned> perTileShape,
                           unsigned kDim, unsigned nonKDim,
                           ConversionPatternRewriter &rewriter, Location loc,
                           ArrayRef<unsigned> order) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  assert(perTileShape.size() == 3);
  assert(elems.size() == product(perTileShape));
  assert(kDim == 1 || kDim == 2);
  assert(nonKDim == 1 || nonKDim == 2);
  const unsigned bDim = 0;

  for (unsigned idx = 0; idx < elems.size(); ++idx) {
    auto spatialIdx = mlir::LLVM::delinearize(idx, perTileShape, order);
    res[{spatialIdx[bDim], spatialIdx[nonKDim], spatialIdx[kDim]}] = elems[idx];
  }
  return res;
}

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());

  SmallVector<int64_t> aShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(aTensorTy)));
  auto dShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(dTensorTy)));

  BlockedEncodingAttr dLayout =
      cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  auto order = expandMatrixOrderWithBatch(dLayout.getOrder());
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread =
      expandMatrixShapeWithBatch(ArrayRef(getSizePerThread(dLayout)));
  auto shapePerCTATile =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTATile(dLayout)));

  unsigned K = aShapePerCTA[2];

  unsigned perThreadShape[3];
  for (int i = 0; i < 3; ++i) {
    unsigned numRep = dShapePerCTA[i] / shapePerCTATile[i];
    numRep = std::max(static_cast<unsigned>(1), numRep);
    perThreadShape[i] = numRep * sizePerThread[i];
  }

  auto has = getValueTableFromStructFMA(
      llA, {perThreadShape[0], perThreadShape[1], K},
      /*kDim*/ 2, /*nonKDim*/ 1, rewriter, loc, order);
  auto hbs = getValueTableFromStructFMA(
      llB, {perThreadShape[0], K, perThreadShape[2]},
      /*kDim*/ 1, /*nonKDim*/ 2, rewriter, loc, order);

  SmallVector<Value> acc = cc;

  for (unsigned b = 0; b < perThreadShape[0]; ++b)
    for (unsigned m = 0; m < perThreadShape[1]; ++m)
      for (unsigned n = 0; n < perThreadShape[2]; ++n) {
        SmallVector<unsigned> multiDimAccumIdx = {b, m, n};
        unsigned linearAccumIdx =
            linearize(multiDimAccumIdx, perThreadShape, order);
        for (unsigned k = 0; k < K; ++k) {
          acc[linearAccumIdx] = rewriter.create<LLVM::FMulAddOp>(
              loc, has[{b, m, k}], hbs[{b, n, k}], acc[linearAccumIdx]);
        }
      }

  auto res = packLLElements(loc, typeConverter, acc, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}
