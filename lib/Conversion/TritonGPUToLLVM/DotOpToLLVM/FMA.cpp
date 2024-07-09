#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

using ValueTableFMA = std::map<std::pair<int, int, int>, Value>;

static ValueTableFMA
getValueTableFromStructFMA(Value val, int K, int n0, int shapePerCTATile,
                           int sizePerThread,
                           ConversionPatternRewriter &rewriter, Location loc,
                           const LLVMTypeConverter *typeConverter, Type type) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTATile)
      for (unsigned mm = 0; mm < sizePerThread; ++mm) {
        res[{m + mm, k}] = elems[index++];
      }
  }
  return res;
}

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto B = op.getB();
  auto C = op.getC();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto bTensorTy = cast<RankedTensorType>(B.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());

  auto aShapePerCTA = getShapePerCTA(aTensorTy);
  auto bShapePerCTA = getShapePerCTA(bTensorTy);

  BlockedEncodingAttr dLayout =
      cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  auto order = dLayout.getOrder();
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = getSizePerThread(dLayout);
  auto shapePerCTATile = getShapePerCTATile(dLayout);

  int Batch = aShapePerCTA[0];
  int K = aShapePerCTA[2];
  int M = aShapePerCTA[1];
  int N = bShapePerCTA[2];

  int bShapePerCTATile = shapePerCTATile[0];
  int bSizePerThread = sizePerThread[0];
  int mShapePerCTATile = shapePerCTATile[1];
  int mSizePerThread = sizePerThread[1];
  int nShapePerCTATile = shapePerCTATile[2];
  int nSizePerThread = sizePerThread[2];

  auto has = getValueTableFromStructFMA(llA, Batch, K, M, mShapePerCTATile,
                                        mSizePerThread, rewriter, loc,
                                        typeConverter, aTensorTy);
  auto hbs =
      getValueTableFromStructFMA(llB, K, N, nShapePerCTATile, nSizePerThread,
                                 rewriter, loc, typeConverter, bTensorTy);

  SmallVector<Value> ret = cc;
  // Number of elements stored over given dimension in ret array
  // i.e. each dot produces 3d matrix of shape [Batch, M, N],
  // Each thread holds part of this matrix, which is 3d as well
  int retDimSize[] = {Batch / bShapePerCTATile * bSizePerThread,
                      M / mShapePerCTATile * mSizePerThread,
                      N / nShapePerCTATile * nSizePerThread};

  for (unsigned k = 0; k < K; ++k)
    for (unsigned b = 0; b < Batch; b += bShapePerCTATile)
      for (unsigned m = 0; m < M; m += mShapePerCTATile)
        for (unsigned n = 0; n < N; n += nShapePerCTATile)
          for (unsigned bb = 0; bb < bSizePerThread; ++bb)
            for (unsigned mm = 0; mm < mSizePerThread; ++mm)
              for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
                int bIdx = b / bShapePerCTATile * bSizePerThread + bb;
                int mIdx = m / mShapePerCTATile * mSizePerThread + mm;
                int nIdx = n / nShapePerCTATile * nSizePerThread + nn;
                int idx[] = {bIdx, mIdx, nIdx};

                int z = 0;
                for (int i = 0; i < order.size(); i++) {
                  int dim = order[i];
                  z = z * retDimSize[dim] + idx[dim];
                }
                ret[z] = rewriter.create<LLVM::FMulAddOp>(
                    loc, has[{m + mm, k}], hbs[{n + nn, k}], ret[z]);
              }

  auto res = packLLElements(loc, typeConverter, ret, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}
