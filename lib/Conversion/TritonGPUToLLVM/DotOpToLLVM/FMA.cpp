#include "../DotOpToLLVM.h"
#include "../Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;

using ValueTableFMA = std::map<std::pair<int, int>, Value>;

static ValueTableFMA getValueTableFromStructFMA(
    Value val, int K, int n0, int shapePerCTA, int sizePerThread,
    ConversionPatternRewriter &rewriter, Location loc,
    TritonGPUToLLVMTypeConverter *typeConverter, Type type) {
  ValueTableFMA res;
  auto elems = typeConverter->unpackLLElements(loc, val, rewriter, type);
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTA)
      for (unsigned mm = 0; mm < sizePerThread; ++mm) {
        res[{m + mm, k}] = elems[index++];
      }
  }
  return res;
}

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToLLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto B = op.getB();
  auto C = op.getC();
  auto D = op.getResult();

  auto aTensorTy = A.getType().cast<RankedTensorType>();
  auto bTensorTy = B.getType().cast<RankedTensorType>();
  auto dTensorTy = D.getType().cast<RankedTensorType>();

  auto aShape = aTensorTy.getShape();
  auto bShape = bTensorTy.getShape();

  BlockedEncodingAttr dLayout =
      dTensorTy.getEncoding().cast<BlockedEncodingAttr>();
  auto order = dLayout.getOrder();
  auto cc =
      typeConverter->unpackLLElements(loc, adaptor.getC(), rewriter, dTensorTy);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = getSizePerThread(dLayout);
  auto shapePerCTA = getShapePerCTA(dLayout);

  int K = aShape[1];
  int M = aShape[0];
  int N = bShape[1];

  int mShapePerCTA =
      order[0] == 1 ? shapePerCTA[order[1]] : shapePerCTA[order[0]];
  int mSizePerThread =
      order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  int nShapePerCTA =
      order[0] == 0 ? shapePerCTA[order[1]] : shapePerCTA[order[0]];
  int nSizePerThread =
      order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];

  auto has =
      getValueTableFromStructFMA(llA, K, M, mShapePerCTA, mSizePerThread,
                                 rewriter, loc, typeConverter, aTensorTy);
  auto hbs =
      getValueTableFromStructFMA(llB, K, N, nShapePerCTA, nSizePerThread,
                                 rewriter, loc, typeConverter, bTensorTy);

  SmallVector<Value> ret = cc;
  bool isCRow = order[0] == 1;

  for (unsigned k = 0; k < K; k++) {
    for (unsigned m = 0; m < M; m += mShapePerCTA)
      for (unsigned n = 0; n < N; n += nShapePerCTA)
        for (unsigned mm = 0; mm < mSizePerThread; ++mm)
          for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
            int mIdx = m / mShapePerCTA * mSizePerThread + mm;
            int nIdx = n / nShapePerCTA * nSizePerThread + nn;

            int z = isCRow ? mIdx * N / nShapePerCTA * mSizePerThread + nIdx
                           : nIdx * M / mShapePerCTA * nSizePerThread + mIdx;
            ret[z] = rewriter.create<LLVM::FMulAddOp>(loc, has[{m + mm, k}],
                                                      hbs[{n + nn, k}], ret[z]);
          }
  }

  auto res = typeConverter->packLLElements(loc, ret, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}
