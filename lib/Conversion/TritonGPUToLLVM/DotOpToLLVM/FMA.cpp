#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

using ::mlir::triton::gpu::expandMatrixOrderWithBatch;
using ::mlir::triton::gpu::expandMatrixShapeWithBatch;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;

using ValueTableFMA = std::map<std::tuple<int, int, int>, Value>;

static ValueTableFMA
getValueTableFromStructFMA(Value val, int batch, int nonK, int K,
                           ConversionPatternRewriter &rewriter, Location loc) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  assert(elems.size() == K * nonK * batch);
  int index = 0;
  for (unsigned b = 0; b < batch; ++b)
    for (unsigned k = 0; k < K; ++k)
      for (unsigned i = 0; i < nonK; ++i)
        res[{b, i, k}] = elems[index++];
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
  auto dTensorTy = cast<RankedTensorType>(D.getType());
  auto dElemTy = dTensorTy.getElementType();

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

  int K = aShapePerCTA[2];

  unsigned retSize[3];
  for (int i = 0; i < 3; ++i) {
    unsigned numRep = dShapePerCTA[i] / shapePerCTATile[i];
    numRep = std::max(static_cast<unsigned>(1), numRep);
    retSize[i] = numRep * sizePerThread[i];
  }

  auto has =
      getValueTableFromStructFMA(llA, retSize[0], retSize[1], K, rewriter, loc);
  auto hbs =
      getValueTableFromStructFMA(llB, retSize[0], retSize[2], K, rewriter, loc);

  SmallVector<Value> ret = cc;

  for (unsigned b = 0; b < retSize[0]; ++b)
    for (unsigned m = 0; m < retSize[1]; ++m)
      for (unsigned n = 0; n < retSize[2]; ++n) {
        unsigned idx[] = {b, m, n};
        unsigned linearIdx = 0;
        for (auto dim : llvm::reverse(order)) {
          linearIdx = linearIdx * retSize[dim] + idx[dim];
        }
        for (unsigned k = 0; k < K; ++k) {
          ret[linearIdx] = rewriter.create<LLVM::FMulAddOp>(
              loc, has[{b, m, k}], hbs[{b, n, k}], ret[linearIdx]);
        }
      }

  auto res = packLLElements(loc, typeConverter, ret, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}
