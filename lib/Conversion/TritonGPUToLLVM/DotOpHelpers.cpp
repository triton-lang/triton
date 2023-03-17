#include "DotOpHelpers.h"
#include "TypeConverter.h"

namespace mlir {
namespace LLVM {

SmallVector<Value> DotOpFMAConversionHelper::getThreadIds(
    Value threadId, ArrayRef<unsigned int> shapePerCTA,
    ArrayRef<unsigned int> sizePerThread, ArrayRef<unsigned int> order,
    ConversionPatternRewriter &rewriter, Location loc) const {
  int dim = order.size();
  SmallVector<Value> threadIds(dim);
  for (unsigned k = 0; k < dim - 1; k++) {
    Value dimK = i32_val(shapePerCTA[order[k]] / sizePerThread[order[k]]);
    Value rem = urem(threadId, dimK);
    threadId = udiv(threadId, dimK);
    threadIds[order[k]] = rem;
  }
  Value dimK = i32_val(shapePerCTA[order[dim - 1]]);
  threadIds[order[dim - 1]] = urem(threadId, dimK);
  return threadIds;
}
Value DotOpFMAConversionHelper::loadA(
    Value A, Value llA, BlockedEncodingAttr dLayout, Value thread, Location loc,
    TritonGPUToLLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter) const {
  auto aTensorTy = A.getType().cast<RankedTensorType>();
  auto aLayout = aTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto aShape = aTensorTy.getShape();

  auto aOrder = aLayout.getOrder();
  auto order = dLayout.getOrder();

  bool isARow = aOrder[0] == 1;

  auto aSmem = getSharedMemoryObjectFromStruct(loc, llA, rewriter);
  Value strideAM = aSmem.strides[0];
  Value strideAK = aSmem.strides[1];
  Value strideA0 = isARow ? strideAK : strideAM;
  Value strideA1 = isARow ? strideAM : strideAK;
  int aNumPtr = 8;
  int K = aShape[1];
  int M = aShape[0];

  auto shapePerCTA = getShapePerCTA(dLayout);
  auto sizePerThread = getSizePerThread(dLayout);

  Value _0 = i32_val(0);

  Value mContig = i32_val(sizePerThread[order[1]]);

  // threadId in blocked layout
  auto threadIds =
      getThreadIds(thread, shapePerCTA, sizePerThread, order, rewriter, loc);
  Value threadIdM = threadIds[0];

  Value offA0 = isARow ? _0 : mul(threadIdM, mContig);
  Value offA1 = isARow ? mul(threadIdM, mContig) : _0;
  SmallVector<Value> aOff(aNumPtr);
  for (int i = 0; i < aNumPtr; ++i) {
    aOff[i] = add(mul(offA0, strideA0), mul(offA1, strideA1));
  }
  auto elemTy = A.getType().cast<RankedTensorType>().getElementType();

  Type ptrTy = ptr_ty(elemTy, 3);
  SmallVector<Value> aPtrs(aNumPtr);
  for (int i = 0; i < aNumPtr; ++i)
    aPtrs[i] = gep(ptrTy, aSmem.base, aOff[i]);

  SmallVector<Value> vas;

  int mShapePerCTA = getShapePerCTAForMN(dLayout, true /*isM*/);
  int mSizePerThread = getSizePerThreadForMN(dLayout, true /*isM*/);

  for (unsigned k = 0; k < K; ++k)
    for (unsigned m = 0; m < M; m += mShapePerCTA)
      for (unsigned mm = 0; mm < mSizePerThread; ++mm) {
        Value offset =
            add(mul(i32_val(m + mm), strideAM), mul(i32_val(k), strideAK));
        Value pa = gep(ptrTy, aPtrs[0], offset);
        Value va = load(pa);
        vas.emplace_back(va);
      }

  return getStructFromValueTable(vas, rewriter, loc, typeConverter, elemTy);
}
Value DotOpFMAConversionHelper::loadB(
    Value B, Value llB, BlockedEncodingAttr dLayout, Value thread, Location loc,
    TritonGPUToLLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter) const {
  auto bTensorTy = B.getType().cast<RankedTensorType>();
  auto bLayout = bTensorTy.getEncoding().cast<SharedEncodingAttr>();
  auto bShape = bTensorTy.getShape();

  auto bOrder = bLayout.getOrder();
  auto order = dLayout.getOrder();

  bool isBRow = bOrder[0] == 1;

  auto bSmem = getSharedMemoryObjectFromStruct(loc, llB, rewriter);
  Value strideBN = bSmem.strides[1];
  Value strideBK = bSmem.strides[0];
  Value strideB0 = isBRow ? strideBN : strideBK;
  Value strideB1 = isBRow ? strideBK : strideBN;
  int bNumPtr = 8;
  int K = bShape[0];
  int N = bShape[1];

  auto shapePerCTA = getShapePerCTA(dLayout);
  auto sizePerThread = getSizePerThread(dLayout);

  Value _0 = i32_val(0);

  Value nContig = i32_val(sizePerThread[order[0]]);

  // threadId in blocked layout
  auto threadIds =
      getThreadIds(thread, shapePerCTA, sizePerThread, order, rewriter, loc);
  Value threadIdN = threadIds[1];

  Value offB0 = isBRow ? mul(threadIdN, nContig) : _0;
  Value offB1 = isBRow ? _0 : mul(threadIdN, nContig);
  SmallVector<Value> bOff(bNumPtr);
  for (int i = 0; i < bNumPtr; ++i) {
    bOff[i] = add(mul(offB0, strideB0), mul(offB1, strideB1));
  }
  auto elemTy = B.getType().cast<RankedTensorType>().getElementType();

  Type ptrTy = ptr_ty(elemTy, 3);
  SmallVector<Value> bPtrs(bNumPtr);
  for (int i = 0; i < bNumPtr; ++i)
    bPtrs[i] = gep(ptrTy, bSmem.base, bOff[i]);

  SmallVector<Value> vbs;

  int nShapePerCTA = getShapePerCTAForMN(dLayout, false /*isM*/);
  int nSizePerThread = getSizePerThreadForMN(dLayout, false /*isM*/);

  for (unsigned k = 0; k < K; ++k)
    for (unsigned n = 0; n < N; n += nShapePerCTA)
      for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
        Value offset =
            add(mul(i32_val(n + nn), strideBN), mul(i32_val(k), strideBK));
        Value pb = gep(ptrTy, bPtrs[0], offset);
        Value vb = load(pb);
        vbs.emplace_back(vb);
      }

  return getStructFromValueTable(vbs, rewriter, loc, typeConverter, elemTy);
}
DotOpFMAConversionHelper::ValueTable
DotOpFMAConversionHelper::getValueTableFromStruct(
    Value val, int K, int n0, int shapePerCTA, int sizePerThread,
    ConversionPatternRewriter &rewriter, Location loc,
    TritonGPUToLLVMTypeConverter *typeConverter, Type type) const {
  ValueTable res;
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
Value DotOpFMAConversionHelper::getStructFromValueTable(
    ArrayRef<Value> vals, ConversionPatternRewriter &rewriter, Location loc,
    TritonGPUToLLVMTypeConverter *typeConverter, Type elemTy) const {
  SmallVector<Type> elemTypes(vals.size(), elemTy);
  SmallVector<Value> elems;
  elems.reserve(vals.size());
  for (auto &val : vals) {
    elems.push_back(val);
  }

  Type structTy = struct_ty(elemTypes);
  return typeConverter->packLLElements(loc, elems, rewriter, structTy);
}

} // namespace LLVM
} // namespace mlir
