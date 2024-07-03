#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using ValueTable = std::map<std::pair<int, int>, Value>;
using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;

SmallVector<Value>
getThreadIds(Value threadId, ArrayRef<unsigned int> shapePerCTATile,
             ArrayRef<unsigned int> sizePerThread, ArrayRef<unsigned int> order,
             ConversionPatternRewriter &rewriter, Location loc) {
  int dim = order.size();
  SmallVector<Value> threadIds(dim);
  for (unsigned k = 0; k < dim - 1; k++) {
    Value dimK = i32_val(shapePerCTATile[order[k]] / sizePerThread[order[k]]);
    Value rem = urem(threadId, dimK);
    threadId = udiv(threadId, dimK);
    threadIds[order[k]] = rem;
  }
  Value dimK = i32_val(shapePerCTATile[order[dim - 1]]);
  threadIds[order[dim - 1]] = urem(threadId, dimK);
  return threadIds;
}

Value getStructFromValueTable(ArrayRef<Value> vals,
                              ConversionPatternRewriter &rewriter, Location loc,
                              const LLVMTypeConverter *typeConverter,
                              Type elemTy) {
  SmallVector<Type> elemTypes(vals.size(), elemTy);
  SmallVector<Value> elems;
  elems.reserve(vals.size());
  for (auto &val : vals) {
    elems.push_back(val);
  }
  MLIRContext *ctx = elemTy.getContext();
  Type structTy = struct_ty(elemTypes);
  return packLLElements(loc, typeConverter, elems, rewriter, structTy);
}

ValueTable getValueTableFromStruct(Value val, int K, int n0, int shapePerCTA,
                                   int sizePerThread,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc,
                                   const LLVMTypeConverter *typeConverter,
                                   Type type) {
  ValueTable res;
  auto elems = unpackLLElements(loc, val, rewriter);
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTA)
      for (unsigned mm = 0; mm < sizePerThread; ++mm) {
        res[{m + mm, k}] = elems[index++];
      }
  }
  return res;
}

Value loadFMAOp(Value A, Value llA, BlockedEncodingAttr dLayout, Value thread,
                Location loc, const LLVMTypeConverter *typeConverter,
                ConversionPatternRewriter &rewriter, const int kDim) {
  const int nonKDim = kDim == 0 ? 1 : 0;
  auto aTensorTy = cast<MemDescType>(A.getType());
  auto aLayout = cast<SharedEncodingAttr>(aTensorTy.getEncoding());
  auto aShapePerCTA = getShapePerCTA(aTensorTy);

  auto order = dLayout.getOrder();

  auto aSmem = getSharedMemoryObjectFromStruct(
      loc, llA, typeConverter->convertType(aTensorTy.getElementType()),
      rewriter);
  Value strideAM = aSmem.strides[nonKDim];
  Value strideAK = aSmem.strides[kDim];
  int K = aShapePerCTA[kDim];
  int M = aShapePerCTA[nonKDim];

  auto shapePerCTATile = getShapePerCTATile(dLayout);
  auto sizePerThread = getSizePerThread(dLayout);

  Value mTileSize = i32_val(sizePerThread[nonKDim]);

  // threadId in blocked layout
  auto threadIds = getThreadIds(thread, shapePerCTATile, sizePerThread, order,
                                rewriter, loc);
  Value threadIdM = threadIds[nonKDim];
  Value nonKTileOffset = mul(threadIdM, mTileSize);

  auto elemTy = typeConverter->convertType(aTensorTy.getElementType());
  Type ptrTy = ptr_ty(rewriter.getContext(), 3);

  SmallVector<Value> vas;

  int mShapePerCTATile = shapePerCTATile[nonKDim];
  int mSizePerThread = sizePerThread[nonKDim];

  for (unsigned k = 0; k < K; ++k)
    for (unsigned m = 0; m < M; m += mShapePerCTATile)
      for (unsigned mm = 0; mm < mSizePerThread; ++mm) {
        Value rawMOffset = add(nonKTileOffset, i32_val(m + mm));
        Value rawKOffset = i32_val(k);
        Value aOffM = mul(urem(rawMOffset, i32_val(M)), strideAM);
        Value aOffK = mul(urem(rawKOffset, i32_val(K)), strideAK);
        Value offset = add(aOffM, aOffK);
        Value pa = gep(ptrTy, elemTy, aSmem.base, offset);
        Value va = load(elemTy, pa);
        vas.emplace_back(va);
      }

  return getStructFromValueTable(vas, rewriter, loc, typeConverter, elemTy);
}

namespace SharedToDotOperandFMA {
Value convertLayout(int opIdx, Value val, Value llVal,
                    BlockedEncodingAttr dLayout, Value thread, Location loc,
                    const LLVMTypeConverter *typeConverter,
                    ConversionPatternRewriter &rewriter) {
  if (opIdx == 0)
    return loadFMAOp(val, llVal, dLayout, thread, loc, typeConverter, rewriter,
                     1);
  else
    return loadFMAOp(val, llVal, dLayout, thread, loc, typeConverter, rewriter,
                     0);
}
} // namespace SharedToDotOperandFMA
