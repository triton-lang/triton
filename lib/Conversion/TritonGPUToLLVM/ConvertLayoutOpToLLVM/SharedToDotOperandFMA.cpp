#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using ValueTable = std::map<std::pair<int, int>, Value>;
using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::expandMatrixOrderWithBatch;
using ::mlir::triton::gpu::expandMatrixShapeWithBatch;
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

SmallVector<Value> swizzleIndices(ConversionPatternRewriter &rewriter,
                                  Location loc, SmallVector<Value> rawIndices,
                                  SharedEncodingAttr layout) {
  const auto &order = layout.getOrder();
  auto rank = order.size();

  if (layout.getMaxPhase() == 1)
    return rawIndices;

  auto vec = i32_val(layout.getVec());
  auto perPhase = i32_val(layout.getPerPhase());
  auto maxPhase = i32_val(layout.getMaxPhase());

  auto fastIdx = rawIndices[order[0]];
  auto secondIdx = rawIndices[order[1]];
  // Original algorithm taken from getSwizzledSharedPtrs function
  // (TritonGPUToLLVMBase.h)
  //
  // phase = (secondIdx // perPhase) % maxPhase
  // swizzledGroup = ((fastIdx // vec) ^ phase) * vec
  // groupRemainder = fastIdx % vec
  // colOff = swizzledGroup + groupRemainder
  auto phase = urem(udiv(secondIdx, perPhase), maxPhase);
  auto swizzledGroup = mul(xor_(udiv(fastIdx, vec), phase), vec);
  auto groupRemainder = urem(fastIdx, vec);
  auto colOff = add(swizzledGroup, groupRemainder);

  SmallVector<Value> swizzledIndices = rawIndices;
  swizzledIndices[order[0]] = colOff;

  return swizzledIndices;
}

Value loadFMAOp(Value dotOp, Value llA, BlockedEncodingAttr dLayout,
                Value thread, Location loc,
                const LLVMTypeConverter *typeConverter,
                ConversionPatternRewriter &rewriter, const int dotOpNo) {
  const int bDim = 0;
  const int kDim = dotOpNo == 0 ? 2 : 1;
  const int nonKDim = dotOpNo == 0 ? 1 : 2;
  auto opTensorTy = cast<MemDescType>(dotOp.getType());
  auto opLayout = cast<SharedEncodingAttr>(opTensorTy.getEncoding());
  auto opShapePerCTA = expandMatrixShapeWithBatch(getShapePerCTA(opTensorTy));

  auto order = expandMatrixOrderWithBatch(dLayout.getOrder());

  auto origSmem = getSharedMemoryObjectFromStruct(
      loc, llA, typeConverter->convertType(opTensorTy.getElementType()),
      rewriter);
  auto smem = getExpandedSharedMemoryObject(rewriter, loc, origSmem,
                                            opTensorTy.getShape());
  auto strides = smem.strides;
  int B = opShapePerCTA[bDim];
  int K = opShapePerCTA[kDim];
  int NonK = opShapePerCTA[nonKDim];

  auto shapePerCTATile =
      expandMatrixShapeWithBatch(getShapePerCTATile(dLayout));
  auto sizePerThread = expandMatrixShapeWithBatch(getSizePerThread(dLayout));

  // threadId in blocked layout
  auto threadIds = getThreadIds(thread, shapePerCTATile, sizePerThread, order,
                                rewriter, loc);

  Value bTileOffset = mul(threadIds[bDim], i32_val(sizePerThread[bDim]));
  Value nonKTileOffset =
      mul(threadIds[nonKDim], i32_val(sizePerThread[nonKDim]));

  auto elemTy = typeConverter->convertType(opTensorTy.getElementType());
  Type ptrTy = ptr_ty(rewriter.getContext(), 3);

  SmallVector<Value> vas;

  int shapePerCTABTile = shapePerCTATile[bDim];
  int shapePerCTANonKTile = shapePerCTATile[nonKDim];
  int sizeBPerThread = sizePerThread[bDim];

  int sizeNonKPerThread = sizePerThread[nonKDim];
  int numBTiles = std::max(1, B / shapePerCTABTile);
  int numTiles = std::max(1, NonK / shapePerCTANonKTile);

  for (unsigned bTile = 0; bTile < numBTiles; ++bTile)
    for (unsigned b = 0; b < sizeBPerThread; ++b)
      for (unsigned k = 0; k < K; ++k)
        for (unsigned nonKTile = 0; nonKTile < numTiles; ++nonKTile)
          for (unsigned elem = 0; elem < sizeNonKPerThread; ++elem) {
            // offsets along named axes in terms of coordinates
            SmallVector<Value> rawIndices(3);
            rawIndices[bDim] =
                add(bTileOffset, i32_val(bTile * shapePerCTABTile + b));
            rawIndices[nonKDim] = add(
                nonKTileOffset, i32_val(nonKTile * shapePerCTANonKTile + elem));
            rawIndices[kDim] = i32_val(k);

            SmallVector<Value> swizzledIndices =
                swizzleIndices(rewriter, loc, rawIndices, opLayout);

            Value offset = i32_val(0);
            for (int dim = 0; dim < order.size(); ++dim)
              offset = add(offset, mul(urem(swizzledIndices[dim],
                                            i32_val(opShapePerCTA[dim])),
                                       strides[dim]));

            Value pa = gep(ptrTy, elemTy, smem.base, offset);
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
  return loadFMAOp(val, llVal, dLayout, thread, loc, typeConverter, rewriter,
                   opIdx);
}
} // namespace SharedToDotOperandFMA
