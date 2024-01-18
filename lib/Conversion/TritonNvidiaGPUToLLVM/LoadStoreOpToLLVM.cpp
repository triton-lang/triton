#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#include <numeric>

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getCTALayout;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

static CUtensorMapDataType getCUtensorMapDataType(Type ty) {
  if (ty.isF16()) {
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if (ty.isBF16()) {
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if (ty.isF32()) {
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if (ty.getIntOrFloatBitWidth() == 8) {
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else {
    llvm::report_fatal_error("Unsupported elemTy for InsertSliceTMAOp");
    return CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  }
}

struct StoreAsyncTMAOpConversion : public ConvertTritonGPUOpToLLVMPattern<
                                       triton::nvidia_gpu::StoreAsyncTMAOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::StoreAsyncTMAOp>::ConvertTritonGPUOpToLLVMPattern;

  StoreAsyncTMAOpConversion(TritonGPUToLLVMTypeConverter &converter,
                            ModuleAllocation &allocation,
                            mlir::triton::gpu::TMAMetadataTy *tmaMetadata,
                            const TensorPtrMapT *tensorPtrMap,
                            PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::nvidia_gpu::StoreAsyncTMAOp>(
            converter, allocation, tmaMetadata, benefit),
        tensorPtrMap(tensorPtrMap) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::StoreAsyncTMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto srcEncoding = srcTy.getEncoding();
    if (srcEncoding.isa<NvidiaMmaEncodingAttr>()) {
      return lowerStoreAsyncWithSlice(op, adaptor, rewriter);
    } else {
      return lowerStoreAsync(op, adaptor, rewriter);
    }
  }

  LogicalResult lowerStoreAsync(triton::nvidia_gpu::StoreAsyncTMAOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto dst = op.getDst();
    auto src = op.getSrc();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto elemTy = srcTy.getElementType();

    auto rank = srcTy.getRank();
    // The sotre async op only supports tensor with ranke <= 5.
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-dimension-size-and-format
    assert(rank > 0 && rank <= 5);

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for StoreAsyncTMAOp");

    auto llFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(llFuncOp && "LLVMFuncOp not found for StoreAsyncTMAOp");

    int numTMADescs = getNumTMADescs(llFuncOp);
    assert(numTMADescs > 0);

    auto sharedLayout = srcTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    assert(sharedLayout && "expected shared encoding");

    mlir::triton::gpu::TMAInfo tmaInfo;

    tmaInfo.tensorDataType = getCUtensorMapDataType(elemTy);
    tmaInfo.tensorRank = rank;
    assert(tmaMetadata);

    auto inOrder = sharedLayout.getOrder();
    unsigned TMADescIdx = tmaMetadata->size();
    unsigned numFuncArgs = llFuncOp.getBody().front().getNumArguments();
    auto makeTensorPtr = tensorPtrMap->lookup(op.getOperation());
    auto dstOrder = makeTensorPtr.getOrder();

    unsigned globalAddressArgIdx = getArgIdx(makeTensorPtr.getBase());
    tmaInfo.globalAddressArgIdx = globalAddressArgIdx;
    tmaInfo.TMADescArgIdx = numFuncArgs - numTMADescs + TMADescIdx;

    auto getDimOfOrder = [](ArrayRef<int32_t> order, int32_t i) {
      auto it = std::find(order.begin(), order.end(), i);
      assert(it != order.end());
      return std::distance(order.begin(), it);
    };

    std::vector<int32_t> globalDimsArgIdx;
    std::vector<int32_t> globalStridesArgIdx;
    // constant values are mapped to (-1 - value)
    for (int i = 0; i < rank; ++i) {
      int32_t argIdx = -1;
      auto dim = getDimOfOrder(dstOrder, i);
      argIdx = getArgIdx(makeTensorPtr.getShape()[dim]);
      globalDimsArgIdx.emplace_back(argIdx);
      // handle constant stride
      argIdx = getArgIdx(makeTensorPtr.getStrides()[dim]);
      globalStridesArgIdx.emplace_back(argIdx);
    }

    tmaInfo.globalDimsArgIdx = globalDimsArgIdx;
    tmaInfo.globalStridesArgIdx = globalStridesArgIdx;
    std::vector<uint32_t> boxDims;
    auto CTAsPerCGA = sharedLayout.getCTALayout().getCTAsPerCGA();
    auto CTAOrder = sharedLayout.getCTALayout().getCTAOrder();
    auto CTASplitNum = sharedLayout.getCTALayout().getCTASplitNum();
    auto tensorShape = makeTensorPtr.getResult()
                           .getType()
                           .cast<triton::PointerType>()
                           .getPointeeType()
                           .cast<RankedTensorType>()
                           .getShape();
    auto shapePerCTA = getShapePerCTA(CTASplitNum, tensorShape);
    const uint32_t bytesPerCacheline = 128;
    uint32_t bytesPerElem = elemTy.getIntOrFloatBitWidth() / 8;
    uint32_t numBox{1};
    for (int i = 0; i < rank; ++i) {
      auto dim = getDimOfOrder(dstOrder, i);
      auto tNumElems = shapePerCTA[dim];
      if (i == 0 && tNumElems * bytesPerElem > bytesPerCacheline) {
        tNumElems = bytesPerCacheline / bytesPerElem;
        numBox = (shapePerCTA[dim] + tNumElems - 1) / tNumElems;
      }
      boxDims.emplace_back(tNumElems);
    }
    std::vector<uint32_t> elementStrides(rank, 1);
    tmaInfo.boxDims = boxDims;
    tmaInfo.elementStrides = elementStrides;

    CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    assert(
        ((elemTy.getIntOrFloatBitWidth() == 16 && sharedLayout.getVec() == 8) or
         (elemTy.getIntOrFloatBitWidth() == 32 &&
          sharedLayout.getVec() == 4)) &&
        "Unexpected shared layout for StoreAsyncTMAOp");
    if (sharedLayout.getPerPhase() == 4 && sharedLayout.getMaxPhase() == 2)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
    else if (sharedLayout.getPerPhase() == 2 && sharedLayout.getMaxPhase() == 4)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
    else if (sharedLayout.getPerPhase() == 1 && sharedLayout.getMaxPhase() == 8)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    else
      llvm::report_fatal_error("Unsupported shared layout for StoreAsyncTMAOp");
    tmaInfo.swizzle = swizzle;
    tmaInfo.interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
    tmaInfo.l2Promotion =
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    tmaInfo.oobFill =
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    tmaMetadata->emplace_back(tmaInfo);

    Value llDst = adaptor.getDst();
    Value llSrc = adaptor.getSrc();
    auto srcShape = srcTy.getShape();
    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, llSrc, elemTy, rewriter);

    SmallVector<Value> offsetVals;
    for (auto i = 0; i < srcShape.size(); ++i) {
      offsetVals.emplace_back(i32_val(0));
    }

    Value tmaDesc =
        llFuncOp.getBody().front().getArgument(tmaInfo.TMADescArgIdx);
    auto ptrSharedTy = LLVM::LLVMPointerType::get(ctx, 3);

    auto threadId = getThreadId(rewriter, loc);
    Value pred = icmp_eq(threadId, i32_val(0));

    auto llCoord = getTypeConverter()->unpackLLElements(loc, llDst, rewriter);
    uint32_t boxStride = std::accumulate(boxDims.begin(), boxDims.end(), 1,
                                         std::multiplies<uint32_t>());

    Value clusterCTAId = getClusterCTAId(rewriter, loc);
    SmallVector<Value> multiDimClusterCTAId =
        delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

    rewriter.create<triton::nvgpu::FenceAsyncSharedOp>(loc, 0);

    for (uint32_t b = 0; b < numBox; ++b) {
      SmallVector<Value> coord;
      // raw coord
      for (int i = 0; i < rank; ++i) {
        auto dim = getDimOfOrder(dstOrder, i);
        coord.push_back(llCoord[dim]);
      }
      // coord with box and cta offset
      for (int i = 0; i < rank; ++i) {
        auto dim = getDimOfOrder(dstOrder, i);
        if (i == 0) {
          coord[i] = add(coord[i], i32_val(b * boxDims[i]));
          auto CTAOffset =
              mul(multiDimClusterCTAId[dim], i32_val(numBox * boxDims[i]));
          coord[i] = add(coord[i], CTAOffset);
        } else {
          coord[i] = add(coord[i],
                         mul(multiDimClusterCTAId[dim], i32_val(boxDims[i])));
        }
      }
      Value srcOffset = i32_val(b * boxStride);
      auto srcPtrTy = ptr_ty(ctx, 3);
      Value srcPtrBase = gep(srcPtrTy, getTypeConverter()->convertType(elemTy),
                             smemObj.base, srcOffset);
      auto addr = bitcast(srcPtrBase, ptrSharedTy);
      rewriter.create<triton::nvgpu::TMAStoreTiledOp>(loc, tmaDesc, addr, pred,
                                                      coord);
    }
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  lowerStoreAsyncWithSlice(triton::nvidia_gpu::StoreAsyncTMAOp op,
                           OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto dst = op.getDst();
    auto src = op.getSrc();
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto makeTensorPtr = tensorPtrMap->lookup(op.getOperation());
    auto dstTensorTy = makeTensorPtr.getResult()
                           .getType()
                           .cast<triton::PointerType>()
                           .getPointeeType()
                           .cast<RankedTensorType>();
    auto tensorShape = dstTensorTy.getShape();
    auto dstOrder = makeTensorPtr.getOrder();
    auto dstElemTy = dstTensorTy.getElementType();

    auto rank = srcTy.getRank();
    // The sotre async op only supports tensor with ranke <= 5.
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-dimension-size-and-format
    assert(rank > 0 && rank <= 5);

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for StoreAsyncTMAOp");

    auto llFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(llFuncOp && "LLVMFuncOp not found for StoreAsyncTMAOp");

    int numTMADescs = getNumTMADescs(llFuncOp);
    assert(numTMADescs > 0);

    auto ctaLayout = getCTALayout(dstTensorTy.getEncoding());
    // The order of smem should be consistent with gmem.
    SmallVector<unsigned> sharedOrder;
    for (auto o : makeTensorPtr.getOrder()) {
      sharedOrder.emplace_back(o);
    }
    auto sharedLayout = SharedEncodingAttr::get(ctx, tensorShape, sharedOrder,
                                                ctaLayout, dstElemTy);

    mlir::triton::gpu::TMAInfo tmaInfo;

    tmaInfo.tensorDataType = getCUtensorMapDataType(dstElemTy);
    tmaInfo.tensorRank = rank;
    assert(tmaMetadata);

    unsigned TMADescIdx = tmaMetadata->size();
    unsigned numFuncArgs = llFuncOp.getBody().front().getNumArguments();

    unsigned globalAddressArgIdx = getArgIdx(makeTensorPtr.getBase());
    tmaInfo.globalAddressArgIdx = globalAddressArgIdx;
    tmaInfo.TMADescArgIdx = numFuncArgs - numTMADescs + TMADescIdx;

    auto getDimOfOrder = [](ArrayRef<int32_t> order, int32_t i) {
      auto it = std::find(order.begin(), order.end(), i);
      assert(it != order.end());
      return std::distance(order.begin(), it);
    };

    std::vector<int32_t> globalDimsArgIdx;
    std::vector<int32_t> globalStridesArgIdx;
    // constant values are mapped to (-1 - value)
    for (int i = 0; i < rank; ++i) {
      int32_t argIdx = -1;
      auto dim = getDimOfOrder(dstOrder, i);
      argIdx = getArgIdx(makeTensorPtr.getShape()[dim]);
      globalDimsArgIdx.emplace_back(argIdx);
      // handle constant stride
      argIdx = getArgIdx(makeTensorPtr.getStrides()[dim]);
      globalStridesArgIdx.emplace_back(argIdx);
    }

    tmaInfo.globalDimsArgIdx = globalDimsArgIdx;
    tmaInfo.globalStridesArgIdx = globalStridesArgIdx;
    std::vector<uint32_t> boxDims;
    auto CTAsPerCGA = sharedLayout.getCTALayout().getCTAsPerCGA();
    auto CTAOrder = sharedLayout.getCTALayout().getCTAOrder();
    auto CTASplitNum = sharedLayout.getCTALayout().getCTASplitNum();
    auto shapePerCTA = getShapePerCTA(CTASplitNum, tensorShape);

    auto srcLayout = srcTy.getEncoding();
    auto mmaLayout = srcLayout.dyn_cast<NvidiaMmaEncodingAttr>();

    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);

    auto instrShape = mmaLayout.getInstrShape();
    auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
    uint32_t repM =
        ceil<unsigned>(shapePerCTA[0], instrShape[0] * warpsPerCTA[0]);
    uint32_t numElemsPerRep = numElems / repM;

    const uint32_t bytesPerCacheline = 128;
    uint32_t bytesPerElem = dstElemTy.getIntOrFloatBitWidth() / 8;
    uint32_t numBox{1};
    for (int i = 0; i < rank; ++i) {
      auto dim = getDimOfOrder(dstOrder, i);
      auto tNumElems = shapePerCTA[dim];
      if (i == 0 && tNumElems * bytesPerElem > bytesPerCacheline) {
        tNumElems = bytesPerCacheline / bytesPerElem;
        numBox = (shapePerCTA[dim] + tNumElems - 1) / tNumElems;
      }
      if (i == 1) {
        tNumElems = tNumElems / repM / warpsPerCTA[0];
      }
      boxDims.emplace_back(tNumElems);
    }
    std::vector<uint32_t> elementStrides(rank, 1);
    tmaInfo.boxDims = boxDims;
    tmaInfo.elementStrides = elementStrides;

    CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    assert(((dstElemTy.getIntOrFloatBitWidth() == 16 &&
             sharedLayout.getVec() == 8) or
            (dstElemTy.getIntOrFloatBitWidth() == 32 &&
             sharedLayout.getVec() == 4)) &&
           "Unexpected shared layout for StoreAsyncTMAOp");
    if (sharedLayout.getPerPhase() == 4 && sharedLayout.getMaxPhase() == 2)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
    else if (sharedLayout.getPerPhase() == 2 && sharedLayout.getMaxPhase() == 4)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
    else if (sharedLayout.getPerPhase() == 1 && sharedLayout.getMaxPhase() == 8)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    else
      llvm::report_fatal_error("Unsupported shared layout for StoreAsyncTMAOp");
    tmaInfo.swizzle = swizzle;
    tmaInfo.interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
    tmaInfo.l2Promotion =
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    tmaInfo.oobFill =
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    tmaMetadata->emplace_back(tmaInfo);

    Value llDst = adaptor.getDst();
    Value llSrc = adaptor.getSrc();
    auto srcShape = srcTy.getShape();
    auto dstElemPtrTy = ptr_ty(ctx, 3);
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getOperation());
    smemBase = bitcast(smemBase, dstElemPtrTy);

    SmallVector<Value> offsetVals;
    for (auto i = 0; i < srcShape.size(); ++i) {
      offsetVals.emplace_back(i32_val(0));
    }

    Value tmaDesc =
        llFuncOp.getBody().front().getArgument(tmaInfo.TMADescArgIdx);
    auto ptrSharedTy = LLVM::LLVMPointerType::get(ctx, 3);

    auto threadId = getThreadId(rewriter, loc);
    Value pred = int_val(1, 1);

    auto llCoord = getTypeConverter()->unpackLLElements(loc, llDst, rewriter);
    uint32_t boxStride = std::accumulate(boxDims.begin(), boxDims.end(), 1,
                                         std::multiplies<uint32_t>());
    boxStride = boxStride * repM * warpsPerCTA[0];

    Value clusterCTAId = getClusterCTAId(rewriter, loc);
    SmallVector<Value> multiDimClusterCTAId =
        delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

    // rowStride in bytes
    uint32_t rowStrideInBytes = shapePerCTA[dstOrder[0]] * bytesPerElem;
    uint32_t swizzlingByteWidth =
        std::min<uint32_t>(rowStrideInBytes, bytesPerCacheline);

    unsigned numElemsPerSwizzlingRow = swizzlingByteWidth / bytesPerElem;
    unsigned leadingDimOffset =
        numElemsPerSwizzlingRow * shapePerCTA[dstOrder[1]];

    uint32_t rowsPerRep = getShapePerCTATile(mmaLayout)[0];

    Value warpId = udiv(threadId, i32_val(32));
    Value warpId0 = urem(urem(warpId, i32_val(warpsPerCTA[0])),
                         i32_val(srcShape[0] / instrShape[0]));
    auto srcOrder = triton::gpu::getOrder(srcLayout);
    unsigned inVec =
        srcOrder == sharedLayout.getOrder()
            ? triton::gpu::getContigPerThread(srcLayout)[srcOrder[0]]
            : 1;
    unsigned outVec = sharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    assert(minVec == 2);

    auto wordTy = vec_ty(dstElemTy, minVec);

    auto inVals =
        getTypeConverter()->unpackLLElements(loc, adaptor.getSrc(), rewriter);
    for (uint32_t b = 0; b < numBox; ++b) {
      for (int rep = 0; rep < repM; ++rep) {
        Value rowOfWarp = add(mul(warpId0, i32_val(instrShape[0])),
                              i32_val(rep * rowsPerRep));
        uint32_t elemIdxOffset = rep * numElemsPerRep;

        for (unsigned idx = 0; idx < numElemsPerRep / numBox; idx += 8) {
          uint32_t elemIdx = elemIdxOffset + b * numElemsPerRep / numBox + idx;

          Value offset = rewriter.create<triton::nvgpu::OffsetOfStmatrixV4Op>(
              loc, i32_ty, threadId, rowOfWarp,
              i32_val(b * numElemsPerRep / numBox + idx), leadingDimOffset,
              numElemsPerSwizzlingRow, true);

          Value addr =
              gep(dstElemPtrTy, getTypeConverter()->convertType(dstElemTy),
                  smemBase, offset);
          Value words[4];
          for (unsigned i = 0; i < 8; ++i) {
            if (i % minVec == 0)
              words[i / 2] = undef(wordTy);
            words[i / 2] = insert_element(
                wordTy, words[i / 2], inVals[elemIdx + i], i32_val(i % minVec));
          }

          rewriter.create<triton::nvgpu::StoreMatrixOp>(
              loc, bitcast(addr, ptrSharedTy),
              ValueRange{bitcast(words[0], i32_ty), bitcast(words[1], i32_ty),
                         bitcast(words[2], i32_ty), bitcast(words[3], i32_ty)});
        }
        rewriter.create<triton::nvgpu::FenceAsyncSharedOp>(loc, 0);

        SmallVector<Value> coord;
        // raw coord
        for (int i = 0; i < rank; ++i) {
          auto dim = getDimOfOrder(dstOrder, i);
          coord.push_back(llCoord[dim]);
        }
        // coord with box and cta offset
        for (int i = 0; i < rank; ++i) {
          auto dim = getDimOfOrder(dstOrder, i);
          if (i == 0) {
            coord[i] = add(coord[i], i32_val(b * boxDims[i]));
            auto CTAOffset =
                mul(multiDimClusterCTAId[dim], i32_val(numBox * boxDims[i]));
            coord[i] = add(coord[i], CTAOffset);
          } else {
            Value blockOffset = i32_val(rep * instrShape[0] * warpsPerCTA[0]);
            Value warpOffset = mul(warpId0, i32_val(instrShape[0]));
            coord[i] = add(add(coord[i], add(blockOffset, warpOffset)),
                           mul(multiDimClusterCTAId[dim],
                               i32_val(boxDims[i] * repM * warpsPerCTA[0])));
          }
        }
        Value srcOffset =
            add(i32_val(b * boxStride + rep * instrShape[0] * warpsPerCTA[0] *
                                            instrShape[1] * warpsPerCTA[1] /
                                            numBox),
                mul(warpId0, i32_val(instrShape[0] * numElemsPerSwizzlingRow)));
        auto srcPtrTy = ptr_ty(ctx, 3);
        Value srcPtrBase =
            gep(srcPtrTy, getTypeConverter()->convertType(dstElemTy), smemBase,
                srcOffset);
        auto addr = bitcast(srcPtrBase, ptrSharedTy);
        rewriter.create<triton::nvgpu::TMAStoreTiledOp>(loc, tmaDesc, addr,
                                                        pred, coord);
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  unsigned getArgIdx(Value v) const {
    if (auto op = v.getDefiningOp<mlir::arith::ConstantOp>()) {
      return -1 -
             op.getValue().dyn_cast<IntegerAttr>().getValue().getZExtValue();
    }
    if (!isa<BlockArgument>(v) &&
        !isa<mlir::UnrealizedConversionCastOp, arith::ExtSIOp>(
            v.getDefiningOp()))
      llvm::report_fatal_error(
          "Operand of `MakeTensorPtrOp` is not the function's argument");
    if (v.getDefiningOp() &&
        isa<mlir::UnrealizedConversionCastOp>(v.getDefiningOp())) {
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else if (v.getParentBlock()->isEntryBlock() && v.isa<BlockArgument>()) {
      // in entryblock and is BlockArgument; Because argument of func are
      // arguments of entryblock bb0 in MLIR
      return v.cast<BlockArgument>().getArgNumber();
    } else if (v.getParentBlock()->isEntryBlock() &&
               (!v.isa<BlockArgument>())) {
      // in entryblock but not BlockArgument
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else if (!v.getParentBlock()->isEntryBlock()) {
      // in non-entryblock
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else {
      llvm::report_fatal_error(
          "Operand of `MakeTensorPtrOp` is not the function's argument");
      return 0;
    }
  }

  int getNumTMADescs(LLVM::LLVMFuncOp func) const {
    if (!func->hasAttr(kAttrNumTMALoadDescsName)) {
      llvm::report_fatal_error("TritonGPU module should contain a "
                               "triton_gpu.num-tma-load attribute");
      return -1;
    }
    if (!func->hasAttr(kAttrNumTMAStoreDescsName)) {
      llvm::report_fatal_error("TritonGPU module should contain a "
                               "triton_gpu.num-tma-store attribute");
      return -1;
    }
    return func->getAttr(kAttrNumTMAStoreDescsName)
               .cast<IntegerAttr>()
               .getInt() +
           func->getAttr(kAttrNumTMALoadDescsName).cast<IntegerAttr>().getInt();
  }

  const TensorPtrMapT *tensorPtrMap;
};

struct InsertSliceTMAOpConversion : public ConvertTritonGPUOpToLLVMPattern<
                                        triton::nvidia_gpu::InsertSliceTMAOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::InsertSliceTMAOp>::ConvertTritonGPUOpToLLVMPattern;

  InsertSliceTMAOpConversion(TritonGPUToLLVMTypeConverter &converter,

                             ModuleAllocation &allocation,
                             mlir::triton::gpu::TMAMetadataTy *tmaMetadata,
                             const TensorPtrMapT *tensorPtrMap,
                             PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::nvidia_gpu::InsertSliceTMAOp>(
            converter, allocation, tmaMetadata, benefit),
        tensorPtrMap(tensorPtrMap) {}

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::InsertSliceTMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    auto resultTy = op.getResult().getType().cast<RankedTensorType>();
    auto elemTy = resultTy.getElementType();
    auto rank = resultTy.getRank() - 1;

    // TODO: support any valid rank in (3, 4, 5)
    // The sotre async op only supports tensor with ranke <= 5.
    // Reference:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-dimension-size-and-format
    assert(rank > 0 && rank <= 5);
    SmallVector<unsigned> shape;
    auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for InsertSliceTMAOp");
    auto llFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(llFuncOp && "LLVMFuncOp not found for InsertSliceTMAOp");
    int numTMADescs = getNumTMADescs(llFuncOp);
    assert(numTMADescs > 0);
    auto sharedLayout = resultTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    assert(sharedLayout && "unexpected layout of InsertSliceTMAOp");
    auto CTAsPerCGA = sharedLayout.getCTALayout().getCTAsPerCGA();
    auto CTAOrder = sharedLayout.getCTALayout().getCTAOrder();
    auto CTASplitNum = sharedLayout.getCTALayout().getCTASplitNum();

    mlir::triton::gpu::TMAInfo tmaInfo;

    tmaInfo.tensorDataType = getCUtensorMapDataType(elemTy);
    tmaInfo.tensorRank = rank;

    assert(tmaMetadata);
    unsigned TMADescIdx = tmaMetadata->size();
    unsigned numFuncArgs = llFuncOp.getBody().front().getNumArguments();
    auto makeTensorPtr = tensorPtrMap->lookup(op.getOperation());
    auto inOrder = makeTensorPtr.getOrder();
    unsigned globalAddressArgIdx = getArgIdx(makeTensorPtr.getBase());
    tmaInfo.globalAddressArgIdx = globalAddressArgIdx;
    tmaInfo.TMADescArgIdx = numFuncArgs - numTMADescs + TMADescIdx;

    auto getDimOfOrder = [](ArrayRef<int32_t> order, int32_t i) {
      auto it = std::find(order.begin(), order.end(), i);
      assert(it != order.end());
      return std::distance(order.begin(), it);
    };

    std::vector<int32_t> globalDimsArgIdx;
    std::vector<int32_t> globalStridesArgIdx;
    // constant values are mapped to (-1 - value)
    for (int i = 0; i < rank; ++i) {
      int32_t argIdx = -1;
      auto dim = getDimOfOrder(inOrder, i);
      argIdx = getArgIdx(makeTensorPtr.getShape()[dim]);
      globalDimsArgIdx.emplace_back(argIdx);
      // handle constant stride
      argIdx = getArgIdx(makeTensorPtr.getStrides()[dim]);
      globalStridesArgIdx.emplace_back(argIdx);
    }

    tmaInfo.globalDimsArgIdx = globalDimsArgIdx;
    tmaInfo.globalStridesArgIdx = globalStridesArgIdx;

    std::vector<uint32_t> boxDims;
    auto tensorShape = makeTensorPtr.getResult()
                           .getType()
                           .cast<triton::PointerType>()
                           .getPointeeType()
                           .cast<RankedTensorType>()
                           .getShape();

    SmallVector<unsigned> numMcast(rank);
    unsigned accNumMcast = 1;
    for (unsigned i = 0; i < rank; ++i) {
      numMcast[i] = CTAsPerCGA[i] / CTASplitNum[i];
      accNumMcast *= numMcast[i];
    }
    auto shapePerCTA = getShapePerCTA(CTASplitNum, tensorShape);
    for (size_t i = 0; i < rank; ++i) {
      auto dim = getDimOfOrder(inOrder, i);
      // in case of TMA multicast, we should always slice along higher order
      // dimensions
      if (i == rank - 1) {
        assert(shapePerCTA[dim] >= accNumMcast &&
               "cases when the size of the highest order is smaller "
               "than numMcasts is not implemented");
        boxDims.emplace_back(shapePerCTA[dim] / accNumMcast);
      } else {
        boxDims.emplace_back(shapePerCTA[dim]);
      }
    }

    std::vector<uint32_t> elementStrides(rank, 1);
    tmaInfo.elementStrides = elementStrides;

    CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    if (sharedLayout.getPerPhase() == 4 && sharedLayout.getMaxPhase() == 2)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
    else if (sharedLayout.getPerPhase() == 2 && sharedLayout.getMaxPhase() == 4)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
    else if (sharedLayout.getPerPhase() == 1 && sharedLayout.getMaxPhase() == 8)
      swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    else
      llvm::report_fatal_error(
          "Unsupported shared layout for InsertSliceTMAOp");

    tmaInfo.swizzle = swizzle;
    tmaInfo.interleave = CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE;
    tmaInfo.l2Promotion =
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    tmaInfo.oobFill =
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    uint32_t numBoxes = 1;
    uint32_t elemSizeOfBytes = elemTy.getIntOrFloatBitWidth() / 8;
    if (swizzle == CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
      while (elemSizeOfBytes * boxDims[0] > 128) {
        boxDims[0] = boxDims[0] / 2;
        numBoxes *= 2;
      }
    }
    tmaInfo.boxDims = boxDims;
    tmaMetadata->emplace_back(tmaInfo);

    uint32_t elemsPerBox =
        std::accumulate(boxDims.begin(), boxDims.end(), 1, std::multiplies{});

    Value clusterCTAId = getClusterCTAId(rewriter, loc);
    SmallVector<Value> multiDimClusterCTAId =
        delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);

    Value llDst = adaptor.getDst();
    Value llIndex = adaptor.getIndex();
    Value src = op.getSrc();
    Value dst = op.getDst();
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShape = dstTy.getShape();
    auto smemObj = getSharedMemoryObjectFromStruct(
        loc, llDst, typeConverter->convertType(dstTy.getElementType()),
        rewriter);

    // the offset of coord considering multicast slicing
    SmallVector<Value> mcastOffsetVals;
    // The index of slice is this CTAId is responsible for
    SmallVector<Value> multiDimSliceIdx(rank);
    for (auto i = 0; i < rank; ++i)
      multiDimSliceIdx[i] =
          udiv(multiDimClusterCTAId[i], i32_val(CTASplitNum[i]));
    Value sliceIdx =
        linearize(rewriter, loc, multiDimSliceIdx, numMcast, CTAOrder);

    Value sliceCoord;
    for (auto i = 0; i < rank; ++i) {
      if (inOrder[i] == rank - 1) {
        // TODO[goostavz]: Cases when the size of the highest order is smaller
        //                 than numMcasts is not implemented.
        sliceCoord = mul(sliceIdx, i32_val(shapePerCTA[i] / accNumMcast));
        mcastOffsetVals.emplace_back(
            mul(sliceIdx, i32_val(shapePerCTA[i] / accNumMcast)));
      } else {
        mcastOffsetVals.emplace_back(i32_val(0));
      }
    }

    uint32_t elemsPerSlice = std::accumulate(
        shapePerCTA.begin(), shapePerCTA.end(), 1, std::multiplies{});
    Value dstOffsetCommon = mul(llIndex, i32_val(elemsPerSlice));
    // [benzh] sliceCoord should be higher dimension's multiplier accumulate.
    // currently only support rank == 2.
    dstOffsetCommon =
        add(dstOffsetCommon, mul(sliceCoord, i32_val(boxDims[0])));
    auto dstPtrTy = ptr_ty(rewriter.getContext(), 3);

    Value tmaDesc =
        llFuncOp.getBody().front().getArgument(tmaInfo.TMADescArgIdx);
    // TODO: sink this logic into Triton::NVGPU dialect and support more
    // cache-policy modes
    Value l2Desc = int_val(64, 0x1000000000000000ll);

    auto ptrSharedTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);

    SmallVector<Value> coordCommon;
    auto llCoord =
        getTypeConverter()->unpackLLElements(loc, adaptor.getSrc(), rewriter);

    for (int i = 0; i < rank; ++i) {
      auto dim = getDimOfOrder(inOrder, i);
      Value coordDim = bitcast(llCoord[dim], i32_ty);
      if (CTASplitNum[dim] != 1) {
        // Add offset for each CTA
        //   boxDims[i] * (multiDimClusterCTAId[i] % CTASplitNum[i]);
        auto CTAOffset =
            mul(i32_val(shapePerCTA[dim]),
                urem(multiDimClusterCTAId[dim], i32_val(CTASplitNum[dim])));
        coordDim = add(coordDim, CTAOffset);
      }

      if (i == rank - 1)
        // Add offset in case of multicast slicing
        coordCommon.push_back(add(coordDim, mcastOffsetVals[dim]));
      else
        coordCommon.push_back(coordDim);
    }

    auto threadId = getThreadId(rewriter, loc);
    Value pred = icmp_eq(threadId, i32_val(0));

    auto mask = adaptor.getMask();
    if (mask) {
      // TODO(thomas): What is the right implementation for this case?
      assert(mask.getType().isInteger(1) &&
             "need to implement cases with tensor mask");
      pred = rewriter.create<arith::AndIOp>(loc, pred, mask);
    }

    Value mcastMask = getMCastMask(sharedLayout, rewriter, loc, clusterCTAId);

    for (size_t i = 0; i < numBoxes; ++i) {
      Value dstOffset =
          add(dstOffsetCommon, i32_val(i * elemsPerBox * accNumMcast));
      Value dstPtrBase = gep(dstPtrTy, getTypeConverter()->convertType(elemTy),
                             smemObj.base, dstOffset);
      SmallVector<Value> coord = coordCommon;
      coord[0] = add(coordCommon[0], i32_val(i * boxDims[0]));
      rewriter.create<triton::nvgpu::TMALoadTiledOp>(
          loc, bitcast(dstPtrBase, ptrSharedTy), adaptor.getMbar(), tmaDesc,
          l2Desc, pred, coord, mcastMask);
    }

    rewriter.replaceOp(op, llDst);
    return success();
  }

private:
  Value getMCastMask(const SharedEncodingAttr &sharedLayout,
                     ConversionPatternRewriter &rewriter, Location loc,
                     Value clusterCTAId) const {
    auto CTAsPerCGA = sharedLayout.getCTALayout().getCTAsPerCGA();
    auto CTAOrder = sharedLayout.getCTALayout().getCTAOrder();
    auto CTASplitNum = sharedLayout.getCTALayout().getCTASplitNum();

    // Short path when no multicast is needed
    if (CTAsPerCGA == CTASplitNum)
      return nullptr;

    // Short path when bcastMask is a constant
    bool isConstMcastMask = true;
    for (unsigned s : CTASplitNum) {
      if (s > 1) {
        isConstMcastMask = false;
        break;
      }
    }
    if (isConstMcastMask) {
      unsigned numCTAs = std::accumulate(CTAsPerCGA.begin(), CTAsPerCGA.end(),
                                         1, std::multiplies{});
      return int_val(/*width*/ 16, (1u << numCTAs) - 1);
    }

    SmallVector<Value> multiDimCTAId =
        delinearize(rewriter, loc, clusterCTAId, CTAsPerCGA, CTAOrder);
    auto rank = CTAOrder.size();
    SmallVector<SmallVector<Value>> multiDimMask(rank);
    unsigned accNumMcast = 1;
    SmallVector<unsigned> numMcast(rank);
    for (unsigned i = 0; i < rank; ++i) {
      // For the ith dimension, CTAsPerCGA[i]/CTASplitNum[i] vals is to be
      // broadcasted, which for this CTAId is:
      //     multiDimCTAId[i] % CTASplitNum[i] + (0 ..
      //     (CTAsPerCGA[i]/CTASplitNum[i] - 1)) * CTASplitNum[i]
      // TODO: will there be cases if CTAsPerCGA[i]/CTASplitNum[i] < 1?
      Value rem = urem(multiDimCTAId[i], i32_val(CTASplitNum[i]));
      numMcast[i] = CTAsPerCGA[i] / CTASplitNum[i];
      accNumMcast *= numMcast[i];
      for (unsigned j = 0; j < numMcast[i]; ++j) {
        if (j == 0) {
          multiDimMask[i].push_back(rem);
        } else {
          multiDimMask[i].push_back(add(rem, i32_val(j * CTASplitNum[i])));
        }
      }
    }

    Value bcastMask = int_val(/*width*/ 16, 0);
    Value _1_i16 = int_val(/*width*/ 16, 1);
    for (unsigned i = 0; i < accNumMcast; ++i) {
      SmallVector<unsigned> multiDimIdx =
          getMultiDimIndex<unsigned>(i, numMcast, CTAOrder);
      SmallVector<Value> multiDimMaskedCTAId(rank);
      for (unsigned dim = 0; dim < rank; ++dim) {
        multiDimMaskedCTAId[dim] = multiDimMask[dim][multiDimIdx[dim]];
      }
      Value bcastCTAId =
          linearize(rewriter, loc, multiDimMaskedCTAId, CTAsPerCGA, CTAOrder);
      // bcastMask |= 1u << bcastCTAId;
      bcastMask = or_(bcastMask, shl(_1_i16, trunc(i16_ty, bcastCTAId)));
    }

    return bcastMask;
  }

  unsigned getArgIdx(Value v) const {
    if (auto op = v.getDefiningOp<mlir::arith::ConstantOp>()) {
      return -1 -
             op.getValue().dyn_cast<IntegerAttr>().getValue().getZExtValue();
    }
    if (!isa<BlockArgument>(v) &&
        !isa<mlir::UnrealizedConversionCastOp, arith::ExtSIOp>(
            v.getDefiningOp()))
      llvm::report_fatal_error(
          "Operand of `MakeTensorPtrOp` is not the function's argument");
    if (v.getDefiningOp() &&
        isa<mlir::UnrealizedConversionCastOp>(v.getDefiningOp())) {
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else if (v.getParentBlock()->isEntryBlock() && v.isa<BlockArgument>()) {
      // in entryblock and is BlockArgument; Because argument of func are
      // arguments of entryblock bb0 in MLIR
      return v.cast<BlockArgument>().getArgNumber();
    } else if (v.getParentBlock()->isEntryBlock() &&
               (!v.isa<BlockArgument>())) {
      // in entryblock but not BlockArgument
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else if (!v.getParentBlock()->isEntryBlock()) {
      // in non-entryblock
      return getArgIdx(v.getDefiningOp()->getOperand(0));
    } else {
      llvm::report_fatal_error(
          "Operand of `MakeTensorPtrOp` is not the function's argument");
      return 0;
    }
  }

  int getNumTMADescs(LLVM::LLVMFuncOp func) const {
    if (!func->hasAttr(kAttrNumTMALoadDescsName)) {
      llvm::report_fatal_error("TritonGPU module should contain a "
                               "triton_gpu.num-tma-load attribute");
      return -1;
    }
    if (!func->hasAttr(kAttrNumTMAStoreDescsName)) {
      llvm::report_fatal_error("TritonGPU module should contain a "
                               "triton_gpu.num-tma-store attribute");
      return -1;
    }
    return func->getAttr(kAttrNumTMAStoreDescsName)
               .cast<IntegerAttr>()
               .getInt() +
           func->getAttr(kAttrNumTMALoadDescsName).cast<IntegerAttr>().getInt();
  }

  const TensorPtrMapT *tensorPtrMap;
};
} // namespace

void mlir::triton::populateLoadStoreOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    mlir::triton::gpu::TMAMetadataTy *tmaMetadata,
    const TensorPtrMapT *tensorPtrMap, PatternBenefit benefit) {
  patterns.add<InsertSliceTMAOpConversion>(typeConverter, allocation,
                                           tmaMetadata, tensorPtrMap, benefit);
  patterns.add<StoreAsyncTMAOpConversion>(typeConverter, allocation,
                                          tmaMetadata, tensorPtrMap, benefit);
}
