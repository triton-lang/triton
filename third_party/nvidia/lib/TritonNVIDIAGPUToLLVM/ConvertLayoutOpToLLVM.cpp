#include "Allocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using mlir::LLVM::NVIDIA::lowerLdStMatrix;

struct ConvertLayoutOpSwizzlingConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
  const NVIDIA::TargetInfo &targetInfo;

  explicit ConvertLayoutOpSwizzlingConversion(
      LLVMTypeConverter &typeConverter, const NVIDIA::TargetInfo &targetInfo,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!cvtNeedsSharedMemory(op))
      return failure();
    auto offsets = getConvertLayoutScratchOffsets(
        op, [&](RankedTensorType srcTy, RankedTensorType dstTy) {
          return nvidia_gpu::getConvertLayoutScratchSize(srcTy, dstTy,
                                                         targetInfo);
        });
    ConvertLayoutBarrier barriers(op, rewriter, targetInfo);
    SmallVector<Value> results;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      barriers.beginConversion();
      if (!cvtNeedsSharedMemory(op, i)) {
        results.push_back(lowerConvertLayout(op, i, adaptor.getSrcs()[i], 0,
                                             *getTypeConverter(), targetInfo,
                                             rewriter));
        continue;
      }
      auto srcTy = op.getSrc(i).getType();
      auto dstTy = op.getType(i);
      auto *ctx = op.getContext();
      auto srcLayout =
          toLinearLayout(srcTy).removeZeroBasesAlongDim(str_attr("register"));
      auto dstLayout =
          toLinearLayout(dstTy).removeZeroBasesAlongDim(str_attr("register"));
      auto loc = op.getLoc();
      auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
      auto smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
      if (offsets[i])
        smemBase = LLVM::GEPOp::create(rewriter, loc, smemBase.getType(),
                                       rewriter.getI8Type(), smemBase,
                                       ArrayRef<LLVM::GEPArg>{offsets[i]});
      auto inVals =
          unpackUniqueTensorElements(loc, adaptor.getSrcs()[i], rewriter);
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, inVals, llvmElemTy, smemBase, op,
          [&] { barriers.emit(); });
      results.push_back(packUniqueTensorElements(loc, getTypeConverter(),
                                                 outVals, rewriter, dstTy));
    }
    rewriter.replaceOp(op, results);
    return success();
  }

  SmallVector<Value> transferWithinBlockSwizzling(
      Location loc, ConversionPatternRewriter &rewriter,
      const LinearLayout &srcLayout, const LinearLayout &dstLayout,
      ArrayRef<Value> inVals, Type llvmElemTy, Value smemBase,
      Operation *sourceOp, llvm::function_ref<void()> sharedBarrier) const {
    auto *ctx = rewriter.getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // We handle transformations recursively as they all need a preprocessing
    // and a postprocessing step.

    // Handle pointer types as 64-bit integers
    if (isa<LLVM::LLVMPointerType>(llvmElemTy)) {
      auto llvmElemTyPtr = i64_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(inVals, [&](Value v) {
        return b.ptrtoint(llvmElemTyPtr, v).getResult();
      }));
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, newInVals, llvmElemTyPtr,
          smemBase, sourceOp, sharedBarrier);
      for (auto &v : outVals) {
        v = b.inttoptr(llvmElemTy, v);
      }
      return outVals;
    }

    // Handle sub-byte elements like i1
    if (llvmElemTy.getIntOrFloatBitWidth() < 8) {
      // Upcast to i8
      auto i8ElemTy = i8_ty;
      auto newInVals = llvm::to_vector(llvm::map_range(
          inVals, [&](Value v) { return b.zext(i8ElemTy, v).getResult(); }));
      auto outVals = transferWithinBlockSwizzling(
          loc, rewriter, srcLayout, dstLayout, newInVals, i8ElemTy, smemBase,
          sourceOp, sharedBarrier);
      for (auto &v : outVals) {
        v = b.trunc(llvmElemTy, v, LLVM::IntegerOverflowFlags::nuw);
      }
      return outVals;
    }

    // At this point we have a type that's at least 8-bit
    // and we don't have broadcasting in the registers
    auto bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    auto kBlock = str_attr("block");
    bool crossCTA =
        !dstLayout.invertAndCompose(srcLayout).isTrivialOver({kBlock});
    auto [srcTiles, dstTiles] = getSrcDstTiles(targetInfo, bitwidth, crossCTA);
    auto [smem, instr] =
        optimalSwizzling(srcLayout, dstLayout, srcTiles, dstTiles, bitwidth);
    auto [idxSrc, idxDst] = instr;

    // Extract reps from smem
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kReps = str_attr("reps");
    auto nReps = smem.getInDimSize(kReps);
    auto reps = LinearLayout::identity1D(nReps, kReg, kReps);

    auto totalStoreCvt = srcLayout.invertAndCompose(smem);
    auto totalLoadCvt = invertAndComposeLocal(smem, dstLayout, {kBlock});

    // The permutation exists by construction of the reps dimension in
    // optimalSwizzling
    auto permStore =
        regPermForDivide(totalStoreCvt, reps, /*left=*/false).value();
    totalStoreCvt = permStore.apply(totalStoreCvt);
    auto permutedInVals = permStore.apply(inVals);
    auto permLoad =
        regPermForDivide(totalLoadCvt, reps, /*left=*/false).value();
    totalLoadCvt = permLoad.apply(totalLoadCvt);

    // Remove the reps and flatten into offset
    auto storeCvt = *divideRight(totalStoreCvt, reps);
    auto loadCvt = *divideRight(totalLoadCvt, reps);
    auto kOffset = str_attr("offset");
    auto nBlock = storeCvt.getInDimSize(kBlock);
    storeCvt = storeCvt.reshapeOuts(
        {{kOffset, storeCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});
    loadCvt = loadCvt.reshapeOuts(
        {{kOffset, loadCvt.getTotalOutDimSize() / nBlock}, {kBlock, nBlock}});

    // We never do cross-CTA writes by construction. We may do cross-CTA reads,
    // but in that case we lower to ld.shared/st.shared
    assert(storeCvt.isTrivialOver({kBlock}));
    assert(loadCvt.isTrivialOver({kBlock}) || idxDst == 0);

    auto tileSize = storeCvt.getInDimSize(kReg);

    assert(permutedInVals.size() == tileSize * nReps);
    SmallVector<Value> outVals;
    auto affineOffset = b.i32_val(0);
    auto maskSpanAffineOffset = 0;
    auto emitBarrier = sharedBarrier;
    auto dropBlock = [&](const LinearLayout &cvt) {
      SmallVector<StringAttr> inDims = {kReg, kLane, kWarp};
      SmallVector<StringAttr> outDims = {kOffset};
      return cvt.sublayout(inDims, outDims);
    };
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    for (int i = 0; i < nReps; ++i) {
      if (i > 0)
        emitBarrier();
      auto tileInVals =
          to_vector(ArrayRef(permutedInVals).slice(i * tileSize, tileSize));
      // Store
      // idxSrc 0: st.shared, idxSrc 1: stmatrix, idxSrc 2: stmatrix.trans
      if (idxSrc == 0) {
        lowerLdSt(loc, ctx, storeCvt, tileInVals, llvmElemTy, smemBase,
                  /*paddingShifts=*/{}, affineOffset, maskSpanAffineOffset,
                  /*affineBlockOffset=*/Value(), /*maskSpanAffineBlock=*/0,
                  laneId, warpId, rewriter, targetInfo,
                  /*maybeMaxVecElems=*/{},
                  makeSharedStoreEmitter(targetInfo, b.true_val()));
      } else {
        assert(idxSrc == 1 || idxSrc == 2);
        bool transpose = idxSrc == 2;
        auto storeCvtNoBlock = dropBlock(storeCvt);
        auto result = lowerLdStMatrix(
            loc, storeCvtNoBlock, transpose, tileInVals, smemBase, affineOffset,
            maskSpanAffineOffset, llvmElemTy, rewriter, targetInfo);
        assert(succeeded(result));
      }
      emitBarrier();
      // Load
      SmallVector<Value> tileOutVals;
      // idxDst 0: ld.shared, idxDst 1: ldmatrix, idxDst 2: ldmatrix.trans
      if (idxDst == 0) {
        tileOutVals = lowerLdSt(
            loc, ctx, loadCvt, {}, llvmElemTy, smemBase, /*paddingShifts=*/{},
            affineOffset, maskSpanAffineOffset, /*affineBlockOffset=*/Value(),
            /*maskSpanAffineBlock=*/0, laneId, warpId, rewriter, targetInfo,
            /*maybeMaxVecElems=*/{}, makeSharedLoadEmitter(targetInfo));
      } else {
        assert(idxDst == 1 || idxDst == 2);
        bool transpose = idxDst == 2;
        auto loadCvtNoBlock = dropBlock(loadCvt);
        auto result = lowerLdStMatrix(
            loc, loadCvtNoBlock, transpose, tileOutVals, smemBase, affineOffset,
            maskSpanAffineOffset, llvmElemTy, rewriter, targetInfo);
        assert(succeeded(result));
      }
      llvm::append_range(outVals, tileOutVals);
    }

    // Undo the permLoad used to divideRight
    outVals = permLoad.inverse().apply(outVals);
    return outVals;
  }
};

} // namespace

void mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Give this convertLayoutOpSwizzlingConversion a higher benefit as it
  // matches optimized ldmatrix/stmatrix cases
  patterns.add<ConvertLayoutOpSwizzlingConversion>(typeConverter, targetInfo,
                                                   benefit.getBenefit() + 1);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
