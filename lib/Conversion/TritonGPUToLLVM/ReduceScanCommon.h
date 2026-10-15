#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCESCANCOMMON_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_REDUCESCANCOMMON_H

// TODO: refactor so that it doesn't fail if Allocation.h
// is included after utility.h (due to conflict in `store` macro
// and <atomic>
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

//
#include "mlir/IR/TypeUtilities.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.h"
#include <iterator>

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {
class ReduceOp;

inline SmallVector<Value>
inlineCombineBlock(ConversionPatternRewriter &rewriter, Block &combineBlock,
                   Block *insertionBlock, Block::iterator insertionPoint,
                   ValueRange combineArgs) {
  auto returnOp = combineBlock.getTerminator();
  rewriter.inlineBlockBefore(&combineBlock, insertionBlock, insertionPoint,
                             combineArgs);

  auto results = SmallVector<Value>(returnOp->getOperands());

  // Delete the terminator, which is no longer used
  rewriter.eraseOp(returnOp);
  return results;
}

inline SmallVector<Value> applyCombineOp(Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         Region &combineOp, ValueRange acc,
                                         ValueRange cur, Value pred = {}) {
  // Allows for passing an uninitialized acc and use cur as the neutral element
  if (acc.size() == 0) {
    return cur;
  }
  assert(cur.size() == acc.size());

  // Create a new copy of the combine block, and try to speculatively inline it
  Block *currentBlock = rewriter.getBlock();
  Region &parent = *currentBlock->getParent();

  rewriter.cloneRegionBefore(combineOp, parent,
                             std::next(currentBlock->getIterator()));
  Block &newCombine = *currentBlock->getNextNode();

  llvm::SmallVector<Value> combineArgs(2 * acc.size());
  for (unsigned i = 0; i < acc.size(); ++i) {
    combineArgs[i] = acc[i];
    combineArgs[acc.size() + i] = cur[i];
  }

  auto isRegionSpeculatable =
      std::all_of(newCombine.begin(), newCombine.end(),
                  [](auto &op) { return isSpeculatable(&op); });

  if (!pred || isRegionSpeculatable) {
    // Fast path, region has no side effects so we can unconditionally execute
    return inlineCombineBlock(rewriter, newCombine, currentBlock,
                              rewriter.getInsertionPoint(), combineArgs);
  }

  // Slow case, create an if to only execute region when pred is true
  // #currentBlock
  // if (pred) {
  //   #newCombine
  //   results = combineOp(cur, acc)
  //   yield results
  // } else {
  //    yield undef
  // }
  // #thenBlock
  Block *thenBlock = currentBlock->splitBlock(rewriter.getInsertionPoint());

  auto returnOp = newCombine.getTerminator();
  auto results = SmallVector<Value>(returnOp->getOperands());

  rewriter.setInsertionPointToEnd(currentBlock);
  SmallVector<Value> thenBlockArgs;
  thenBlockArgs.reserve(results.size());
  for (auto result : results) {
    auto ty = result.getType();
    auto undef = LLVM::UndefOp::create(rewriter, loc, ty);
    thenBlockArgs.push_back(undef);
    thenBlock->addArgument(ty, loc);
  }
  LLVM::CondBrOp::create(rewriter, loc, pred, &newCombine, combineArgs,
                         thenBlock, thenBlockArgs);

  // Split a block after the call.
  rewriter.setInsertionPointToEnd(&newCombine);
  rewriter.replaceOpWithNewOp<LLVM::BrOp>(returnOp, results, thenBlock);
  rewriter.setInsertionPointToStart(thenBlock);
  return SmallVector<Value>(thenBlock->getArguments());
}

// Operands have disjoint scratch regions. When each conversion fits in one
// tile, store all operands before synchronizing and loading their results.
inline std::optional<SmallVector<SmallVector<Value>>>
convertLayoutValuesTogether(Location loc, ConversionPatternRewriter &rewriter,
                            Operation *op, LinearLayout src, LinearLayout dst,
                            const SmallVector<SmallVector<Value>> &inVals,
                            const LayoutConversionScratchConfig &scratch,
                            const TargetInfoBase &targetInfo) {
  auto *ctx = rewriter.getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");
  auto kReps = StringAttr::get(ctx, "reps");
  auto kOffset = StringAttr::get(ctx, "offset");
  if (inVals.size() < 2 || !scratch.sizeInBytes ||
      !isCvtDimSync(src, dst, kBlock))
    return std::nullopt;

  src = src.removeZeroBasesAlongDim(kReg);
  dst = dst.removeZeroBasesAlongDim(kReg);
  SmallVector<LinearLayout> sharedLayouts;
  for (const auto &values : inVals) {
    auto type = values.front().getType();
    if (!type.isIntOrFloat() || type.getIntOrFloatBitWidth() < 8)
      return std::nullopt;
    unsigned bitwidth = type.getIntOrFloatBitWidth();
    auto vecBitwidth = triton::gpu::getVecBitwidthLdSt(src, dst, bitwidth);
    auto [dstTile, srcTile] = targetInfo.getSharedLdStTiles(vecBitwidth);
    auto shared = optimalSwizzlingLdSt(src, dst, bitwidth,
                                       targetInfo.getSharedMemoryBanks(),
                                       srcTile, dstTile);
    if (shared.getInDimSize(kReps) != 1)
      return std::nullopt;
    sharedLayouts.push_back(std::move(shared));
  }

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value smem = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  SmallVector<SmallVector<Value>> outVals(inVals.size());
  for (bool store : {true, false}) {
    for (unsigned i = 0; i < inVals.size(); ++i) {
      auto cvt = store ? src.invertAndCompose(sharedLayouts[i])
                       : invertAndComposeLocal(sharedLayouts[i], dst, {kBlock});
      unsigned blocks = cvt.getOutDimSize(kBlock);
      cvt = cvt.reshapeOuts(
          {{kOffset, cvt.getTotalOutDimSize() / blocks}, {kBlock, blocks}});
      Value base = b.gep(smem.getType(), IntegerType::get(ctx, 8), smem,
                         b.i32_val(scratch.offsets[i]));
      auto values = lowerLdSt(
          loc, ctx, cvt, store ? ArrayRef<Value>(inVals[i]) : ArrayRef<Value>{},
          inVals[i].front().getType(), base, /*paddingShifts=*/{}, b.i32_val(0),
          /*maskSpanAffineOffset=*/0,
          /*affineBlockOffset=*/Value(), /*maskSpanAffineBlock=*/0, laneId,
          warpId, rewriter, targetInfo, /*maybeMaxVecElems=*/{},
          store ? makeSharedStoreEmitter(targetInfo, b.true_val())
                : makeSharedLoadEmitter(targetInfo));
      if (!store)
        outVals[i] = std::move(values);
    }
    if (store) {
      if (isCvtDimSync(src, dst, kWarp))
        targetInfo.warpSync(loc, rewriter);
      else
        targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
    }
  }
  return outVals;
}

template <typename SourceOp>
SmallVector<SmallVector<Value>> convertLayoutValues(
    Location loc, ConversionPatternRewriter &rewriter, SourceOp op,
    const LinearLayout &srcLayout, const LinearLayout &dstLayout,
    const SmallVector<SmallVector<Value>> &inVals,
    const LLVMTypeConverter *typeConverter, const TargetInfoBase &targetInfo,
    bool forceWarpShuffle = false) {
  SmallVector<SmallVector<Value>> outVals(op.getNumOperands());
  auto *ctx = rewriter.getContext();
  SmallVector<int64_t> shape;
  for (auto dim : srcLayout.getOutDimNames()) {
    shape.push_back(srcLayout.getOutDimSize(dim));
  }
  auto srcEnc = triton::gpu::LinearEncodingAttr::get(ctx, srcLayout);
  auto dstEnc = triton::gpu::LinearEncodingAttr::get(ctx, dstLayout);
  // The proper way to lower reduce would be to lower it to:
  // reduce_threads / reduce_lanes / convert_layout
  // and let AllocationAnalysis handle the shared memory allocation
  // and Membar the barriers.
  // Forced warp-local conversions never need an allocation, even when the
  // default conversion heuristic would choose shared memory.
  LayoutConversionScratchConfig scratch;
  if (!forceWarpShuffle)
    scratch = getLayoutConversionScratchConfig(
        srcLayout, dstLayout, op.getElementTypes(),
        [&](const LinearLayout &src, const LinearLayout &dst,
            unsigned bitwidth) {
          auto vecBitwidth =
              triton::gpu::getVecBitwidthLdSt(src, dst, bitwidth);
          auto [dstTile, srcTile] = targetInfo.getSharedLdStTiles(vecBitwidth);
          return getNumScratchElemsSwizzledCvt(
              src, dst, bitwidth, targetInfo.getSharedMemoryBanks(), srcTile,
              dstTile);
        });
  auto baseOffsetAttr =
      op->template getAttrOfType<IntegerAttr>("allocation.offset");
  assert((!scratch.sizeInBytes || baseOffsetAttr) &&
         "expected allocation.offset for shared-memory conversion");
  int64_t baseOffset = baseOffsetAttr ? baseOffsetAttr.getInt() : 0;
  if (auto values = convertLayoutValuesTogether(
          loc, rewriter, op, srcLayout, dstLayout, inVals, scratch, targetInfo))
    return std::move(*values);
  auto offsetTy = IntegerType::get(ctx, 32);
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    auto elemTy = op.getElementTypes()[i];
    auto srcTy = RankedTensorType::get(shape, elemTy, srcEnc);
    auto dstTy = RankedTensorType::get(shape, elemTy, dstEnc);
    Value packed = packUniqueTensorElements(loc, typeConverter, inVals[i],
                                            rewriter, srcTy);
    auto srcTensor =
        UnrealizedConversionCastOp::create(rewriter, loc, srcTy, packed)
            .getResult(0);
    auto cvt =
        triton::gpu::ConvertLayoutOp::create(rewriter, loc, dstTy, srcTensor);
    if (forceWarpShuffle)
      cvt.setForceWarpShuffleAttr(rewriter.getUnitAttr());
    triton::nvidia_gpu::copyClusterBarrierMbarOffset(op, cvt);
    if (scratch.sizeInBytes)
      cvt->setAttr("allocation.offset",
                   IntegerAttr::get(offsetTy, baseOffset + scratch.offsets[i]));
    Type packedDstTy = typeConverter->convertType(dstTy);
    auto packedDst = UnrealizedConversionCastOp::create(
                         rewriter, loc, packedDstTy, cvt.getResult())
                         .getResult(0);
    outVals[i] = unpackUniqueTensorElements(loc, packedDst, rewriter);
  }
  return outVals;
}

} // namespace mlir::triton

#endif
