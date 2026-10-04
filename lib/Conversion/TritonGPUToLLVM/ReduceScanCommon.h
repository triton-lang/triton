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
class ScanOp;

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

inline void visitScanTotals(
    Location loc, ConversionPatternRewriter &rewriter, triton::ScanOp op,
    const LinearLayout &src, const LinearLayout &dst,
    const SmallVector<SmallVector<Value>> &values,
    const LLVMTypeConverter *typeConverter, const TargetInfoBase &targetInfo,
    Value storePred, Value laneId, Value warpId, ArrayRef<unsigned> storeOrder,
    ArrayRef<unsigned> loadOrder, llvm::function_ref<void()> initialize,
    llvm::function_ref<void(unsigned, ValueRange)> visit) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  auto kOffset = StringAttr::get(ctx, "offset");
  auto scratch = getScanScratchConfig(src, dst, op.getElementTypes());
  auto base = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
  if (!storePred) {
    storePred = b.true_val();
    auto free = src.getFreeVariableMasks();
    for (auto [dim, id] : SmallVector<std::pair<StringAttr, Value>>{
             {StringAttr::get(ctx, "lane"), laneId},
             {StringAttr::get(ctx, "warp"), warpId}})
      if (free[dim])
        storePred =
            b.and_(storePred,
                   b.icmp_eq(b.and_(id, b.i32_val(free[dim])), b.i32_val(0)));
  }
  // Preserve native register order in shared memory. Independent hardware
  // scans occupy low bits, followed by warp totals, then native register bits.
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kReg = StringAttr::get(ctx, "register");
  SmallVector<size_t> permutation;
  for (unsigned bit = 0; (1u << bit) < storeOrder.size(); ++bit)
    permutation.push_back(llvm::Log2_32(storeOrder[1u << bit]));
  ColumnAction registerOrder(permutation, kReg, permutation.size());
  auto storeSource = registerOrder.apply(src);
  auto free = storeSource.getFreeVariableMasks();
  std::vector<std::vector<int32_t>> hardware;
  for (auto dim : {kLane, kWarp})
    for (auto [bit, basis] :
         llvm::enumerate(storeSource.getBases().lookup(dim)))
      if (!(free[dim] & (1u << bit)))
        hardware.push_back(basis);
  std::stable_partition(
      hardware.begin(), hardware.end(),
      [&](const auto &basis) { return !basis[op.getAxis()]; });
  for (auto [bit, basis] : llvm::enumerate(storeSource.getBases().lookup(kReg)))
    if (!(free[kReg] & (1u << bit)))
      hardware.push_back(basis);
  LinearLayout shared({{kOffset, std::move(hardware)},
                       {kBlock, storeSource.getBases().lookup(kBlock)}},
                      llvm::to_vector(src.getOutDimNames()));
  auto storeLayout = storeSource.invertAndCompose(shared);
  auto loadLayout = shared.pseudoinvert();
  struct SharedOperand {
    Value base;
    Type type;
  };
  SmallVector<SharedOperand> operands;
  SmallVector<SmallVector<Value>> storedValues;
  for (auto [i, input] : llvm::enumerate(values)) {
    Type type = typeConverter->convertType(op.getElementTypes()[i]);
    Type storageType = type;
    if (isa<LLVM::LLVMPointerType>(type))
      storageType = rewriter.getI64Type();
    else if (type.getIntOrFloatBitWidth() < 8)
      storageType = rewriter.getI8Type();
    Value operandBase = b.gep(base.getType(), rewriter.getI8Type(), base,
                              b.i32_val(scratch.offsets[i]));
    operands.push_back({operandBase, storageType});
    storedValues.push_back(registerOrder.apply(input));
  }
  // Warp coordinates are already bounded by the execution region. Preserve
  // an identity mapping directly instead of masking it again.
  auto warpOffset = storeLayout.sublayout({kWarp}, {kOffset});
  bool warpIsOffset =
      warpOffset.getNumConsecutiveInOut() == warpOffset.getInDimSize(kWarp) &&
      storeLayout.sublayoutIsZero({kLane}, {kOffset});
  // Publish all operands of one native chunk together, as in main. Keeping
  // tuple stores together avoids extending the other operands' lifetimes.
  for (unsigned reg = 0; reg < storeSource.getInDimSize(kReg); ++reg) {
    Value offset;
    if (warpIsOffset) {
      unsigned regOffset =
          storeLayout.apply({{kReg, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}})
              .front()
              .second;
      offset = b.add(warpId, b.i32_val(regOffset));
    } else {
      offset = applyLinearLayout(loc, rewriter, storeLayout,
                                 {{kReg, b.i32_val(reg)},
                                  {kLane, laneId},
                                  {kWarp, warpId},
                                  {kBlock, b.i32_val(0)}})
                   .front()
                   .second;
    }
    for (auto [i, operand] : llvm::enumerate(operands)) {
      Value value = storedValues[i][reg];
      if (value.getType() != operand.type)
        value = isa<LLVM::LLVMPointerType>(value.getType())
                    ? Value(b.ptrtoint(operand.type, value))
                    : Value(b.zext(operand.type, value));
      Value ptr =
          b.gep(operand.base.getType(), operand.type, operand.base, offset);
      targetInfo.storeShared(rewriter, loc, ptr, value, storePred);
    }
  }
  targetInfo.barrier(loc, rewriter, gpu::AddrSpace::Local);
  initialize();
  // Emit each load next to its accumulation, as in main. Loading the full
  // sequence up front would keep many raw totals live unnecessarily.
  for (unsigned reg : loadOrder) {
    auto coords = applyLinearLayout(loc, rewriter, dst,
                                    {{kReg, b.i32_val(reg)},
                                     {StringAttr::get(ctx, "lane"), laneId},
                                     {StringAttr::get(ctx, "warp"), warpId},
                                     {kBlock, b.i32_val(0)}});
    SmallVector<Value> loaded;
    for (auto [operandIdx, operand] : llvm::enumerate(operands)) {
      Value offset =
          applyLinearLayout(loc, rewriter, loadLayout, coords).front().second;
      Value ptr =
          b.gep(operand.base.getType(), operand.type, operand.base, offset);
      Value value =
          targetInfo.loadShared(rewriter, loc, ptr, operand.type, b.true_val());
      Type type = typeConverter->convertType(op.getElementTypes()[operandIdx]);
      if (type != operand.type)
        value = isa<LLVM::LLVMPointerType>(type)
                    ? Value(b.inttoptr(type, value))
                    : Value(b.trunc(type, value));
      loaded.push_back(value);
    }
    visit(reg, loaded);
  }
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
