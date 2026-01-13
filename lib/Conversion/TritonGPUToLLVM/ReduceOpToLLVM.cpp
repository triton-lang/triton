#include "ReduceScanCommon.h"

#include <memory>
#include <tuple>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct ReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpConversion(LLVMTypeConverter &typeConverter,
                     const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(typeConverter,
                                                                  benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ReduceOpHelper helper(op);
    Location loc = op->getLoc();
    auto accs = unpackInputs(loc, op, adaptor, rewriter);
    unsigned axis = op.getAxis();

    // The lowering already supports cross-CTA reductions in principle
    // We are only missing:
    // - Supporting them in convert_layout for LinearLayouts
    // - Emitting cross-CTA barriers between convert_layouts when the second
    //   convert_layout crosses CTAs
    // After this, we can uncomment the tests in test_reduce_funky_layout
    if (helper.isReduceWithinCTA()) {
      return failure();
    }

    auto *ctx = op.getContext();

    // Remove block as we don't currently support it
    LinearLayout regLl = triton::gpu::toLinearLayout(helper.getSrcTy());
    // Remove broadcasting in registers as SliceLayout removes them
    auto removeBroadcast = actionRemoveBroadcastedRegs(regLl);
    if (!removeBroadcast.isIdentity()) {
      regLl = removeBroadcast.apply(regLl);
      for (auto &vals : accs) {
        vals = removeBroadcast.apply(vals);
      }
    }

    // First reduce all the values along axis within each thread.
    std::tie(regLl, accs) =
        reduceWithinThreads(op, std::move(regLl), std::move(accs), rewriter);

    // Then reduce across threads within a warp.
    std::tie(regLl, accs) =
        reduceWithinWarps(op, std::move(regLl), std::move(accs), rewriter);

    // reducedRegLaneLayout is used in the AllocationAnalysis to get the size
    // of the scratch space.
    assert(regLl ==
           ReduceOpHelper::reducedRegLaneLayout(helper.getSrcTy(), axis));

    // If we still need to reduce along warps / blocks:
    // Create temporary layout for reduction within warps.
    // By construction of tmpLl, we will iterate at most 2 times, as the maximum
    // number of warp / block bases is 64 * 16 = 32 * 32
    // That is, they fit in 2 rounds of warp reductions
    // Even more, if we do two rounds, getInterLayout will make sure that the
    // first one does not cross CTAs
    for (int i = 0; i < 2; ++i) {
      LinearLayout tmpLl = ReduceOpHelper::getInterLayout(regLl, axis);

      auto smemBaseOffsets = getSmemBaseOffsets(op, regLl, tmpLl);
      // Emit a barrier if we are reusing the shmem
      if (i > 0) {
        TritonLLVMOpBuilder(loc, rewriter).barrier();
      }
      accs = convertLayoutValues(loc, rewriter, op, regLl, tmpLl, accs,
                                 smemBaseOffsets);

      std::tie(tmpLl, accs) =
          reduceWithinWarps(op, std::move(tmpLl), std::move(accs), rewriter);
      regLl = std::move(tmpLl);
      if (to_vector(regLl.getOutDimSizes())[axis] == 1) {
        break;
      }
    }
    // Remove the axis dimension, which at this point is of size 1
    regLl = removeStandardDim(regLl, axis);

    // Convert to output layout if we didn't fit the warp bases within zero
    // bases in the tmpLl
    if (auto resultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto outputLayout = triton::gpu::toLinearLayout(resultTy);
      if (regLl != outputLayout) {
        // Reuse the shmem
        auto b = TritonLLVMOpBuilder(loc, rewriter);
        b.barrier();
        auto smemBaseOffsets = getSmemBaseOffsets(op, regLl, outputLayout);
        accs = convertLayoutValues(loc, rewriter, op, regLl, outputLayout, accs,
                                   smemBaseOffsets);
      }
    }

    packResults(op, accs, rewriter);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  static bool useTernaryTreeReduction(triton::ReduceOp op) {
    if (op.getNumOperands() != 1)
      return false;
    auto elemTy = op.getElementTypes()[0];
    if (!isa<IntegerType>(elemTy))
      return false;
    Operation *combiner = op.getSingleCombiner();
    if (!combiner)
      return false;
    return isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::AddIOp>(
        combiner);
  }

  std::unique_ptr<Region>
  maybePackValuesf32x2(Location loc, ConversionPatternRewriter &rewriter,
                       Region &combineOp,
                       SmallVector<SmallVector<Value>> &values) const {
    if (values.size() < 2 || (values.size() % 2) != 0)
      return nullptr;
    if (values.front().size() != 1)
      return nullptr;
    auto elemTy = values.front().front().getType();
    if (!(elemTy.isF16() || elemTy.isBF16() || elemTy.isF32()))
      return nullptr;
    Operation *combiner = nullptr;
    if (!combineOp.empty()) {
      auto &block = combineOp.front();
      if (block.getOperations().size() == 2)
        combiner = &block.front();
    }
    if (!combiner)
      return nullptr;
    if (!isa<arith::AddFOp, arith::MulFOp>(combiner))
      return nullptr;
    bool isMul = isa<arith::MulFOp>(combiner);
    // Pack the values into 2-element vectors
    SmallVector<SmallVector<Value>> packed;
    for (size_t i = 0; i < values.size(); i += 2) {
      SmallVector<Value> vecTuple(values.front().size());
      for (unsigned opIdx = 0; opIdx < values.front().size(); ++opIdx) {
        vecTuple[opIdx] = packLLVector(
            loc, {values[i][opIdx], values[i + 1][opIdx]}, rewriter);
      }
      packed.push_back(std::move(vecTuple));
    }
    values = std::move(packed);
    // Create a new region that takes 2-element vectors as inputs and returns a
    // 2-element vector
    auto region = std::make_unique<Region>();
    auto *block = new Block();
    region->push_back(block);
    auto vecTy = vec_ty(elemTy, 2);
    block->addArgument(vecTy, loc);
    block->addArgument(vecTy, loc);
    auto *ctx = rewriter.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(block);
    Value lhs = block->getArgument(0);
    Value rhs = block->getArgument(1);
    Value result =
        isMul ? LLVM::FMulOp::create(builder, loc, lhs, rhs).getResult()
              : LLVM::FAddOp::create(builder, loc, lhs, rhs).getResult();
    triton::ReduceReturnOp::create(builder, loc, ValueRange{result});
    return region;
  }

  void maybeUnpackValuesf32x2(Location loc, ConversionPatternRewriter &rewriter,
                              Region &combineOp,
                              SmallVector<Value> &values) const {
    // If it has more than one output, it's not vectorized
    // There is a world where we just check whether the region has all arith ops
    // and we vectorize them all with a pass... but that's for another day.
    if (values.size() != 1)
      return;
    auto vecTy = dyn_cast<VectorType>(values.front().getType());
    if (!vecTy || vecTy.getNumElements() != 2 ||
        (!vecTy.getElementType().isF16() && !vecTy.getElementType().isF32()))
      return;
    // Perform the last (non-vectorized) combine operation.
    auto elems = unpackLLVector(loc, values.front(), rewriter);
    SmallVector<Value> acc = {elems[0]};
    accumulate(loc, rewriter, combineOp, acc, {elems[1]});
    values = std::move(acc);
  }

  SmallVector<Value>
  treeReduceTernary(Location loc, ConversionPatternRewriter &rewriter,
                    Region &combineOp,
                    SmallVector<SmallVector<Value>> values) const {
    while (values.size() > 1) {
      SmallVector<SmallVector<Value>> next;
      size_t i = 0;
      for (; i + 2 < values.size(); i += 3) {
        SmallVector<Value> acc = values[i];
        accumulate(loc, rewriter, combineOp, acc, values[i + 1]);
        accumulate(loc, rewriter, combineOp, acc, values[i + 2]);
        next.push_back(std::move(acc));
      }
      // Process tail
      if (values.size() - i == 1) {
        next.push_back(std::move(values[i]));
      } else if (values.size() - i == 2) {
        SmallVector<Value> acc = values[i];
        accumulate(loc, rewriter, combineOp, acc, values[i + 1]);
        next.push_back(std::move(acc));
      }
      values = std::move(next);
    }
    return std::move(values.front());
  }

  SmallVector<Value>
  treeReduceBinary(Location loc, ConversionPatternRewriter &rewriter,
                   Region &combineOp,
                   SmallVector<SmallVector<Value>> values) const {
    // The number of elements is always a power of two
    assert(llvm::isPowerOf2_64(values.size()) && !values.empty());
    auto vectorCombine = maybePackValuesf32x2(loc, rewriter, combineOp, values);
    Region &accumulateRegion = vectorCombine ? *vectorCombine : combineOp;
    while (values.size() > 1) {
      SmallVector<SmallVector<Value>> next;
      for (size_t i = 0; i + 1 < values.size(); i += 2) {
        SmallVector<Value> acc = values[i];
        accumulate(loc, rewriter, accumulateRegion, acc, values[i + 1]);
        next.push_back(std::move(acc));
      }
      values = std::move(next);
    }
    SmallVector<Value> val = std::move(values.front());
    maybeUnpackValuesf32x2(loc, rewriter, combineOp, val);
    return val;
  }

  void accumulate(Location loc, ConversionPatternRewriter &rewriter,
                  Region &combineOp, SmallVector<Value> &acc, ValueRange cur,
                  Value pred = {}) const {
    auto results = applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
    if (acc.size() < results.size()) {
      acc.resize(results.size());
    }
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto operands = adaptor.getOperands();
    SmallVector<SmallVector<Value>> srcValues(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      srcValues[i] = unpackLLElements(loc, operands[i], rewriter);
    }
    return srcValues;
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinThreads(triton::ReduceOp op, LinearLayout layout,
                      SmallVector<SmallVector<Value>> accs,
                      ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto kReg = str_attr("register");
    auto linearAttr = triton::gpu::LinearEncodingAttr::get(ctx, layout);
    auto basesPerDim = linearAttr.basesPerDim(kReg, /*skipBroadcast=*/true);
    unsigned axisPack = basesPerDim[op.getAxis()];
    if (axisPack == 1) {
      return {std::move(layout), std::move(accs)};
    }

    bool useTernary = useTernaryTreeReduction(op);
    // Bring the registers that move the axis to the front
    auto perm = ReduceOpHelper::makeAxisContiguous(layout, op.getAxis());
    if (!perm.isIdentity()) {
      layout = perm.apply(layout);
      for (auto &vals : accs) {
        vals = perm.apply(vals);
      }
    }

    // Reduce with a tree. Use a ternary tree when it can map to 3-input SASS
    // ops (IADD3/LOP3); otherwise use a binary tree.
    SmallVector<SmallVector<Value>> reduced(op.getNumOperands());
    for (unsigned regBase = 0; regBase < layout.getInDimSize(kReg);
         regBase += axisPack) {
      SmallVector<SmallVector<Value>> vals;
      for (unsigned i = 0; i < axisPack; ++i) {
        SmallVector<Value> cur(op.getNumOperands());
        for (unsigned opIdx = 0; opIdx < op.getNumOperands(); ++opIdx) {
          cur[opIdx] = accs[opIdx][regBase + i];
        }
        vals.push_back(std::move(cur));
      }
      auto acc = useTernary
                     ? treeReduceTernary(op.getLoc(), rewriter,
                                         op.getCombineOp(), std::move(vals))
                     : treeReduceBinary(op.getLoc(), rewriter,
                                        op.getCombineOp(), std::move(vals));
      for (unsigned opIdx = 0; opIdx < op.getNumOperands(); ++opIdx) {
        reduced[opIdx].push_back(acc[opIdx]);
      }
    }
    accs = std::move(reduced);

    // Update layout killing the axis bases along registers
    layout =
        ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, op.getAxis(), kReg);
    layout = actionRemoveBroadcastedRegs(layout).apply(layout);
    return {std::move(layout), std::move(accs)};
  }

  // Reduce across threads within each warp.
  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinWarps(triton::ReduceOp op, LinearLayout layout,
                    SmallVector<SmallVector<Value>> accs,
                    ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto kLane = str_attr("lane");
    const auto &laneBases = layout.getBases().lookup(kLane);
    unsigned activeLanes = 0;
    for (unsigned bit = 0; bit < laneBases.size(); ++bit) {
      if (laneBases[bit][op.getAxis()] != 0) {
        activeLanes |= 1u << bit;
      }
    }
    if (activeLanes == 0) {
      return {std::move(layout), std::move(accs)};
    }

    unsigned regs = accs.front().size();
    for (unsigned reg = 0; reg < regs; ++reg) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        acc[i] = accs[i][reg];
      }
      warpReduce(op, activeLanes, acc, rewriter);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        accs[i][reg] = acc[i];
      }
    }

    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, op.getAxis(),
                                                         kLane);
    return {std::move(layout), std::move(accs)};
  }

  void warpReduce(triton::ReduceOp op, unsigned activeLanes,
                  SmallVector<Value> &acc,
                  ConversionPatternRewriter &rewriter) const {
    // No reduction to do
    if (activeLanes == 0)
      return;
    auto moduleOp = op->getParentOfType<ModuleOp>();
    unsigned warpSize =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);
    assert(activeLanes < warpSize &&
           "expected active lanes mask to be strictly less than warp size");
    // Try to use the redux op if it is supported by the target
    if (targetInfo.warpReduce(rewriter, op.getLoc(), acc, op, activeLanes)) {
      return;
    }
    for (unsigned bit = 0; bit < llvm::Log2_32(warpSize); ++bit) {
      unsigned mask = 1u << bit;
      if ((activeLanes & mask) == 0)
        continue;
      SmallVector<Value> shfl(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        shfl[i] = targetInfo.shuffleXor(rewriter, op.getLoc(), acc[i], mask);
      }
      accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl);
    }
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(triton::ReduceOp op, SmallVector<SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              dyn_cast<RankedTensorType>(op.getResult()[i].getType())) {
        results[i] = packLLElements(loc, getTypeConverter(), accs[i], rewriter,
                                    resultTy);
      } else {
        results[i] = accs[i].front();
      }
    }
    rewriter.replaceOp(op, results);
  }

  SmallVector<SmallVector<Value>>
  convertLayoutValues(Location loc, ConversionPatternRewriter &rewriter,
                      triton::ReduceOp op, const LinearLayout &srcLayout,
                      const LinearLayout &dstLayout,
                      const SmallVector<SmallVector<Value>> &inVals,
                      ArrayRef<int64_t> smemBaseOffsets) const {
    SmallVector<SmallVector<Value>> outVals(op.getNumOperands());
    auto *ctx = rewriter.getContext();
    SmallVector<int64_t> shape;
    for (auto dim : srcLayout.getOutDimNames()) {
      shape.push_back(srcLayout.getOutDimSize(dim));
    }
    auto srcEnc = triton::gpu::LinearEncodingAttr::get(ctx, srcLayout);
    auto dstEnc = triton::gpu::LinearEncodingAttr::get(ctx, dstLayout);
    auto baseOffsetAttr = op->getAttrOfType<IntegerAttr>("allocation.offset");
    assert(baseOffsetAttr && "expected allocation.offset on reduce op");
    int64_t baseOffset = baseOffsetAttr.getValue().getZExtValue();
    auto offsetTy = IntegerType::get(ctx, 32);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = op.getElementTypes()[i];
      auto srcTy = RankedTensorType::get(shape, elemTy, srcEnc);
      auto dstTy = RankedTensorType::get(shape, elemTy, dstEnc);
      Value packed =
          packLLElements(loc, getTypeConverter(), inVals[i], rewriter, srcTy);
      auto srcTensor =
          UnrealizedConversionCastOp::create(rewriter, loc, srcTy, packed)
              .getResult(0);
      auto cvt =
          triton::gpu::ConvertLayoutOp::create(rewriter, loc, dstTy, srcTensor);
      cvt->setAttr("allocation.offset",
                   IntegerAttr::get(offsetTy, baseOffset + smemBaseOffsets[i]));
      Type packedDstTy = getTypeConverter()->convertType(dstTy);
      auto packedDst = UnrealizedConversionCastOp::create(
                           rewriter, loc, packedDstTy, cvt.getResult())
                           .getResult(0);
      outVals[i] = unpackLLElements(loc, packedDst, rewriter);
    }
    return outVals;
  }

  Type getReduceMemElemTy(Type elemTy, MLIRContext *ctx) const {
    if (elemTy.isIntOrFloat() && elemTy.getIntOrFloatBitWidth() < 8)
      return IntegerType::get(ctx, 8);
    return elemTy;
  }

  SmallVector<int64_t> getSmemBaseOffsets(triton::ReduceOp op,
                                          const LinearLayout &srcLayout,
                                          const LinearLayout &dstLayout) const {
    constexpr int64_t kReduceScratchAlign = 16;
    auto bytesPerOperand =
        ReduceOpHelper(op).getScratchBytesForCvt(srcLayout, dstLayout);
    std::vector<unsigned> indices(op.getNumOperands());
    std::iota(indices.begin(), indices.end(), 0);
    auto *ctx = op.getContext();
    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      auto lhsTy = getReduceMemElemTy(op.getElementTypes()[i], ctx);
      auto rhsTy = getReduceMemElemTy(op.getElementTypes()[j], ctx);
      return getIntOrFloatOrPtrBitWidth(lhsTy) >
             getIntOrFloatOrPtrBitWidth(rhsTy);
    });
    SmallVector<int64_t> offsets(op.getNumOperands());
    int64_t offset = 0;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      unsigned idx = indices[i];
      offset = llvm::alignTo(offset, kReduceScratchAlign);
      offsets[idx] = offset;
      offset += bytesPerOperand[idx];
    }
    return offsets;
  }
};
} // namespace

void mlir::triton::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, targetInfo, benefit);
}
