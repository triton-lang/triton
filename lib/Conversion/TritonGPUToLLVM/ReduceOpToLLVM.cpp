#include "ReduceScanCommon.h"

#include <memory>
#include <tuple>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Allocation.h"
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
    if (!helper.isReduceWithinCTA()) {
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
    auto kAxis = *(regLl.getOutDimNames().begin() + axis);
    int i = 0;
    while (regLl.getOutDimSize(kAxis) != 1) {
      LinearLayout tmpLl = ReduceOpHelper::getInterLayout(regLl, axis);

      // Emit a barrier if we are reusing the shmem
      if (i > 0) {
        sync(rewriter, loc);
      }
      accs = convertLayoutValues(loc, rewriter, op, regLl, tmpLl, accs);

      std::tie(regLl, accs) =
          reduceWithinWarps(op, std::move(tmpLl), std::move(accs), rewriter);
      ++i;
    }
    assert(i <= 2 && "expected at most 2 rounds of warp reductions");
    // Remove the axis dimension, which at this point is of size 1
    regLl = removeStandardDim(regLl, axis);

    // Convert to output layout if we didn't fit the warp bases within zero
    // bases in the tmpLl
    if (auto resultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto outputLayout = triton::gpu::toLinearLayout(resultTy);
      if (regLl != outputLayout) {
        // Reuse the shmem
        sync(rewriter, loc);
        accs =
            convertLayoutValues(loc, rewriter, op, regLl, outputLayout, accs);
      }
    }

    packResults(op, accs, rewriter);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  SmallVector<Value>
  treeReduceBinary(Location loc, ConversionPatternRewriter &rewriter,
                   Region &combineOp,
                   SmallVector<SmallVector<Value>> values) const {
    // The number of elements is always a power of two
    assert(llvm::isPowerOf2_64(values.size()) && !values.empty());
    while (values.size() > 1) {
      SmallVector<SmallVector<Value>> next;
      for (size_t i = 0; i + 1 < values.size(); i += 2) {
        SmallVector<Value> acc = values[i];
        accumulate(loc, rewriter, combineOp, acc, values[i + 1]);
        next.push_back(std::move(acc));
      }
      values = std::move(next);
    }
    return values.front();
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

  void sync(ConversionPatternRewriter &rewriter, Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    b.barrier(triton::gpu::AddrSpace::Local);
  }

  void packVectorized(SmallVector<SmallVector<Value>> &accs,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = accs.front().front().getLoc();
    for (auto &acc : accs) {
      SmallVector<Value> packedAcc;
      for (unsigned reg = 0; reg < acc.size(); reg += 2) {
        auto vector = packLLVector(loc, {acc[reg], acc[reg + 1]}, rewriter);
        packedAcc.emplace_back(std::move(vector));
      }
      acc = std::move(packedAcc);
    }
  }

  std::unique_ptr<Region> createVectorCombineRegion(
      Location loc, Type elemTy,
      ReduceOpHelper::InThreadVectorizeOpKind vectorizeKind,
      ConversionPatternRewriter &rewriter) const {
    if (vectorizeKind == ReduceOpHelper::InThreadVectorizeOpKind::None)
      return nullptr;
    MLIRContext *ctx = rewriter.getContext();
    auto vecTy = vec_ty(elemTy, 2);

    auto storage = std::make_unique<Region>();
    auto *block = new Block();
    storage->push_back(block);
    block->addArgument(vecTy, loc);
    block->addArgument(vecTy, loc);

    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(block);
    Value result = ReduceOpHelper::createInThreadVectorizedCombineOp(
        builder, loc, vectorizeKind, block->getArgument(0),
        block->getArgument(1));
    triton::ReduceReturnOp::create(builder, loc, ValueRange{result});
    return storage;
  }

  void unpackVectorized(Location loc, SmallVector<SmallVector<Value>> &accs,
                        ConversionPatternRewriter &rewriter,
                        Region *reduction) const {
    for (auto &acc : accs) {
      SmallVector<Value> unpacked;
      for (Value val : acc) {
        auto elems = unpackLLVector(loc, val, rewriter);
        assert(elems.size() == 2 && "expected a 2-lane packed vector");
        if (reduction) {
          SmallVector<Value> cur = {elems[0]};
          accumulate(loc, rewriter, *reduction, cur, {elems[1]});
          unpacked.emplace_back(cur[0]);
        } else {
          unpacked.emplace_back(elems[0]);
          unpacked.emplace_back(elems[1]);
        }
      }
      acc = std::move(unpacked);
    }
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinThreads(triton::ReduceOp op, LinearLayout layout,
                      SmallVector<SmallVector<Value>> accs,
                      ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto loc = op.getLoc();
    unsigned axis = op.getAxis();
    auto kReg = str_attr("register");
    auto linearAttr = triton::gpu::LinearEncodingAttr::get(ctx, layout);
    auto basesPerDim = linearAttr.basesPerDim(kReg, /*skipBroadcast=*/true);
    unsigned axisPack = basesPerDim[axis];
    if (axisPack == 1) {
      return {std::move(layout), std::move(accs)};
    }

    ReduceOpHelper helper(op);
    auto vectorizeKind = helper.getInThreadVectorizeOpKind(
        axisPack, targetInfo.supportBitwidth16Elementwise(),
        targetInfo.supportBitwidth32Elementwise());
    bool vectorize =
        vectorizeKind != ReduceOpHelper::InThreadVectorizeOpKind::None;

    // Bring the registers that move the axis to the front
    auto perm = ReduceOpHelper::moveAxisBasesToFront(layout, axis, vectorize);
    if (!perm.isIdentity()) {
      layout = perm.apply(layout);
      for (auto &vals : accs) {
        vals = perm.apply(vals);
      }
    }

    // Pack the inputs into vector values
    if (vectorize)
      packVectorized(accs, rewriter);

    // If we pack along the reduction axis we need to process half the registers
    const auto &regBases = layout.getBases().lookup(kReg);
    bool packAlongAxis = vectorize && regBases.front()[axis] != 0;
    if (packAlongAxis)
      axisPack /= 2;

    // Create the vectorized region if needed
    auto elemTy =
        cast<RankedTensorType>(op.getOperandTypes().front()).getElementType();
    std::unique_ptr<Region> vectorCombineRegion =
        createVectorCombineRegion(loc, elemTy, vectorizeKind, rewriter);
    Region &combineRegion =
        vectorCombineRegion ? *vectorCombineRegion : op.getCombineOp();

    // Perform a tree reduction
    unsigned numOperands = accs.size();
    SmallVector<SmallVector<Value>> reduced(numOperands);
    unsigned regs = accs.front().size();
    for (unsigned regBase = 0; regBase < regs; regBase += axisPack) {
      // Transpose from [opIdx][reg] into [reg][opIdx]
      SmallVector<SmallVector<Value>> vals;
      for (unsigned i = 0; i < axisPack; ++i) {
        SmallVector<Value> cur(numOperands);
        for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx) {
          cur[opIdx] = accs[opIdx][regBase + i];
        }
        vals.push_back(std::move(cur));
      }
      auto acc =
          treeReduceBinary(loc, rewriter, combineRegion, std::move(vals));
      for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx) {
        reduced[opIdx].push_back(acc[opIdx]);
      }
    }
    accs = std::move(reduced);

    // Unpack the vector values into the accumulator values
    // Reduce one last time via the scalar combine op if we packed along the
    // axis
    if (vectorize) {
      Region *reduceAfterUnpacking =
          packAlongAxis ? &op.getCombineOp() : nullptr;
      unpackVectorized(loc, accs, rewriter, reduceAfterUnpacking);
    }

    // Update layout killing the axis bases along registers
    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, axis, kReg);
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
    unsigned reduceLaneIdMask = 0;
    for (unsigned bit = 0; bit < laneBases.size(); ++bit) {
      if (laneBases[bit][op.getAxis()] != 0) {
        reduceLaneIdMask |= 1u << bit;
      }
    }
    if (reduceLaneIdMask == 0) {
      return {std::move(layout), std::move(accs)};
    }

    unsigned regs = accs.front().size();
    for (unsigned reg = 0; reg < regs; ++reg) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        acc[i] = accs[i][reg];
      }
      warpReduce(op, reduceLaneIdMask, acc, rewriter);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        accs[i][reg] = acc[i];
      }
    }

    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, op.getAxis(),
                                                         kLane);
    return {std::move(layout), std::move(accs)};
  }

  void warpReduce(triton::ReduceOp op, unsigned reduceLaneIdMask,
                  SmallVector<Value> &acc,
                  ConversionPatternRewriter &rewriter) const {
    // No reduction to do
    if (reduceLaneIdMask == 0)
      return;
    auto moduleOp = op->getParentOfType<ModuleOp>();
    unsigned warpSize =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);
    assert(reduceLaneIdMask < warpSize &&
           "expected reduce lane ID mask to be strictly less than warp size");
    // Try to use the redux op if it is supported by the target
    if (targetInfo.warpReduce(rewriter, op.getLoc(), acc, op,
                              reduceLaneIdMask)) {
      return;
    }
    // Not that it matters a lot, but a more reasonble iteration order would be
    // from bit 0 to bit llvm::Log2_32(warpSize) - 1. Changing this breaks a ton
    // of bitwise comparisons so we stick with the legacy inverse order
    for (int bit = llvm::Log2_32(warpSize) - 1; bit >= 0; --bit) {
      unsigned mask = 1u << bit;
      if ((reduceLaneIdMask & mask) == 0)
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
                      const SmallVector<SmallVector<Value>> &inVals) const {
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
    auto smemBaseOffsets = getSmemBaseOffsets(op, srcLayout, dstLayout);
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
    // Hack:
    // Here we know that we are never going to use ldmatrix/stmatrix
    // instructions as by the time we go through shared memory, we have already
    // reduced all the registers As such, we can use
    // `getNumScratchElemsSwizzledCvt` which assumes ld.shared/st.shared
    // instructions
    // The proper way to lower reduce would be to lower it to:
    // reduce_threads / reduce_lanes / convert_layout
    // And let the AllocationAnalysis handle the shared memory allocation
    // and membar the barriers
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
      offsets[idx] = offset;
      auto inputTy = op.getInputTypes()[idx];
      auto bytes = getNumScratchElemsSwizzledCvt(srcLayout, dstLayout,
                                                 getBitwidth(inputTy)) *
                   (getBitwidth(inputTy) / 8);
      offset += bytes;
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
