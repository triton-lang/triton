#include "ReduceScanCommon.h"

#include <tuple>
#include <utility>

#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

LinearLayout zeroBasesAlongDimAndReorder(const LinearLayout &layout,
                                         unsigned axis, StringAttr dim) {
  LinearLayout::BasesT newBases;
  for (auto [inDim, bases] : layout.getBases()) {
    std::vector<std::vector<int32_t>> newInBases = bases;
    if (inDim == dim) {
      for (auto &basis : newInBases)
        basis[axis] = 0;
    }
    newBases[inDim] = std::move(newInBases);
  }

  // We zero bases for one in-dim along the reduction axis, then compact the
  // remaining axis bases to 1, 2, 4, ...
  // We assume the reduction is commutative (and obviously associative), so we
  // can reorder axis contributions and shrink the axis size to keep the layout
  // surjective.
  int32_t nextAxisBase = 1;
  for (auto &[inDim, inDimBases] : newBases) {
    for (auto &basis : inDimBases) {
      if (basis[axis] == 0)
        continue;
      basis[axis] = nextAxisBase;
      nextAxisBase *= 2;
    }
  }

  return LinearLayout(std::move(newBases), to_vector(layout.getOutDimNames()));
}

ColumnAction makeAxisContiguous(const LinearLayout &layout, int axis) {
  auto *ctx = layout.getOutDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  const auto &bases = layout.getBases().lookup(kReg);
  SmallVector<size_t> perm;
  SmallVector<size_t> back;
  for (size_t i = 0; i < bases.size(); ++i) {
    if (bases[i][axis] != 0)
      perm.push_back(i);
    else
      back.push_back(i);
  }
  perm.append(back.begin(), back.end());
  return ColumnAction(perm, kReg, bases.size());
}

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
    assert(helper.isReduceWithinCTA() &&
           "Unexpected srcLayout in ReduceOpConversion");
    Location loc = op->getLoc();
    auto accs = unpackInputs(loc, op, adaptor, rewriter);
    unsigned axis = op.getAxis();

    auto *ctx = op.getContext();
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");

    // Remove block as we don't currently support it
    LinearLayout regLl = triton::gpu::toLinearLayout(helper.getSrcTy());
    regLl = regLl.sublayout({kReg, kLane, kWarp},
                            to_vector(regLl.getOutDimNames()));
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

    if (helper.isWarpSynchronous()) {
      // If all the values to be reduced are within the same warp there is
      // nothing left to do.
      packResults(op, accs, rewriter);
      return success();
    }

    // Create temporary layout for reduction within warps
    auto regBases = regLl.getBases();
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    int laneBits = regLl.getInDimSizeLog2(kLane);
    int neededLaneBits = llvm::Log2_32(sizeInterWarps);
    // TODO Will fix when we implement block-level reductions
    assert(neededLaneBits <= laneBits && "NYI: more inter-warps than lanes");
    // Move the warp axis bases we need to reduce into lane bases, while
    // keeping non-axis components in their original in-dim.
    auto &laneBases = regBases[kLane];
    auto &warpBases = regBases[kWarp];
    int moved = 0;
    for (auto &warpBasis : warpBases) {
      if (warpBasis[axis] == 0)
        continue;
      std::swap(laneBases[moved], warpBasis);
      moved++;
    }

    LinearLayout tmpLl(std::move(regBases), to_vector(regLl.getOutDimNames()));

    // Compute shared memory offsets per operand.
    auto nelemsSmem = product<unsigned>(helper.getScratchRepShape());
    auto smemBaseOffsets = getSmemBaseOffsets(op, nelemsSmem);
    accs = convertLayoutValues(loc, rewriter, op, regLl, tmpLl, accs,
                               smemBaseOffsets);

    std::tie(tmpLl, accs) =
        reduceWithinWarps(op, std::move(tmpLl), std::move(accs), rewriter);
    // Remove the axis dimension
    assert(to_vector(tmpLl.getOutDimSizes())[axis] == 1);
    tmpLl = removeStandardDim(tmpLl, axis);

    // Convert to output layout if we didn't fit the warp bases within zero
    // bases in the tmpLl
    // TODO Prefer tmp layouts that omit this conversion whenever possible
    if (auto resultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto outputLayout = triton::gpu::toLinearLayout(resultTy);
      outputLayout = outputLayout.sublayout(
          {kReg, kLane, kWarp}, to_vector(outputLayout.getOutDimNames()));
      if (tmpLl != outputLayout) {
        // Reuse the shmem
        auto b = TritonLLVMOpBuilder(loc, rewriter);
        b.barrier();
        accs = convertLayoutValues(loc, rewriter, op, tmpLl, outputLayout, accs,
                                   smemBaseOffsets);
      }
    }

    packResults(op, accs, rewriter);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
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

    // Bring the registers that move the axis to the front
    auto perm = makeAxisContiguous(layout, op.getAxis());
    if (!perm.isIdentity()) {
      layout = perm.apply(layout);
      for (auto &vals : accs) {
        vals = perm.apply(vals);
      }
    }

    // Reduce linearly
    // TODO Perform a tree reduction
    SmallVector<SmallVector<Value>> reduced(op.getNumOperands());
    for (unsigned regBase = 0; regBase < layout.getInDimSize(kReg);
         regBase += axisPack) {
      SmallVector<Value> acc;
      for (unsigned i = 0; i < axisPack; ++i) {
        SmallVector<Value> cur(op.getNumOperands());
        for (unsigned opIdx = 0; opIdx < op.getNumOperands(); ++opIdx) {
          cur[opIdx] = accs[opIdx][regBase + i];
        }
        accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, cur);
      }
      for (unsigned opIdx = 0; opIdx < op.getNumOperands(); ++opIdx) {
        reduced[opIdx].push_back(acc[opIdx]);
      }
    }
    accs = std::move(reduced);

    // Update layout killing the axis bases along registers
    layout = zeroBasesAlongDimAndReorder(layout, op.getAxis(), kReg);
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
    SmallVector<unsigned> laneMasks;
    for (unsigned bit = 0; bit < laneBases.size(); ++bit) {
      if (laneBases[bit][op.getAxis()] != 0) {
        laneMasks.push_back(1u << bit);
      }
    }
    if (laneMasks.empty()) {
      return {std::move(layout), std::move(accs)};
    }

    unsigned regs = accs.front().size();
    for (unsigned reg = 0; reg < regs; ++reg) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        acc[i] = accs[i][reg];
      }
      for (unsigned mask : laneMasks) {
        SmallVector<Value> shfl(op.getNumOperands());
        for (unsigned i = 0; i < op.getNumOperands(); ++i) {
          shfl[i] = targetInfo.shuffleXor(rewriter, op.getLoc(), acc[i], mask);
        }
        accumulate(op.getLoc(), rewriter, op.getCombineOp(), acc, shfl);
      }
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        accs[i][reg] = acc[i];
      }
    }

    layout = zeroBasesAlongDimAndReorder(layout, op.getAxis(), kLane);
    return {std::move(layout), std::move(accs)};
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

  SmallVector<int64_t> getSmemBaseOffsets(triton::ReduceOp op,
                                          unsigned nelems) const {
    std::vector<unsigned> indices(op.getNumOperands());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](unsigned i, unsigned j) {
      return op.getElementTypes()[i].getIntOrFloatBitWidth() >
             op.getElementTypes()[j].getIntOrFloatBitWidth();
    });
    SmallVector<int64_t> offsets(op.getNumOperands());
    int64_t offset = 0;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      unsigned idx = indices[i];
      offsets[idx] = offset;
      unsigned bitwidth = op.getElementTypes()[idx].getIntOrFloatBitWidth();
      bitwidth = std::max(bitwidth, 8u);
      offset += nelems * ceil(bitwidth, 8u);
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
