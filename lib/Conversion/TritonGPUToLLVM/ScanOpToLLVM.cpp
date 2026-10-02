#include "ReduceScanCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct ScanOpConversion : public ConvertOpToLLVMPattern<triton::ScanOp> {
  // Values are indexed by register, then by combiner operand.
  using ScanValues = SmallVector<SmallVector<Value>>;

  ScanOpConversion(LLVMTypeConverter &typeConverter,
                   const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::ScanOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ScanLoweringHelper helper(op);
    if (!helper.isSupported())
      return op.emitError("scan axis distributed across CTAs is not supported");

    auto loc = op.getLoc();
    ScanValues values;
    for (Value operand : adaptor.getOperands()) {
      auto unpacked = unpackUniqueTensorElements(loc, operand, rewriter);
      values.resize(unpacked.size());
      for (unsigned r = 0; r < unpacked.size(); ++r)
        values[r].push_back(unpacked[r]);
    }

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value threadId = getThreadId(rewriter, loc);
    unsigned warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    Value laneId = b.urem(threadId, b.i32_val(warpSize));
    Value warpId = b.udiv(threadId, b.i32_val(warpSize));

    // Keep native thread prefixes in place and communicate only segment totals.
    permuteRegisters(values, helper.getRegisterOrder());
    scanWithinThreads(op, values, helper.getThreadLocalSegmentSize(), rewriter);
    ScanValues intraWarpTotals, interWarpTotals;
    if (helper.getIntraWarpLayout())
      intraWarpTotals = scanWithinWarps(op, helper, values, laneId, rewriter);
    if (helper.getInterWarpLayout())
      interWarpTotals = scanAcrossWarps(
          op, helper, intraWarpTotals.empty() ? values : intraWarpTotals,
          laneId, warpId, rewriter);
    applyScanCarries(op, helper, values, intraWarpTotals, interWarpTotals,
                     laneId, warpId, rewriter);
    permuteRegisters(values, helper.getRegisterOrder().inverse());

    SmallVector<Value> results;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      SmallVector<Value> unpacked;
      for (const auto &row : values)
        unpacked.push_back(row[i]);
      results.push_back(packUniqueTensorElements(loc, getTypeConverter(),
                                                 unpacked, rewriter,
                                                 op.getResult()[i].getType()));
    }
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  static unsigned getSegmentMask(const LinearLayout &layout, StringAttr dim,
                                 unsigned axis, unsigned segmentSize) {
    unsigned mask = 0;
    for (auto [bit, basis] : llvm::enumerate(layout.getBases().lookup(dim)))
      if (basis[axis] && basis[axis] < segmentSize)
        mask |= 1u << bit;
    return mask;
  }

  static ScanValues transposeValues(const ScanValues &values) {
    ScanValues result(values.front().size());
    for (const auto &row : values)
      for (auto [i, value] : llvm::enumerate(row))
        result[i].push_back(value);
    return result;
  }

  static ScanValues extractSegmentTotals(const ScanValues &values,
                                         unsigned numRegs, bool reverse) {
    ScanValues totals;
    for (unsigned base = 0; base < values.size(); base += numRegs)
      totals.push_back(values[base + (reverse ? 0 : numRegs - 1)]);
    return totals;
  }

  SmallVector<Value> shuffleValues(Location loc, SmallVector<Value> values,
                                   Value lane,
                                   ConversionPatternRewriter &rewriter) const {
    for (Value &value : values)
      value = targetInfo.shuffleIdx(rewriter, loc, value, lane);
    return values;
  }

  // Shift within a segment, preserving bits that select other segments.
  static Value shiftWithinSegment(TritonLLVMOpBuilder &b, Value index,
                                  int distance, unsigned segmentSize,
                                  unsigned axisSize) {
    unsigned mask = segmentSize - 1;
    Value shifted = b.and_(b.add(index, b.i32_val(distance)), b.i32_val(mask));
    if (segmentSize == axisSize)
      return shifted;
    return b.or_(b.and_(index, b.i32_val(~mask)), shifted);
  }

  // Reindex SSA values without emitting data movement.
  void permuteRegisters(ScanValues &values, const ColumnAction &order) const {
    if (order.isIdentity())
      return;
    auto operands = transposeValues(values);
    for (auto &operand : operands)
      operand = order.apply(operand);
    values = transposeValues(operands);
  }

  // Convert segment totals using warp-local shuffles.
  void convertScanValues(triton::ScanOp op, ScanValues &values,
                         const LinearLayout &src, const LinearLayout &dst,
                         ConversionPatternRewriter &rewriter) const {
    if (src == dst)
      return;
    auto operands = convertLayoutValues(
        op.getLoc(), rewriter, op, src, dst, transposeValues(values),
        getTypeConverter(), targetInfo, /*forceWarpShuffle=*/true);
    values = transposeValues(operands);
  }

  // Scan contiguous register groups in traversal order.
  void scanWithinThreads(triton::ScanOp op, ScanValues &values,
                         unsigned numRegs,
                         ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    bool reverse = op.getReverse();
    for (unsigned base = 0; base < values.size(); base += numRegs)
      for (unsigned i = 1; i < numRegs; ++i) {
        unsigned r = base + (reverse ? numRegs - 1 - i : i);
        unsigned prev = reverse ? r + 1 : r - 1;
        values[r] = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                   values[prev], values[r]);
      }
  }

  // Scan lane totals, reusing each round's shuffle for the exclusive carries.
  void scanLaneTotals(triton::ScanOp op, ScanValues &values,
                      const LinearLayout &layout, unsigned numRegs,
                      unsigned segmentSize, Value laneId,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kLane = StringAttr::get(op.getContext(), "lane");
    unsigned axis = op.getAxis();
    bool reverse = op.getReverse();
    unsigned numLanes = segmentSize / numRegs;
    if (numLanes == 1)
      return;
    auto dims = llvm::to_vector(layout.getOutDimNames());
    auto laneLayout = layout.sublayout({kLane}, dims);
    auto inverseLaneLayout = layout.pseudoinvert().sublayout(dims, {kLane});

    // Shift logical coordinates and invert the layout to find the source lane.
    // Preserve coordinates belonging to independent scans.
    auto coords =
        applyLinearLayout(loc, rewriter, laneLayout, {{kLane, laneId}});
    Value index = coords[axis].second;
    unsigned axisSize = layout.getOutDimSize(dims[axis]);
    Value segmentIndex = index;
    if (segmentSize != axisSize)
      segmentIndex = b.and_(index, b.i32_val(segmentSize - 1));
    // Source lanes and predicates are shared by every register group.
    SmallVector<std::pair<Value, Value>> rounds;
    for (unsigned offset = 1; offset < numLanes; offset *= 2) {
      int distance = offset * numRegs;
      auto source = coords;
      source[axis].second = shiftWithinSegment(
          b, index, reverse ? distance : -distance, segmentSize, axisSize);
      Value lane = applyLinearLayout(loc, rewriter, inverseLaneLayout, source)
                       .front()
                       .second;
      Value pred =
          reverse ? b.icmp_ult(segmentIndex, b.i32_val(segmentSize - distance))
                  : b.icmp_uge(segmentIndex, b.i32_val(distance));
      rounds.emplace_back(lane, pred);
    }
    for (unsigned base = 0; base < values.size(); base += numRegs) {
      unsigned last = base + (reverse ? 0 : numRegs - 1);
      auto acc = values[last];
      SmallVector<Value> prefix;
      for (auto [lane, pred] : rounds) {
        auto incoming = shuffleValues(loc, acc, lane, rewriter);
        if (numRegs > 1) {
          // Initialize the exclusive carry from the first shuffle, then prepend
          // earlier groups. The boundary lane never consumes its carry.
          if (prefix.empty())
            prefix = incoming;
          else
            prefix = combineWithPrefix(op, incoming, prefix, rewriter, pred);
        }
        acc = combineWithPrefix(op, incoming, acc, rewriter, pred);
      }
      values[last] = acc;
      if (numRegs == 1)
        continue;
      // Skip the boundary lane without assuming an identity value.
      Value pred = rounds.front().second;
      for (unsigned r = base; r < base + numRegs; ++r)
        if (r != last)
          values[r] = combineWithPrefix(op, prefix, values[r], rewriter, pred);
    }
  }

  // Scan thread-segment totals, leaving native prefixes untouched.
  ScanValues scanWithinWarps(triton::ScanOp op,
                             const ScanLoweringHelper &helper,
                             const ScanValues &values, Value laneId,
                             ConversionPatternRewriter &rewriter) const {
    const auto &intraWarpLayout = *helper.getIntraWarpLayout();
    const auto &scanLayout = *helper.getIntraWarpScanLayout();
    unsigned segmentRegs = helper.getThreadLocalSegmentSize();
    unsigned numSegments = helper.getWarpLocalSegmentSize() / segmentRegs;
    bool reverse = op.getReverse();

    auto totals = extractSegmentTotals(values, segmentRegs, reverse);
    convertScanValues(op, totals, intraWarpLayout, scanLayout, rewriter);

    auto kReg = StringAttr::get(op.getContext(), "register");
    auto axis =
        StringAttr::get(op.getContext(), "dim" + std::to_string(op.getAxis()));
    unsigned numRegs =
        scanLayout.sublayout({kReg}, {axis}).getNumConsecutiveInOut();
    scanWithinThreads(op, totals, numRegs, rewriter);
    scanLaneTotals(op, totals, scanLayout, numRegs, numSegments, laneId,
                   rewriter);
    // Keep this layout for extracting inter-warp totals.
    return totals;
  }

  // Replicate the warp-segment totals in each warp and scan their full
  // sequence.
  ScanValues scanAcrossWarps(triton::ScanOp op,
                             const ScanLoweringHelper &helper,
                             const ScanValues &values, Value laneId,
                             Value warpId,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kLane = StringAttr::get(ctx, "lane");
    // Use converted thread totals when available, otherwise native prefixes.
    const auto &sourceLayout = helper.getIntraWarpScanLayout()
                                   ? *helper.getIntraWarpScanLayout()
                                   : helper.getPermutedLayout();
    unsigned segmentSize = helper.getWarpLocalSegmentSize();
    // Express the segment size in source-layout units.
    if (helper.getIntraWarpScanLayout())
      segmentSize /= helper.getThreadLocalSegmentSize();
    const auto &interWarpLayout = *helper.getInterWarpLayout();
    const auto &totalsLayout = *helper.getInterWarpScanLayout();
    bool reverse = op.getReverse();
    auto kReg = StringAttr::get(ctx, "register");
    unsigned segmentRegs =
        sourceLayout.getInDimSize(kReg) / interWarpLayout.getInDimSize(kReg);
    unsigned segmentLaneMask =
        getSegmentMask(sourceLayout, kLane, op.getAxis(), segmentSize);

    // Only terminal lanes store complete segment totals; conversion loads
    // distribute them to the destination layout.
    auto totals = extractSegmentTotals(values, segmentRegs, reverse);
    Value storePred;
    if (segmentLaneMask)
      storePred = b.icmp_eq(b.and_(laneId, b.i32_val(segmentLaneMask)),
                            b.i32_val(reverse ? 0 : segmentLaneMask));
    auto operands = convertLayoutValues(loc, rewriter, op, interWarpLayout,
                                        totalsLayout, transposeValues(totals),
                                        getTypeConverter(), targetInfo,
                                        /*forceWarpShuffle=*/false, storePred);
    totals = transposeValues(operands);

    // Reuse the warp-local scan, including sequences spanning registers.
    ScanLoweringHelper totalsHelper(totalsLayout, op.getAxis());
    permuteRegisters(totals, totalsHelper.getRegisterOrder());
    scanWithinThreads(op, totals, totalsHelper.getThreadLocalSegmentSize(),
                      rewriter);
    ScanValues intraWarpTotals;
    if (totalsHelper.getIntraWarpLayout())
      intraWarpTotals =
          scanWithinWarps(op, totalsHelper, totals, laneId, rewriter);
    applyScanCarries(op, totalsHelper, totals, intraWarpTotals, {}, laneId,
                     warpId, rewriter);
    permuteRegisters(totals, totalsHelper.getRegisterOrder().inverse());

    return totals;
  }

  // Apply intra-warp carries before inter-warp carries to preserve operand
  // order.
  void applyScanCarries(triton::ScanOp op, const ScanLoweringHelper &helper,
                        ScanValues &values, ScanValues &intraWarpTotals,
                        const ScanValues &interWarpTotals, Value laneId,
                        Value warpId,
                        ConversionPatternRewriter &rewriter) const {
    if (helper.getIntraWarpLayout()) {
      const auto &intraWarpLayout = *helper.getIntraWarpLayout();
      unsigned segmentRegs = helper.getThreadLocalSegmentSize();
      unsigned numSegments = helper.getWarpLocalSegmentSize() / segmentRegs;
      convertScanValues(op, intraWarpTotals, *helper.getIntraWarpScanLayout(),
                        intraWarpLayout, rewriter);

      // Terminal prefixes are already scanned; only other registers need
      // carries.
      for (unsigned r = 0; r < intraWarpTotals.size(); ++r)
        values[r * segmentRegs + (op.getReverse() ? 0 : segmentRegs - 1)] =
            intraWarpTotals[r];
      if (segmentRegs > 1)
        applySegmentCarries(op, values, intraWarpTotals, intraWarpLayout,
                            intraWarpLayout, segmentRegs, numSegments, laneId,
                            warpId, rewriter, /*skipTerminal=*/true);
    }
    if (helper.getInterWarpLayout()) {
      const auto &interWarpLayout = *helper.getInterWarpLayout();
      auto kReg = StringAttr::get(op.getContext(), "register");
      unsigned segmentRegs = helper.getPermutedLayout().getInDimSize(kReg) /
                             interWarpLayout.getInDimSize(kReg);
      auto axis =
          *std::next(interWarpLayout.getOutDimNames().begin(), op.getAxis());
      applySegmentCarries(op, values, interWarpTotals, interWarpLayout,
                          *helper.getInterWarpScanLayout(), segmentRegs,
                          interWarpLayout.getOutDimSize(axis), laneId, warpId,
                          rewriter);
    }
  }

  // Enumerate the source registers that the lanes of this segment can request.
  static SmallVector<unsigned>
  getCarryRegisters(const LinearLayout &segmentLayout,
                    const LinearLayout &inverseTotalsLayout, unsigned reg,
                    unsigned axis, unsigned segmentsPerScan, bool reverse) {
    auto *ctx = segmentLayout.getInDimNames().begin()->getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    unsigned mask = segmentsPerScan - 1;
    llvm::SmallSetVector<unsigned, 8> candidates;
    for (unsigned warp = 0; warp < segmentLayout.getInDimSize(kWarp); ++warp)
      for (unsigned lane = 0; lane < segmentLayout.getInDimSize(kLane);
           ++lane) {
        auto coords = segmentLayout.apply(
            {{kReg, reg}, {kLane, lane}, {kWarp, warp}, {kBlock, 0}});
        auto &index = coords[axis].second;
        index = (index & ~mask) | ((index + (reverse ? 1 : -1)) & mask);
        candidates.insert(inverseTotalsLayout.apply(coords).front().second);
      }
    return llvm::to_vector(candidates);
  }

  // Select after shuffling: each destination lane may request a different
  // register from its source lane.
  SmallVector<Value> shuffleCarry(Location loc, const ScanValues &totals,
                                  ArrayRef<unsigned> candidates, Value srcReg,
                                  Value srcLane,
                                  ConversionPatternRewriter &rewriter) const {
    assert(!candidates.empty() && "each segment has a warp-local carry");
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto carry =
        shuffleValues(loc, totals[candidates.front()], srcLane, rewriter);
    for (unsigned candidate : candidates.drop_front()) {
      auto incoming = shuffleValues(loc, totals[candidate], srcLane, rewriter);
      Value select = b.icmp_eq(srcReg, b.i32_val(candidate));
      for (auto [value, source] : llvm::zip(carry, incoming))
        value = b.select(select, source, value);
    }
    return carry;
  }

  // Map each native segment to its exclusive carry in the totals layout.
  void applySegmentCarries(triton::ScanOp op, ScanValues &values,
                           const ScanValues &totals,
                           const LinearLayout &segmentLayout,
                           const LinearLayout &totalsLayout,
                           unsigned segmentRegs, unsigned segmentsPerScan,
                           Value laneId, Value warpId,
                           ConversionPatternRewriter &rewriter,
                           bool skipTerminal = false) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    auto axis =
        *std::next(segmentLayout.getOutDimNames().begin(), op.getAxis());
    bool reverse = op.getReverse();
    // All required totals are present in this warp.
    auto inverse = totalsLayout.pseudoinvert().sublayout(
        llvm::to_vector(totalsLayout.getOutDimNames()), {kReg, kLane});
    unsigned numSegments = segmentLayout.getOutDimSize(axis);
    unsigned begin = skipTerminal && reverse ? 1 : 0;
    unsigned end = segmentRegs - (skipTerminal && !reverse);
    for (unsigned r = 0; r < values.size() / segmentRegs; ++r) {
      // Evaluate the logical segment before shifting. CTA ownership is
      // unchanged.
      auto coords = applyLinearLayout(loc, rewriter, segmentLayout,
                                      {{kReg, b.i32_val(r)},
                                       {kLane, laneId},
                                       {kWarp, warpId},
                                       {kBlock, b.i32_val(0)}});
      Value segment = coords[op.getAxis()].second;
      Value segmentInScan = segment;
      if (segmentsPerScan != numSegments)
        segmentInScan = b.and_(segment, b.i32_val(segmentsPerScan - 1));
      Value pred = b.icmp_ne(segmentInScan,
                             b.i32_val(reverse ? segmentsPerScan - 1 : 0));
      // Wrap within this scan; pred excludes the boundary without an identity.
      coords[op.getAxis()].second = shiftWithinSegment(
          b, segment, reverse ? 1 : -1, segmentsPerScan, numSegments);
      auto src = applyLinearLayout(loc, rewriter, inverse, coords);
      Value srcReg = src[0].second;
      Value srcLane = src[1].second;
      auto candidates = getCarryRegisters(
          segmentLayout, inverse, r, op.getAxis(), segmentsPerScan, reverse);
      auto carry =
          shuffleCarry(loc, totals, candidates, srcReg, srcLane, rewriter);
      for (unsigned j = begin; j < end; ++j) {
        unsigned reg = r * segmentRegs + j;
        values[reg] = combineWithPrefix(op, carry, values[reg], rewriter, pred);
      }
    }
  }

  // Guard the combine (including side effects) and preserve boundary values.
  SmallVector<Value> combineWithPrefix(triton::ScanOp op, ValueRange prefix,
                                       ValueRange values,
                                       ConversionPatternRewriter &rewriter,
                                       Value pred) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto combined =
        applyCombineOp(loc, rewriter, op.getCombineOp(), prefix, values, pred);
    if (pred)
      for (unsigned i = 0; i < combined.size(); ++i)
        combined[i] = b.select(pred, combined[i], values[i]);
    return combined;
  }

  const TargetInfoBase &targetInfo;
};
} // namespace

void mlir::triton::populateScanOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ScanOpConversion>(typeConverter, targetInfo, benefit);
}
