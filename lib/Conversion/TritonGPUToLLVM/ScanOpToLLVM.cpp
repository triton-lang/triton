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

  struct ScanCarry {
    SmallVector<Value> values;
    Value pred;
  };

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

    scanWithinCTA(op, helper, values, laneId, warpId, rewriter);

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
  void scanWithinCTA(triton::ScanOp op, const ScanLoweringHelper &helper,
                     ScanValues &values, Value laneId, Value warpId,
                     ConversionPatternRewriter &rewriter) const {
    // Keep native thread prefixes in place and communicate only segment totals.
    permuteRegisters(values, helper.getRegisterOrder());
    scanWithinThreads(op, values, helper.getThreadSegmentSize(), rewriter);
    ScanValues intraWarpTotals;
    SmallVector<ScanCarry> interWarpCarries;
    if (helper.getIntraWarpTotalsLayout())
      intraWarpTotals = scanWithinWarps(op, helper, values, laneId, rewriter);
    if (helper.getInterWarpTotalsLayout())
      interWarpCarries = scanAcrossWarps(
          op, helper, intraWarpTotals.empty() ? values : intraWarpTotals,
          laneId, warpId, rewriter);
    applyScanCarries(op, helper, values, intraWarpTotals, interWarpCarries,
                     laneId, warpId, rewriter);
    permuteRegisters(values, helper.getRegisterOrder().inverse());
  }

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

  SmallVector<Value>
  shuffleValues(Location loc, SmallVector<Value> values, Value lane,
                ConversionPatternRewriter &rewriter,
                std::optional<unsigned> upDistance = {}) const {
    for (Value &value : values)
      value = upDistance
                  ? targetInfo.shuffleUp(rewriter, loc, value, *upDistance)
                  : targetInfo.shuffleIdx(rewriter, loc, value, lane);
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

  // Convert segment totals using the shared reduction/scan conversion helper.
  void convertScanValues(triton::ScanOp op, ScanValues &values,
                         const LinearLayout &src, const LinearLayout &dst,
                         ConversionPatternRewriter &rewriter,
                         bool forceWarpShuffle = true) const {
    if (src == dst)
      return;
    auto operands = convertLayoutValues(
        op.getLoc(), rewriter, op, src, dst, transposeValues(values),
        getTypeConverter(), targetInfo, forceWarpShuffle);
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

  // Each round shuffles the preceding lanes' terminal total and combines it
  // with every register prefix in the group.
  void scanLaneTotals(triton::ScanOp op, ScanValues &values,
                      const LinearLayout &layout, unsigned numRegs,
                      unsigned segmentSize, Value laneId,
                      ConversionPatternRewriter &rewriter) const {
    unsigned numLanes = segmentSize / numRegs;
    if (numLanes == 1)
      return;
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kLane = StringAttr::get(op.getContext(), "lane");
    auto kReg = StringAttr::get(op.getContext(), "register");
    unsigned axis = op.getAxis();
    bool reverse = op.getReverse();
    auto dims = llvm::to_vector(layout.getOutDimNames());
    auto laneLayout = layout.sublayout({kLane}, dims);
    auto inverseLayout = layout.pseudoinvert().sublayout(dims, {kReg, kLane});
    auto coords =
        applyLinearLayout(loc, rewriter, laneLayout, {{kLane, laneId}});
    Value index = coords[axis].second;
    unsigned axisSize = layout.getOutDimSize(dims[axis]);
    Value segmentIndex = index;
    if (segmentSize != axisSize)
      segmentIndex = b.and_(index, b.i32_val(segmentSize - 1));

    // Shift logical coordinates and invert the layout to find the source lane.
    // Preserve coordinates belonging to independent scans.
    SmallVector<std::tuple<Value, Value, std::optional<unsigned>>> rounds;
    for (unsigned offset = 1; offset < numLanes; offset *= 2) {
      int distance = offset * numRegs;
      auto upDistance =
          getShuffleUpDistance(layout, inverseLayout, axis, segmentSize,
                               reverse ? distance : -distance);
      Value lane;
      if (!upDistance) {
        auto source = coords;
        source[axis].second = shiftWithinSegment(
            b, index, reverse ? distance : -distance, segmentSize, axisSize);
        lane = applyLinearLayout(loc, rewriter, inverseLayout, source)
                   .back()
                   .second;
      }
      Value pred =
          reverse ? b.icmp_ult(segmentIndex, b.i32_val(segmentSize - distance))
                  : b.icmp_uge(segmentIndex, b.i32_val(distance));
      rounds.emplace_back(lane, pred, upDistance);
    }
    for (unsigned base = 0; base < values.size(); base += numRegs) {
      unsigned last = base + (reverse ? 0 : numRegs - 1);
      for (auto [lane, pred, upDistance] : rounds) {
        auto incoming =
            shuffleValues(loc, values[last], lane, rewriter, upDistance);
        for (unsigned r = base; r < base + numRegs; ++r)
          values[r] =
              combineWithPrefix(op, incoming, values[r], rewriter, pred);
      }
    }
  }

  // Scan thread-segment totals, leaving native prefixes untouched.
  ScanValues scanWithinWarps(triton::ScanOp op,
                             const ScanLoweringHelper &helper,
                             const ScanValues &values, Value laneId,
                             ConversionPatternRewriter &rewriter) const {
    const auto &intraWarpTotalsLayout = *helper.getIntraWarpTotalsLayout();
    const auto &scanLayout = *helper.getIntraWarpScanLayout();
    unsigned segmentRegs = helper.getThreadSegmentSize();
    unsigned numSegments = helper.getWarpSegmentSize() / segmentRegs;
    bool reverse = op.getReverse();

    auto totals = extractSegmentTotals(values, segmentRegs, reverse);
    convertScanValues(op, totals, intraWarpTotalsLayout, scanLayout, rewriter);

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

  // Replicate the full segment sequence in each warp and scan it there.
  SmallVector<ScanCarry>
  scanAcrossWarps(triton::ScanOp op, const ScanLoweringHelper &helper,
                  const ScanValues &values, Value laneId, Value warpId,
                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kLane = StringAttr::get(ctx, "lane");
    // Use converted thread totals when available, otherwise native prefixes.
    const auto &sourceLayout = helper.getIntraWarpScanLayout()
                                   ? *helper.getIntraWarpScanLayout()
                                   : helper.getPermutedLayout();
    unsigned segmentSize = helper.getWarpSegmentSize();
    // Express the segment size in source-layout units.
    if (helper.getIntraWarpScanLayout())
      segmentSize /= helper.getThreadSegmentSize();
    const auto &interWarpTotalsLayout = *helper.getInterWarpTotalsLayout();
    const auto &totalsLayout = *helper.getInterWarpScanLayout();
    bool reverse = op.getReverse();
    auto kReg = StringAttr::get(ctx, "register");
    unsigned segmentRegs = sourceLayout.getInDimSize(kReg) /
                           interWarpTotalsLayout.getInDimSize(kReg);
    unsigned segmentLaneMask =
        getSegmentMask(sourceLayout, kLane, op.getAxis(), segmentSize);

    // The totals layout replicates each segment across its lanes. Broadcast
    // the terminal value so every replica contains the complete total.
    auto totals = extractSegmentTotals(values, segmentRegs, reverse);
    if (segmentLaneMask) {
      Value terminalLane;
      if (reverse)
        terminalLane = b.and_(laneId, b.i32_val(~segmentLaneMask));
      else
        terminalLane = b.or_(laneId, b.i32_val(segmentLaneMask));
      for (auto &total : totals)
        total = shuffleValues(loc, total, terminalLane, rewriter);
    }
    convertScanValues(op, totals, interWarpTotalsLayout, totalsLayout, rewriter,
                      /*forceWarpShuffle=*/false);

    // Reuse the warp-local scan, including sequences spanning registers.
    ScanLoweringHelper totalsHelper(totalsLayout, op.getAxis());
    assert(!totalsHelper.getInterWarpTotalsLayout() &&
           "the full totals sequence must be warp-local");
    scanWithinCTA(op, totalsHelper, totals, laneId, warpId, rewriter);
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    return getSegmentCarries(op, totals, interWarpTotalsLayout, totalsLayout,
                             interWarpTotalsLayout.getOutDimSize(axis), laneId,
                             warpId, rewriter);
  }

  // Include inter-warp carries in the thread totals before fetching the
  // preceding thread's total, so each local prefix receives one complete carry.
  void applyScanCarries(triton::ScanOp op, const ScanLoweringHelper &helper,
                        ScanValues &values, ScanValues &intraWarpTotals,
                        ArrayRef<ScanCarry> interWarpCarries, Value laneId,
                        Value warpId,
                        ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    if (!helper.getIntraWarpTotalsLayout()) {
      if (!interWarpCarries.empty())
        applySegmentCarries(op, values, interWarpCarries, rewriter);
      return;
    }

    const auto &intraWarpTotalsLayout = *helper.getIntraWarpTotalsLayout();
    unsigned segmentRegs = helper.getThreadSegmentSize();
    unsigned numSegments = helper.getWarpSegmentSize() / segmentRegs;
    const auto &scanLayout = *helper.getIntraWarpScanLayout();
    if (!interWarpCarries.empty())
      applySegmentCarries(op, intraWarpTotals, interWarpCarries, rewriter);
    convertScanValues(op, intraWarpTotals, scanLayout, intraWarpTotalsLayout,
                      rewriter);

    // Terminal prefixes are complete; only the other registers need carries.
    for (unsigned r = 0; r < intraWarpTotals.size(); ++r)
      values[r * segmentRegs + (op.getReverse() ? 0 : segmentRegs - 1)] =
          intraWarpTotals[r];
    if (segmentRegs == 1)
      return;

    auto carries = getSegmentCarries(op, intraWarpTotals, intraWarpTotalsLayout,
                                     intraWarpTotalsLayout, numSegments, laneId,
                                     warpId, rewriter);
    if (!interWarpCarries.empty()) {
      unsigned segmentsPerWarp = carries.size() / interWarpCarries.size();
      for (auto [r, carry] : llvm::enumerate(carries)) {
        const auto &interWarpCarry = interWarpCarries[r / segmentsPerWarp];
        // The first thread segment has no preceding local total. Use the
        // inter-warp carry there, and leave the scan boundary untouched.
        for (auto [value, interWarpValue] :
             llvm::zip(carry.values, interWarpCarry.values))
          value = b.select(carry.pred, value, interWarpValue);
        carry.pred = b.or_(carry.pred, interWarpCarry.pred);
      }
    }
    applySegmentCarries(op, values, carries, rewriter,
                        /*skipTerminal=*/true);
  }

  // Subtracting 2^k logical positions borrows through axis bits k and above.
  // Consecutive physical lane bits make that subtraction a constant shuffle-up.
  static std::optional<unsigned>
  getShuffleUpDistance(const LinearLayout &layout,
                       const LinearLayout &inverseLayout, unsigned axis,
                       unsigned segmentSize, int distance) {
    if (distance >= 0 || unsigned(-distance) >= segmentSize)
      return std::nullopt;
    auto *ctx = layout.getInDimNames().begin()->getContext();
    auto kLane = StringAttr::get(ctx, "lane");
    if (!layout.compose(inverseLayout).isIdentityOnOutDim(kLane))
      return std::nullopt;

    auto axisDim = StringAttr::get(ctx, "dim" + std::to_string(axis));
    const auto &bases = inverseLayout.getBases().lookup(axisDim);
    unsigned laneDim = inverseLayout.getOutDimIndex(kLane);
    unsigned first = llvm::Log2_32(-distance);
    unsigned stride = bases[first][laneDim];
    if (!llvm::isPowerOf2_32(stride))
      return std::nullopt;
    for (unsigned bit = first + 1; bit < llvm::Log2_32(segmentSize); ++bit)
      if (bases[bit][laneDim] != (stride << (bit - first)))
        return std::nullopt;
    return stride;
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
                                  ConversionPatternRewriter &rewriter,
                                  std::optional<unsigned> upDistance) const {
    assert(!candidates.empty() && "each segment has a warp-local carry");
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto carry = shuffleValues(loc, totals[candidates.front()], srcLane,
                               rewriter, upDistance);
    for (unsigned candidate : candidates.drop_front()) {
      auto incoming =
          shuffleValues(loc, totals[candidate], srcLane, rewriter, upDistance);
      Value select = b.icmp_eq(srcReg, b.i32_val(candidate));
      for (auto [value, source] : llvm::zip(carry, incoming))
        value = b.select(select, source, value);
    }
    return carry;
  }

  // Map each native segment to its exclusive carry in the totals layout.
  SmallVector<ScanCarry>
  getSegmentCarries(triton::ScanOp op, const ScanValues &totals,
                    const LinearLayout &segmentLayout,
                    const LinearLayout &totalsLayout, unsigned segmentsPerScan,
                    Value laneId, Value warpId,
                    ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    bool reverse = op.getReverse();
    // All required totals are present in this warp.
    auto inverse = totalsLayout.pseudoinvert().sublayout(
        llvm::to_vector(totalsLayout.getOutDimNames()), {kReg, kLane});
    unsigned numSegments = segmentLayout.getOutDimSize(axis);
    SmallVector<ScanCarry> carries;
    auto upDistance = getShuffleUpDistance(segmentLayout, inverse, op.getAxis(),
                                           segmentsPerScan, reverse ? 1 : -1);
    for (unsigned r = 0; r < segmentLayout.getInDimSize(kReg); ++r) {
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
      auto carry = shuffleCarry(loc, totals, candidates, srcReg, srcLane,
                                rewriter, upDistance);
      carries.push_back({std::move(carry), pred});
    }
    return carries;
  }

  void applySegmentCarries(triton::ScanOp op, ScanValues &values,
                           ArrayRef<ScanCarry> carries,
                           ConversionPatternRewriter &rewriter,
                           bool skipTerminal = false) const {
    assert(!carries.empty() && values.size() % carries.size() == 0);
    unsigned segmentRegs = values.size() / carries.size();
    unsigned begin = skipTerminal && op.getReverse() ? 1 : 0;
    unsigned end = segmentRegs - (skipTerminal && !op.getReverse());
    for (auto [r, carry] : llvm::enumerate(carries))
      for (unsigned j = begin; j < end; ++j) {
        unsigned reg = r * segmentRegs + j;
        values[reg] = combineWithPrefix(op, carry.values, values[reg], rewriter,
                                        carry.pred);
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
