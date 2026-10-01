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
  // Values are indexed by register, then by scan operand. The scan phases use
  // logical axis order; unpacking and packing use the original layout order.
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

    // Match the helper's layout: axis register bits first, ordered by logical
    // significance. This only reorders SSA values within each thread.
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
  void permuteRegisters(ScanValues &values, const ColumnAction &order) const {
    if (order.isIdentity())
      return;
    for (unsigned i = 0; i < values.front().size(); ++i) {
      SmallVector<Value> operand;
      for (const auto &row : values)
        operand.push_back(row[i]);
      operand = order.apply(operand);
      for (unsigned r = 0; r < values.size(); ++r)
        values[r][i] = operand[r];
    }
  }

  // Convert only the segment totals using warp-local shuffles. The original
  // prefixes stay in their owning threads. Identity conversions need no work.
  void convertScanValues(triton::ScanOp op, ScanValues &values,
                         const LinearLayout &src, const LinearLayout &dst,
                         ConversionPatternRewriter &rewriter) const {
    if (src == dst)
      return;
    SmallVector<SmallVector<Value>> operands(op.getNumOperands());
    for (const auto &row : values)
      for (unsigned i = 0; i < operands.size(); ++i)
        operands[i].push_back(row[i]);
    operands = convertLayoutValues(op.getLoc(), rewriter, op, src, dst,
                                   operands, getTypeConverter(), targetInfo,
                                   /*forceWarpShuffle=*/true);
    for (unsigned r = 0; r < values.size(); ++r)
      for (unsigned i = 0; i < operands.size(); ++i)
        values[r][i] = operands[i][r];
  }

  // Scan each group of numRegs consecutive logical elements within a thread,
  // retaining every prefix. The caller ensures registers are contiguous.
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

  // Scan normalized totals across lanes with shuffle-up rounds, then add each
  // lane's exclusive carry to the register prefixes of those totals.
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
    SmallVector<unsigned> laneBits;
    unsigned laneMask = 0;
    for (auto [bit, basis] : llvm::enumerate(layout.getBases().lookup(kLane)))
      if (basis[axis] && basis[axis] < segmentSize) {
        laneBits.push_back(bit);
        laneMask |= 1u << bit;
      }

    // The scan lane bits are ordered, but may be separated by bits for
    // independent scans or other segments. Shift in logical lane order.
    Value laneIndex = b.i32_val(0);
    bool contiguous = true;
    for (auto [i, bit] : llvm::enumerate(laneBits)) {
      Value digit = b.and_(b.lshr(laneId, b.i32_val(bit)), b.i32_val(1));
      laneIndex = b.or_(laneIndex, b.shl(digit, b.i32_val(i)));
      contiguous &= bit == laneBits.front() + i;
    }
    auto shuffle = [&](SmallVector<Value> input, unsigned offset) {
      if (contiguous && !reverse) {
        for (Value &value : input)
          value = targetInfo.shuffleUp(rewriter, loc, value,
                                       offset << laneBits.front());
      } else {
        // Boundary sources wrap; the combine predicate excludes them.
        Value index = b.add(laneIndex, b.i32_val(reverse ? offset : -offset));
        Value lane = b.and_(laneId, b.i32_val(~laneMask));
        for (auto [i, bit] : llvm::enumerate(laneBits)) {
          Value digit = b.and_(b.lshr(index, b.i32_val(i)), b.i32_val(1));
          lane = b.or_(lane, b.shl(digit, b.i32_val(bit)));
        }
        for (Value &value : input)
          value = targetInfo.shuffleIdx(rewriter, loc, value, lane);
      }
      return input;
    };
    auto hasPrefix = [&](unsigned offset) {
      return reverse ? b.icmp_ult(laneIndex, b.i32_val(numLanes - offset))
                     : b.icmp_uge(laneIndex, b.i32_val(offset));
    };
    for (unsigned base = 0; base < values.size(); base += numRegs) {
      unsigned last = base + (reverse ? 0 : numRegs - 1);
      auto acc = values[last];
      for (unsigned offset = 1; offset < numLanes; offset *= 2)
        acc = combineWithPrefix(op, shuffle(acc, offset), acc, rewriter,
                                hasPrefix(offset));
      values[last] = acc;
      if (numRegs == 1)
        continue;
      // One more shuffle gives the exclusive carry for the local prefixes.
      // Skip the first lane without assuming an identity value.
      auto prefix = shuffle(acc, 1);
      Value pred = hasPrefix(1);
      for (unsigned r = base; r < base + numRegs; ++r)
        if (r != last)
          values[r] = combineWithPrefix(op, prefix, values[r], rewriter, pred);
    }
  }

  ScanValues scanWithinWarps(triton::ScanOp op,
                             const ScanLoweringHelper &helper,
                             const ScanValues &values, Value laneId,
                             ConversionPatternRewriter &rewriter) const {
    const auto &intraWarpLayout = *helper.getIntraWarpLayout();
    const auto &scanLayout = *helper.getIntraWarpScanLayout();
    unsigned segmentRegs = helper.getThreadLocalSegmentSize();
    unsigned numSegments = helper.getWarpLocalSegmentSize() / segmentRegs;
    bool reverse = op.getReverse();

    // Keep the thread-local prefixes in place and extract only their totals.
    ScanValues totals;
    for (unsigned base = 0; base < values.size(); base += segmentRegs)
      totals.push_back(values[base + (reverse ? 0 : segmentRegs - 1)]);
    convertScanValues(op, totals, intraWarpLayout, scanLayout, rewriter);

    // After conversion, consecutive totals can share a register group. Scan
    // those groups first, then combine across lanes in logical order.
    auto kReg = StringAttr::get(op.getContext(), "register");
    unsigned numRegs = 1;
    for (const auto &basis : scanLayout.getBases().lookup(kReg))
      if (basis[op.getAxis()] && basis[op.getAxis()] < numSegments)
        numRegs *= 2;
    scanWithinThreads(op, totals, numRegs, rewriter);
    scanLaneTotals(op, totals, scanLayout, numRegs, numSegments, laneId,
                   rewriter);
    // Keep totals in the scan layout: the inter-warp stage can extract its
    // terminal values directly, before mapping any carries to local prefixes.
    return totals;
  }

  ScanValues scanAcrossWarps(triton::ScanOp op,
                             const ScanLoweringHelper &helper,
                             const ScanValues &values, Value laneId,
                             Value warpId,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    // If a lane scan was needed, values are still the converted thread totals.
    // Otherwise they are the native thread-local prefixes.
    const auto &sourceLayout = helper.getIntraWarpScanLayout()
                                   ? *helper.getIntraWarpScanLayout()
                                   : helper.getPermutedLayout();
    unsigned segmentSize = helper.getWarpLocalSegmentSize();
    if (helper.getIntraWarpScanLayout())
      segmentSize /= helper.getThreadLocalSegmentSize();
    const auto &interWarpLayout = *helper.getInterWarpLayout();
    const auto &totalsLayout = *helper.getInterWarpScanLayout();
    bool reverse = op.getReverse();
    unsigned segmentRegs = 1;
    for (const auto &basis : sourceLayout.getBases().lookup(kReg))
      if (basis[op.getAxis()] && basis[op.getAxis()] < segmentSize)
        segmentRegs *= 2;
    unsigned segmentLaneMask = 0;
    for (auto [i, basis] :
         llvm::enumerate(sourceLayout.getBases().lookup(kLane)))
      if (basis[op.getAxis()] && basis[op.getAxis()] < segmentSize)
        segmentLaneMask |= 1u << i;

    // Extract the terminal register and broadcast the terminal lane. The
    // resulting values have exactly the collapsed segment layout.
    Value terminalLane = b.or_(b.and_(laneId, b.i32_val(~segmentLaneMask)),
                               b.i32_val(reverse ? 0 : segmentLaneMask));
    SmallVector<SmallVector<Value>> operands(op.getNumOperands());
    for (unsigned base = 0; base < values.size(); base += segmentRegs) {
      unsigned last = base + (reverse ? 0 : segmentRegs - 1);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        Value total = values[last][i];
        if (segmentLaneMask)
          total = targetInfo.shuffleIdx(rewriter, loc, total, terminalLane);
        operands[i].push_back(total);
      }
    }
    operands =
        convertLayoutValues(loc, rewriter, op, interWarpLayout, totalsLayout,
                            operands, getTypeConverter(), targetInfo);
    ScanValues totals(operands.front().size());
    for (unsigned r = 0; r < totals.size(); ++r)
      for (const auto &operand : operands)
        totals[r].push_back(operand[r]);

    // Reuse the same ordered scan, including register groups when the complete
    // sequence is longer than the available lanes.
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

  // Restore native ownership only after both scans have consumed their totals.
  // Apply the warp-local carry first, then prepend the inter-warp carry,
  // keeping the combiner's logical operand order without assuming an identity
  // value.
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

      // The scanned totals are already the terminal prefixes. Only the other
      // registers need an exclusive carry; one-element segments are now done.
      for (unsigned r = 0; r < intraWarpTotals.size(); ++r)
        values[r * segmentRegs + (op.getReverse() ? 0 : segmentRegs - 1)] =
            intraWarpTotals[r];
      if (segmentRegs > 1)
        applySegmentCarries(op, values, intraWarpTotals, intraWarpLayout,
                            intraWarpLayout, segmentRegs, numSegments, laneId,
                            warpId, rewriter, /*skipTerminal=*/true);
    }
    if (helper.getInterWarpLayout()) {
      auto kReg = StringAttr::get(op.getContext(), "register");
      unsigned segmentRegs = 1;
      for (const auto &basis :
           helper.getPermutedLayout().getBases().lookup(kReg))
        if (basis[op.getAxis()] &&
            basis[op.getAxis()] < helper.getWarpLocalSegmentSize())
          segmentRegs *= 2;
      const auto &interWarpLayout = *helper.getInterWarpLayout();
      auto axis =
          *std::next(interWarpLayout.getOutDimNames().begin(), op.getAxis());
      applySegmentCarries(op, values, interWarpTotals, interWarpLayout,
                          *helper.getInterWarpScanLayout(), segmentRegs,
                          interWarpLayout.getOutDimSize(axis), laneId, warpId,
                          rewriter);
    }
  }

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
    // Read the preceding segment's inclusive total as an exclusive carry.
    // All required totals are present in this warp, so only register selects
    // and lane shuffles are needed, even when the sequence spans registers.
    auto inverse = totalsLayout.pseudoinvert();
    unsigned numSegments = segmentLayout.getOutDimSize(axis);
    for (unsigned r = 0; r < values.size() / segmentRegs; ++r) {
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
      // Wrap within this scan's sequence, so an excluded boundary source
      // cannot introduce register candidates from another warp-local segment.
      // pred guards the combine without assuming an identity value.
      Value preceding = b.and_(b.add(segment, b.i32_val(reverse ? 1 : -1)),
                               b.i32_val(segmentsPerScan - 1));
      if (segmentsPerScan != numSegments)
        preceding = b.or_(b.and_(segment, b.i32_val(~(segmentsPerScan - 1))),
                          preceding);
      coords[op.getAxis()].second = preceding;
      auto src = applyLinearLayout(loc, rewriter, inverse, coords);
      Value srcReg = src[0].second;
      Value srcLane = src[1].second;
      // Enumerate only the registers reachable by this segment's threads.
      // This also handles the borrow/carry at a register sequence boundary.
      llvm::SmallSetVector<unsigned, 8> candidates;
      for (unsigned warp = 0; warp < segmentLayout.getInDimSize(kWarp);
           ++warp) {
        for (unsigned lane = 0; lane < segmentLayout.getInDimSize(kLane);
             ++lane) {
          auto coordinates = segmentLayout.apply(
              {{kReg, r}, {kLane, lane}, {kWarp, warp}, {kBlock, 0}});
          auto &index = coordinates[op.getAxis()].second;
          index = (index & ~(segmentsPerScan - 1)) |
                  ((index + (reverse ? 1 : -1)) & (segmentsPerScan - 1));
          candidates.insert(inverse.apply(coordinates)[0].second);
        }
      }
      SmallVector<Value> carry;
      for (unsigned candidate : candidates) {
        auto incoming = totals[candidate];
        for (Value &value : incoming)
          value = targetInfo.shuffleIdx(rewriter, loc, value, srcLane);
        if (carry.empty()) {
          carry = std::move(incoming);
        } else {
          Value select = b.icmp_eq(srcReg, b.i32_val(candidate));
          for (unsigned i = 0; i < carry.size(); ++i)
            carry[i] = b.select(select, incoming[i], carry[i]);
        }
      }
      assert(!carry.empty() && "each segment has a warp-local carry");
      for (unsigned j = 0; j < segmentRegs; ++j) {
        if (skipTerminal && j == (reverse ? 0 : segmentRegs - 1))
          continue;
        unsigned reg = r * segmentRegs + j;
        values[reg] = combineWithPrefix(op, carry, values[reg], rewriter, pred);
      }
    }
  }

  // Keep the existing prefix where the carry does not apply. The predicate
  // also guards the combine region, which may contain side effects.
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
