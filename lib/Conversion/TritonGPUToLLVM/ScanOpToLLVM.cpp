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
    if (helper.getIntraWarpLayout())
      convertScanValues(op, values, helper.getPermutedLayout(),
                        *helper.getIntraWarpLayout(), rewriter);
    scanWithinThreads(op, values, helper.getThreadLocalSegmentSize(), rewriter);
    if (helper.getIntraWarpLayout())
      scanWithinWarps(op, helper, values, laneId, rewriter);
    if (helper.getInterWarpLayout())
      scanAcrossWarps(op, helper, values, laneId, warpId, rewriter);
    if (helper.getIntraWarpLayout())
      convertScanValues(op, values, *helper.getIntraWarpLayout(),
                        helper.getPermutedLayout(), rewriter);
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

  // Convert values using warp-local shuffles. Keep the scan layout through
  // the inter-warp stage. Identity conversions need no work.
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

  // Scan thread totals across lanes in logical order, then add each lane's
  // exclusive carry to the prefixes computed by scanWithinThreads.
  void scanWithinWarps(triton::ScanOp op, const ScanLoweringHelper &helper,
                       ScanValues &values, Value laneId,
                       ConversionPatternRewriter &rewriter) const {
    const auto &layout = *helper.getIntraWarpLayout();
    unsigned numRegs = helper.getThreadLocalSegmentSize();
    unsigned segmentSize = helper.getWarpLocalSegmentSize();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kLane = StringAttr::get(op.getContext(), "lane");
    unsigned axis = op.getAxis();
    bool reverse = op.getReverse();
    unsigned numLanes = segmentSize / numRegs;
    auto dims = llvm::to_vector(layout.getOutDimNames());
    auto laneLayout = layout.sublayout({kLane}, dims);
    auto inverseLaneLayout = layout.pseudoinvert().sublayout(dims, {kLane});

    // Example: lane=[1,0,2] maps physical lanes 0,1,4,5 to logical positions
    // 0,1,2,3. Shift the logical coordinate, then use the inverse layout to
    // find its owning lane. A pseudoinverse picks a valid copy for broadcasts.
    // Register and warp coordinates stay fixed, so only the lane contribution
    // is needed. Preserve all coordinates belonging to independent scans.
    auto coords =
        applyLinearLayout(loc, rewriter, laneLayout, {{kLane, laneId}});
    Value index = coords[axis].second;
    Value segmentIndex = b.and_(index, b.i32_val(segmentSize - 1));
    auto shuffle = [&](SmallVector<Value> input, unsigned offset) {
      // Each lane owns numRegs consecutive elements. Shift by whole groups,
      // wrapping within this segment; the combine predicate excludes
      // wraparound.
      int distance = offset * numRegs;
      Value shifted =
          b.and_(b.add(segmentIndex, b.i32_val(reverse ? distance : -distance)),
                 b.i32_val(segmentSize - 1));
      auto source = coords;
      source[axis].second =
          b.or_(b.and_(index, b.i32_val(~(segmentSize - 1))), shifted);
      Value lane = applyLinearLayout(loc, rewriter, inverseLaneLayout, source)
                       .front()
                       .second;
      for (Value &value : input)
        value = targetInfo.shuffleIdx(rewriter, loc, value, lane);
      return input;
    };
    auto hasPrefix = [&](unsigned offset) {
      unsigned distance = offset * numRegs;
      return reverse
                 ? b.icmp_ult(segmentIndex, b.i32_val(segmentSize - distance))
                 : b.icmp_uge(segmentIndex, b.i32_val(distance));
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

  void scanAcrossWarps(triton::ScanOp op, const ScanLoweringHelper &helper,
                       ScanValues &values, Value laneId, Value warpId,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kLane = StringAttr::get(ctx, "lane");
    // Extract totals directly from the current scan layout. Full prefixes
    // stay in this layout until the inter-warp carries have been applied.
    const auto &sourceLayout = helper.getIntraWarpLayout()
                                   ? *helper.getIntraWarpLayout()
                                   : helper.getPermutedLayout();
    unsigned segmentSize = helper.getWarpLocalSegmentSize();
    const auto &interWarpLayout = *helper.getInterWarpLayout();
    const auto &totalsLayout = *helper.getInterWarpScanLayout();
    bool reverse = op.getReverse();
    unsigned segmentRegs = helper.getThreadLocalSegmentSize();
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
    if (totalsHelper.getIntraWarpLayout())
      convertScanValues(op, totals, totalsHelper.getPermutedLayout(),
                        *totalsHelper.getIntraWarpLayout(), rewriter);
    scanWithinThreads(op, totals, totalsHelper.getThreadLocalSegmentSize(),
                      rewriter);
    if (totalsHelper.getIntraWarpLayout()) {
      scanWithinWarps(op, totalsHelper, totals, laneId, rewriter);
      convertScanValues(op, totals, *totalsHelper.getIntraWarpLayout(),
                        totalsHelper.getPermutedLayout(), rewriter);
    }
    permuteRegisters(totals, totalsHelper.getRegisterOrder().inverse());

    applySegmentCarries(op, values, totals, interWarpLayout, totalsLayout,
                        segmentRegs, laneId, warpId, rewriter);
  }

  void applySegmentCarries(triton::ScanOp op, ScanValues &values,
                           const ScanValues &totals,
                           const LinearLayout &segmentLayout,
                           const LinearLayout &totalsLayout,
                           unsigned segmentRegs, Value laneId, Value warpId,
                           ConversionPatternRewriter &rewriter) const {
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
      Value pred = b.icmp_ne(segment, b.i32_val(reverse ? numSegments - 1 : 0));
      // Wrap the boundary source into the sequence. pred guards the combine
      // without assuming an identity value.
      Value preceding = b.and_(b.add(segment, b.i32_val(reverse ? 1 : -1)),
                               b.i32_val(numSegments - 1));
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
          index = (index + (reverse ? 1 : -1)) & (numSegments - 1);
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
