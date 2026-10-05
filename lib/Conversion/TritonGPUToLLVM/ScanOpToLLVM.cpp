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

    scanLinearLayout(op, helper, values, laneId, warpId, rewriter);

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
  void scanLinearLayout(triton::ScanOp op, const ScanLoweringHelper &helper,
                        ScanValues &values, Value laneId, Value warpId,
                        ConversionPatternRewriter &rewriter) const {
    // Keep native ownership and communicate only segment totals.
    permuteRegisters(values, helper.getRegisterOrder());
    if (!scanWarpChunks(op, helper, values, laneId, rewriter)) {
      // Consume carries as soon as their raw totals have been accumulated,
      // as in main. A layout conversion defers consumption until all totals
      // are available; unchanged ownership can finish each native chunk here.
      unsigned lanes =
          helper.getWarpLocalSegmentSize() / helper.getThreadLocalSegmentSize();
      bool consumeImmediately =
          helper.getInterWarpLayout() &&
          (!helper.getIntraWarpLayout() ||
           (*helper.getIntraWarpLayout() == *helper.getIntraWarpScanLayout() &&
            getScanLaneStride(*helper.getIntraWarpLayout(), op.getAxis(),
                              lanes)));
      auto order = getNativeRegisterOrder(
          helper, values.size() / helper.getThreadLocalSegmentSize());
      bool deferFirst = consumeImmediately && op.getReverse();
      scanWithinThreads(op, values, helper.getThreadLocalSegmentSize(),
                        rewriter, order, deferFirst);
      ScanValues intraWarpTotals;
      SmallVector<ScanCarry> interWarpCarries;
      if (helper.getIntraWarpLayout())
        intraWarpTotals = scanWithinWarps(op, helper, values, laneId, rewriter);
      auto consume = [&](unsigned reg, const ScanCarry &carry) {
        applyWarpCarry(op, helper, values, intraWarpTotals, reg, carry, laneId,
                       rewriter, deferFirst);
      };
      llvm::function_ref<void(unsigned, const ScanCarry &)> consumer;
      if (consumeImmediately)
        consumer = consume;
      if (helper.getInterWarpLayout())
        interWarpCarries = scanAcrossWarps(
            op, helper, intraWarpTotals.empty() ? values : intraWarpTotals,
            laneId, warpId, rewriter, consumer);
      if (!consumeImmediately)
        applyScanCarries(op, helper, values, intraWarpTotals, interWarpCarries,
                         laneId, warpId, rewriter);
    }
    permuteRegisters(values, helper.getRegisterOrder().inverse());
  }

  // Reverse logical traversal without moving values between their owners.
  static unsigned getScanIndex(triton::ScanOp op, unsigned index,
                               unsigned size) {
    return op.getReverse() ? size - 1 - index : index;
  }

  static unsigned getSegmentMask(const LinearLayout &layout, StringAttr dim,
                                 unsigned axis, unsigned segmentSize) {
    unsigned mask = 0;
    for (auto [bit, basis] : llvm::enumerate(layout.getBases().lookup(dim)))
      if (basis[axis] && basis[axis] < segmentSize)
        mask |= 1u << bit;
    return mask;
  }

  static Value getTerminalLane(TritonLLVMOpBuilder &b, Value laneId,
                               unsigned laneMask, bool reverse) {
    return reverse ? Value(b.and_(laneId, b.i32_val(~laneMask)))
                   : Value(b.or_(laneId, b.i32_val(laneMask)));
  }

  static ScanValues transposeValues(const ScanValues &values) {
    ScanValues result(values.front().size());
    for (const auto &row : values)
      for (auto [i, value] : llvm::enumerate(row))
        result[i].push_back(value);
    return result;
  }

  static ScanValues extractSegmentTotals(triton::ScanOp op,
                                         const ScanValues &values,
                                         unsigned numRegs) {
    ScanValues totals;
    for (unsigned base = 0; base < values.size(); base += numRegs)
      totals.push_back(values[base + getScanIndex(op, numRegs - 1, numRegs)]);
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

  // Renumbering registers must not reorder independent work unnecessarily:
  // retain native order to keep the same register lifetimes as main.
  static SmallVector<unsigned>
  getNativeRegisterOrder(const ScanLoweringHelper &helper, unsigned count) {
    auto kReg = StringAttr::get(
        helper.getPermutedLayout().getInDimNames().begin()->getContext(),
        "register");
    unsigned size = helper.getPermutedLayout().getInDimSize(kReg);
    auto ranks = helper.getRegisterOrder().apply(
        LinearLayout::identity1D(size, kReg, kReg));
    auto order = llvm::to_vector(llvm::seq<unsigned>(count));
    llvm::sort(order, [&](unsigned a, unsigned b) {
      return ranks.apply({{kReg, a * (size / count)}}).front().second <
             ranks.apply({{kReg, b * (size / count)}}).front().second;
    });
    return order;
  }

  static void preserveLogicalOrder(SmallVectorImpl<unsigned> &order,
                                   unsigned segmentsPerScan) {
    SmallVector<int> previous(order.size() / segmentsPerScan, -1);
    for (unsigned reg : order) {
      int &prev = previous[reg / segmentsPerScan];
      if (int(reg % segmentsPerScan) <= prev) {
        llvm::sort(order);
        return;
      }
      prev = reg % segmentsPerScan;
    }
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

  // Scan contiguous register groups in traversal order. When deferring the
  // first value, save (x0, x1, x1*x2, total) instead of full local prefixes:
  // x0 absorbs the carry once, and subsequent combines need no boundary guard.
  void scanWithinThreads(triton::ScanOp op, ScanValues &values,
                         unsigned numRegs, ConversionPatternRewriter &rewriter,
                         ArrayRef<unsigned> order = {},
                         bool deferFirst = false) const {
    auto loc = op.getLoc();
    for (unsigned group = 0; group < values.size() / numRegs; ++group) {
      unsigned base = (order.empty() ? group : order[group]) * numRegs;
      auto at = [&](unsigned i) -> SmallVector<Value> & {
        return values[base + getScanIndex(op, i, numRegs)];
      };
      for (unsigned i = deferFirst ? 2 : 1; i < numRegs; ++i)
        at(i) =
            applyCombineOp(loc, rewriter, op.getCombineOp(), at(i - 1), at(i));
      if (deferFirst && numRegs > 1)
        at(numRegs - 1) = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                         at(0), at(numRegs - 1));
    }
  }

  // A regular group of lane bases admits main's shuffle-up instruction.
  // For lane=[dim0:1, dim0:2, dim1:1, ...], the stride is one. Interleaved
  // scan bits use the inverse layout to find their logical predecessor.
  static unsigned getScanLaneStride(const LinearLayout &layout, unsigned axis,
                                    unsigned numLanes) {
    auto kLane =
        StringAttr::get(layout.getInDimNames().begin()->getContext(), "lane");
    const auto &bases = layout.getBases().lookup(kLane);
    unsigned first = bases.size();
    for (unsigned value = 1, i = 0; value < numLanes; value *= 2, ++i) {
      auto it = llvm::find_if(bases, [&](const auto &basis) {
        return basis[axis] == value &&
               llvm::count_if(basis, [](int32_t x) { return x != 0; }) == 1;
      });
      if (it == bases.end())
        return 0;
      unsigned bit = std::distance(bases.begin(), it);
      if (!i)
        first = bit;
      if (bit != first + i)
        return 0;
    }
    return first == bases.size() ? 0 : 1u << first;
  }

  // Main's logarithmic shuffle scan, applied independently to each chunk.
  void scanLaneTotals(triton::ScanOp op, ScanValues &values,
                      const LinearLayout &layout, unsigned numLanes,
                      Value laneId, ConversionPatternRewriter &rewriter,
                      ArrayRef<unsigned> order) const {
    if (numLanes == 1)
      return;
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kLane = StringAttr::get(op.getContext(), "lane");
    auto dims = llvm::to_vector(layout.getOutDimNames());
    unsigned stride = getScanLaneStride(layout, op.getAxis(), numLanes);
    auto coords = applyLinearLayout(
        loc, rewriter, layout.sublayout({kLane}, dims), {{kLane, laneId}});
    Value index = coords[op.getAxis()].second;
    unsigned axisSize = layout.getOutDimSize(dims[op.getAxis()]);
    Value laneIndex = axisSize == numLanes
                          ? index
                          : Value(b.and_(index, b.i32_val(numLanes - 1)));
    auto inverse = layout.pseudoinvert().sublayout(dims, {kLane});
    struct Round {
      unsigned offset;
      Value source, pred;
    };
    SmallVector<Round> rounds;
    for (unsigned offset = 1; offset < numLanes; offset *= 2) {
      Value source;
      if (!stride) {
        auto previous = coords;
        previous[op.getAxis()].second = shiftWithinSegment(
            b, index, op.getReverse() ? int(offset) : -int(offset), numLanes,
            axisSize);
        source =
            applyLinearLayout(loc, rewriter, inverse, previous).front().second;
      }
      rounds.push_back({offset, source, {}});
    }
    // Expose independent indexed shuffles in bounded batches. Keep the native
    // shuffle-up emission order for forward scans.
    unsigned batchSize = op.getReverse() ? 8 : 1;
    for (unsigned base = 0; base < order.size(); base += batchSize) {
      auto batch =
          order.slice(base, std::min<unsigned>(batchSize, order.size() - base));
      for (auto &[offset, source, pred] : rounds) {
        // In the last reverse round, every participating lane has this bit
        // clear. Its higher neighbor differs by exactly that physical lane bit.
        bool butterfly = op.getReverse() && stride && offset * 2 == numLanes;
        if (!source && stride && op.getReverse() && !butterfly)
          source = b.add(laneId, b.i32_val(offset * stride));
        for (unsigned reg : batch) {
          auto &acc = values[reg];
          auto incoming = acc;
          for (auto &value : incoming) {
            if (butterfly)
              value =
                  targetInfo.shuffleXor(rewriter, loc, value, offset * stride);
            else if (source)
              value = targetInfo.shuffleIdx(rewriter, loc, value, source);
            else
              value =
                  targetInfo.shuffleUp(rewriter, loc, value, offset * stride);
          }
          if (!pred)
            pred = op.getReverse()
                       ? b.icmp_ult(laneIndex, b.i32_val(numLanes - offset))
                       : b.icmp_uge(laneIndex, b.i32_val(offset));
          acc = combineWithPrefix(op, incoming, acc, rewriter, pred);
        }
      }
    }
  }

  // Main scans all lane totals first, then carries each completed chunk into
  // the next one. Keep this order when totals already have native ownership.
  bool scanWarpChunks(triton::ScanOp op, const ScanLoweringHelper &helper,
                      ScanValues &values, Value laneId,
                      ConversionPatternRewriter &rewriter) const {
    if (!helper.getIntraWarpLayout() || helper.getInterWarpLayout())
      return false;
    const auto &src = *helper.getIntraWarpLayout();
    const auto &dst = *helper.getIntraWarpScanLayout();
    if (src != dst)
      return false;
    auto *ctx = op.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    unsigned numSegments = src.getOutDimSize(axis);
    unsigned laneMask = getSegmentMask(dst, kLane, op.getAxis(), numSegments);
    unsigned chunkSize = 1u << llvm::popcount(laneMask);
    unsigned numChunks = numSegments / chunkSize;
    if (numChunks == 1 || !getScanLaneStride(dst, op.getAxis(), chunkSize))
      return false;

    unsigned segmentRegs = helper.getThreadLocalSegmentSize();
    auto order = getNativeRegisterOrder(helper, values.size() / segmentRegs);
    scanWithinThreads(op, values, segmentRegs, rewriter, order);
    auto totals = extractSegmentTotals(op, values, segmentRegs);
    scanLaneTotals(op, totals, dst, chunkSize, laneId, rewriter, order);

    auto chunkLayout = helper.getPermutedLayout()
                           .resizeOutDim(axis, chunkSize * segmentRegs)
                           .removeZeroBasesAlongDim(kReg);
    ScanLoweringHelper chunkHelper(chunkLayout, op.getAxis());
    unsigned nativeLaneMask = getSegmentMask(chunkLayout, kLane, op.getAxis(),
                                             chunkSize * segmentRegs);
    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    Value terminalLane =
        getTerminalLane(b, laneId, nativeLaneMask, op.getReverse());
    ScanValues carries(totals.size() / numChunks);
    preserveLogicalOrder(order, numChunks);
    for (unsigned index : order) {
      unsigned reg = index / numChunks * numChunks +
                     getScanIndex(op, index % numChunks, numChunks);
      auto &prefix = carries[reg / numChunks];
      ScanCarry carry{prefix.empty() ? totals[reg] : prefix,
                      prefix.empty() ? Value(b.false_val())
                                     : Value(b.true_val())};
      applyWarpCarry(op, chunkHelper, values, totals, reg, carry, laneId,
                     rewriter);
      if (index % numChunks + 1 < numChunks) {
        unsigned terminalReg =
            reg * segmentRegs + getScanIndex(op, segmentRegs - 1, segmentRegs);
        prefix = shuffleValues(op.getLoc(), values[terminalReg], terminalLane,
                               rewriter);
      }
    }
    return true;
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

    auto totals = extractSegmentTotals(op, values, segmentRegs);
    convertScanValues(op, totals, intraWarpLayout, scanLayout, rewriter);

    auto kReg = StringAttr::get(op.getContext(), "register");
    auto axis =
        StringAttr::get(op.getContext(), "dim" + std::to_string(op.getAxis()));
    unsigned numRegs =
        scanLayout.sublayout({kReg}, {axis}).getNumConsecutiveInOut();
    auto kLane = StringAttr::get(op.getContext(), "lane");
    unsigned laneMask =
        getSegmentMask(scanLayout, kLane, op.getAxis(), numSegments);
    unsigned chunkSize = numRegs * (1u << llvm::popcount(laneMask));
    assert(numRegs == 1 && "warp scan uses one total per lane and chunk");
    auto order = getNativeRegisterOrder(helper, totals.size());
    scanLaneTotals(op, totals, scanLayout, chunkSize, laneId, rewriter, order);
    unsigned numChunks = numSegments / chunkSize;
    if (numChunks > 1)
      scanChunkTotals(op, totals, numRegs, numChunks, laneMask, laneId,
                      rewriter);
    // Keep this layout for extracting inter-warp totals.
    return totals;
  }

  // Broadcast chunk totals, scan them, and apply the preceding chunk's carry.
  void scanChunkTotals(triton::ScanOp op, ScanValues &totals, unsigned numRegs,
                       unsigned numChunks, unsigned laneMask, Value laneId,
                       ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    auto chunkTotals = extractSegmentTotals(op, totals, numRegs);
    Value terminal = getTerminalLane(b, laneId, laneMask, op.getReverse());
    for (auto &total : chunkTotals)
      total = shuffleValues(op.getLoc(), total, terminal, rewriter);
    scanWithinThreads(op, chunkTotals, numChunks, rewriter);
    // The first chunk has no carry. Every other chunk uses its predecessor's
    // complete total, in traversal order, without requiring an identity.
    for (unsigned base = 0; base < chunkTotals.size(); base += numChunks)
      for (unsigned i = 1; i < numChunks; ++i) {
        unsigned chunk = base + getScanIndex(op, i, numChunks);
        unsigned prev = base + getScanIndex(op, i - 1, numChunks);
        for (unsigned r = chunk * numRegs; r < (chunk + 1) * numRegs; ++r)
          totals[r] =
              combineWithPrefix(op, chunkTotals[prev], totals[r], rewriter, {});
      }
  }

  // Publish raw warp totals once, then accumulate earlier totals directly
  // in each consumer, preserving main's single shared-memory exchange.
  SmallVector<ScanCarry> scanAcrossWarps(
      triton::ScanOp op, const ScanLoweringHelper &helper,
      const ScanValues &values, Value laneId, Value warpId,
      ConversionPatternRewriter &rewriter,
      llvm::function_ref<void(unsigned, const ScanCarry &)> consume) const {
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
    auto kReg = StringAttr::get(ctx, "register");
    unsigned segmentRegs =
        sourceLayout.getInDimSize(kReg) / interWarpLayout.getInDimSize(kReg);
    unsigned segmentLaneMask =
        getSegmentMask(sourceLayout, kLane, op.getAxis(), segmentSize);

    // Only terminal lanes store complete segment totals; conversion loads
    // distribute them to the destination layout.
    auto totals = extractSegmentTotals(op, values, segmentRegs);
    Value storePred;
    if (segmentLaneMask)
      storePred = b.icmp_eq(b.and_(laneId, b.i32_val(segmentLaneMask)),
                            b.i32_val(op.getReverse() ? 0 : segmentLaneMask));
    auto free = sourceLayout.getFreeVariableMasks();
    for (auto [dim, id] : SmallVector<std::pair<StringAttr, Value>>{
             {kLane, laneId}, {StringAttr::get(ctx, "warp"), warpId}})
      if (free[dim]) {
        Value representative =
            b.icmp_eq(b.and_(id, b.i32_val(free[dim])), b.i32_val(0));
        storePred =
            storePred ? b.and_(storePred, representative) : representative;
      }
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    unsigned numSegments = interWarpLayout.getOutDimSize(axis);
    unsigned numRegs = interWarpLayout.sublayout({kReg}, {axis})
                           .removeZeroBasesAlongDim(kReg)
                           .getInDimSize(kReg);
    auto inverseRegs = interWarpLayout.pseudoinvert().sublayout(
        llvm::to_vector(interWarpLayout.getOutDimNames()), {kReg});
    Value threadSegment;
    unsigned registerMask = 0;
    for (const auto &basis : interWarpLayout.getBases().lookup(kReg))
      registerMask |= basis[op.getAxis()];
    SmallVector<ScanCarry> carries(interWarpLayout.getInDimSize(kReg));
    // Materialize consumer predicates after the shared-memory publication.
    auto initializeCarries = [&] {
      threadSegment = applyLinearLayout(loc, rewriter, interWarpLayout,
                                        {{kReg, b.i32_val(0)},
                                         {kLane, laneId},
                                         {kWarp, warpId},
                                         {kBlock, b.i32_val(0)}})[op.getAxis()]
                          .second;
      for (unsigned r = 0; r < carries.size(); ++r) {
        unsigned regSegment = interWarpLayout
                                  .apply({{kReg, r},
                                          {kLane, 0},
                                          {kWarp, 0},
                                          {kBlock, 0}})[op.getAxis()]
                                  .second;
        unsigned firstReg = op.getReverse() ? registerMask : 0;
        unsigned firstThread =
            op.getReverse() ? (numSegments - 1) & ~registerMask : 0;
        carries[r].pred =
            regSegment != firstReg
                ? Value(b.true_val())
                : b.icmp_ne(threadSegment, b.i32_val(firstThread));
      }
    };
    auto nativeOrder = getNativeRegisterOrder(helper, carries.size());
    SmallVector<unsigned> ranks(carries.size());
    for (auto [rank, reg] : llvm::enumerate(nativeOrder))
      ranks[reg] = rank;
    auto loadOrder =
        llvm::to_vector(llvm::seq<unsigned>(totalsLayout.getInDimSize(kReg)));
    auto sourceRegister = [&](unsigned reg) {
      return inverseRegs
          .apply(totalsLayout.apply(
              {{kReg, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}}))
          .front()
          .second;
    };
    llvm::stable_sort(loadOrder, [&](unsigned a, unsigned b) {
      return ranks[sourceRegister(a)] < ranks[sourceRegister(b)];
    });
    // Native register bits may themselves be reordered. Never let emission
    // order violate the logical dependencies of any one scan.
    preserveLogicalOrder(loadOrder, numSegments);
    for (unsigned &reg : loadOrder)
      reg = reg / numSegments * numSegments +
            getScanIndex(op, reg % numSegments, numSegments);
    ScanValues accumulators(loadOrder.size() / numSegments);
    visitScanTotals(
        loc, rewriter, op, interWarpLayout, totalsLayout,
        transposeValues(totals), getTypeConverter(), targetInfo, storePred,
        laneId, warpId, nativeOrder, loadOrder, initializeCarries,
        [&](unsigned reg, ValueRange rawTotal) {
          unsigned segment = reg % numSegments;
          auto &acc = accumulators[reg / numSegments];
          auto coords = llvm::to_vector(llvm::map_range(
              interWarpLayout.getOutDimNames(),
              [](StringAttr dim) { return std::make_pair(dim, int32_t(0)); }));
          auto ownerOf = [&](unsigned index) {
            coords[op.getAxis()].second = index;
            return inverseRegs.apply(coords).front().second +
                   (reg / numSegments) * numRegs;
          };
          unsigned index = getScanIndex(op, segment, numSegments);
          unsigned next = op.getReverse() ? segment - 1 : segment + 1;
          auto *carry = index + 1 < numSegments ? &carries[ownerOf(next)].values
                                                : nullptr;
          Value before;
          if (carry && !carry->empty()) {
            unsigned threshold = next & ~registerMask;
            before = op.getReverse()
                         ? b.icmp_ule(threadSegment, b.i32_val(threshold))
                         : b.icmp_uge(threadSegment, b.i32_val(threshold));
          }
          acc = applyCombineOp(loc, rewriter, op.getCombineOp(), acc, rawTotal);
          if (carry) {
            if (carry->empty()) {
              *carry = acc;
            } else {
              for (auto [value, prefix] : llvm::zip(*carry, acc))
                value = b.select(before, prefix, value);
            }
          }
          unsigned lastOwner =
              op.getReverse() ? 0 : (numSegments - 1) & ~registerMask;
          if (consume && (segment & ~registerMask) == lastOwner) {
            unsigned owner = ownerOf(segment);
            consume(owner, carries[owner]);
          }
        });
    return carries;
  }

  // Finish one chunk while its warp carry is live. Only the terminal prefix
  // is shuffled; each remaining thread-local prefix receives one full carry.
  void applyWarpCarry(triton::ScanOp op, const ScanLoweringHelper &helper,
                      ScanValues &values, const ScanValues &totals,
                      unsigned reg, const ScanCarry &carry, Value laneId,
                      ConversionPatternRewriter &rewriter,
                      bool deferFirst = false) const {
    unsigned count = helper.getThreadLocalSegmentSize();
    unsigned begin = reg * count;
    unsigned last = begin + getScanIndex(op, count - 1, count);
    auto total = combineWithPrefix(op, carry.values,
                                   totals.empty() ? values[last] : totals[reg],
                                   rewriter, carry.pred);
    values[last] = total;
    if (count == 1)
      return;
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kLane = StringAttr::get(op.getContext(), "lane");
    auto axis =
        StringAttr::get(op.getContext(), "dim" + std::to_string(op.getAxis()));
    unsigned lanes = helper.getWarpLocalSegmentSize() / count;
    const auto &layout = helper.getIntraWarpLayout()
                             ? *helper.getIntraWarpLayout()
                             : helper.getPermutedLayout();
    unsigned stride = getScanLaneStride(layout, op.getAxis(), lanes);
    if (!stride)
      stride = layout.getInDimSize(kLane);
    Value index =
        applyLinearLayout(loc, rewriter, layout.sublayout({kLane}, {axis}),
                          {{kLane, laneId}})
            .front()
            .second;
    Value hasLocalCarry = b.icmp_ne(b.and_(index, b.i32_val(lanes - 1)),
                                    b.i32_val(op.getReverse() ? lanes - 1 : 0));
    Value source;
    if (op.getReverse())
      source = b.add(laneId, b.i32_val(stride));
    for (auto [value, warpCarry] : llvm::zip(total, carry.values)) {
      value = source ? targetInfo.shuffleIdx(rewriter, loc, value, source)
                     : targetInfo.shuffleUp(rewriter, loc, value, stride);
      value = b.select(hasLocalCarry, value, warpCarry);
    }
    Value pred = b.or_(hasLocalCarry, carry.pred);
    if (deferFirst) {
      unsigned first = begin + getScanIndex(op, 0, count);
      auto prefix = combineWithPrefix(op, total, values[first], rewriter, pred);
      values[first] = prefix;
      for (unsigned i = 1; i + 1 < count; ++i) {
        unsigned r = begin + getScanIndex(op, i, count);
        values[r] = combineWithPrefix(op, prefix, values[r], rewriter, {});
      }
      return;
    }
    for (unsigned i = 1; i < count; ++i) {
      unsigned r = begin + getScanIndex(op, count - 1 - i, count);
      values[r] = combineWithPrefix(op, total, values[r], rewriter, pred);
    }
  }

  // Include inter-warp carries in the thread totals before fetching the
  // preceding thread's total, so each local prefix receives one complete carry.
  void applyScanCarries(triton::ScanOp op, const ScanLoweringHelper &helper,
                        ScanValues &values, ScanValues &intraWarpTotals,
                        ArrayRef<ScanCarry> interWarpCarries, Value laneId,
                        Value warpId,
                        ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    auto kReg = StringAttr::get(op.getContext(), "register");
    unsigned interWarpRegs = 0;
    if (helper.getInterWarpLayout()) {
      const auto &layout = *helper.getInterWarpLayout();
      interWarpRegs = helper.getPermutedLayout().getInDimSize(kReg) /
                      layout.getInDimSize(kReg);
    }
    if (!helper.getIntraWarpLayout()) {
      if (!interWarpCarries.empty())
        applySegmentCarries(op, values, interWarpCarries, interWarpRegs,
                            rewriter);
      return;
    }

    const auto &intraWarpLayout = *helper.getIntraWarpLayout();
    unsigned segmentRegs = helper.getThreadLocalSegmentSize();
    unsigned numSegments = helper.getWarpLocalSegmentSize() / segmentRegs;
    convertScanValues(op, intraWarpTotals, *helper.getIntraWarpScanLayout(),
                      intraWarpLayout, rewriter);
    if (!interWarpCarries.empty())
      applySegmentCarries(op, intraWarpTotals, interWarpCarries,
                          interWarpRegs / segmentRegs, rewriter);

    // Terminal prefixes are complete; only the other registers need carries.
    for (unsigned r = 0; r < intraWarpTotals.size(); ++r)
      values[r * segmentRegs + getScanIndex(op, segmentRegs - 1, segmentRegs)] =
          intraWarpTotals[r];
    if (segmentRegs == 1)
      return;

    auto carries =
        getSegmentCarries(op, intraWarpTotals, intraWarpLayout, intraWarpLayout,
                          numSegments, laneId, warpId, rewriter);
    if (!interWarpCarries.empty()) {
      for (auto [r, carry] : llvm::enumerate(carries)) {
        const auto &interWarpCarry =
            interWarpCarries[r * segmentRegs / interWarpRegs];
        // The first thread segment has no preceding local total. Use the
        // inter-warp carry there, and leave the scan boundary untouched.
        for (auto [value, interWarpValue] :
             llvm::zip(carry.values, interWarpCarry.values))
          value = b.select(carry.pred, value, interWarpValue);
        carry.pred = b.or_(carry.pred, interWarpCarry.pred);
      }
    }
    applySegmentCarries(op, values, carries, segmentRegs, rewriter,
                        /*skipTerminal=*/true);
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
    // All required totals are present in this warp.
    auto inverse = totalsLayout.pseudoinvert().sublayout(
        llvm::to_vector(totalsLayout.getOutDimNames()), {kReg, kLane});
    unsigned numSegments = segmentLayout.getOutDimSize(axis);
    unsigned stride =
        segmentLayout == totalsLayout
            ? getScanLaneStride(segmentLayout, op.getAxis(), segmentsPerScan)
            : 0;
    SmallVector<ScanCarry> carries;
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
      Value pred = b.icmp_ne(
          segmentInScan, b.i32_val(op.getReverse() ? segmentsPerScan - 1 : 0));
      if (stride) {
        auto carry = totals[r];
        Value source;
        if (op.getReverse())
          source = b.add(laneId, b.i32_val(stride));
        for (auto &value : carry)
          value = source ? targetInfo.shuffleIdx(rewriter, loc, value, source)
                         : targetInfo.shuffleUp(rewriter, loc, value, stride);
        carries.push_back({std::move(carry), pred});
        continue;
      }
      // Wrap within this scan; pred excludes the boundary without an identity.
      coords[op.getAxis()].second = shiftWithinSegment(
          b, segment, op.getReverse() ? 1 : -1, segmentsPerScan, numSegments);
      auto src = applyLinearLayout(loc, rewriter, inverse, coords);
      Value srcReg = src[0].second;
      Value srcLane = src[1].second;
      auto candidates =
          getCarryRegisters(segmentLayout, inverse, r, op.getAxis(),
                            segmentsPerScan, op.getReverse());
      auto carry =
          shuffleCarry(loc, totals, candidates, srcReg, srcLane, rewriter);
      carries.push_back({std::move(carry), pred});
    }
    return carries;
  }

  void applySegmentCarries(triton::ScanOp op, ScanValues &values,
                           ArrayRef<ScanCarry> carries, unsigned segmentRegs,
                           ConversionPatternRewriter &rewriter,
                           bool skipTerminal = false) const {
    assert(values.size() == carries.size() * segmentRegs);
    unsigned begin = skipTerminal && op.getReverse();
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
