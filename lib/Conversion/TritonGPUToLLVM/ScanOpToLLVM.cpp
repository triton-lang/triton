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
                std::optional<int> shuffleDistance = {}) const {
    for (Value &value : values) {
      if (!shuffleDistance)
        value = targetInfo.shuffleIdx(rewriter, loc, value, lane);
      else if (*shuffleDistance > 0)
        value = targetInfo.shuffleUp(rewriter, loc, value, *shuffleDistance);
      else
        value = targetInfo.shuffleDown(rewriter, loc, value, -*shuffleDistance);
    }
    return values;
  }

  // Fetch preceding segment indices using a signed scan distance, preserving
  // bits for other scans. Positive distances scan forward; negative scan
  // reverse.
  static Value shiftWithinSegment(TritonLLVMOpBuilder &b, Value index,
                                  int distance, unsigned segmentsPerScan,
                                  unsigned numSegments) {
    unsigned mask = segmentsPerScan - 1;
    Value shifted = b.and_(b.sub(index, b.i32_val(distance)), b.i32_val(mask));
    if (segmentsPerScan == numSegments)
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

  // Scan each thread-local segment in traversal order, preserving combiner
  // operand order for both forward and reverse scans.
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
                      unsigned numSegments, Value laneId,
                      ConversionPatternRewriter &rewriter) const {
    unsigned numLanes = numSegments / numRegs;
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
    if (numSegments != axisSize)
      segmentIndex = b.and_(index, b.i32_val(numSegments - 1));

    // Shift logical coordinates and invert the layout to find the source lane.
    // Preserve coordinates belonging to independent scans.
    SmallVector<std::tuple<Value, Value, std::optional<int>>> rounds;
    for (unsigned offset = 1; offset < numLanes; offset *= 2) {
      int distance = offset * numRegs;
      auto shuffleDistance =
          getShuffleDistance(layout, inverseLayout, axis, numSegments,
                             reverse ? -distance : distance);
      Value lane;
      if (!shuffleDistance) {
        auto source = coords;
        source[axis].second = shiftWithinSegment(
            b, index, reverse ? -distance : distance, numSegments, axisSize);
        lane = applyLinearLayout(loc, rewriter, inverseLayout, source)
                   .back()
                   .second;
      }
      Value pred =
          reverse ? b.icmp_ult(segmentIndex, b.i32_val(numSegments - distance))
                  : b.icmp_uge(segmentIndex, b.i32_val(distance));
      rounds.emplace_back(lane, pred, shuffleDistance);
    }
    for (unsigned base = 0; base < values.size(); base += numRegs) {
      unsigned last = base + (reverse ? 0 : numRegs - 1);
      for (auto [lane, pred, shuffleDistance] : rounds) {
        auto incoming =
            shuffleValues(loc, values[last], lane, rewriter, shuffleDistance);
        for (unsigned r = base; r < base + numRegs; ++r)
          values[r] =
              combineWithPrefix(op, incoming, values[r], rewriter, pred);
      }
    }
  }

  // Extract thread-local totals and convert to contiguous register groups.
  // Scan those groups, then their terminal values across lanes. Save the
  // element prefixes for carry propagation.
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
    return totals;
  }

  // Convert warp-local segment totals into the full sequence in each
  // participating warp, then scan it. Return exclusive carries in the
  // original warp-segment ownership.
  SmallVector<ScanCarry>
  scanAcrossWarps(triton::ScanOp op, const ScanLoweringHelper &helper,
                  ScanValues &values, Value laneId, Value warpId,
                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kLane = StringAttr::get(ctx, "lane");
    const auto &sourceLayout = helper.getIntraWarpScanLayout()
                                   ? *helper.getIntraWarpScanLayout()
                                   : helper.getPermutedLayout();
    unsigned valuesPerSegment = helper.getWarpSegmentSize();
    // Count source values per warp-local segment: thread-local totals when
    // available, otherwise original elements.
    if (helper.getIntraWarpScanLayout())
      valuesPerSegment /= helper.getThreadSegmentSize();
    const auto &interWarpTotalsLayout = *helper.getInterWarpTotalsLayout();
    const auto &totalsLayout = *helper.getInterWarpScanLayout();
    bool reverse = op.getReverse();
    auto kReg = StringAttr::get(ctx, "register");
    unsigned segmentRegs = sourceLayout.getInDimSize(kReg) /
                           interWarpTotalsLayout.getInDimSize(kReg);
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    unsigned segmentLaneMask = getInputBasisMask(
        sourceLayout.resizeOutDim(axis, valuesPerSegment), kLane, {axis});

    // Replicate each complete total according to interWarpTotalsLayout before
    // the conversion selects a source owner for its shared-memory store.
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
    if (helper.getInterWarpScanGroupSize()) {
      scanInterWarpGroups(op, helper, totals, values, laneId, warpId, rewriter);
      return {};
    }
    convertScanValues(op, totals, interWarpTotalsLayout, totalsLayout, rewriter,
                      /*forceWarpShuffle=*/false);

    ScanLoweringHelper totalsHelper(totalsLayout, op.getAxis());
    assert(!totalsHelper.getInterWarpTotalsLayout() &&
           "the full totals sequence must be warp-local");
    scanWithinCTA(op, totalsHelper, totals, laneId, warpId, rewriter);
    return getSegmentCarries(op, totals, interWarpTotalsLayout, totalsLayout,
                             interWarpTotalsLayout.getOutDimSize(axis), laneId,
                             warpId, rewriter);
  }

  // Store the complete totals table once. Scan register groups in traversal
  // order, selecting carries only at positions requested by their consumers.
  // Complete each consumer's local prefixes as soon as its carry is available.
  void scanInterWarpGroups(triton::ScanOp op, const ScanLoweringHelper &helper,
                           const ScanValues &totals, ScanValues &values,
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
    const auto &source = *helper.getInterWarpTotalsLayout();
    ScanLoweringHelper totalsHelper(*helper.getInterWarpScanLayout(),
                                    op.getAxis());
    const auto &scan = totalsHelper.getPermutedLayout();
    unsigned numSegments = scan.getOutDimSize(axis);
    unsigned numRegs = scan.getInDimSize(kReg);
    unsigned maxGroupSize = helper.getInterWarpScanGroupSize();
    bool reverse = op.getReverse();

    // Adjacent independent scans occupy adjacent shared-memory elements.
    auto dims = llvm::to_vector(scan.getOutDimNames());
    llvm::erase(dims, axis);
    dims.push_back(axis);
    auto storeLayout = source.transposeOuts(dims).flattenOuts();
    auto loadLayout = scan.transposeOuts(dims).flattenOuts();
    auto scratch = helper.getGroupedInterWarpScratchConfig();
    Value smem = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
    SmallVector<Value> bases;
    SmallVector<Type> types;
    for (auto [i, type] : llvm::enumerate(op.getElementTypes())) {
      bases.push_back(
          b.gep(smem.getType(), i8_ty, smem, b.i32_val(scratch.offsets[i])));
      types.push_back(getTypeConverter()->convertType(type));
    }
    auto offset = [&](const LinearLayout &layout, unsigned reg) {
      return applyLinearLayout(loc, rewriter, layout,
                               {{kReg, b.i32_val(reg)},
                                {kLane, laneId},
                                {kWarp, warpId},
                                {kBlock, b.i32_val(0)}})
          .front()
          .second;
    };
    auto free = source.getFreeVariableMasks();
    Value storePred =
        b.and_(b.icmp_eq(b.and_(laneId, b.i32_val(free[kLane])), b.i32_val(0)),
               b.icmp_eq(b.and_(warpId, b.i32_val(free[kWarp])), b.i32_val(0)));
    for (auto [r, total] : llvm::enumerate(totals)) {
      Value index = offset(storeLayout, r);
      for (unsigned i = 0; i < total.size(); ++i) {
        Value ptr = b.gep(smem.getType(), types[i], bases[i], index);
        targetInfo.storeDShared(rewriter, loc, ptr, Value(), total[i],
                                storePred);
      }
    }
    targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);

    auto inverse = scan.pseudoinvert().sublayout(
        llvm::to_vector(scan.getOutDimNames()), {kReg, kLane});
    SmallVector<Value> sourceRegs;
    SmallVector<SmallVector<unsigned>> consumers(numRegs);
    SmallVector<unsigned> remaining;
    SmallVector<ScanCarry> carries;
    for (unsigned r = 0; r < totals.size(); ++r) {
      auto coords = applyLinearLayout(loc, rewriter, source,
                                      {{kReg, b.i32_val(r)},
                                       {kLane, laneId},
                                       {kWarp, warpId},
                                       {kBlock, b.i32_val(0)}});
      Value index = coords[op.getAxis()].second;
      Value pred = b.icmp_ne(index, b.i32_val(reverse ? numSegments - 1 : 0));
      coords[op.getAxis()].second = shiftWithinSegment(
          b, index, reverse ? -1 : 1, numSegments, numSegments);
      sourceRegs.push_back(
          applyLinearLayout(loc, rewriter, inverse, coords).front().second);
      auto candidates = getCarryRegisters(source, inverse, r, op.getAxis(),
                                          numSegments, reverse,
                                          /*excludeBoundary=*/true);
      for (unsigned reg : candidates)
        consumers[reg].push_back(r);
      remaining.push_back(candidates.size());
      carries.push_back({{}, pred});
    }

    SmallVector<Value> acc;
    unsigned segmentRegs = values.size() / totals.size();
    unsigned numOperands = types.size();
    unsigned rowStride = scan.getTotalOutDimSize() / numSegments;
    auto registerAt = [&](unsigned step) {
      return reverse ? numRegs - 1 - step : step;
    };
    for (unsigned step = 0; step < numRegs;) {
      const auto &groupConsumers = consumers[registerAt(step)];
      // Consecutive prefixes with the same consumers share one small loop.
      // End the loop before an independent scan or a different consumer set.
      unsigned groupSize = 1;
      unsigned available =
          std::min(maxGroupSize, numSegments - step % numSegments);
      while (groupSize < available &&
             consumers[registerAt(step + groupSize)] == groupConsumers)
        ++groupSize;
      unsigned base = reverse ? numRegs - step - groupSize : step;
      bool firstGroup = step % numSegments == 0;
      if (firstGroup) {
        acc.clear();
        for (Type type : types)
          acc.push_back(LLVM::UndefOp::create(rewriter, loc, type));
      }
      for (unsigned r : groupConsumers)
        if (carries[r].values.empty())
          for (Type type : types)
            carries[r].values.push_back(
                LLVM::UndefOp::create(rewriter, loc, type));
      SmallVector<Value> initial = acc;
      for (unsigned r : groupConsumers)
        llvm::append_range(initial, carries[r].values);
      SmallVector<Type> resultTypes;
      for (Value value : initial)
        resultTypes.push_back(value.getType());
      SmallVector<Type> loopTypes{rewriter.getI32Type()};
      llvm::append_range(loopTypes, resultTypes);
      SmallVector<Location> loopLocs(loopTypes.size(), loc);
      Value initialOffset = offset(loadLayout, base);
      Block *before = rewriter.getBlock();
      Block *after = rewriter.splitBlock(before, rewriter.getInsertionPoint());
      for (Type type : resultTypes)
        after->addArgument(type, loc);
      Block *loop = rewriter.createBlock(after, loopTypes, loopLocs);
      rewriter.setInsertionPointToEnd(before);
      SmallVector<Value> loopArgs{b.i32_val(0)};
      llvm::append_range(loopArgs, initial);
      LLVM::BrOp::create(rewriter, loc, loopArgs, loop);
      rewriter.setInsertionPointToStart(loop);
      Value iteration = loop->getArgument(0);
      Value position =
          reverse ? b.sub(b.i32_val(groupSize - 1), iteration) : iteration;
      Value index = b.add(initialOffset, b.mul(position, b.i32_val(rowStride)));
      SmallVector<Value> current;
      for (unsigned i = 0; i < numOperands; ++i) {
        Value ptr = b.gep(smem.getType(), types[i], bases[i], index);
        current.push_back(targetInfo.loadDShared(rewriter, loc, ptr, Value(),
                                                 types[i], b.true_val()));
      }
      ValueRange incoming = loop->getArguments().slice(1, numOperands);
      auto combined =
          firstGroup ? combineWithPrefix(op, incoming, current, rewriter,
                                         b.icmp_ne(iteration, b.i32_val(0)))
                     : applyCombineOp(loc, rewriter, op.getCombineOp(),
                                      incoming, current);
      SmallVector<Value> results = combined;
      Value reg = b.add(b.i32_val(base), position);
      for (auto [j, r] : llvm::enumerate(groupConsumers)) {
        Value selected = b.icmp_eq(sourceRegs[r], reg);
        for (unsigned i = 0; i < numOperands; ++i)
          results.push_back(
              b.select(selected, combined[i],
                       loop->getArgument(1 + (j + 1) * numOperands + i)));
      }
      Value next = b.add(iteration, b.i32_val(1));
      SmallVector<Value> nextArgs{next};
      llvm::append_range(nextArgs, results);
      auto branch = LLVM::CondBrOp::create(
          rewriter, loc, b.icmp_ult(next, b.i32_val(groupSize)), loop, nextArgs,
          after, results);
      // Keep the group loop rolled to bound the live shared-memory loads.
      auto unroll = LLVM::LoopUnrollAttr::get(ctx, rewriter.getBoolAttr(true),
                                              {}, {}, {}, {}, {}, {});
      branch.setLoopAnnotationAttr(LLVM::LoopAnnotationAttr::get(
          ctx, {}, {}, {}, unroll, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}));
      rewriter.setInsertionPointToStart(after);
      acc.assign(after->getArguments().begin(),
                 after->getArguments().begin() + numOperands);
      for (auto [j, r] : llvm::enumerate(groupConsumers)) {
        auto selected =
            after->getArguments().slice((j + 1) * numOperands, numOperands);
        carries[r].values.assign(selected.begin(), selected.end());
        remaining[r] -= groupSize;
        if (remaining[r])
          continue;
        auto &carry = carries[r];
        for (unsigned i = 0; i < segmentRegs; ++i) {
          auto &value = values[r * segmentRegs + i];
          value =
              combineWithPrefix(op, carry.values, value, rewriter, carry.pred);
        }
        carry.values.clear();
      }
      step += groupSize;
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

  // Positive distances fetch earlier positions; negative distances fetch later
  // positions. Consecutive lane bits turn logical subtraction/addition into a
  // constant shuffle-up/down distance.
  static std::optional<int>
  getShuffleDistance(const LinearLayout &layout,
                     const LinearLayout &inverseLayout, unsigned axis,
                     unsigned segmentsPerScan, int distance) {
    unsigned magnitude = std::abs(distance);
    if (magnitude == 0 || magnitude >= segmentsPerScan)
      return std::nullopt;
    auto *ctx = layout.getInDimNames().begin()->getContext();
    auto kLane = StringAttr::get(ctx, "lane");
    if (!layout.compose(inverseLayout).isIdentityOnOutDim(kLane))
      return std::nullopt;

    auto axisDim = StringAttr::get(ctx, "dim" + std::to_string(axis));
    const auto &bases = inverseLayout.getBases().lookup(axisDim);
    unsigned laneDim = inverseLayout.getOutDimIndex(kLane);
    unsigned first = llvm::Log2_32(magnitude);
    unsigned stride = bases[first][laneDim];
    if (stride == 0)
      return std::nullopt;
    for (unsigned bit = first + 1; bit < llvm::Log2_32(segmentsPerScan); ++bit)
      if (bases[bit][laneDim] != (stride << (bit - first)))
        return std::nullopt;
    return distance > 0 ? int(stride) : -int(stride);
  }

  // Enumerate the source registers that the lanes of this segment can request.
  static SmallVector<unsigned>
  getCarryRegisters(const LinearLayout &segmentLayout,
                    const LinearLayout &inverseTotalsLayout, unsigned reg,
                    unsigned axis, unsigned segmentsPerScan, bool reverse,
                    bool excludeBoundary = false) {
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
        if (excludeBoundary &&
            (index & mask) == (reverse ? segmentsPerScan - 1 : 0))
          continue;
        index = (index & ~mask) | ((index + (reverse ? 1 : -1)) & mask);
        candidates.insert(inverseTotalsLayout.apply(coords).front().second);
      }
    return llvm::to_vector(candidates);
  }

  // Select after fetching: each destination lane may request a different
  // register from its source lane.
  SmallVector<Value> gatherCarry(Location loc, const ScanValues &totals,
                                 ArrayRef<unsigned> candidates, Value srcReg,
                                 Value srcLane,
                                 ConversionPatternRewriter &rewriter,
                                 std::optional<int> shuffleDistance,
                                 bool laneLocal) const {
    assert(!candidates.empty() && "each segment has a warp-local carry");
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto fetch = [&](unsigned reg) {
      return laneLocal ? totals[reg]
                       : shuffleValues(loc, totals[reg], srcLane, rewriter,
                                       shuffleDistance);
    };
    auto carry = fetch(candidates.front());
    for (unsigned candidate : candidates.drop_front()) {
      auto incoming = fetch(candidate);
      Value select = b.icmp_eq(srcReg, b.i32_val(candidate));
      for (auto [value, source] : llvm::zip(carry, incoming))
        value = b.select(select, source, value);
    }
    return carry;
  }

  // Map each consumer segment to its preceding prefix in the totals layout.
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
    // The totals sequence is available in every consuming warp. CTA ownership
    // is preserved, so lookup uses only register and lane coordinates.
    auto inverse = totalsLayout.pseudoinvert().sublayout(
        llvm::to_vector(totalsLayout.getOutDimNames()), {kReg, kLane});
    // The unshifted lookup preserves lanes, and shifting the scan coordinate
    // cannot change the source lane when axis bits map only to registers.
    bool laneLocal = inverse.sublayoutIsZero({axis}, {kLane}) &&
                     segmentLayout.compose(inverse).isIdentityOnOutDim(kLane);
    unsigned numSegments = segmentLayout.getOutDimSize(axis);
    SmallVector<ScanCarry> carries;
    auto shuffleDistance =
        getShuffleDistance(segmentLayout, inverse, op.getAxis(),
                           segmentsPerScan, reverse ? -1 : 1);
    for (unsigned r = 0; r < segmentLayout.getInDimSize(kReg); ++r) {
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
          b, segment, reverse ? -1 : 1, segmentsPerScan, numSegments);
      auto src = applyLinearLayout(loc, rewriter, inverse, coords);
      Value srcReg = src[0].second;
      Value srcLane = src[1].second;
      auto candidates = getCarryRegisters(
          segmentLayout, inverse, r, op.getAxis(), segmentsPerScan, reverse);
      auto carry = gatherCarry(loc, totals, candidates, srcReg, srcLane,
                               rewriter, shuffleDistance, laneLocal);
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
