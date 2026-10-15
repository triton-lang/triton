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
                         bool forceWarpShuffle = true,
                         Value storePred = {}) const {
    if (src == dst)
      return;
    auto operands = convertLayoutValues(
        op.getLoc(), rewriter, op, src, dst, transposeValues(values),
        getTypeConverter(), targetInfo, forceWarpShuffle, storePred);
    values = transposeValues(operands);
  }

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

  // Scan terminal totals across lanes, then apply each preceding lane's
  // complete total to the remaining register prefixes once.
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
      auto acc = values[last];
      for (auto [lane, pred, shuffleDistance] : rounds) {
        auto incoming =
            shuffleValues(loc, acc, lane, rewriter, shuffleDistance);
        acc = combineWithPrefix(op, incoming, acc, rewriter, pred);
      }
      values[last] = acc;
      if (numRegs == 1)
        continue;
      auto [lane, pred, shuffleDistance] = rounds.front();
      auto prefix = shuffleValues(loc, acc, lane, rewriter, shuffleDistance);
      for (unsigned r = base; r < base + numRegs; ++r)
        if (r != last)
          values[r] = combineWithPrefix(op, prefix, values[r], rewriter, pred);
    }
  }

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

    // Select the lanes holding complete totals for the shared-memory stores.
    // Zero lane bases still describe the common address of each segment.
    auto totals = extractSegmentTotals(values, segmentRegs, reverse);
    Value storePred;
    if (segmentLaneMask)
      storePred = b.icmp_eq(b.and_(laneId, b.i32_val(segmentLaneMask)),
                            b.i32_val(reverse ? 0 : segmentLaneMask));
    // 1. Register efficient path: Load totals incrementally and accumulate
    // within each thread with a short register lifetime
    if (helper.getInterWarpScanType() ==
        ScanLoweringHelper::InterWarpScanType::Grouped) {
      scanInterWarpGroups(op, helper, totals, values, laneId, warpId, rewriter);
      return {};
    }
    // 2. General path: Load totals in each warp at once and perform intra-warp
    // scan, then compute carries of each warp through shuffles. It enjoys fewer
    // arith instructions and more parallelism.
    convertScanValues(op, totals, interWarpTotalsLayout, totalsLayout, rewriter,
                      /*forceWarpShuffle=*/false, storePred);

    ScanLoweringHelper totalsHelper(totalsLayout, op.getAxis());
    assert(!totalsHelper.getInterWarpTotalsLayout() &&
           "the full totals sequence must be warp-local");
    if (totalsLayout.sublayoutIsZero({kLane}, {axis})) {
      permuteRegisters(totals, totalsHelper.getRegisterOrder());
      return scanRegisterCarries(
          op, totals, interWarpTotalsLayout, totalsHelper.getPermutedLayout(),
          laneId, warpId, rewriter,
          helper.getIntraWarpTotalsLayout() ? nullptr : &values);
    }
    scanWithinCTA(op, totalsHelper, totals, laneId, warpId, rewriter);
    return getSegmentCarries(op, totals, interWarpTotalsLayout, totalsLayout,
                             interWarpTotalsLayout.getOutDimSize(axis), laneId,
                             warpId, rewriter);
  }

  // Scan register-resident warp totals in logical order. Thread-owned segments
  // reuse inclusive totals for their terminal values and exclusive carries for
  // their other prefixes. Other layouts receive exclusive carries.
  SmallVector<ScanCarry>
  scanRegisterCarries(triton::ScanOp op, const ScanValues &totals,
                      const LinearLayout &consumers, const LinearLayout &scan,
                      Value laneId, Value warpId,
                      ConversionPatternRewriter &rewriter,
                      ScanValues *consumerValues) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    unsigned numSegments = scan.getOutDimSize(axis);
    bool reverse = op.getReverse();
    auto inverse = scan.pseudoinvert().sublayout(
        llvm::to_vector(scan.getOutDimNames()), {kReg, kLane});
    SmallVector<ScanCarry> carries;
    SmallVector<Value> consumerRegs;
    SmallVector<SmallVector<unsigned>> requested(totals.size());
    SmallVector<unsigned> remaining(consumers.getInDimSize(kReg), 0);
    unsigned segmentRegs =
        consumerValues ? consumerValues->size() / remaining.size() : 0;
    ScanValues inclusiveTotals(consumerValues ? remaining.size() : 0);
    for (unsigned r = 0; r < consumers.getInDimSize(kReg); ++r) {
      auto coords = applyLinearLayout(loc, rewriter, consumers,
                                      {{kReg, b.i32_val(r)},
                                       {kLane, laneId},
                                       {kWarp, warpId},
                                       {kBlock, b.i32_val(0)}});
      Value index = coords[op.getAxis()].second;
      Value pred = b.icmp_ne(index, b.i32_val(reverse ? numSegments - 1 : 0));
      carries.push_back({{}, pred});
      consumerRegs.push_back(
          applyLinearLayout(loc, rewriter, inverse, coords).front().second);
      for (unsigned candidate :
           getScanRegisters(consumers, inverse, r, op.getAxis(), numSegments,
                            consumerValues ? 0 : (reverse ? 1 : -1),
                            /*excludeBoundary=*/!consumerValues)) {
        requested[candidate].push_back(r);
        ++remaining[r];
      }
    }
    for (unsigned base = 0; base < totals.size(); base += numSegments) {
      SmallVector<Value> acc;
      for (unsigned i = 0; i < numSegments - (consumerValues ? 0 : 1); ++i) {
        unsigned r = base + (reverse ? numSegments - 1 - i : i);
        auto exclusive = acc;
        acc = applyCombineOp(loc, rewriter, op.getCombineOp(), acc, totals[r]);
        for (unsigned consumer : requested[r]) {
          Value selected;
          if (consumerValues)
            selected = reverse
                           ? b.icmp_uge(b.i32_val(r), consumerRegs[consumer])
                           : b.icmp_ule(b.i32_val(r), consumerRegs[consumer]);
          else
            selected = reverse
                           ? b.icmp_ugt(b.i32_val(r), consumerRegs[consumer])
                           : b.icmp_ult(b.i32_val(r), consumerRegs[consumer]);
          if (consumerValues) {
            selectPrefix(acc, inclusiveTotals[consumer], selected, b);
            if (segmentRegs > 1)
              selectPrefix(exclusive, carries[consumer].values, selected, b);
          } else {
            selectPrefix(acc, carries[consumer].values, selected, b);
          }
          --remaining[consumer];
          if (consumerValues && remaining[consumer] == 0)
            completeThreadSegment(op, *consumerValues, consumer, segmentRegs,
                                  inclusiveTotals[consumer], carries[consumer],
                                  rewriter);
        }
      }
    }
    return consumerValues ? SmallVector<ScanCarry>{} : carries;
  }

  // Store the complete totals table once. Scan each independent sequence,
  // selecting carries only at positions requested by their consumers.
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
    auto scan = *helper.getInterWarpScanLayout();
    unsigned numTotals = scan.getOutDimSize(axis);
    bool reverse = op.getReverse();

    // Factor the contiguous scan axis out of the register layout. The
    // quotient maps each independent scan to its shared-memory base.
    // e.g., R = [[0, 1], [0, 2], [0, 4], [32, 0]] -> R = [[32, 0]]
    auto scanAxis = LinearLayout::identity1D(numTotals, kReg, axis);
    scan = regPermForDivide(scan, scanAxis, /*left=*/true)->apply(scan);
    auto independentScans = divideLeft(scan, scanAxis).value();
    auto dims = llvm::to_vector(scan.getOutDimNames());
    llvm::erase(dims, axis);
    dims.push_back(axis);
    auto storeLayout = source.transposeOuts(dims).flattenOuts();
    auto loadLayout = independentScans.transposeOuts(dims).flattenOuts();
    unsigned stride = independentScans.getTotalOutDimSize();
    auto scratch = helper.getGroupedInterWarpScratchConfig();
    Value smem = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
    SmallVector<Value> bases;
    SmallVector<Type> types;
    for (auto [i, type] : llvm::enumerate(op.getElementTypes())) {
      bases.push_back(
          b.gep(smem.getType(), i8_ty, smem, b.i32_val(scratch.offsets[i])));
      types.push_back(getTypeConverter()->convertType(type));
    }
    auto getOffset = [&](const LinearLayout &layout, unsigned reg) {
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
      Value offset = getOffset(storeLayout, r);
      for (unsigned i = 0; i < total.size(); ++i) {
        Value ptr = b.gep(smem.getType(), types[i], bases[i], offset);
        targetInfo.storeDShared(rewriter, loc, ptr, Value(), total[i],
                                storePred);
      }
    }
    targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);

    auto inverse = scan.pseudoinvert().sublayout(
        llvm::to_vector(scan.getOutDimNames()), {kReg, kLane});
    unsigned segmentRegs = values.size() / totals.size();
    SmallVector<Value> consumerRegs;
    SmallVector<SmallVector<std::pair<unsigned, bool>>> consumers(
        scan.getInDimSize(kReg));
    SmallVector<ScanCarry> carries;
    ScanValues inclusiveTotals(totals.size());
    for (unsigned r = 0; r < totals.size(); ++r) {
      auto coords = applyLinearLayout(loc, rewriter, source,
                                      {{kReg, b.i32_val(r)},
                                       {kLane, laneId},
                                       {kWarp, warpId},
                                       {kBlock, b.i32_val(0)}});
      Value index = coords[op.getAxis()].second;
      Value pred = b.icmp_ne(index, b.i32_val(reverse ? numTotals - 1 : 0));
      consumerRegs.push_back(
          applyLinearLayout(loc, rewriter, inverse, coords).front().second);
      // A simple implementation visits every prefix for every destination:
      //   for (each prefix Qj)
      //     for (each destination r)
      //       carry[r] = select(consumerRegs[r] == j + 1, Qj, carry[r]);
      //
      // Enumerating candidate prefixes reduces the selections:
      //   for (each destination r)
      //     for (each candidate prefix Qc)
      //       carry[r] = select(consumerRegs[r] == c + 1, Qc, carry[r]);
      auto candidates =
          getScanRegisters(source, inverse, r, op.getAxis(), numTotals,
                           /*distance=*/0);
      llvm::sort(candidates, [reverse](unsigned a, unsigned b) {
        return reverse ? a > b : a < b;
      });
      for (unsigned reg : candidates)
        consumers[reg].emplace_back(r, reg == candidates.back());
      carries.push_back({{}, pred});
    }

    unsigned numScans = independentScans.getInDimSize(kReg);
    for (unsigned n = 0; n < numScans; ++n) {
      unsigned scanIndex = reverse ? numScans - 1 - n : n;
      Value base = getOffset(loadLayout, scanIndex);
      SmallVector<Value> acc;
      for (unsigned i = 0; i < numTotals; ++i) {
        unsigned segment = reverse ? numTotals - 1 - i : i;
        unsigned reg = scanIndex * numTotals + segment;
        Value index = b.add(base, b.i32_val(segment * stride));
        SmallVector<Value> current;
        for (unsigned i = 0; i < types.size(); ++i) {
          Value ptr = b.gep(smem.getType(), types[i], bases[i], index);
          current.push_back(targetInfo.loadDShared(rewriter, loc, ptr, Value(),
                                                   types[i], b.true_val()));
        }
        auto exclusive = acc;
        acc = applyCombineOp(loc, rewriter, op.getCombineOp(), acc, current);
        // `complete` marks the last candidate producer for consumer r in scan
        // order.
        // Once all candidates have been considered, apply the selected carry to
        // the elements of segment r. For example:
        //   consumers[1] = {{0, false}};
        //   consumers[2] = {{0, true}, {1, false}};
        //   consumers[3] = {{1, true}};
        for (auto [r, complete] : consumers[reg]) {
          Value selected = reverse
                               ? b.icmp_uge(b.i32_val(reg), consumerRegs[r])
                               : b.icmp_ule(b.i32_val(reg), consumerRegs[r]);
          selectPrefix(acc, inclusiveTotals[r], selected, b);
          if (segmentRegs > 1)
            selectPrefix(exclusive, carries[r].values, selected, b);
          if (complete)
            completeThreadSegment(op, values, r, segmentRegs,
                                  inclusiveTotals[r], carries[r], rewriter);
        }
      }
    }
  }

  // The first candidate initializes a prefix without requiring an identity.
  static void selectPrefix(ValueRange prefix, SmallVector<Value> &result,
                           Value selected, TritonLLVMOpBuilder &b) {
    if (prefix.empty())
      return;
    if (result.empty()) {
      result.assign(prefix.begin(), prefix.end());
      return;
    }
    for (auto [value, candidate] : llvm::zip(result, prefix))
      value = b.select(selected, candidate, value);
  }

  // The inclusive total already contains the terminal result. Only earlier
  // local prefixes need the exclusive carry; guard the scan boundary.
  void completeThreadSegment(triton::ScanOp op, ScanValues &values,
                             unsigned segment, unsigned segmentRegs,
                             ValueRange total, const ScanCarry &carry,
                             ConversionPatternRewriter &rewriter) const {
    unsigned terminal = op.getReverse() ? 0 : segmentRegs - 1;
    for (unsigned i = 0; i < segmentRegs; ++i) {
      auto &value = values[segment * segmentRegs + i];
      if (i == terminal)
        value.assign(total.begin(), total.end());
      else
        value =
            combineWithPrefix(op, carry.values, value, rewriter, carry.pred);
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
      // Case 1: No intra-warp shuffle, only apply inter-warp carries
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

    for (unsigned r = 0; r < intraWarpTotals.size(); ++r)
      values[r * segmentRegs + (op.getReverse() ? 0 : segmentRegs - 1)] =
          intraWarpTotals[r];
    if (segmentRegs == 1)
      // Case 2: Even if we have intra-warp shuffles, each thread segment has a
      // single value, so the intra-warp totals are already the final output.
      return;

    // Case 3: Otherwise, we have to also apply intra warp shuffle results to
    // the individual thread segments
    auto carries = getSegmentCarries(op, intraWarpTotals, intraWarpTotalsLayout,
                                     intraWarpTotalsLayout, numSegments, laneId,
                                     warpId, rewriter);
    if (!interWarpCarries.empty()) {
      unsigned segmentsPerWarp = carries.size() / interWarpCarries.size();
      for (auto [r, carry] : llvm::enumerate(carries)) {
        const auto &interWarpCarry = interWarpCarries[r / segmentsPerWarp];
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
    // layout and inverseLayout.inverse() may be different
    if (!layout.compose(inverseLayout).isIdentityOnOutDim(kLane))
      return std::nullopt;

    auto axisDim = StringAttr::get(ctx, "dim" + std::to_string(axis));
    const auto &bases = inverseLayout.getBases().lookup(axisDim);
    unsigned laneDim = inverseLayout.getOutDimIndex(kLane);
    unsigned first = llvm::Log2_32(magnitude);
    unsigned stride = bases[first][laneDim];
    if (stride == 0)
      return std::nullopt;
    // After the first stride, each subsequent stride must be a power-of-2
    // multiple of the first.
    for (unsigned bit = first + 1; bit < llvm::Log2_32(segmentsPerScan); ++bit)
      if (bases[bit][laneDim] != (stride << (bit - first)))
        return std::nullopt;
    return distance > 0 ? int(stride) : -int(stride);
  }

  static SmallVector<unsigned>
  getScanRegisters(const LinearLayout &segmentLayout,
                   const LinearLayout &inverseTotalsLayout, unsigned reg,
                   unsigned axis, unsigned segmentsPerScan, int offset,
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
            (index & mask) == (offset > 0 ? segmentsPerScan - 1 : 0))
          continue;
        index = (index & ~mask) | ((index + offset) & mask);
        candidates.insert(inverseTotalsLayout.apply(coords).front().second);
      }
    return llvm::to_vector(candidates);
  }

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
    auto inverse = totalsLayout.pseudoinvert().sublayout(
        llvm::to_vector(totalsLayout.getOutDimNames()), {kReg, kLane});
    bool laneLocal = inverse.sublayoutIsZero({axis}, {kLane});
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
      coords[op.getAxis()].second = shiftWithinSegment(
          b, segment, reverse ? -1 : 1, segmentsPerScan, numSegments);
      auto src = applyLinearLayout(loc, rewriter, inverse, coords);
      Value srcReg = src[0].second;
      Value srcLane = src[1].second;
      auto candidates =
          getScanRegisters(segmentLayout, inverse, r, op.getAxis(),
                           segmentsPerScan, reverse ? 1 : -1);
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
