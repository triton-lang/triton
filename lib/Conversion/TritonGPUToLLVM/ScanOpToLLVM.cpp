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
                std::optional<unsigned> upDistance = {}) const {
    for (Value &value : values)
      value = upDistance
                  ? targetInfo.shuffleUp(rewriter, loc, value, *upDistance)
                  : targetInfo.shuffleIdx(rewriter, loc, value, lane);
    return values;
  }

  static Value shiftWithinSegment(TritonLLVMOpBuilder &b, Value index,
                                  int distance, unsigned segmentsPerScan,
                                  unsigned numSegments) {
    unsigned mask = segmentsPerScan - 1;
    Value shifted = b.and_(b.add(index, b.i32_val(distance)), b.i32_val(mask));
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

    // Depending on the interval, we maybe able to use shuffleUp or shuffleIdx.
    // The former is more efficient, but it only works when the distance is a
    // power of 2. The latter is more general, but it requires a lane index to
    // be computed and passed in.
    SmallVector<std::tuple<Value, Value, std::optional<unsigned>>> rounds;
    for (unsigned offset = 1; offset < numLanes; offset *= 2) {
      int distance = offset * numRegs;
      auto upDistance =
          getShuffleUpDistance(layout, inverseLayout, axis, numSegments,
                               reverse ? distance : -distance);
      Value lane;
      if (!upDistance) {
        auto source = coords;
        source[axis].second = shiftWithinSegment(
            b, index, reverse ? distance : -distance, numSegments, axisSize);
        lane = applyLinearLayout(loc, rewriter, inverseLayout, source)
                   .back()
                   .second;
      }
      Value pred =
          reverse ? b.icmp_ult(segmentIndex, b.i32_val(numSegments - distance))
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
    if (helper.useGroupedInterWarpScan()) {
      scanInterWarpGroups(op, helper, totals, values, laneId, warpId, rewriter);
      return {};
    }
    // 2. General path: Load totals in each warp at once and perform intra-warp
    // scan, then compute carries of each warp through shuffles. It enjoys fewer
    // arith instructions and more parallelism.
    convertScanValues(op, totals, interWarpTotalsLayout, totalsLayout, rewriter,
                      /*forceWarpShuffle=*/false, storePred);

    ScanLoweringHelper totalsHelper(totalsLayout, op.getAxis());
    scanWithinCTA(op, totalsHelper, totals, laneId, warpId, rewriter);
    return getSegmentCarries(op, totals, interWarpTotalsLayout, totalsLayout,
                             interWarpTotalsLayout.getOutDimSize(axis), laneId,
                             warpId, rewriter);
  }

  // The general conversion materializes the replicated totals in registers.
  // For long, wide tuples this can keep many registers live at once. Grouped
  // traversal loads one group at a time and maintains a running prefix and
  // the carries.
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
    // For one-value segments, the inclusive prefix is already the final
    // output. Select it directly to avoid a carry combine and its boundary
    // select.
    bool inclusive = segmentRegs == 1;
    SmallVector<Value> consumerRegs;
    SmallVector<SmallVector<std::pair<unsigned, bool>>> consumers(
        scan.getInDimSize(kReg));
    SmallVector<ScanCarry> carries;
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
                           inclusive ? 0 : (reverse ? 1 : -1),
                           /*excludeBoundary=*/!inclusive);
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
        acc = applyCombineOp(loc, rewriter, op.getCombineOp(), acc, current);
        // `complete` marks the last candidate producer for consumer r in scan
        // order.
        // Once all candidates have been considered, apply the selected carry to
        // the elements of segment r. For example:
        //   consumers[1] = {{0, false}};
        //   consumers[2] = {{0, true}, {1, false}};
        //   consumers[3] = {{1, true}};
        for (auto [r, complete] : consumers[reg]) {
          auto &carry = carries[r];
          if (carry.values.empty()) {
            carry.values = acc;
          } else {
            unsigned consumer = inclusive ? reg : (reverse ? reg - 1 : reg + 1);
            Value selected = b.icmp_eq(consumerRegs[r], b.i32_val(consumer));
            for (unsigned i = 0; i < types.size(); ++i)
              carry.values[i] = b.select(selected, acc[i], carry.values[i]);
          }
          if (!complete)
            continue;
          if (inclusive) {
            values[r] = carry.values;
          } else {
            for (unsigned i = 0; i < segmentRegs; ++i) {
              auto &value = values[r * segmentRegs + i];
              value = combineWithPrefix(op, carry.values, value, rewriter,
                                        carry.pred);
            }
          }
        }
      }
    }
  }

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

  static std::optional<unsigned>
  getShuffleUpDistance(const LinearLayout &layout,
                       const LinearLayout &inverseLayout, unsigned axis,
                       unsigned segmentsPerScan, int distance) {
    if (distance >= 0 || unsigned(-distance) >= segmentsPerScan)
      return std::nullopt;
    auto *ctx = layout.getInDimNames().begin()->getContext();
    auto kLane = StringAttr::get(ctx, "lane");
    // layout and inverseLayout.inverse() may be different
    if (!layout.compose(inverseLayout).isIdentityOnOutDim(kLane))
      return std::nullopt;

    auto axisDim = StringAttr::get(ctx, "dim" + std::to_string(axis));
    const auto &bases = inverseLayout.getBases().lookup(axisDim);
    unsigned laneDim = inverseLayout.getOutDimIndex(kLane);
    unsigned first = llvm::Log2_32(-distance);
    unsigned stride = bases[first][laneDim];
    if (!llvm::isPowerOf2_32(stride))
      return std::nullopt;
    // After the first stride, each subsequent stride must be a power-of-2
    // multiple of the first.
    for (unsigned bit = first + 1; bit < llvm::Log2_32(segmentsPerScan); ++bit)
      if (bases[bit][laneDim] != (stride << (bit - first)))
        return std::nullopt;
    return stride;
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
    unsigned numSegments = segmentLayout.getOutDimSize(axis);
    SmallVector<ScanCarry> carries;
    auto upDistance = getShuffleUpDistance(segmentLayout, inverse, op.getAxis(),
                                           segmentsPerScan, reverse ? 1 : -1);
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
          b, segment, reverse ? 1 : -1, segmentsPerScan, numSegments);
      auto src = applyLinearLayout(loc, rewriter, inverse, coords);
      Value srcReg = src[0].second;
      Value srcLane = src[1].second;
      auto candidates =
          getScanRegisters(segmentLayout, inverse, r, op.getAxis(),
                           segmentsPerScan, reverse ? 1 : -1);
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
