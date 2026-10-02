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
  // A row is a complete combiner tuple, e.g. (value, index) for an argmax scan.
  // Communicate every component together, even when their bit widths differ.
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
    //
    // The three arrays below have different meanings and lifetimes:
    //   values: original elements -> thread-local prefixes, in permutedLayout;
    //   intraWarpTotals: one per thread segment, scanned within its warp-local
    //                    segment and left in intraWarpScanLayout;
    //   interWarpTotals: one per warp-local segment, scanned across the full
    //                    axis and replicated in interWarpScanLayout.
    // Keep values in their original threads while communicating only totals.
    // applyScanCarries joins these results, first within a warp, then across
    // warps. An absent layout means that communication phase is unnecessary.
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
  // Reindex the compiler's SSA-value array, without emitting data movement.
  // For example, swapping register basis columns [8,1] to [1,8] changes the
  // array [x0,x8,x1,x9] to [x0,x1,x8,x9] in the same hardware thread.
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
  // Example: two registers holding T0,T4 in lane 0 become T0,T1 after the
  // register/lane transpose illustrated in buildIntraWarpScanLayout. Force
  // shuffles here; the separate inter-warp conversion may use shared memory.
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
  // For numRegs=2 and [x0,x1,x8,x9], produce
  // [x0, combine(x0,x1), x8, combine(x8,x9)]. Do not combine x1 with x8:
  // elements x2..x7 are elsewhere. Reverse traversal starts at each group's
  // last register, producing [combine(x1,x0), x1, combine(x9,x8), x9].
  // The same routine later scans segment totals, not the original elements
  // again. All examples use combine(a,b) in traversal order; associativity
  // allows grouping, but commutativity and an identity value are not required.
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

  // Scan normalized totals across lanes in logical order, then add each
  // lane's exclusive carry to the register prefixes of those totals.
  // Example: after scanning two totals per lane, four logical lanes hold
  //   [T0, T0+T1], [T2, T2+T3], [T4, T4+T5], [T6, T6+T7].
  // Scan only their terminal registers using shuffle distances 1,2. They
  // become sums through T1,T3,T5,T7. Reuse each round's incoming value to
  // accumulate the exclusive carry too, then apply it to the other registers
  // to obtain sums through T0,T2,T4,T6 without another shuffle.
  // '+' here abbreviates the ordered combiner, not necessarily numeric add.
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
    auto dims = llvm::to_vector(layout.getOutDimNames());
    auto laneLayout = layout.sublayout({kLane}, dims);
    auto inverseLaneLayout = layout.pseudoinvert().sublayout(dims, {kLane});

    // Example: lane=[1,0,2] maps physical lanes 0,1,4,5 to logical positions
    // 0,1,2,3. Shift the logical coordinate, then use the inverse layout to
    // find its owning lane. A pseudoinverse picks a valid copy for broadcasts.
    // Register and warp coordinates stay fixed, so only the lane contribution
    // is needed. Preserve all coordinates belonging to independent scans.
    // In the lane=[1,0,2] example, physical lane 4 has logical index 2. Its
    // predecessor is index 1, whose canonical inverse image is physical lane 1.
    // Lane 3 is another valid copy of index 1 because bit 1 is
    // broadcast. If bit 1 instead indexes another tensor axis, preserving
    // that coordinate ensures the lookup stays in the same independent scan.
    auto coords =
        applyLinearLayout(loc, rewriter, laneLayout, {{kLane, laneId}});
    Value index = coords[axis].second;
    Value segmentIndex = b.and_(index, b.i32_val(segmentSize - 1));
    auto shuffle = [&](SmallVector<Value> input, unsigned offset) {
      // Each lane owns numRegs consecutive totals. Shift by whole groups,
      // wrapping within this segment; the combine predicate excludes
      // wraparound.
      // For numRegs=2, segmentSize=8, a lane starting at logical index 4
      // reads index 2 at offset 1 and index 0 at offset 2. Keeping the high
      // bits of index unchanged prevents crossing into a different segment.
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
      SmallVector<Value> prefix;
      // Each round consumes the previous round's inclusive prefixes:
      // distance 1 joins neighboring lanes, distance 2 joins pairs, etc.
      // Reverse scans read toward higher logical indices instead.
      for (unsigned offset = 1; offset < numLanes; offset *= 2) {
        auto incoming = shuffle(acc, offset);
        Value pred = hasPrefix(offset);
        if (numRegs > 1) {
          // The first round supplies the preceding lane's total directly;
          // later rounds prepend earlier groups to that exclusive prefix.
          // For lane 3 with lane totals A,B,C,D, incoming is C at offset 1
          // and A+B at offset 2: prefix becomes A+B+C while acc becomes
          // A+B+C+D. Both accumulators reuse the same shuffled tuple.
          // The boundary lane's first incoming value is only a placeholder:
          // all later predicates are false, and hasPrefix(1) prevents its use.
          if (offset == 1)
            prefix = incoming;
          else
            prefix = combineWithPrefix(op, incoming, prefix, rewriter, pred);
        }
        acc = combineWithPrefix(op, incoming, acc, rewriter, pred);
      }
      values[last] = acc;
      if (numRegs == 1 || numLanes == 1)
        continue;
      // The exclusive carry is already available. Skip the first lane in
      // traversal order without assuming an identity value.
      Value pred = hasPrefix(1);
      for (unsigned r = base; r < base + numRegs; ++r)
        if (r != last)
          values[r] = combineWithPrefix(op, prefix, values[r], rewriter, pred);
    }
  }

  // Input: saved thread-local prefixes in permutedLayout. Output: inclusive
  // thread-segment totals in intraWarpScanLayout, with values left untouched.
  // For register=[1,8], lane=[2,4,...], define Ti=combine(x[2*i],x[2*i+1]).
  // Extract T0,T4 from lane 0, T1,T5 from lane 1, etc.; convert these totals
  // into adjacent register pairs, then scan registers and lanes. This second
  // thread scan combines summaries of disjoint pairs, not each x a second time.
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

  // Example: warp 0 owns [0..3],[8..11],[16..19],[24..27], while warp 1
  // owns [4..7],[12..15],[20..23],[28..31]. Let Wi summarize [4*i..4*i+3].
  // The local scans supply W0,W2,W4,W6 in warp 0 and W1,W3,W5,W7 in warp 1.
  // Broadcast each terminal value within its segment, then convert ownership
  // so each warp has W0..W7. Scan that sequence independently in each warp.
  // Return inclusive totals Cj=W0+...+Wj; later, segment j uses C(j-1) as its
  // exclusive carry. Reverse scans instead read the carry from segment j+1.
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
    // Express the segment length in the units of values: original elements
    // without a lane scan, or thread-segment totals after a lane scan.
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
    // For segment lane bits 0 and 2, the mask is 0b00101. Setting those bits
    // picks the forward terminal lane; clearing them picks the reverse one.
    // Other bits keep selecting the same segment or independent scan. This
    // broadcast makes those lane bases zero in interWarpLayout: every lane
    // of the segment now holds its total, not just the lane that computed it.
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
    // This is the cross-warp exchange. Conversion lowering supplies shared
    // memory and synchronization if required; the following scan is warp-local
    // because totalsLayout replicates the complete sequence in each warp.
    operands =
        convertLayoutValues(loc, rewriter, op, interWarpLayout, totalsLayout,
                            operands, getTypeConverter(), targetInfo);
    ScanValues totals(operands.front().size());
    for (unsigned r = 0; r < totals.size(); ++r)
      for (const auto &operand : operands)
        totals[r].push_back(operand[r]);

    // Reuse the same ordered scan, including register groups when the complete
    // sequence is longer than the available lanes.
    // For 64 totals and 32 lanes, totalsLayout has register=[32], lane=[1,2,
    // 4,8,16], warp=[0,...]. totalsHelper plans the register/lane scan for this
    // new tensor. Its warp axis bases are zero, so it needs no further
    // inter-warp scan. Restore totalsLayout before returning for carry lookup.
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
  // Example (forward): a native thread segment [x4,x5] saved [x4,x4+x5].
  // Its scanned total now includes preceding thread segments in the same
  // warp-local interval. Replace x4+x5 with that inclusive total; prepend the
  // preceding thread segment's total to x4. Then prepend any preceding warps'
  // segment total to BOTH results. Applying the warp carry first would put
  // these operands in the wrong order for a noncommutative combiner.
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

  // Map each native segment to its exclusive carry in the totals' layout.
  // segmentLayout describes the destination segment IDs; totalsLayout locates
  // their scanned totals, possibly in different registers/lanes. segmentRegs
  // counts local registers updated per destination segment, not its full
  // logical size (a warp segment can span multiple lanes).
  //
  // The two callers use different segment units:
  // - Thread carries: segmentsPerScan is the number of thread segments in ONE
  //   warp-local interval. The already-replaced terminal register is skipped.
  // - Warp carries: segmentsPerScan is the number of warp-local intervals in
  //   the WHOLE axis. Update every register, including terminal registers.
  //
  // Example: four thread segments per warp-local interval means segment 6
  // reads scanned total 5, while segment 4 has no carry. Segment 4 must not
  // read total 3 from another warp-local interval. For inter-warp carries,
  // interval 4 DOES read interval 3, since all intervals form one sequence.
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
      // A fixed r can represent different logical segments in different
      // lanes/warps. Evaluate the destination layout before taking j-1 (or
      // j+1 in reverse); subtracting from a physical lane ID is insufficient.
      // CTA ID 0 is sufficient here because neither layout moves data across
      // CTAs: other CTAs perform the identical lookup for their own scans.
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
      // For totals in register=[32], lane=[1,2,4,8,16], segment 32 needs
      // register 0, lane 31, while segment 33 needs register 1, lane 0. Thus
      // srcReg may vary by lane. SSA registers cannot be indexed dynamically:
      // enumerate possible source registers at compile time, shuffle each
      // candidate from srcLane, then select using this destination's srcReg.
      // Shuffling before selection matters: the source lane's own srcReg
      // need not equal the destination lane's requested source register.
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
  // At the first segment in traversal order there is no preceding total.
  // A wrapped shuffle source is only a valid address, not a neutral element:
  // pred must suppress that combine and preserve values. For tuple scans the
  // region receives the entire prefix tuple followed by the current tuple.
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
