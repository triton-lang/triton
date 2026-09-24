#include "ReduceScanCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct ScanOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ScanOp> {
  // Values are indexed by register, then by scan operand. The scan phases use
  // logical axis order; unpacking and packing use the original layout order.
  using ScanValues = SmallVector<SmallVector<Value>>;

  ScanOpConversion(LLVMTypeConverter &typeConverter,
                   const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ScanOp>(typeConverter,
                                                                benefit),
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

    // Match the helper's layout: axis register bits first, ordered by logical
    // significance. This only reorders SSA values within each thread.
    const auto &registerOrder = helper.getRegisterOrder();
    permuteRegisters(values, registerOrder);

    unsigned localSize = helper.getLocalScanSize();
    bool scanEndpoints = localSize > 2 && helper.getStages().size() > 1 &&
                         !deferWarpPrefixes(op, helper) &&
                         !warpRegisterLaneMask(helper);

    // First scan contiguous groups of registers owned by each thread.
    scanRegisterGroups(op, values, localSize, scanEndpoints, rewriter);

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value threadId = getThreadId(rewriter, loc);
    unsigned warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    Value laneId = b.urem(threadId, b.i32_val(warpSize));
    Value warpId = b.udiv(threadId, b.i32_val(warpSize));

    // Merge the groups into contiguous warp-local prefixes.
    scanWithinWarps(op, helper, values, laneId, scanEndpoints, rewriter);

    // Finally add the carries from preceding warp-local segments.
    if (helper.getScratchLayout())
      scanAcrossWarps(op, helper, values, laneId, warpId, rewriter);

    // Restore the input register order before packing the results.
    permuteRegisters(values, registerOrder.inverse());
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

  // Each group contains the consecutive low axis bits owned by registers.
  // Higher register bits can still be interleaved with lane/warp bits.
  void scanRegisterGroups(triton::ScanOp op, ScanValues &values,
                          unsigned localSize, bool scanEndpoints,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    bool reverse = op.getReverse();
    for (unsigned base = 0; base < values.size(); base += localSize) {
      // For the endpoint scan, retain prefixes of the suffix excluding the
      // leading element. They can all combine with the scanned leading value
      // independently afterwards, without another serial register scan.
      for (unsigned i = scanEndpoints ? 2 : 1; i < localSize; ++i) {
        unsigned r = base + (reverse ? localSize - 1 - i : i);
        unsigned prev = reverse ? r + 1 : r - 1;
        values[r] = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                   values[prev], values[r]);
      }
      if (scanEndpoints) {
        unsigned first = base + (reverse ? localSize - 1 : 0);
        unsigned last = base + (reverse ? 0 : localSize - 1);
        values[last] = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                      values[first], values[last]);
      }
    }
  }

  // Every tree stage reads the terminal register of a local group. Scanning
  // only each group's leading element and total saves combines without adding
  // shuffles. Reconstruct the interior prefixes after the tree is complete.
  void scanWithinWarps(triton::ScanOp op, const ScanLoweringHelper &helper,
                       ScanValues &values, Value laneId, bool scanEndpoints,
                       ConversionPatternRewriter &rewriter) const {
    if (unsigned mask = warpRegisterLaneMask(helper)) {
      scanWarpRegisterGroups(op, helper, values, laneId, mask, rewriter);
      return;
    }
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    unsigned localSize = helper.getLocalScanSize();
    bool reverse = op.getReverse();
    bool totalsOnly = deferWarpPrefixes(op, helper);
    // Complete independent register groups before advancing to the next one.
    // Split at changes to the register dependency mask so interleaved register
    // and lane bits still read the completed preceding tree stages.
    bool hasShared = helper.getScratchLayout().has_value();
    bool groupWarpRegisters = hasShared || reverse;
    unsigned laneMask =
        hasShared && !reverse ? getLaneScanMask(helper).value_or(0) : 0;
    auto stages = helper.getStages();
    for (unsigned begin = 0; begin < stages.size();) {
      unsigned mask = stages[begin].lower[0] | stages[begin].current[0];
      unsigned end = begin + 1;
      while (end < stages.size() &&
             (stages[end].lower[0] | stages[end].current[0]) == mask)
        ++end;
      unsigned groupSize = groupWarpRegisters ? mask + 1 : values.size();
      SmallVector<unsigned> groupOrder;
      for (unsigned base = 0; base < values.size(); base += groupSize)
        groupOrder.push_back(base);
      auto kReg = StringAttr::get(op.getContext(), "register");
      unsigned axisRegs = getAxisRegisterCount(op, helper);
      if (values.size() / axisRegs >= 4) {
        auto physical = helper.getRegisterOrder().apply(
            LinearLayout::identity1D(values.size(), kReg, kReg));
        llvm::sort(groupOrder, [&](unsigned a, unsigned b) {
          unsigned x = physical.apply({{kReg, a}}).front().second;
          unsigned y = physical.apply({{kReg, b}}).front().second;
          return reverse ? x > y : x < y;
        });
      }
      for (unsigned base : groupOrder) {
        for (const auto &stage : stages.slice(begin, end - begin)) {
          // In logical coordinates the lower-half endpoint is
          // (x & ~((1 << (k + 1)) - 1)) | ((1 << k) - 1).
          // LinearEncodingAttr is a permutation of bits (plus broadcasts), so
          // translate this affine map to one clear/set mask per hardware dim.
          std::array<unsigned, 3> clear, set;
          for (unsigned d = 0; d < 3; ++d) {
            clear[d] = stage.lower[d] | stage.current[d];
            set[d] = reverse ? stage.current[d] : stage.lower[d];
          }
          Value pred;
          if (laneMask) {
            pred = b.icmp_uge(b.and_(laneId, b.i32_val(laneMask)),
                              b.i32_val(stage.current[1]));
          } else if (stage.current[1]) {
            Value bit = b.and_(laneId, b.i32_val(stage.current[1]));
            pred = b.icmp_eq(bit, b.i32_val(reverse ? 0 : stage.current[1]));
          }
          // Every stage reads the previous stage, including when its source
          // register is also one of its destinations. Cache exchanged endpoints
          // shared by multiple destination registers.
          auto previous = values;
          DenseMap<unsigned, SmallVector<Value>> endpoints;
          for (unsigned r = base; r < base + groupSize; ++r) {
            unsigned localIndex = r % localSize;
            if (totalsOnly && localIndex != (reverse ? 0 : localSize - 1))
              continue;
            if (scanEndpoints && localIndex != 0 && localIndex != localSize - 1)
              continue;
            if (stage.current[0] && bool(r & stage.current[0]) == reverse)
              continue;
            unsigned src = (r & ~clear[0]) | set[0];
            auto it = endpoints.find(src);
            if (it == endpoints.end()) {
              SmallVector<Value> endpoint = previous[src];
              if (clear[1]) {
                Value lane = b.or_(b.and_(laneId, b.i32_val(~clear[1])),
                                   b.i32_val(set[1]));
                for (Value &value : endpoint)
                  value = laneMask ? targetInfo.shuffleUp(rewriter, loc, value,
                                                          stage.current[1])
                                   : targetInfo.shuffleIdx(rewriter, loc, value,
                                                           lane);
              }
              it = endpoints.try_emplace(src, std::move(endpoint)).first;
            }
            values[r] =
                combineWithPrefix(op, it->second, previous[r], rewriter, pred);
          }
        }
      }
      begin = end;
    }

    // Combine the scanned leading element with each saved local suffix
    // prefix. Both endpoints already contain their complete warp-local scan.
    if (!scanEndpoints)
      return;
    for (unsigned base = 0; base < values.size(); base += localSize) {
      unsigned first = base + (reverse ? localSize - 1 : 0);
      for (unsigned i = 1; i + 1 < localSize; ++i) {
        unsigned r = base + (reverse ? localSize - 1 - i : i);
        values[r] = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                   values[first], values[r]);
      }
    }
  }

  unsigned getAxisRegisterCount(triton::ScanOp op,
                                const ScanLoweringHelper &helper) const {
    auto kReg = StringAttr::get(op.getContext(), "register");
    const auto &layout = helper.getLayout();
    auto axis = *std::next(layout.getOutDimNames().begin(), op.getAxis());
    return getInputBasisMask(layout, kReg, {axis}) + 1;
  }

  // Distance shuffles require consecutive physical lane bits. Warp-local
  // register groups may follow those bits, but must not be interleaved.
  // A zero mask is valid for a scan with no lane stages.
  std::optional<unsigned>
  getLaneScanMask(const ScanLoweringHelper &helper,
                  bool allowRegisterSuffix = false) const {
    unsigned mask = 0;
    bool registerSuffix = false;
    for (const auto &stage : helper.getStages()) {
      if (stage.current[0] && allowRegisterSuffix) {
        registerSuffix = true;
        continue;
      }
      unsigned bit = stage.current[1];
      if (stage.current[0] || registerSuffix || !bit ||
          (mask && bit != llvm::bit_floor(mask) * 2))
        return std::nullopt;
      mask |= bit;
    }
    return mask;
  }

  unsigned warpRegisterLaneMask(const ScanLoweringHelper &helper) const {
    if (helper.getScratchLayout() || helper.getLocalScanSize() < 2)
      return 0;
    unsigned regMask = helper.getLocalScanSize() - 1;
    for (const auto &stage : helper.getStages())
      regMask |= stage.current[0];
    // With long local prefixes and few groups, the endpoint tree exposes
    // more independent work than propagating one total through every group.
    if (helper.getLocalScanSize() >= 16 &&
        (regMask + 1) / helper.getLocalScanSize() < 4)
      return 0;
    return getLaneScanMask(helper, /*allowRegisterSuffix=*/true).value_or(0);
  }

  void scanWarpRegisterGroups(triton::ScanOp op,
                              const ScanLoweringHelper &helper,
                              ScanValues &values, Value laneId,
                              unsigned laneMask,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    unsigned axisRegs = getAxisRegisterCount(op, helper);
    unsigned localSize = helper.getLocalScanSize();
    unsigned stride = 1u << llvm::countr_zero(laneMask);
    bool reverse = op.getReverse();
    Value lane = b.and_(laneId, b.i32_val(laneMask));
    Value notFirst = b.icmp_ne(lane, b.i32_val(reverse ? laneMask : 0));
    Value lastLane = b.or_(b.and_(laneId, b.i32_val(~laneMask)),
                           b.i32_val(reverse ? 0 : laneMask));
    SmallVector<unsigned> groups;
    for (unsigned parallel = 0; parallel < values.size(); parallel += axisRegs)
      for (unsigned step = 0; step < axisRegs; step += localSize)
        groups.push_back(parallel +
                         (reverse ? axisRegs - localSize - step : step));
    for (unsigned base : groups) {
      unsigned last = base + (reverse ? 0 : localSize - 1);
      auto acc = values[last];
      for (unsigned distance = stride; distance <= laneMask; distance *= 2) {
        auto prefix = acc;
        for (Value &value : prefix)
          value = shufflePrevious(op, value, laneId, distance, rewriter);
        Value pred = reverse ? b.icmp_ule(lane, b.i32_val(laneMask - distance))
                             : b.icmp_uge(lane, b.i32_val(distance));
        acc = combineWithPrefix(op, prefix, acc, rewriter, pred);
      }
      values[last] = acc;
    }
    SmallVector<Value> carry;
    unsigned groupsPerAxis = axisRegs / localSize;
    for (auto [i, base] : llvm::enumerate(groups)) {
      bool first = i % groupsPerAxis == 0;
      unsigned last = base + (reverse ? 0 : localSize - 1);
      addSegmentCarry(op, helper, values, base, localSize,
                      first ? ValueRange{} : ValueRange(carry), {}, laneId,
                      rewriter, notFirst);
      if ((i + 1) % groupsPerAxis) {
        carry = values[last];
        for (Value &value : carry)
          value = targetInfo.shuffleIdx(rewriter, loc, value, lastLane);
      }
    }
  }

  bool deferWarpPrefixes(triton::ScanOp op,
                         const ScanLoweringHelper &helper) const {
    if (!helper.getScratchLayout() || helper.getLocalScanSize() == 1)
      return false;
    if (!getLaneScanMask(helper).has_value())
      return false;
    auto kLane = StringAttr::get(op.getContext(), "lane");
    const auto &layout = helper.getLayout();
    auto axis = *std::next(layout.getOutDimNames().begin(), op.getAxis());
    return getOutputBasisMask(layout, {kLane}, axis) < helper.getSegmentSize();
  }

  Value shufflePrevious(triton::ScanOp op, Value value, Value laneId,
                        unsigned distance,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    return op.getReverse()
               ? targetInfo.shuffleIdx(rewriter, loc, value,
                                       b.add(laneId, b.i32_val(distance)))
               : targetInfo.shuffleUp(rewriter, loc, value, distance);
  }

  // Add the complete carry once. For a total-only warp scan, reconstruct the
  // interior prefixes from the preceding lane's already updated total. An
  // empty carry denotes the first warp-local group. Reuse its lane predicate
  // when the warp scan has already computed one.
  void addSegmentCarry(triton::ScanOp op, const ScanLoweringHelper &helper,
                       ScanValues &values, unsigned base, unsigned count,
                       ValueRange carry, Value pred, Value laneId,
                       ConversionPatternRewriter &rewriter,
                       Value notFirstLane = {}) const {
    if (helper.getScratchLayout() && !deferWarpPrefixes(op, helper)) {
      for (unsigned r = base; r < base + count; ++r)
        values[r] = combineWithPrefix(op, carry, values[r], rewriter, pred);
      return;
    }
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    bool reverse = op.getReverse();
    unsigned last = base + (reverse ? 0 : count - 1);
    if (!carry.empty())
      values[last] = combineWithPrefix(op, carry, values[last], rewriter, pred);
    unsigned laneMask = 0;
    for (const auto &stage : helper.getStages())
      laneMask |= stage.current[1];
    SmallVector<Value> prefix(carry);
    Value interiorPred = pred;
    if (laneMask) {
      unsigned distance = 1u << llvm::countr_zero(laneMask);
      if (!notFirstLane)
        notFirstLane = b.icmp_ne(b.and_(laneId, b.i32_val(laneMask)),
                                 b.i32_val(reverse ? laneMask : 0));
      prefix = values[last];
      for (unsigned i = 0; i < prefix.size(); ++i) {
        Value prev =
            shufflePrevious(op, values[last][i], laneId, distance, rewriter);
        prefix[i] =
            carry.empty() ? prev : b.select(notFirstLane, prev, carry[i]);
      }
      if (carry.empty())
        interiorPred = notFirstLane;
      else if (pred)
        interiorPred = b.or_(pred, notFirstLane);
    }
    for (unsigned j = 1; j < count; ++j) {
      unsigned r = reverse ? base + j : base + count - 1 - j;
      values[r] =
          combineWithPrefix(op, prefix, values[r], rewriter, interiorPred);
    }
  }

  // Regrouping short chains adds carry combines without enough dependency
  // reduction. Long chains benefit even with a single carry stream; reverse
  // scans also benefit when several independent columns hide the extra work.
  // Keep large register sets on the batched schedule to limit live values.
  bool useGroupedCarries(triton::ScanOp op, const ScanLoweringHelper &helper,
                         unsigned numRegisters) const {
    if (!helper.getScratchLayout() || numRegisters > 64)
      return false;
    const auto &layout = helper.getLayout();
    auto axis = *std::next(layout.getOutDimNames().begin(), op.getAxis());
    unsigned axisRegisters = getAxisRegisterCount(op, helper);
    unsigned numSegments = layout.getOutDimSize(axis) / helper.getSegmentSize();
    if (numSegments < 16 ||
        (numSegments < 128 &&
         (!op.getReverse() || numRegisters / axisRegisters < 4)))
      return false;
    auto *ctx = op.getContext();
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    unsigned threadMask = getOutputBasisMask(layout, {kLane, kWarp}, axis) /
                          helper.getSegmentSize();
    // All thread-owned segment bits must be lower than the register bits.
    return llvm::isPowerOf2_32(threadMask + 1);
  }

  // Publish one total per contiguous warp-local segment, then scan those
  // totals in logical order and prepend the exclusive carry to each prefix.
  void scanAcrossWarps(triton::ScanOp op, const ScanLoweringHelper &helper,
                       ScanValues &values, Value laneId, Value warpId,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    const auto &layout = helper.getLayout();
    const auto &scratch = *helper.getScratchLayout();
    auto axis = *std::next(layout.getOutDimNames().begin(), op.getAxis());
    unsigned segmentSize = helper.getSegmentSize();
    unsigned numSegments = layout.getOutDimSize(axis) / segmentSize;
    unsigned numParallel = scratch.getTotalOutDimSize() / numSegments;
    unsigned numOperands = op.getNumOperands();
    bool reverse = op.getReverse();

    auto laneMask = getInputBasisMask(layout, kLane, {axis});
    auto warpMask = getInputBasisMask(layout, kWarp, {axis});
    // Register order puts all axis bits first, sorted by logical significance.
    unsigned axisRegs = getAxisRegisterCount(op, helper);
    unsigned segmentRegs = 1;
    for (const auto &basis : layout.getBases().lookup(kReg))
      if (basis[op.getAxis()] && basis[op.getAxis()] < segmentSize)
        segmentRegs *= 2;

    auto smemBases = storeWarpTotals(op, helper, values, segmentRegs, laneId,
                                     warpId, rewriter);
    b.barrier(triton::gpu::AddrSpace::Local);

    SmallVector<Type> smemTypes;
    for (unsigned i = 0; i < numOperands; ++i)
      smemTypes.push_back(getElementType(op, i));
    auto threadLayout = layout.sublayout({kLane, kWarp}, {axis});
    Value threadSegment = applyLinearLayout(loc, rewriter, threadLayout,
                                            {{kLane, laneId}, {kWarp, warpId}})
                              .front()
                              .second;
    threadSegment =
        b.lshr(threadSegment, b.i32_val(llvm::Log2_32(segmentSize)));
    unsigned threadMask =
        getOutputBasisMask(layout, {kLane, kWarp}, axis) / segmentSize;
    unsigned regSegmentMask =
        getOutputBasisMask(layout, {kReg}, axis) / segmentSize;
    DenseMap<unsigned, unsigned> segmentToReg;
    for (unsigned r = 0; r < axisRegs; r += segmentRegs) {
      unsigned segment = layout
                             .apply({{kReg, r},
                                     {kLane, 0},
                                     {kWarp, 0},
                                     {kBlock, 0}})[op.getAxis()]
                             .second /
                         segmentSize;
      segmentToReg[segment] = r;
    }
    Value parallelLane = b.and_(laneId, b.i32_val(~laneMask));
    Value parallelWarp = b.and_(warpId, b.i32_val(~warpMask));
    Value notFirst =
        b.icmp_ne(threadSegment, b.i32_val(reverse ? threadMask : 0));

    // Interleave independent parallel scans so their shared loads and carry
    // chains can overlap. Each scan still accumulates segments in logical
    // order and consumes a carry as soon as its register group is complete.
    struct ParallelScan {
      Value parallelOffset;
      SmallVector<Value> acc;
      bool prefetched = false;
      DenseMap<unsigned, SmallVector<Value>> carries;
    };
    SmallVector<ParallelScan> parallelScans(values.size() / axisRegs);
    for (auto [parallel, state] : llvm::enumerate(parallelScans))
      state.parallelOffset =
          getScratchOffset(loc, rewriter, scratch, parallel * axisRegs,
                           parallelLane, parallelWarp);

    auto loadTotal = [&](Value parallelOffset, unsigned segment) {
      Value index = b.add(parallelOffset, b.i32_val(segment * numParallel));
      SmallVector<Value> total;
      for (unsigned i = 0; i < numOperands; ++i) {
        Value ptr =
            b.gep(smemBases[i].getType(), smemTypes[i], smemBases[i], index);
        total.push_back(targetInfo.loadShared(rewriter, loc, ptr, smemTypes[i],
                                              b.true_val()));
      }
      return total;
    };
    auto selectCarry = [&](SmallVector<Value> &carry, ValueRange acc,
                           unsigned segment) {
      if (carry.empty()) {
        carry.assign(acc.begin(), acc.end());
        return;
      }
      Value pred = reverse ? b.icmp_ule(threadSegment, b.i32_val(segment))
                           : b.icmp_uge(threadSegment, b.i32_val(segment));
      for (unsigned i = 0; i < numOperands; ++i)
        carry[i] = b.select(pred, acc[i], carry[i]);
    };

    // A register group spans the low segment bits owned by lanes/warps.
    // Keep short scans together: interleaving small batches can hurt backend
    // scheduling and register pressure. For longer scans, expose independent
    // carry chains in batches of at least 64 segments, without splitting a
    // contiguous register group.
    unsigned segmentsPerGroup =
        regSegmentMask ? 1u << llvm::countr_zero(regSegmentMask) : numSegments;
    // When thread-owned segment bits precede register-owned bits, each
    // register group has an independent local prefix. Carry its total across
    // groups rather than serializing every shared load through one accumulator.
    if (useGroupedCarries(op, helper, values.size())) {
      for (unsigned first = 0; first < numSegments; first += segmentsPerGroup) {
        unsigned regSegment =
            reverse ? numSegments - first - segmentsPerGroup : first;
        unsigned reg = segmentToReg.lookup(regSegment);
        for (auto [parallel, state] : llvm::enumerate(parallelScans)) {
          SmallVector<Value> groupAcc, carry;
          unsigned count = std::min(segmentsPerGroup, numSegments - first - 1);
          for (unsigned step = 0; step < count; ++step) {
            unsigned segment =
                reverse ? numSegments - 1 - first - step : first + step;
            auto total = loadTotal(state.parallelOffset, segment);
            groupAcc = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                      groupAcc, total);
            if (step + 1 == segmentsPerGroup)
              continue;
            unsigned next = reverse ? segmentsPerGroup - 2 - step : step + 1;
            selectCarry(carry, groupAcc, next);
          }
          if (!state.acc.empty()) {
            if (carry.empty()) {
              carry = state.acc;
            } else {
              auto combined = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                             state.acc, carry, notFirst);
              for (unsigned i = 0; i < numOperands; ++i)
                carry[i] = b.select(notFirst, combined[i], state.acc[i]);
            }
          }
          if (!carry.empty()) {
            unsigned base = parallel * axisRegs + reg;
            Value pred = first == 0 ? notFirst : Value{};
            addSegmentCarry(op, helper, values, base, segmentRegs, carry, pred,
                            laneId, rewriter);
          }
          if (first + segmentsPerGroup < numSegments)
            state.acc = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                       state.acc, groupAcc);
        }
      }
      return;
    }
    unsigned segmentsPerBatch =
        parallelScans.size() >= 4
            ? segmentsPerGroup
            : std::min(numSegments, std::max(64u, segmentsPerGroup));
    for (unsigned first = 0; first < numSegments; first += segmentsPerBatch) {
      for (auto [parallel, state] : llvm::enumerate(parallelScans)) {
        for (unsigned step = std::max(1u, first);
             step < first + segmentsPerBatch; ++step) {
          unsigned previous = reverse ? numSegments - step : step - 1;
          unsigned current = reverse ? previous - 1 : step;
          unsigned regSegment = current & regSegmentMask;
          unsigned reg = segmentToReg.lookup(regSegment);
          unsigned base = parallel * axisRegs;
          if (!state.prefetched) {
            auto total = loadTotal(state.parallelOffset, previous);
            state.acc = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                       state.acc, total);
          }
          state.prefetched = false;
          auto &carry = state.carries[reg];
          selectCarry(carry, state.acc, current & threadMask);
          unsigned last = reverse ? regSegment : regSegment | threadMask;
          if (current != last)
            continue;
          // Read the group's last total before consuming its carry. Otherwise
          // vectorized shared loads can read this word twice, on either side
          // of the prefix reconstruction.
          if (step + 1 < numSegments) {
            auto nextTotal = loadTotal(state.parallelOffset, current);
            state.acc = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                       state.acc, nextTotal);
            state.prefetched = true;
          }
          Value pred =
              regSegment == (reverse ? regSegmentMask : 0) ? notFirst : Value{};
          addSegmentCarry(op, helper, values, base + reg, segmentRegs, carry,
                          pred, laneId, rewriter);
          state.carries.erase(reg);
        }
      }
    }
    for (const auto &state : parallelScans)
      assert(state.carries.empty() && "all carries must be consumed");
  }

  // Only the terminal element of each segment writes its total. When the
  // layout broadcasts lanes or warps, elect one representative writer.
  SmallVector<Value>
  storeWarpTotals(triton::ScanOp op, const ScanLoweringHelper &helper,
                  const ScanValues &values, unsigned segmentRegs, Value laneId,
                  Value warpId, ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kLane = StringAttr::get(rewriter.getContext(), "lane");
    auto kWarp = StringAttr::get(rewriter.getContext(), "warp");
    const auto &layout = helper.getLayout();
    unsigned segmentLaneMask = 0;
    for (auto [i, basis] : llvm::enumerate(layout.getBases().lookup(kLane)))
      if (basis[op.getAxis()] && basis[op.getAxis()] < helper.getSegmentSize())
        segmentLaneMask |= 1u << i;

    auto smemBases =
        getSmemBases(op, helper.getScratchSizeInElems(), rewriter, targetInfo);
    auto free = layout.getFreeVariableMasks();
    Value representative =
        b.icmp_eq(b.or_(b.and_(laneId, b.i32_val(free.lookup(kLane))),
                        b.and_(warpId, b.i32_val(free.lookup(kWarp)))),
                  b.i32_val(0));
    bool reverse = op.getReverse();
    Value writer = b.and_(representative,
                          b.icmp_eq(b.and_(laneId, b.i32_val(segmentLaneMask)),
                                    b.i32_val(reverse ? 0 : segmentLaneMask)));
    unsigned lastReg = reverse ? 0 : segmentRegs - 1;
    const auto &scratch = *helper.getScratchLayout();
    auto kReg = StringAttr::get(rewriter.getContext(), "register");
    auto kBlock = StringAttr::get(rewriter.getContext(), "block");
    auto offsetDim = *scratch.getOutDimNames().begin();
    unsigned threadMask =
        getOutputBasisMask(scratch, {kLane, kWarp, kBlock}, offsetDim);
    SmallVector<std::pair<unsigned, unsigned>> stores;
    for (unsigned r = lastReg; r < values.size(); r += segmentRegs) {
      unsigned offset =
          scratch.apply({{kReg, r}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}})
              .front()
              .second;
      stores.emplace_back(offset, r);
    }
    // Group adjacent scratch words, even when their original registers were
    // interleaved. Thread-owned offset bits must preserve vector alignment.
    llvm::sort(stores);
    for (unsigned first = 0; first < stores.size();) {
      unsigned width = 1;
      while (width < 4 && first + 2 * width <= stores.size() &&
             !(stores[first].first & (2 * width - 1)) &&
             !(threadMask & (2 * width - 1))) {
        bool contiguous = true;
        for (unsigned j = width; j < 2 * width; ++j)
          contiguous &= stores[first + j].first == stores[first].first + j;
        if (!contiguous)
          break;
        width *= 2;
      }
      Value index = getScratchOffset(loc, rewriter, scratch,
                                     stores[first].second, laneId, warpId);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        Value ptr = b.gep(smemBases[i].getType(), getElementType(op, i),
                          smemBases[i], index);
        SmallVector<Value> packed;
        for (unsigned j = 0; j < width; ++j)
          packed.push_back(values[stores[first + j].second][i]);
        Value value =
            width == 1 ? packed.front() : packLLVector(loc, packed, rewriter);
        targetInfo.storeShared(rewriter, loc, ptr, value, writer);
      }
      first += width;
    }
    return smemBases;
  }

  Value getScratchOffset(Location loc, ConversionPatternRewriter &rewriter,
                         const LinearLayout &scratch, unsigned reg, Value lane,
                         Value warp) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();
    return applyLinearLayout(
               loc, rewriter, scratch,
               {{StringAttr::get(ctx, "register"), b.i32_val(reg)},
                {StringAttr::get(ctx, "lane"), lane},
                {StringAttr::get(ctx, "warp"), warp},
                {StringAttr::get(ctx, "block"), b.i32_val(0)}})
        .front()
        .second;
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
