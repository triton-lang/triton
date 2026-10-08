#include "ReduceScanCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct SharedScanOperand {
  Value base;
  Type type;
};

// Map shared offsets to consecutive segment totals, followed by independent
// scan coordinates. Block bases preserve CTA ownership. For a 1D i32 scan,
// totals T0, T1, T2, ... occupy byte offsets 0, 4, 8, ... .
LinearLayout getSharedScanLayout(const LinearLayout &src, unsigned axis) {
  auto *ctx = src.getInDimNames().begin()->getContext();
  auto kOffset = StringAttr::get(ctx, "offset");
  auto kBlock = StringAttr::get(ctx, "block");
  SmallVector<StringAttr> localDims = {StringAttr::get(ctx, "register"),
                                       StringAttr::get(ctx, "lane"),
                                       StringAttr::get(ctx, "warp")};
  auto outDims = llvm::to_vector(src.getOutDimNames());
  auto order = llvm::to_vector(llvm::seq<unsigned>(outDims.size()));
  std::rotate(order.begin(), order.begin() + axis, order.begin() + axis + 1);
  std::vector<std::vector<int32_t>> offsets;
  for (unsigned dim : order) {
    unsigned mask = getOutputBasisMask(src, localDims, outDims[dim]);
    for (unsigned bit = 1; bit < src.getOutDimSize(outDims[dim]); bit <<= 1)
      if (mask & bit) {
        std::vector<int32_t> basis(outDims.size(), 0);
        basis[dim] = bit;
        offsets.push_back(std::move(basis));
      }
  }
  return LinearLayout(
      {{kOffset, std::move(offsets)}, {kBlock, src.getBases().lookup(kBlock)}},
      outDims);
}

SmallVector<SharedScanOperand>
storeScanTotals(Location loc, ConversionPatternRewriter &rewriter,
                triton::ScanOp op, const LinearLayout &src,
                const LinearLayout &shared,
                const SmallVector<SmallVector<Value>> &values,
                const LLVMTypeConverter *typeConverter,
                const TargetInfoBase &targetInfo, Value storePred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto scratch = getScanScratchConfig(src, op.getElementTypes());
  auto base = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  auto toShared = src.invertAndCompose(shared);
  SmallVector<SharedScanOperand> operands;
  // Keep every operand in its own part of the allocation, then synchronize
  // once after storing all segment totals.
  for (auto [i, input] : llvm::enumerate(values)) {
    Type type = typeConverter->convertType(op.getElementTypes()[i]);
    Type storageType = type;
    auto stored = input;
    if (isa<LLVM::LLVMPointerType>(type)) {
      storageType = rewriter.getI64Type();
      for (auto &v : stored)
        v = b.ptrtoint(storageType, v);
    } else if (type.getIntOrFloatBitWidth() < 8) {
      storageType = rewriter.getI8Type();
      for (auto &v : stored)
        v = b.zext(storageType, v);
    }
    Value operandBase = b.gep(base.getType(), rewriter.getI8Type(), base,
                              b.i32_val(scratch.offsets[i]));
    lowerLdSt(loc, ctx, toShared, stored, storageType, operandBase, {},
              b.i32_val(0), 0, {}, 0, laneId, warpId, rewriter, targetInfo, {},
              makeSharedStoreEmitter(targetInfo, storePred));
    operands.push_back({operandBase, storageType});
  }
  targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
  return operands;
}

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

    // Keep native thread prefixes in place and communicate only segment totals.
    permuteRegisters(values, helper.getRegisterOrder());
    scanWithinThreads(op, values, helper.getThreadLocalSegmentSize(), rewriter);
    ScanValues intraWarpTotals;
    if (helper.getIntraWarpLayout())
      intraWarpTotals = scanWithinWarps(op, helper, values, laneId, rewriter);
    if (helper.getInterWarpLayout())
      scanAcrossWarps(op, helper, values, intraWarpTotals, laneId, warpId,
                      rewriter);
    else
      completeScanGroup(op, helper, values, intraWarpTotals, {}, laneId, warpId,
                        rewriter);
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

  // Find the preceding logical lane at the requested scan distance.
  std::pair<Value, Value>
  getLaneScanStep(triton::ScanOp op, const LinearLayout &layout,
                  unsigned numRegs, unsigned segmentSize, Value laneId,
                  unsigned offset, ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kLane = StringAttr::get(op.getContext(), "lane");
    unsigned axis = op.getAxis();
    bool reverse = op.getReverse();
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
    return {lane, pred};
  }

  // Compute inclusive lane totals. Local prefixes are completed after the
  // inter-warp carry is available.
  void scanLaneTotals(triton::ScanOp op, ScanValues &values,
                      const LinearLayout &layout, unsigned numRegs,
                      unsigned segmentSize, Value laneId,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    unsigned numLanes = segmentSize / numRegs;
    if (numLanes == 1)
      return;
    SmallVector<std::pair<Value, Value>> rounds;
    for (unsigned offset = 1; offset < numLanes; offset *= 2)
      rounds.push_back(getLaneScanStep(op, layout, numRegs, segmentSize, laneId,
                                       offset, rewriter));
    for (unsigned base = 0; base < values.size(); base += numRegs) {
      unsigned last = base + (op.getReverse() ? 0 : numRegs - 1);
      auto acc = values[last];
      for (auto [lane, pred] : rounds) {
        auto incoming = shuffleValues(loc, acc, lane, rewriter);
        acc = combineWithPrefix(op, incoming, acc, rewriter, pred);
      }
      values[last] = acc;
    }
  }

  // Complete one warp-local segment's totals and exclusive thread carries.
  // Its first logical lane takes the incoming inter-warp carry.
  ScanValues completeLaneTotals(triton::ScanOp op, ScanValues &values,
                                const LinearLayout &layout,
                                unsigned segmentSize, const ScanCarry *carry,
                                Value laneId, bool needCarries,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    unsigned numRegs = values.size();
    unsigned last = op.getReverse() ? 0 : numRegs - 1;
    if (carry)
      values[last] = combineWithPrefix(op, carry->values, values[last],
                                       rewriter, carry->pred);
    if (numRegs == 1 && !needCarries)
      return {};

    auto [lane, pred] =
        getLaneScanStep(op, layout, numRegs, segmentSize, laneId, 1, rewriter);
    auto prefix = shuffleValues(loc, values[last], lane, rewriter);
    if (carry) {
      for (auto [value, incoming] : llvm::zip(prefix, carry->values))
        value = b.select(pred, value, incoming);
      pred = b.or_(pred, carry->pred);
    }
    for (unsigned r = 0; r < numRegs; ++r)
      if (r != last)
        values[r] = combineWithPrefix(op, prefix, values[r], rewriter, pred);

    ScanValues carries;
    if (needCarries)
      for (unsigned r = 0; r < numRegs; ++r) {
        bool first = r == (op.getReverse() ? numRegs - 1 : 0);
        carries.push_back(first ? prefix
                                : values[op.getReverse() ? r + 1 : r - 1]);
      }
    return carries;
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

  // Store segment totals, then compute the carries in their consuming lanes.
  void scanAcrossWarps(triton::ScanOp op, const ScanLoweringHelper &helper,
                       ScanValues &values, ScanValues &intraWarpTotals,
                       Value laneId, Value warpId,
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
    bool reverse = op.getReverse();
    auto kReg = StringAttr::get(ctx, "register");
    unsigned segmentRegs =
        sourceLayout.getInDimSize(kReg) / interWarpLayout.getInDimSize(kReg);
    unsigned segmentLaneMask =
        getSegmentMask(sourceLayout, kLane, op.getAxis(), segmentSize);

    // Only terminal lanes store complete segment totals.
    auto totals =
        extractSegmentTotals(intraWarpTotals.empty() ? values : intraWarpTotals,
                             segmentRegs, reverse);
    Value storePred = b.icmp_eq(b.and_(laneId, b.i32_val(segmentLaneMask)),
                                b.i32_val(reverse ? 0 : segmentLaneMask));
    auto free = sourceLayout.getFreeVariableMasks();
    for (auto [dim, id] : SmallVector<std::pair<StringAttr, Value>>{
             {kLane, laneId}, {StringAttr::get(ctx, "warp"), warpId}})
      if (free[dim]) {
        Value representative =
            b.icmp_eq(b.and_(id, b.i32_val(free[dim])), b.i32_val(0));
        storePred = b.and_(storePred, representative);
      }
    auto sharedLayout = getSharedScanLayout(interWarpLayout, op.getAxis());
    auto operands = storeScanTotals(loc, rewriter, op, interWarpLayout,
                                    sharedLayout, transposeValues(totals),
                                    getTypeConverter(), targetInfo, storePred);
    unsigned valuesPerSegment =
        values.size() / interWarpLayout.getInDimSize(kReg);
    unsigned totalsPerSegment =
        intraWarpTotals.size() / interWarpLayout.getInDimSize(kReg);
    // Each group contains all lane/warp axis bits. Every lane scans its shared
    // totals and applies its carry before advancing to the next group. Only the
    // running prefix crosses groups.
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    unsigned numSegments = interWarpLayout.getOutDimSize(axis);
    unsigned numRegs = interWarpLayout.getInDimSize(kReg);
    unsigned groupSize = 1;
    for (auto dim : {kLane, kWarp})
      for (const auto &basis : interWarpLayout.getBases().lookup(dim))
        groupSize = std::max(groupSize, unsigned(2 * basis[op.getAxis()]));
    SmallVector<Value> undef;
    for (Type type : op.getElementTypes())
      undef.push_back(LLVM::UndefOp::create(
          rewriter, loc, getTypeConverter()->convertType(type)));

    auto independentDims = llvm::to_vector(interWarpLayout.getOutDimNames());
    independentDims.erase(independentDims.begin() + op.getAxis());
    auto scanBaseLayout =
        interWarpLayout
            .sublayout(llvm::to_vector(interWarpLayout.getInDimNames()),
                       independentDims)
            .invertAndCompose(sharedLayout.sublayout(
                llvm::to_vector(sharedLayout.getInDimNames()),
                independentDims));
    auto axisRegisters = interWarpLayout.sublayout({kReg}, {axis});
    unsigned regsPerScan =
        axisRegisters.removeZeroBasesAlongDim(kReg).getInDimSize(kReg);
    unsigned regsPerGroup = axisRegisters.resizeOutDim(axis, groupSize)
                                .removeZeroBasesAlongDim(kReg)
                                .getInDimSize(kReg);
    for (unsigned column = 0; column < numRegs; column += regsPerScan) {
      Value scanBase = applyLinearLayout(loc, rewriter, scanBaseLayout,
                                         {{kReg, b.i32_val(column)},
                                          {kLane, laneId},
                                          {kWarp, warpId},
                                          {kBlock, b.i32_val(0)}})
                           .front()
                           .second;
      // The scan axis occupies the low offset bits. All operands share this
      // element offset, with separate bases and storage types.
      SmallVector<Value> acc;
      for (unsigned g = 0; g < regsPerScan; g += regsPerGroup) {
        unsigned firstReg =
            column + (reverse ? regsPerScan - g - regsPerGroup : g);
        auto groupRegs = llvm::seq(firstReg, firstReg + regsPerGroup);
        unsigned firstIndex = interWarpLayout
                                  .apply({{kReg, firstReg},
                                          {kLane, 0},
                                          {kWarp, 0},
                                          {kBlock, 0}})[op.getAxis()]
                                  .second;
        SmallVector<Value> indices;
        SmallVector<ScanCarry> carries;
        for (unsigned r : groupRegs) {
          auto coords = applyLinearLayout(loc, rewriter, interWarpLayout,
                                          {{kReg, b.i32_val(r)},
                                           {kLane, laneId},
                                           {kWarp, warpId},
                                           {kBlock, b.i32_val(0)}});
          Value index = coords[op.getAxis()].second;
          indices.push_back(index);
          carries.push_back(
              {undef,
               b.icmp_ne(index, b.i32_val(reverse ? numSegments - 1 : 0))});
        }
        for (unsigned step = 0; step < groupSize; ++step) {
          unsigned index = firstIndex + (reverse ? groupSize - 1 - step : step);
          if (!acc.empty())
            for (unsigned r = 0; r < groupRegs.size(); ++r) {
              Value take = b.icmp_eq(indices[r], b.i32_val(index));
              for (auto [carry, prefix] : llvm::zip(carries[r].values, acc))
                carry = b.select(take, prefix, carry);
            }
          // No consumer needs the total after the final segment.
          if (index == (reverse ? 0 : numSegments - 1))
            continue;
          Value offset = b.add(scanBase, b.i32_val(index));
          SmallVector<Value> incoming;
          for (auto [i, operand] : llvm::enumerate(operands)) {
            Value ptr = b.gep(operand.base.getType(), operand.type,
                              operand.base, offset);
            Value value = targetInfo.loadShared(rewriter, loc, ptr,
                                                operand.type, b.true_val());
            Type type =
                getTypeConverter()->convertType(op.getElementTypes()[i]);
            if (type != operand.type)
              value = isa<LLVM::LLVMPointerType>(type)
                          ? Value(b.inttoptr(type, value))
                          : Value(b.trunc(type, value));
            incoming.push_back(value);
          }
          acc = applyCombineOp(loc, rewriter, op.getCombineOp(), acc, incoming);
        }
        auto groupValues = MutableArrayRef(values).slice(
            firstReg * valuesPerSegment, carries.size() * valuesPerSegment);
        auto groupTotals = ArrayRef(intraWarpTotals)
                               .slice(firstReg * totalsPerSegment,
                                      carries.size() * totalsPerSegment);
        completeScanGroup(op, helper, groupValues, groupTotals, carries, laneId,
                          warpId, rewriter);
      }
    }
  }

  // Restrict register ownership to one group and compact its coordinates.
  // Omitted register bits select other groups or independent scans.
  static LinearLayout getGroupLayout(const LinearLayout &layout,
                                     unsigned numRegs) {
    auto kReg = StringAttr::get(layout.getInDimNames().begin()->getContext(),
                                "register");
    auto group = layout.resizeInDim(kReg, numRegs);
    auto bases = group.getBases();
    auto inDims = llvm::to_vector(group.getInDimNames());
    auto outDims = llvm::to_vector(group.getOutDimNames());
    for (auto [d, dim] : llvm::enumerate(outDims)) {
      unsigned mask = getOutputBasisMask(group, inDims, dim);
      for (auto &[inDim, dimBases] : bases)
        for (auto &basis : dimBases)
          if (basis[d])
            basis[d] = 1u << llvm::popcount(mask & (basis[d] - 1));
    }
    return LinearLayout(std::move(bases), outDims);
  }

  // Include inter-warp carries in the thread totals before fetching the
  // preceding thread's total, so each local prefix receives one complete carry.
  void completeScanGroup(triton::ScanOp op, const ScanLoweringHelper &helper,
                         MutableArrayRef<SmallVector<Value>> values,
                         ArrayRef<SmallVector<Value>> intraWarpTotals,
                         ArrayRef<ScanCarry> interWarpCarries, Value laneId,
                         Value warpId,
                         ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    auto kReg = StringAttr::get(op.getContext(), "register");
    if (!helper.getIntraWarpLayout()) {
      unsigned interWarpRegs = interWarpCarries.empty()
                                   ? 0
                                   : values.size() / interWarpCarries.size();
      for (auto [r, carry] : llvm::enumerate(interWarpCarries))
        for (unsigned j = 0; j < interWarpRegs; ++j) {
          auto &value = values[r * interWarpRegs + j];
          value =
              combineWithPrefix(op, carry.values, value, rewriter, carry.pred);
        }
      return;
    }

    unsigned segmentRegs = helper.getThreadLocalSegmentSize();
    unsigned numSegments = helper.getWarpLocalSegmentSize() / segmentRegs;
    auto axis =
        StringAttr::get(op.getContext(), "dim" + std::to_string(op.getAxis()));
    unsigned numRegs = helper.getIntraWarpScanLayout()
                           ->sublayout({kReg}, {axis})
                           .getNumConsecutiveInOut();
    // Each completion handles one warp-local segment. Other register bits
    // select separate segments or independent scans.
    auto intraWarpLayout =
        getGroupLayout(*helper.getIntraWarpLayout(), numRegs);
    auto scanLayout = getGroupLayout(*helper.getIntraWarpScanLayout(), numRegs);
    auto kLane = StringAttr::get(op.getContext(), "lane");
    auto kWarp = StringAttr::get(op.getContext(), "warp");
    auto kBlock = StringAttr::get(op.getContext(), "block");
    for (unsigned base = 0; base < intraWarpTotals.size(); base += numRegs) {
      const ScanCarry *carry = interWarpCarries.empty()
                                   ? nullptr
                                   : &interWarpCarries[base / numRegs];
      auto totals = llvm::to_vector(intraWarpTotals.slice(base, numRegs));
      auto prefixes =
          completeLaneTotals(op, totals, scanLayout, numSegments, carry, laneId,
                             segmentRegs > 1, rewriter);
      convertScanValues(op, totals, scanLayout, intraWarpLayout, rewriter);
      if (!prefixes.empty())
        convertScanValues(op, prefixes, scanLayout, intraWarpLayout, rewriter);

      for (unsigned r = 0; r < numRegs; ++r) {
        auto local = values.slice((base + r) * segmentRegs, segmentRegs);
        unsigned terminal = op.getReverse() ? 0 : segmentRegs - 1;
        local[terminal] = totals[r];
        if (segmentRegs == 1)
          continue;
        auto coords = applyLinearLayout(op.getLoc(), rewriter, intraWarpLayout,
                                        {{kReg, b.i32_val(r)},
                                         {kLane, laneId},
                                         {kWarp, warpId},
                                         {kBlock, b.i32_val(0)}});
        Value index =
            b.and_(coords[op.getAxis()].second, b.i32_val(numSegments - 1));
        Value pred =
            b.icmp_ne(index, b.i32_val(op.getReverse() ? numSegments - 1 : 0));
        if (carry)
          pred = b.or_(pred, carry->pred);
        for (unsigned j = 0; j < segmentRegs; ++j)
          if (j != terminal)
            local[j] =
                combineWithPrefix(op, prefixes[r], local[j], rewriter, pred);
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
