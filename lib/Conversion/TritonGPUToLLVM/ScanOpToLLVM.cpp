#include "ReduceScanCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct SharedScanOperand {
  LinearLayout layout;
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
                const LinearLayout &dst,
                const SmallVector<SmallVector<Value>> &values,
                const LLVMTypeConverter *typeConverter,
                const TargetInfoBase &targetInfo, Value storePred = {}) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto shared = getSharedScanLayout(src, op.getAxis());
  auto scratch = getScanScratchConfig(src, dst, op.getElementTypes());
  auto base = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  if (!storePred) {
    storePred = b.true_val();
    auto free = src.getFreeVariableMasks();
    for (auto [dim, id] : SmallVector<std::pair<StringAttr, Value>>{
             {StringAttr::get(ctx, "lane"), laneId},
             {StringAttr::get(ctx, "warp"), warpId}})
      if (free[dim])
        storePred =
            b.and_(storePred,
                   b.icmp_eq(b.and_(id, b.i32_val(free[dim])), b.i32_val(0)));
  }
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
    lowerLdSt(loc, ctx, src.invertAndCompose(shared), stored, storageType,
              operandBase, {}, b.i32_val(0), 0, {}, 0, laneId, warpId, rewriter,
              targetInfo, {}, makeSharedStoreEmitter(targetInfo, storePred));
    operands.push_back({shared, operandBase, storageType});
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
      applyScanCarries(op, helper, values, intraWarpTotals, {}, laneId, warpId,
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

  // Complete lane totals with the inter-warp carry before shifting them to
  // the local prefixes. Each nonterminal register receives one complete carry.
  void completeLaneTotals(triton::ScanOp op, ScanValues &values,
                          const LinearLayout &layout, unsigned numRegs,
                          unsigned segmentSize,
                          ArrayRef<ScanCarry> interWarpCarries, Value laneId,
                          ConversionPatternRewriter &rewriter,
                          unsigned firstSegment = 0) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    std::pair<Value, Value> step;
    if (numRegs > 1)
      step = getLaneScanStep(op, layout, numRegs, segmentSize, laneId, 1,
                             rewriter);
    assert(interWarpCarries.empty() ||
           values.size() >= (firstSegment + interWarpCarries.size()) * numRegs);
    unsigned end = interWarpCarries.empty()
                       ? values.size()
                       : (firstSegment + interWarpCarries.size()) * numRegs;
    for (unsigned base = firstSegment * numRegs; base < end; base += numRegs) {
      unsigned last = base + (op.getReverse() ? 0 : numRegs - 1);
      const ScanCarry *interWarp =
          interWarpCarries.empty()
              ? nullptr
              : &interWarpCarries[base / numRegs - firstSegment];
      if (interWarp)
        values[last] = combineWithPrefix(op, interWarp->values, values[last],
                                         rewriter, interWarp->pred);
      if (numRegs == 1)
        continue;
      auto [lane, pred] = step;
      auto prefix = shuffleValues(loc, values[last], lane, rewriter);
      if (interWarp) {
        // The first logical lane receives only the inter-warp carry.
        for (auto [value, carry] : llvm::zip(prefix, interWarp->values))
          value = b.select(pred, value, carry);
        pred = b.or_(pred, interWarp->pred);
      }
      for (unsigned r = base; r < base + numRegs; ++r)
        if (r != last)
          values[r] = combineWithPrefix(op, prefix, values[r], rewriter, pred);
    }
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

  // Each group contains all lane/warp axis bits. Every lane scans its shared
  // totals and applies its carry before advancing to the next group. Only the
  // running prefix crosses groups.
  void scanSharedTotals(
      triton::ScanOp op, const LinearLayout &layout,
      ArrayRef<SharedScanOperand> operands, Value laneId, Value warpId,
      ConversionPatternRewriter &rewriter,
      llvm::function_ref<void(unsigned, ArrayRef<ScanCarry>)> consume) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    auto axis = StringAttr::get(ctx, "dim" + std::to_string(op.getAxis()));
    unsigned numSegments = layout.getOutDimSize(axis);
    unsigned numRegs = layout.getInDimSize(kReg);
    unsigned groupSize = 1;
    for (auto dim : {kLane, kWarp})
      for (const auto &basis : layout.getBases().lookup(dim))
        groupSize = std::max(groupSize, unsigned(2 * basis[op.getAxis()]));
    bool reverse = op.getReverse();
    unsigned numOperands = op.getNumOperands();
    SmallVector<Value> undef;
    SmallVector<LinearLayout> loadLayouts;
    for (auto [i, operand] : llvm::enumerate(operands)) {
      undef.push_back(LLVM::UndefOp::create(
          rewriter, loc,
          getTypeConverter()->convertType(op.getElementTypes()[i])));
      loadLayouts.push_back(operand.layout.pseudoinvert());
    }

    llvm::MapVector<SmallVector<int32_t>, SmallVector<unsigned>> columns;
    SmallVector<unsigned> registerSegments;
    for (unsigned r = 0; r < numRegs; ++r) {
      auto coords =
          layout.apply({{kReg, r}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      SmallVector<int32_t> column;
      for (auto [dim, coord] : coords)
        if (dim != axis)
          column.push_back(coord);
      columns[column].push_back(r);
      registerSegments.push_back(coords[op.getAxis()].second);
    }
    for (const auto &[column, registers] : columns) {
      llvm::MapVector<unsigned, SmallVector<unsigned>> groups;
      for (unsigned r : registers)
        groups[registerSegments[r] / groupSize].push_back(r);
      SmallVector<Value> acc;
      for (unsigned g = 0; g < groups.size(); ++g) {
        auto &[group, groupRegs] =
            *(groups.begin() + (reverse ? groups.size() - 1 - g : g));
        assert(groupRegs.back() - groupRegs.front() + 1 == groupRegs.size());
        SmallVector<Value> indices;
        SmallVector<Value> predicates;
        SmallVector<Value> carries;
        for (unsigned r : groupRegs) {
          auto coords = applyLinearLayout(loc, rewriter, layout,
                                          {{kReg, b.i32_val(r)},
                                           {kLane, laneId},
                                           {kWarp, warpId},
                                           {kBlock, b.i32_val(0)}});
          Value index = coords[op.getAxis()].second;
          indices.push_back(index);
          predicates.push_back(
              b.icmp_ne(index, b.i32_val(reverse ? numSegments - 1 : 0)));
          llvm::append_range(carries, undef);
        }
        auto coords = applyLinearLayout(loc, rewriter, layout,
                                        {{kReg, b.i32_val(groupRegs.front())},
                                         {kLane, laneId},
                                         {kWarp, warpId},
                                         {kBlock, b.i32_val(0)}});
        for (unsigned step = 0; step < groupSize; ++step) {
          unsigned index =
              group * groupSize + (reverse ? groupSize - 1 - step : step);
          if (!acc.empty())
            for (unsigned r = 0; r < groupRegs.size(); ++r) {
              Value take = b.icmp_eq(indices[r], b.i32_val(index));
              for (unsigned i = 0; i < numOperands; ++i) {
                unsigned slot = r * numOperands + i;
                carries[slot] = b.select(take, acc[i], carries[slot]);
              }
            }
          // No consumer needs the total after the final segment.
          if (index == (reverse ? 0 : numSegments - 1))
            continue;
          coords[op.getAxis()].second = b.i32_val(index);
          SmallVector<Value> incoming;
          for (auto [i, operand] : llvm::enumerate(operands)) {
            Value offset =
                applyLinearLayout(loc, rewriter, loadLayouts[i], coords)
                    .front()
                    .second;
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
        SmallVector<ScanCarry> result;
        for (unsigned r = 0; r < groupRegs.size(); ++r) {
          SmallVector<Value> values(carries.begin() + r * numOperands,
                                    carries.begin() + (r + 1) * numOperands);
          result.push_back({std::move(values), predicates[r]});
        }
        consume(groupRegs.front(), result);
      }
    }
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
    const auto &totalsLayout = *helper.getInterWarpScanLayout();
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
    Value storePred;
    if (segmentLaneMask)
      storePred = b.icmp_eq(b.and_(laneId, b.i32_val(segmentLaneMask)),
                            b.i32_val(reverse ? 0 : segmentLaneMask));
    auto free = sourceLayout.getFreeVariableMasks();
    for (auto [dim, id] : SmallVector<std::pair<StringAttr, Value>>{
             {kLane, laneId}, {StringAttr::get(ctx, "warp"), warpId}})
      if (free[dim]) {
        Value representative =
            b.icmp_eq(b.and_(id, b.i32_val(free[dim])), b.i32_val(0));
        storePred =
            storePred ? b.and_(storePred, representative) : representative;
      }
    auto operands = storeScanTotals(loc, rewriter, op, interWarpLayout,
                                    totalsLayout, transposeValues(totals),
                                    getTypeConverter(), targetInfo, storePred);
    scanSharedTotals(op, interWarpLayout, operands, laneId, warpId, rewriter,
                     [&](unsigned firstSegment, ArrayRef<ScanCarry> carries) {
                       applyScanCarries(op, helper, values, intraWarpTotals,
                                        carries, laneId, warpId, rewriter,
                                        firstSegment);
                     });
  }

  // Include inter-warp carries in the thread totals before fetching the
  // preceding thread's total, so each local prefix receives one complete carry.
  void applyScanCarries(triton::ScanOp op, const ScanLoweringHelper &helper,
                        ScanValues &values, ScanValues &intraWarpTotals,
                        ArrayRef<ScanCarry> interWarpCarries, Value laneId,
                        Value warpId, ConversionPatternRewriter &rewriter,
                        unsigned firstSegment = 0) const {
    auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
    auto kReg = StringAttr::get(op.getContext(), "register");
    unsigned interWarpRegs = 0;
    if (helper.getInterWarpLayout()) {
      const auto &layout = *helper.getInterWarpLayout();
      interWarpRegs = helper.getPermutedLayout().getInDimSize(kReg) /
                      layout.getInDimSize(kReg);
    }
    unsigned firstReg = firstSegment * interWarpRegs;
    unsigned endReg = interWarpCarries.empty()
                          ? values.size()
                          : firstReg + interWarpCarries.size() * interWarpRegs;
    if (!helper.getIntraWarpLayout()) {
      if (!interWarpCarries.empty())
        applySegmentCarries(op, values, interWarpCarries, interWarpRegs,
                            rewriter, /*skipTerminal=*/false, firstSegment);
      return;
    }

    const auto &intraWarpLayout = *helper.getIntraWarpLayout();
    unsigned segmentRegs = helper.getThreadLocalSegmentSize();
    unsigned numSegments = helper.getWarpLocalSegmentSize() / segmentRegs;
    const auto &scanLayout = *helper.getIntraWarpScanLayout();
    auto axis =
        StringAttr::get(op.getContext(), "dim" + std::to_string(op.getAxis()));
    unsigned numRegs =
        scanLayout.sublayout({kReg}, {axis}).getNumConsecutiveInOut();
    auto completedTotals = intraWarpTotals;
    completeLaneTotals(op, completedTotals, scanLayout, numRegs, numSegments,
                       interWarpCarries, laneId, rewriter, firstSegment);
    convertScanValues(op, completedTotals, scanLayout, intraWarpLayout,
                      rewriter);

    // Terminal prefixes are complete; only the other registers need carries.
    for (unsigned r = firstReg / segmentRegs; r < endReg / segmentRegs; ++r)
      values[r * segmentRegs + (op.getReverse() ? 0 : segmentRegs - 1)] =
          completedTotals[r];
    if (segmentRegs == 1)
      return;

    auto carries =
        getSegmentCarries(op, completedTotals, intraWarpLayout, intraWarpLayout,
                          numSegments, laneId, warpId, rewriter);
    if (!interWarpCarries.empty()) {
      for (unsigned r = firstReg / segmentRegs; r < endReg / segmentRegs; ++r) {
        auto &carry = carries[r];
        const auto &interWarpCarry =
            interWarpCarries[r * segmentRegs / interWarpRegs - firstSegment];
        // The first thread segment has no preceding local total. Use the
        // inter-warp carry there, and leave the scan boundary untouched.
        for (auto [value, interWarpValue] :
             llvm::zip(carry.values, interWarpCarry.values))
          value = b.select(carry.pred, value, interWarpValue);
        carry.pred = b.or_(carry.pred, interWarpCarry.pred);
      }
    }
    applySegmentCarries(
        op, values,
        ArrayRef<ScanCarry>(carries).slice(firstReg / segmentRegs,
                                           (endReg - firstReg) / segmentRegs),
        segmentRegs, rewriter, /*skipTerminal=*/true, firstReg / segmentRegs);
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
    bool reverse = op.getReverse();
    // All required totals are present in this warp.
    auto inverse = totalsLayout.pseudoinvert().sublayout(
        llvm::to_vector(totalsLayout.getOutDimNames()), {kReg, kLane});
    unsigned numSegments = segmentLayout.getOutDimSize(axis);
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
      auto carry =
          shuffleCarry(loc, totals, candidates, srcReg, srcLane, rewriter);
      carries.push_back({std::move(carry), pred});
    }
    return carries;
  }

  void applySegmentCarries(triton::ScanOp op, ScanValues &values,
                           ArrayRef<ScanCarry> carries, unsigned segmentRegs,
                           ConversionPatternRewriter &rewriter,
                           bool skipTerminal = false,
                           unsigned firstSegment = 0) const {
    assert(values.size() >= (firstSegment + carries.size()) * segmentRegs);
    unsigned begin = skipTerminal && op.getReverse() ? 1 : 0;
    unsigned end = segmentRegs - (skipTerminal && !op.getReverse());
    for (auto [r, carry] : llvm::enumerate(carries))
      for (unsigned j = begin; j < end; ++j) {
        unsigned reg = (firstSegment + r) * segmentRegs + j;
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
