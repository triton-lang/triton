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
    scanWithinWarps(op, helper, values, laneId, rewriter);
    if (helper.getSegmentLayout())
      scanSegmentTotals(op, helper, values, laneId, warpId, rewriter);
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

  // Scan in logical axis order: accumulate register groups sequentially and
  // scan lane groups with shuffle-up distances 1, 2, 4, ... . Register and lane
  // groups may alternate in a linear layout.
  void scanWithinWarps(triton::ScanOp op, const ScanLoweringHelper &helper,
                       ScanValues &values, Value laneId,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    const auto &layout = helper.getLayout();
    const auto &regBases = layout.getBases().lookup(kReg);
    const auto &laneBases = layout.getBases().lookup(kLane);
    unsigned axis = op.getAxis();
    bool reverse = op.getReverse();

    // Each group of this size already contains its local prefixes. Track the
    // registers and lanes holding the group so we can read its terminal value.
    unsigned size = 1;
    unsigned numRegs = 1;
    unsigned laneMask = 0;
    auto terminal = [&](unsigned base) {
      auto result = values[base + (reverse ? 0 : numRegs - 1)];
      if (laneMask) {
        Value lane = b.or_(b.and_(laneId, b.i32_val(~laneMask)),
                           b.i32_val(reverse ? 0 : laneMask));
        for (Value &value : result)
          value = targetInfo.shuffleIdx(rewriter, loc, value, lane);
      }
      return result;
    };

    while (size < helper.getSegmentSize()) {
      unsigned count = 1;
      while (
          size * count < helper.getSegmentSize() &&
          llvm::any_of(
              regBases,
              [&](const auto &basis) { return basis[axis] == size * count; }))
        count *= 2;
      if (count > 1) {
        // Registers are already sorted by axis significance. Prepend each
        // preceding group's terminal prefix to the next group's values.
        for (unsigned base = 0; base < values.size(); base += numRegs * count) {
          for (unsigned i = 1; i < count; ++i) {
            unsigned cur = base + (reverse ? count - 1 - i : i) * numRegs;
            unsigned prev = reverse ? cur + numRegs : cur - numRegs;
            auto prefix = terminal(prev);
            for (unsigned r = cur; r < cur + numRegs; ++r)
              values[r] = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                         prefix, values[r]);
          }
        }
        numRegs *= count;
        size *= count;
        continue;
      }

      // Collect the next consecutive logical lane bits. Their physical lane
      // bits can be permuted or separated by bits for independent scans.
      SmallVector<unsigned> laneBits;
      unsigned groupMask = 0;
      while (size * count < helper.getSegmentSize()) {
        auto it = llvm::find_if(laneBases, [&](const auto &basis) {
          return basis[axis] == size * count;
        });
        if (it == laneBases.end())
          break;
        unsigned bit = std::distance(laneBases.begin(), it);
        laneBits.push_back(bit);
        groupMask |= 1u << bit;
        count *= 2;
      }
      assert(count > 1 && "warp-local axis bits belong to registers or lanes");
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
          // Shift in logical lane order, then restore the physical lane bits.
          // Boundary sources wrap; the combine predicate excludes them.
          Value index = b.add(laneIndex, b.i32_val(reverse ? offset : -offset));
          Value lane = b.and_(laneId, b.i32_val(~groupMask));
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
        return reverse ? b.icmp_ult(laneIndex, b.i32_val(count - offset))
                       : b.icmp_uge(laneIndex, b.i32_val(offset));
      };
      for (unsigned base = 0; base < values.size(); base += numRegs) {
        // Scan just the group totals, retaining the local prefixes below.
        auto acc = terminal(base);
        for (unsigned offset = 1; offset < count; offset *= 2)
          acc = combineWithPrefix(op, shuffle(acc, offset), acc, rewriter,
                                  hasPrefix(offset));
        if (size == 1) {
          values[base] = std::move(acc);
          continue;
        }
        // The preceding total is an exclusive carry for this group's local
        // prefixes. Skip the first group without assuming an identity value.
        auto prefix = shuffle(acc, 1);
        Value pred = hasPrefix(1);
        for (unsigned r = base; r < base + numRegs; ++r) {
          if (!laneMask && r == base + (reverse ? 0 : numRegs - 1))
            values[r] = acc;
          else
            values[r] =
                combineWithPrefix(op, prefix, values[r], rewriter, pred);
        }
      }
      laneMask |= groupMask;
      size *= count;
    }
  }

  void scanSegmentTotals(triton::ScanOp op, const ScanLoweringHelper &helper,
                         ScanValues &values, Value laneId, Value warpId,
                         ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = op.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    auto kBlock = StringAttr::get(ctx, "block");
    const auto &layout = helper.getLayout();
    const auto &segments = *helper.getSegmentLayout();
    const auto &totalsLayout = *helper.getWarpTotalsLayout();
    auto axis = *std::next(layout.getOutDimNames().begin(), op.getAxis());
    bool reverse = op.getReverse();
    unsigned segmentRegs = 1;
    unsigned segmentLaneMask = 0;
    for (auto basis : layout.getBases().lookup(kReg))
      if (basis[op.getAxis()] && basis[op.getAxis()] < helper.getSegmentSize())
        segmentRegs *= 2;
    for (auto [i, basis] : llvm::enumerate(layout.getBases().lookup(kLane)))
      if (basis[op.getAxis()] && basis[op.getAxis()] < helper.getSegmentSize())
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
    operands = convertLayoutValues(loc, rewriter, op, segments, totalsLayout,
                                   operands, getTypeConverter(), targetInfo);
    ScanValues totals(operands.front().size());
    for (unsigned r = 0; r < totals.size(); ++r)
      for (const auto &operand : operands)
        totals[r].push_back(operand[r]);

    // Reuse the same ordered scan, including register groups when the complete
    // sequence is longer than the available lanes.
    ScanLoweringHelper totalsHelper(totalsLayout, op.getAxis());
    permuteRegisters(totals, totalsHelper.getRegisterOrder());
    scanWithinWarps(op, totalsHelper, totals, laneId, rewriter);
    permuteRegisters(totals, totalsHelper.getRegisterOrder().inverse());

    // Read the preceding segment's inclusive total as an exclusive carry.
    // All segment totals are present in this warp, so only register selects
    // and lane shuffles are needed, even when the sequence spans registers.
    auto inverse = totalsLayout.pseudoinvert();
    unsigned numSegments = segments.getOutDimSize(axis);
    for (unsigned r = 0; r < values.size() / segmentRegs; ++r) {
      auto coords = applyLinearLayout(loc, rewriter, segments,
                                      {{kReg, b.i32_val(r)},
                                       {kLane, laneId},
                                       {kWarp, warpId},
                                       {kBlock, b.i32_val(0)}});
      Value segment = coords[op.getAxis()].second;
      Value pred = b.icmp_ne(segment, b.i32_val(reverse ? numSegments - 1 : 0));
      // Wrapping is harmless at the boundary: pred guards the combine, with
      // no assumed identity and no out-of-range source register or lane.
      coords[op.getAxis()].second =
          b.and_(b.add(segment, b.i32_val(reverse ? 1 : -1)),
                 b.i32_val(numSegments - 1));
      auto src = applyLinearLayout(loc, rewriter, inverse, coords);
      Value srcReg = src[0].second;
      Value srcLane = src[1].second;
      // Enumerate only the registers reachable by this segment's threads.
      // This also handles the borrow/carry at a register sequence boundary.
      llvm::SmallSetVector<unsigned, 8> candidates;
      for (unsigned warp = 0; warp < segments.getInDimSize(kWarp); ++warp) {
        for (unsigned lane = 0; lane < segments.getInDimSize(kLane); ++lane) {
          auto coordinates = segments.apply(
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
