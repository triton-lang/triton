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

  // Scan contiguous register groups, then merge them in logical bit order.
  // Higher register bits may be interleaved with lane bits. Each tree stage
  // reads the preceding stage's endpoints, preserving noncommutative order.
  void scanWithinWarps(triton::ScanOp op, const ScanLoweringHelper &helper,
                       ScanValues &values, Value laneId,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    bool reverse = op.getReverse();
    unsigned localSize = helper.getLocalScanSize();
    for (unsigned base = 0; base < values.size(); base += localSize) {
      for (unsigned i = 1; i < localSize; ++i) {
        unsigned r = base + (reverse ? localSize - 1 - i : i);
        unsigned prev = reverse ? r + 1 : r - 1;
        values[r] = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                   values[prev], values[r]);
      }
    }
    for (const auto &stage : helper.getStages()) {
      // In logical coordinates the lower-half endpoint is
      // (x & ~((1 << (k + 1)) - 1)) | ((1 << k) - 1).
      // LinearEncodingAttr is a permutation of bits (plus broadcasts), so
      // translate this affine map to one clear/set mask per hardware dim.
      unsigned clearReg = stage.lower[0] | stage.current[0];
      unsigned setReg = reverse ? stage.current[0] : stage.lower[0];
      unsigned clearLane = stage.lower[1] | stage.current[1];
      unsigned setLane = reverse ? stage.current[1] : stage.lower[1];
      Value pred;
      if (stage.current[1]) {
        Value bit = b.and_(laneId, b.i32_val(stage.current[1]));
        pred = b.icmp_eq(bit, b.i32_val(reverse ? 0 : stage.current[1]));
      }
      auto previous = values;
      DenseMap<unsigned, SmallVector<Value>> endpoints;
      for (unsigned r = 0; r < values.size(); ++r) {
        if (stage.current[0] && bool(r & stage.current[0]) == reverse)
          continue;
        unsigned src = (r & ~clearReg) | setReg;
        auto it = endpoints.find(src);
        if (it == endpoints.end()) {
          SmallVector<Value> endpoint = previous[src];
          if (clearLane) {
            Value lane = b.or_(b.and_(laneId, b.i32_val(~clearLane)),
                               b.i32_val(setLane));
            for (Value &value : endpoint)
              value = targetInfo.shuffleIdx(rewriter, loc, value, lane);
          }
          it = endpoints.try_emplace(src, std::move(endpoint)).first;
        }
        values[r] =
            combineWithPrefix(op, it->second, previous[r], rewriter, pred);
      }
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

    // Reuse the same ordered scan, including register stages when the complete
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
