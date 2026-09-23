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
    bool scanEndpoints = localSize > 2 && helper.getStages().size() > 1;

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
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    unsigned localSize = helper.getLocalScanSize();
    bool reverse = op.getReverse();
    for (const auto &stage : helper.getStages()) {
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
      if (stage.current[1]) {
        Value bit = b.and_(laneId, b.i32_val(stage.current[1]));
        pred = b.icmp_eq(bit, b.i32_val(reverse ? 0 : stage.current[1]));
      }
      // Every stage reads the previous stage, including when its source
      // register is also one of its destinations. Cache exchanged endpoints
      // shared by multiple destination registers.
      auto previous = values;
      DenseMap<unsigned, SmallVector<Value>> endpoints;
      for (unsigned r = 0; r < values.size(); ++r) {
        unsigned localIndex = r % localSize;
        if (scanEndpoints && localIndex != 0 && localIndex != localSize - 1)
          continue;
        if (stage.current[0] && bool(r & stage.current[0]) == reverse)
          continue;
        unsigned src = (r & ~clear[0]) | set[0];
        auto it = endpoints.find(src);
        if (it == endpoints.end()) {
          SmallVector<Value> endpoint = previous[src];
          if (clear[1]) {
            Value lane =
                b.or_(b.and_(laneId, b.i32_val(~clear[1])), b.i32_val(set[1]));
            for (Value &value : endpoint)
              value = targetInfo.shuffleIdx(rewriter, loc, value, lane);
          }
          it = endpoints.try_emplace(src, std::move(endpoint)).first;
        }
        values[r] =
            combineWithPrefix(op, it->second, previous[r], rewriter, pred);
      }
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

    auto regMask = getInputBasisMask(layout, kReg, {axis});
    auto laneMask = getInputBasisMask(layout, kLane, {axis});
    auto warpMask = getInputBasisMask(layout, kWarp, {axis});
    // Register order puts all axis bits first, sorted by logical significance.
    unsigned axisRegs = regMask + 1;
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

    // Every thread reads the ordered segment totals for its parallel scan.
    // Select its exclusive carry while accumulating, then consume that carry
    // as soon as the last possible segment for a register group is reached.
    // This keeps only one carry live for ordinary blocked layouts, but also
    // handles register/lane/warp axis bits interleaved in any order.
    for (unsigned base = 0; base < values.size(); base += axisRegs) {
      Value parallelOffset = getScratchOffset(loc, rewriter, scratch, base,
                                              parallelLane, parallelWarp);
      SmallVector<Value> acc;
      DenseMap<unsigned, SmallVector<Value>> carries;
      for (unsigned step = 1; step < numSegments; ++step) {
        unsigned previous = reverse ? numSegments - step : step - 1;
        unsigned current = reverse ? previous - 1 : step;
        Value index = b.add(parallelOffset, b.i32_val(previous * numParallel));
        SmallVector<Value> total;
        for (unsigned i = 0; i < numOperands; ++i) {
          Value ptr =
              b.gep(smemBases[i].getType(), smemTypes[i], smemBases[i], index);
          total.push_back(targetInfo.loadShared(rewriter, loc, ptr,
                                                smemTypes[i], b.true_val()));
        }
        acc = applyCombineOp(loc, rewriter, op.getCombineOp(), acc, total);
        unsigned regSegment = current & regSegmentMask;
        unsigned reg = segmentToReg.lookup(regSegment);
        auto &carry = carries[reg];
        if (carry.empty()) {
          carry = acc;
        } else {
          Value pred =
              b.icmp_eq(threadSegment, b.i32_val(current & threadMask));
          for (unsigned i = 0; i < numOperands; ++i)
            carry[i] = b.select(pred, acc[i], carry[i]);
        }
        unsigned last = reverse ? regSegment : regSegment | threadMask;
        if (current != last)
          continue;
        Value pred =
            regSegment == (reverse ? regSegmentMask : 0) ? notFirst : Value{};
        for (unsigned r = base + reg; r < base + reg + segmentRegs; ++r)
          values[r] = combineWithPrefix(op, carry, values[r], rewriter, pred);
        carries.erase(reg);
      }
      assert(carries.empty() && "all carries must be consumed");
    }
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
    for (unsigned r = lastReg; r < values.size(); r += segmentRegs) {
      Value index = getScratchOffset(loc, rewriter, *helper.getScratchLayout(),
                                     r, laneId, warpId);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        Value ptr = b.gep(smemBases[i].getType(), getElementType(op, i),
                          smemBases[i], index);
        targetInfo.storeShared(rewriter, loc, ptr, values[r][i], writer);
      }
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
