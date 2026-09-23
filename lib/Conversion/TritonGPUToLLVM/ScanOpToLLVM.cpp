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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();
    auto kReg = StringAttr::get(ctx, "register");
    const auto &layout = helper.getLayout();
    unsigned numRegs = layout.getInDimSize(kReg);
    unsigned numOperands = op.getNumOperands();
    bool reverse = op.getReverse();

    SmallVector<SmallVector<Value>> values(numRegs);
    for (Value operand : adaptor.getOperands()) {
      auto unpacked = unpackUniqueTensorElements(loc, operand, rewriter);
      unpacked = helper.getRegisterOrder().apply(unpacked);
      for (unsigned r = 0; r < numRegs; ++r)
        values[r].push_back(unpacked[r]);
    }

    // Factor out the consecutive low axis bits owned by registers. Each group
    // is a contiguous logical segment, even when later axis bits belong to
    // registers interleaved with lane/warp bits.
    unsigned localSize = helper.getLocalScanSize();
    // Every tree stage reads the terminal register of a local group. Scan
    // only each group's leading element and total through the entire tree,
    // then reconstruct its interior prefixes from local suffix prefixes.
    // This saves combines without adding shuffles, including for register
    // bits interleaved with lane bits above the local register prefix.
    bool scanEndpoints = localSize > 2 && helper.getStages().size() > 1;
    for (unsigned base = 0; base < numRegs; base += localSize) {
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

    Value threadId = getThreadId(rewriter, loc);
    unsigned warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    Value laneId = b.urem(threadId, b.i32_val(warpSize));
    Value warpId = b.udiv(threadId, b.i32_val(warpSize));
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
      auto sourceReg = [&](unsigned r) { return (r & ~clear[0]) | set[0]; };
      auto sourceLane = [&]() {
        return b.or_(b.and_(laneId, b.i32_val(~clear[1])), b.i32_val(set[1]));
      };

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
      for (unsigned r = 0; r < numRegs; ++r) {
        unsigned localIndex = r % localSize;
        if (scanEndpoints && localIndex != 0 && localIndex != localSize - 1)
          continue;
        if (stage.current[0] && bool(r & stage.current[0]) == reverse)
          continue;
        unsigned src = sourceReg(r);
        auto it = endpoints.find(src);
        if (it == endpoints.end()) {
          SmallVector<Value> endpoint = previous[src];
          if (clear[1]) {
            Value lane = sourceLane();
            for (Value &value : endpoint)
              value = targetInfo.shuffleIdx(rewriter, loc, value, lane);
          }
          it = endpoints.try_emplace(src, std::move(endpoint)).first;
        }
        auto combined = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                       it->second, previous[r], pred);
        for (unsigned i = 0; i < numOperands; ++i)
          values[r][i] =
              pred ? b.select(pred, combined[i], previous[r][i]) : combined[i];
      }
    }
    if (scanEndpoints) {
      for (unsigned base = 0; base < numRegs; base += localSize) {
        unsigned first = base + (reverse ? localSize - 1 : 0);
        for (unsigned i = 1; i + 1 < localSize; ++i) {
          unsigned r = base + (reverse ? localSize - 1 - i : i);
          values[r] = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                     values[first], values[r]);
        }
      }
    }

    if (helper.getScratchLayout())
      scanAcrossWarps(op, helper, values, laneId, warpId, rewriter);

    SmallVector<Value> results;
    auto inverseOrder = helper.getRegisterOrder().inverse();
    for (unsigned i = 0; i < numOperands; ++i) {
      SmallVector<Value> unpacked;
      for (const auto &row : values)
        unpacked.push_back(row[i]);
      unpacked = inverseOrder.apply(unpacked);
      results.push_back(packUniqueTensorElements(loc, getTypeConverter(),
                                                 unpacked, rewriter,
                                                 op.getResult()[i].getType()));
    }
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  void scanAcrossWarps(triton::ScanOp op, ScanLoweringHelper &helper,
                       SmallVector<SmallVector<Value>> &values, Value laneId,
                       Value warpId,
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
    unsigned segmentLaneMask = 0;
    for (const auto &basis : layout.getBases().lookup(kReg))
      if (basis[op.getAxis()] && basis[op.getAxis()] < segmentSize)
        segmentRegs *= 2;
    for (auto [i, basis] : llvm::enumerate(layout.getBases().lookup(kLane)))
      if (basis[op.getAxis()] && basis[op.getAxis()] < segmentSize)
        segmentLaneMask |= 1u << i;

    auto smemBases =
        getSmemBases(op, helper.getScratchSizeInElems(), rewriter, targetInfo);
    SmallVector<Type> smemTypes;
    for (unsigned i = 0; i < numOperands; ++i)
      smemTypes.push_back(getElementType(op, i));
    auto ptr = [&](unsigned i, Value offset) {
      return b.gep(smemBases[i].getType(), smemTypes[i], smemBases[i], offset);
    };
    auto offset = [&](unsigned reg, Value lane, Value warp) {
      return applyLinearLayout(loc, rewriter, scratch,
                               {{kReg, b.i32_val(reg)},
                                {kLane, lane},
                                {kWarp, warp},
                                {kBlock, b.i32_val(0)}})
          .front()
          .second;
    };
    auto free = layout.getFreeVariableMasks();
    Value representative =
        b.icmp_eq(b.or_(b.and_(laneId, b.i32_val(free.lookup(kLane))),
                        b.and_(warpId, b.i32_val(free.lookup(kWarp)))),
                  b.i32_val(0));
    Value writer = b.and_(representative,
                          b.icmp_eq(b.and_(laneId, b.i32_val(segmentLaneMask)),
                                    b.i32_val(reverse ? 0 : segmentLaneMask)));
    unsigned lastReg = reverse ? 0 : segmentRegs - 1;
    for (unsigned r = lastReg; r < values.size(); r += segmentRegs) {
      Value index = offset(r, laneId, warpId);
      for (unsigned i = 0; i < numOperands; ++i)
        targetInfo.storeShared(rewriter, loc, ptr(i, index), values[r][i],
                               writer);
    }
    b.barrier(triton::gpu::AddrSpace::Local);

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
      Value parallelOffset = offset(base, parallelLane, parallelWarp);
      SmallVector<Value> acc;
      DenseMap<unsigned, SmallVector<Value>> carries;
      for (unsigned step = 1; step < numSegments; ++step) {
        unsigned previous = reverse ? numSegments - step : step - 1;
        unsigned current = reverse ? previous - 1 : step;
        Value index = b.add(parallelOffset, b.i32_val(previous * numParallel));
        SmallVector<Value> total;
        for (unsigned i = 0; i < numOperands; ++i)
          total.push_back(targetInfo.loadShared(rewriter, loc, ptr(i, index),
                                                smemTypes[i], b.true_val()));
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
        for (unsigned r = base + reg; r < base + reg + segmentRegs; ++r) {
          auto combined = applyCombineOp(loc, rewriter, op.getCombineOp(),
                                         carry, values[r], pred);
          for (unsigned i = 0; i < numOperands; ++i)
            values[r][i] =
                pred ? b.select(pred, combined[i], values[r][i]) : combined[i];
        }
        carries.erase(reg);
      }
      assert(carries.empty() && "all carries must be consumed");
    }
  }

  const TargetInfoBase &targetInfo;
};
} // namespace

void mlir::triton::populateScanOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ScanOpConversion>(typeConverter, targetInfo, benefit);
}
