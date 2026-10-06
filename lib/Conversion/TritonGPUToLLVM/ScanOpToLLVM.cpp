#include "ReduceScanCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
using ScanValues = SmallVector<SmallVector<Value>>;

struct ScanOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ScanOp> {
  ScanOpConversion(LLVMTypeConverter &typeConverter,
                   const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ScanOp>(typeConverter,
                                                                benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  const TargetInfoBase &targetInfo;

  ScanValues reverseValues(triton::ScanOp op, const ScanValues &values,
                           unsigned warpSize,
                           ConversionPatternRewriter &rewriter) const {
    ScanValues result(values.size());
    for (unsigned reg = 0; reg < values.size(); ++reg)
      for (Value value : values[values.size() - 1 - reg])
        result[reg].push_back(
            targetInfo.shuffleXor(rewriter, op.getLoc(), value, warpSize - 1));
    return result;
  }

  SmallVector<Value> combine(triton::ScanOp op, ValueRange prefix,
                             ValueRange values,
                             ConversionPatternRewriter &rewriter,
                             Value pred = {}) const {
    auto result = applyCombineOp(op.getLoc(), rewriter, op.getCombineOp(),
                                 prefix, values, pred);
    if (pred) {
      auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
      for (auto [i, value] : llvm::enumerate(result))
        result[i] = b.select(pred, value, values[i]);
    }
    return result;
  }

  SmallVector<Value> shuffle(triton::ScanOp op, ValueRange values, Value source,
                             ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> result;
    for (Value value : values)
      result.push_back(
          targetInfo.shuffleIdx(rewriter, op.getLoc(), value, source));
    return result;
  }
};

LogicalResult
ScanOpConversion::matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  ScanLoweringHelper helper(op);
  if (!helper.isSupported())
    return op.emitError("unsupported scan layout: scans across CTAs");
  auto loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto kReg = rewriter.getStringAttr("register");
  auto kLane = rewriter.getStringAttr("lane");
  auto kWarp = rewriter.getStringAttr("warp");
  auto kBlock = rewriter.getStringAttr("block");
  auto axis = rewriter.getStringAttr("dim" + std::to_string(op.getAxis()));
  const auto &layout = helper.getLayout();
  auto dims = llvm::to_vector(layout.getOutDimNames());
  auto groups = helper.getThreadGroups();
  unsigned threadSize = helper.getThreadLocalSize();
  unsigned chunkSize = helper.getWarpChunkSize();
  unsigned numLanes = chunkSize / threadSize;
  unsigned numChunks = layout.getOutDimSize(axis) / chunkSize;
  unsigned laneMask = helper.getAxisMask(kLane, chunkSize);
  unsigned regMask = helper.getAxisMask(kReg, threadSize);
  unsigned warpSize = layout.getInDimSize(kLane);
  Value threadId = getThreadId(rewriter, loc);
  Value laneId = b.urem(threadId, b.i32_val(warpSize));
  Value warpId = b.and_(b.udiv(threadId, b.i32_val(warpSize)),
                        b.i32_val(layout.getInDimSize(kWarp) - 1));

  auto terminalReg = [&](unsigned group) { return groups[group].back(); };
  auto reflectIndex = [&](Value index, unsigned mask, unsigned size) -> Value {
    // A complete field reversal is subtraction, as in main. A partial
    // reversal preserves the other layout bits with XOR.
    if (mask == size - 1)
      return b.sub(b.i32_val(mask), index);
    return b.xor_(index, b.i32_val(mask));
  };

  // As in main, reverse the register/lane traversal around a forward scan.
  // The layout gives the remaining axis offset, including interleaved or
  // swizzled warp bits; no layout or physical warp ownership is changed.
  unsigned axisOffset = 0;
  if (op.getReverse()) {
    axisOffset = (layout.getOutDimSize(axis) - 1) ^
                 layout
                     .apply({{kReg, layout.getInDimSize(kReg) - 1},
                             {kLane, warpSize - 1},
                             {kWarp, 0},
                             {kBlock, 0}})[op.getAxis()]
                     .second;
  }

  // Apply main's logarithmic warp scan to each thread chunk's terminal value.
  // A contiguous physical lane range uses shuffle-up. Otherwise invert the
  // lane layout to find the preceding logical lane, preserving other scans.
  auto laneBases = layout.sublayout({kLane}, {axis}).getBases();
  for (auto &basis : laneBases[kLane])
    basis[0] = (basis[0] % chunkSize) / threadSize;
  LinearLayout laneLayout(laneBases, {{axis, numLanes}}, true);
  Value laneIndex =
      applyLinearLayout(loc, rewriter, laneLayout, {{kLane, laneId}})
          .front()
          .second;
  auto laneInverse = laneLayout.pseudoinvert();
  unsigned stride = numLanes == 1 ? 1 : 0;
  const auto &columns = laneBases[kLane];
  for (unsigned bit = 0; bit < columns.size(); ++bit)
    if (columns[bit][0] == 1) {
      stride = 1u << bit;
      for (unsigned i = 0; (1u << i) < numLanes; ++i)
        if (bit + i >= columns.size() || columns[bit + i][0] != (1u << i))
          stride = 0;
      break;
    }
  auto sourceLane = [&](unsigned delta) {
    Value previous = b.sub(laneIndex, b.i32_val(delta));
    previous = b.and_(previous, b.i32_val(numLanes - 1));
    Value mapped =
        applyLinearLayout(loc, rewriter, laneInverse, {{axis, previous}})
            .front()
            .second;
    return Value(b.or_(b.and_(laneId, b.i32_val(~laneMask)), mapped));
  };
  auto shufflePrevious = [&](ValueRange input) {
    if (stride) {
      SmallVector<Value> result;
      for (Value value : input)
        result.push_back(targetInfo.shuffleUp(rewriter, loc, value, stride));
      return result;
    }
    return shuffle(op, input, sourceLane(1), rewriter);
  };

  bool interWarp = helper.hasInterWarpScan();
  unsigned warpReflection =
      op.getReverse() ? helper.getAxisMask(kWarp, layout.getOutDimSize(axis))
                      : 0;
  // Reflect axis-warp slots once for reverse traversal. Regular layouts
  // then read neighboring totals at increasing shared-memory addresses.
  Value scratchWarpId =
      reflectIndex(warpId, warpReflection, layout.getInDimSize(kWarp));
  unsigned scratchReflection = 0;
  if (interWarp && op.getReverse())
    scratchReflection =
        helper.getScratchAddressLayout()
            .apply(
                {{kReg, 0}, {kLane, 0}, {kWarp, warpReflection}, {kBlock, 0}})
            .front()
            .second;
  auto hardware = [&](unsigned reg, Value lane, Value warp) {
    return SmallVector<std::pair<StringAttr, Value>>{{kReg, b.i32_val(reg)},
                                                     {kLane, lane},
                                                     {kWarp, warp},
                                                     {kBlock, b.i32_val(0)}};
  };
  auto constantHardware = [&](unsigned reg, unsigned lane, unsigned warp) {
    return SmallVector<std::pair<StringAttr, int32_t>>{
        {kReg, reg}, {kLane, lane}, {kWarp, warp}, {kBlock, 0}};
  };
  Value notFirstLane = b.icmp_ne(laneIndex, b.i32_val(0));
  auto baseCoords = applyLinearLayout(loc, rewriter, helper.getTotalsLayout(),
                                      hardware(0, laneId, warpId));
  Value threadChunk = baseCoords[op.getAxis()].second;
  auto regAxis = helper.getTotalsLayout().sublayout({kReg}, {axis});
  unsigned varyingMask = 0;
  for (auto dim : {kLane, kWarp})
    for (const auto &basis : helper.getTotalsLayout().getBases().lookup(dim))
      varyingMask |= basis[op.getAxis()];
  unsigned varyingSize = llvm::PowerOf2Ceil(varyingMask + 1);
  unsigned varyingBits = varyingSize - 1;
  threadChunk = reflectIndex(
      threadChunk, (axisOffset / chunkSize) & varyingBits, varyingSize);
  ScanValues values(layout.getInDimSize(kReg));
  for (auto [i, operand] : llvm::enumerate(adaptor.getOperands())) {
    auto unpacked = unpackUniqueTensorElements(loc, operand, rewriter);
    for (auto [reg, value] : llvm::enumerate(unpacked))
      values[reg].push_back(value);
  }

  if (op.getReverse())
    values = reverseValues(op, values, warpSize, rewriter);

  // Scan contiguous elements within a thread. Follow native register emission
  // order, but take each chunk's elements in logical axis order.
  SmallVector<unsigned> groupForReg(values.size()), nextReg(groups.size(), 0);
  ScanValues accumulators(groups.size());
  for (unsigned g = 0; g < groups.size(); ++g)
    for (unsigned reg : groups[g])
      groupForReg[reg] = g;
  for (unsigned nativeReg = 0; nativeReg < values.size(); ++nativeReg) {
    unsigned g = groupForReg[nativeReg];
    unsigned reg = groups[g][nextReg[g]++];
    accumulators[g] = combine(op, accumulators[g], values[reg], rewriter);
    values[reg] = accumulators[g];
  }

  ScanValues totals(groups.size());
  for (unsigned g = 0; g < groups.size(); ++g) {
    auto acc = values[terminalReg(g)];
    for (unsigned delta = 1; delta < numLanes; delta *= 2) {
      SmallVector<Value> incoming;
      if (stride) {
        for (Value value : acc)
          incoming.push_back(
              targetInfo.shuffleUp(rewriter, loc, value, delta * stride));
      } else {
        incoming = shuffle(op, acc, sourceLane(delta), rewriter);
      }
      Value pred = b.icmp_uge(laneIndex, b.i32_val(delta));
      acc = combine(op, incoming, acc, rewriter, pred);
    }
    totals[g] = std::move(acc);
  }

  SmallVector<Value> smemBases;
  SmallVector<Type> smemTypes;
  if (interWarp) {
    smemBases =
        getSmemBases(op, helper.getScratchSizeInElems(), rewriter, targetInfo);
    for (unsigned i = 0; i < op.getNumOperands(); ++i)
      smemTypes.push_back(getElementType(op, i));
    // Store one total per contiguous chunk. All independent scans and all
    // register chunks publish before the single CTA barrier, as in main.
    unsigned laneZeros = 0, warpZeros = 0;
    for (auto [dim, mask] :
         {std::pair{kLane, &laneZeros}, std::pair{kWarp, &warpZeros}})
      for (auto [bit, basis] : llvm::enumerate(layout.getBases().lookup(dim)))
        if (llvm::all_of(basis, [](int32_t x) { return x == 0; }))
          *mask |= 1u << bit;
    Value pred = b.icmp_eq(b.and_(laneId, b.i32_val(laneMask | laneZeros)),
                           b.i32_val(laneMask));
    if (warpZeros)
      pred = b.and_(
          pred, b.icmp_eq(b.and_(warpId, b.i32_val(warpZeros)), b.i32_val(0)));
    for (unsigned g = 0; g < groups.size(); ++g) {
      Value offset =
          applyLinearLayout(loc, rewriter, helper.getScratchAddressLayout(),
                            {{kReg, b.i32_val(terminalReg(g))},
                             {kLane, laneId},
                             {kWarp, scratchWarpId},
                             {kBlock, b.i32_val(0)}})
              .front()
              .second;
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        Value ptr =
            b.gep(smemBases[i].getType(), smemTypes[i], smemBases[i], offset);
        targetInfo.storeShared(rewriter, loc, ptr, totals[g][i], pred);
      }
    }
    b.barrier(triton::gpu::AddrSpace::Local);
  }

  SmallVector<Value> chunkIndices(groups.size());
  auto isAfter = [&](unsigned g, unsigned index) -> Value {
    unsigned reg = regAxis.apply({{kReg, terminalReg(g)}}).front().second;
    unsigned fixed = (reg ^ (axisOffset / chunkSize)) & ~varyingBits;
    if (fixed != (index & ~varyingBits))
      return b.i1_val(fixed > (index & ~varyingBits));
    // Compare only bits that can vary between threads. This shares main's
    // warp-prefix predicate across repeated register chunks.
    if (!chunkIndices[g])
      chunkIndices[g] = b.xor_(threadChunk, b.i32_val(reg & varyingBits));
    return b.icmp_ugt(chunkIndices[g], b.i32_val(index & varyingBits));
  };
  auto finishChunk = [&](unsigned g, ValueRange carry) {
    Value hasCarry;
    if (!carry.empty())
      hasCarry = isAfter(g, 0);
    auto terminal = combine(op, carry, totals[g], rewriter, hasCarry);
    values[terminalReg(g)] = terminal;
    if (groups[g].size() == 1 && !interWarp)
      return terminal;
    auto previous = numLanes > 1 || interWarp ? shufflePrevious(terminal)
                                              : SmallVector<Value>(carry);
    if (groups[g].size() == 1)
      return terminal;
    if (!carry.empty() && (numLanes > 1 || interWarp))
      for (auto [i, value] : llvm::enumerate(previous))
        previous[i] = b.select(notFirstLane, value, carry[i]);
    if (previous.empty())
      return terminal;
    Value pred = hasCarry ? Value(b.or_(hasCarry, notFirstLane)) : notFirstLane;
    for (unsigned i = 1; i < groups[g].size(); ++i) {
      unsigned reg = groups[g][groups[g].size() - 1 - i];
      values[reg] = combine(op, previous, values[reg], rewriter, pred);
    }
    return terminal;
  };

  auto localLayout = layout.sublayout({kReg, kLane}, dims)
                         .removeZeroBasesAlongDim(kReg)
                         .removeZeroBasesAlongDim(kLane);
  auto free = localLayout.getFreeVariableMasks();
  bool oneChunkPerLaneGroup =
      helper.getAxisMask(kLane, layout.getOutDimSize(axis)) == laneMask &&
      triton::gpu::hasPowerOfTwoBases(localLayout) && !free[kReg] &&
      !free[kLane];
  if (numChunks == 1) {
    for (unsigned g = 0; g < groups.size(); ++g)
      finishChunk(g, {});
  } else if (!interWarp && oneChunkPerLaneGroup) {
    // Main's one-warp carry chain: broadcast the completed chunk prefix,
    // instead of separately recomputing the same prefix from raw totals.
    unsigned axisRegs = helper.getAxisMask(kReg, layout.getOutDimSize(axis));
    llvm::MapVector<unsigned, SmallVector<unsigned>> rows;
    for (unsigned g = 0; g < groups.size(); ++g)
      rows[terminalReg(g) & ~axisRegs].push_back(g);
    for (auto &[row, order] : rows)
      llvm::sort(order, [&](unsigned a, unsigned b) {
        return regAxis.apply({{kReg, terminalReg(a)}}).front().second <
               regAxis.apply({{kReg, terminalReg(b)}}).front().second;
      });
    llvm::DenseMap<unsigned, unsigned> nextChunk;
    llvm::DenseMap<unsigned, SmallVector<Value>> prefixes;
    Value lastLane =
        b.or_(b.and_(laneId, b.i32_val(~laneMask)), b.i32_val(laneMask));
    for (unsigned native = 0; native < groups.size(); ++native) {
      unsigned row = terminalReg(native) & ~axisRegs;
      unsigned g = rows[row][nextChunk[row]++];
      auto terminal = finishChunk(g, prefixes[row]);
      prefixes[row] =
          numLanes > 1 ? shuffle(op, terminal, lastLane, rewriter) : terminal;
    }
  } else {
    // Read chunk totals in logical order and retain the exclusive prefix for
    // each consumer. Main's repeated CTA tiles are disjoint ranges of this
    // sequence; interleaved register/lane/warp chunks can have overlapping
    // ranges. Visit each total once per independent scan and finish a chunk
    // as soon as all its possible predecessors have been read.
    unsigned axisRegs = helper.getAxisMask(kReg, layout.getOutDimSize(axis));
    struct Chunk {
      unsigned first, last, row;
      SmallVector<Value> carry;
    };
    struct Row {
      unsigned next = 0;
      SmallVector<Value> acc;
      SmallVector<unsigned> groups;
    };
    SmallVector<Chunk> chunks(groups.size());
    SmallVector<Row> rows;
    llvm::DenseMap<unsigned, unsigned> rowIds;
    llvm::DenseMap<unsigned, unsigned> groupIds;
    for (unsigned g = 0; g < groups.size(); ++g) {
      unsigned reg = terminalReg(g);
      groupIds[reg & ~regMask] = g;
      auto [it, inserted] =
          rowIds.try_emplace(reg & ~(axisRegs | regMask), rows.size());
      if (inserted)
        rows.emplace_back();
      chunks[g] = {numChunks - 1, 0, it->second, {}};
      rows[it->second].groups.push_back(g);
      for (unsigned lane = 0; lane < layout.getInDimSize(kLane); ++lane)
        for (unsigned warp = 0; warp < layout.getInDimSize(kWarp); ++warp) {
          unsigned index =
              helper.getTotalsLayout()
                  .apply(constantHardware(reg, lane, warp))[op.getAxis()]
                  .second;
          index ^= axisOffset / chunkSize;
          chunks[g].first = std::min(chunks[g].first, index);
          chunks[g].last = std::max(chunks[g].last, index);
        }
    }
    auto queryBases = helper.getTotalsLayout().getBases();
    for (auto &[dim, columns] : queryBases)
      for (auto &basis : columns) {
        if (dim == kBlock || (!interWarp && dim == kWarp))
          std::fill(basis.begin(), basis.end(), 0);
        basis[op.getAxis()] = 0;
      }
    for (unsigned bit = 1; bit < numChunks; bit *= 2) {
      std::vector<int32_t> basis(dims.size(), 0);
      basis[op.getAxis()] = bit;
      queryBases[axis].push_back(std::move(basis));
    }
    LinearLayout query(queryBases, helper.getTotalsLayout().getOutDims(),
                       false);
    auto ownerLayout =
        interWarp ? helper.getScratchLayout()
                  : helper.getTotalsLayout().sublayout({kReg, kLane}, dims);
    // Invert only coordinates reachable by this CTA (or warp), rather than
    // demanding a full-tensor inverse for broadcasts or CTA-local views.
    auto lookup = query.invertAndCompose(ownerLayout);
    for (unsigned g = 0; g < groups.size(); ++g) {
      auto &row = rows[chunks[g].row];
      while (row.next <= chunks[g].last) {
        unsigned index = row.next++;
        unsigned logical = index ^ (axisOffset / chunkSize);
        auto queryIndices = hardware(terminalReg(g), laneId,
                                     interWarp ? warpId : Value(b.i32_val(0)));
        queryIndices.push_back({axis, b.i32_val(logical)});
        SmallVector<Value> total;
        if (interWarp) {
          Value offset = applyLinearLayout(loc, rewriter, lookup, queryIndices)
                             .front()
                             .second;
          offset = b.xor_(offset, b.i32_val(scratchReflection));
          for (unsigned i = 0; i < op.getNumOperands(); ++i) {
            Value ptr = b.gep(smemBases[i].getType(), smemTypes[i],
                              smemBases[i], offset);
            total.push_back(targetInfo.loadShared(rewriter, loc, ptr,
                                                  smemTypes[i], b.true_val()));
          }
        } else {
          llvm::SmallSet<unsigned, 32> candidates;
          for (unsigned lane = 0; lane < warpSize; ++lane) {
            auto queryConstants = constantHardware(terminalReg(g), lane, 0);
            queryConstants.push_back({axis, logical});
            auto owner = lookup.apply(queryConstants);
            unsigned reg = llvm::find_if(owner, [&](auto x) {
                             return x.first == kReg;
                           })->second;
            candidates.insert(groupIds.lookup(reg & ~regMask));
          }
          auto owner = applyLinearLayout(loc, rewriter, lookup, queryIndices);
          Value reg = llvm::find_if(owner, [&](auto x) {
                        return x.first == kReg;
                      })->second;
          Value lane = llvm::find_if(owner, [&](auto x) {
                         return x.first == kLane;
                       })->second;
          lane = b.or_(b.and_(lane, b.i32_val(~laneMask)), b.i32_val(laneMask));
          bool needsShuffle = !layout.sublayoutIsZero({kLane}, {axis});
          for (unsigned candidate : candidates) {
            auto incoming = needsShuffle
                                ? shuffle(op, totals[candidate], lane, rewriter)
                                : totals[candidate];
            if (total.empty()) {
              total = incoming;
            } else {
              Value pred =
                  b.icmp_eq(reg, b.i32_val(terminalReg(candidate) & ~regMask));
              for (auto [i, value] : llvm::enumerate(total))
                total[i] = b.select(pred, incoming[i], value);
            }
          }
        }
        row.acc = combine(op, row.acc, total, rewriter);
        for (unsigned consumer : row.groups) {
          auto &chunk = chunks[consumer];
          if (index + 1 == chunk.first ||
              (index >= chunk.first && index < chunk.last)) {
            if (chunk.carry.empty()) {
              chunk.carry = row.acc;
            } else {
              Value pred = isAfter(consumer, index);
              for (auto [i, value] : llvm::enumerate(chunk.carry))
                chunk.carry[i] = b.select(pred, row.acc[i], value);
            }
          }
        }
      }
      finishChunk(g, chunks[g].carry);
    }
  }

  if (op.getReverse())
    values = reverseValues(op, values, warpSize, rewriter);
  SmallVector<Value> results;
  for (unsigned operand = 0; operand < op.getNumOperands(); ++operand) {
    SmallVector<Value> unpacked;
    for (const auto &value : values)
      unpacked.push_back(value[operand]);
    results.push_back(
        packUniqueTensorElements(loc, getTypeConverter(), unpacked, rewriter,
                                 op.getResult()[operand].getType()));
  }
  rewriter.replaceOp(op, results);
  return success();
}
} // namespace

void mlir::triton::populateScanOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ScanOpConversion>(typeConverter, targetInfo, benefit);
}
