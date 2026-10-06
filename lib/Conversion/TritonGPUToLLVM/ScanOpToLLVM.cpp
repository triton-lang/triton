#include "ReduceScanCommon.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct ScanOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ScanOp> {
  // Values are indexed by register, then by combiner operand.
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
      return op.emitError("unsupported scan layout: scans across CTAs");
    auto indices = getScanIndices(op, helper, rewriter);
    auto loc = op.getLoc();
    auto kReg = rewriter.getStringAttr("register");
    ScanValues values(helper.getLayout().getInDimSize(kReg));
    for (auto [i, operand] : llvm::enumerate(adaptor.getOperands())) {
      auto unpacked = unpackUniqueTensorElements(loc, operand, rewriter);
      for (auto [reg, value] : llvm::enumerate(unpacked))
        values[reg].push_back(value);
    }

    if (op.getReverse())
      values = reverseValues(op, helper, values, rewriter);

    scanWithinThreads(op, helper, values, rewriter);
    auto totals = scanWithinWarps(op, helper, indices, values, rewriter);
    if (helper.hasInterWarpScan())
      scanAcrossWarps(op, helper, indices, values, totals, rewriter);

    if (op.getReverse())
      values = reverseValues(op, helper, values, rewriter);
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

private:
  // Generated coordinates reused by the scan stages.
  struct ScanIndices {
    Value laneId, warpId, laneIndex, scratchWarpId, notFirstLane, threadChunk;
    LinearLayout laneInverse;
    unsigned stride = 0, axisOffset = 0, scratchReflection = 0, varyingBits = 0;
    SmallVector<Value> chunkIndices;
  };

  static SmallVector<std::pair<StringAttr, Value>>
  getHardwareCoordinates(triton::ScanOp op, unsigned reg, Value lane,
                         Value warp, ConversionPatternRewriter &rewriter) {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    return {{rewriter.getStringAttr("register"), b.i32_val(reg)},
            {rewriter.getStringAttr("lane"), lane},
            {rewriter.getStringAttr("warp"), warp},
            {rewriter.getStringAttr("block"), b.i32_val(0)}};
  }

  static SmallVector<std::pair<StringAttr, int32_t>>
  getHardwareCoordinates(unsigned reg, unsigned lane, unsigned warp,
                         ConversionPatternRewriter &rewriter) {
    return {{rewriter.getStringAttr("register"), reg},
            {rewriter.getStringAttr("lane"), lane},
            {rewriter.getStringAttr("warp"), warp},
            {rewriter.getStringAttr("block"), 0}};
  }

  static Value reflectIndex(TritonLLVMOpBuilder &b, Value index, unsigned mask,
                            unsigned size) {
    // A complete field reversal is subtraction, as in main. A partial
    // reversal preserves the other layout bits with XOR.
    if (mask == size - 1)
      return b.sub(b.i32_val(mask), index);
    return b.xor_(index, b.i32_val(mask));
  }

  SmallVector<Value> combineWithPrefix(triton::ScanOp op, ValueRange prefix,
                                       ValueRange values,
                                       ConversionPatternRewriter &rewriter,
                                       Value pred = {}) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto result =
        applyCombineOp(loc, rewriter, op.getCombineOp(), prefix, values, pred);
    if (pred)
      for (auto [value, original] : llvm::zip(result, values))
        value = b.select(pred, value, original);
    return result;
  }

  SmallVector<Value> shuffleValues(triton::ScanOp op, ValueRange values,
                                   Value source,
                                   ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    SmallVector<Value> result;
    for (Value value : values)
      result.push_back(targetInfo.shuffleIdx(rewriter, loc, value, source));
    return result;
  }

  ScanValues reverseValues(triton::ScanOp op, const ScanLoweringHelper &helper,
                           const ScanValues &values,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto kLane = rewriter.getStringAttr("lane");
    const auto &layout = helper.getLayout();
    unsigned warpSize = layout.getInDimSize(kLane);
    ScanValues result(values.size());
    for (unsigned reg = 0; reg < values.size(); ++reg)
      for (Value value : values[values.size() - 1 - reg])
        result[reg].push_back(
            targetInfo.shuffleXor(rewriter, loc, value, warpSize - 1));
    return result;
  }

  ScanIndices getScanIndices(triton::ScanOp op,
                             const ScanLoweringHelper &helper,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kReg = rewriter.getStringAttr("register");
    auto kLane = rewriter.getStringAttr("lane");
    auto kWarp = rewriter.getStringAttr("warp");
    auto kBlock = rewriter.getStringAttr("block");
    auto axis = rewriter.getStringAttr("dim" + std::to_string(op.getAxis()));
    const auto &layout = helper.getLayout();
    auto groups = helper.getThreadGroups();
    unsigned chunkSize = helper.getWarpChunkSize();
    unsigned warpSize = layout.getInDimSize(kLane);
    bool interWarp = helper.hasInterWarpScan();
    ScanIndices indices;
    Value threadId = getThreadId(rewriter, loc);
    indices.laneId = b.urem(threadId, b.i32_val(warpSize));
    indices.warpId = b.and_(b.udiv(threadId, b.i32_val(warpSize)),
                            b.i32_val(layout.getInDimSize(kWarp) - 1));

    // As in main, reverse the register/lane traversal around a forward scan.
    // The layout gives the remaining axis offset, including interleaved or
    // swizzled warp bits; no layout or physical warp ownership is changed.
    indices.axisOffset = helper.getAxisOffset();
    // Physical lane -> position within a chunk, in thread-local range units.
    unsigned threadSize = helper.getThreadLocalSize();
    unsigned numLanes = chunkSize / threadSize;
    auto laneBases = layout.sublayout({kLane}, {axis}).getBases();
    for (auto &basis : laneBases[kLane])
      basis[0] = (basis[0] % chunkSize) / threadSize;
    LinearLayout laneLayout(laneBases, {{axis, numLanes}}, true);
    indices.laneIndex =
        applyLinearLayout(loc, rewriter, laneLayout, {{kLane, indices.laneId}})
            .front()
            .second;
    indices.laneInverse = laneLayout.pseudoinvert();
    // Use shuffle-up when logical neighbors have a fixed physical stride.
    // Otherwise leave stride zero and use the inverse layout for lane lookup.
    if (numLanes == 1)
      indices.stride = 1;
    else {
      const auto &columns = laneBases[kLane];
      for (unsigned bit = 0; bit < columns.size(); ++bit)
        if (columns[bit][0] == 1) {
          indices.stride = 1u << bit;
          for (unsigned i = 0; (1u << i) < numLanes; ++i)
            if (bit + i >= columns.size() || columns[bit + i][0] != (1u << i)) {
              indices.stride = 0;
              break;
            }
          break;
        }
    }

    unsigned warpReflection =
        op.getReverse() ? helper.getAxisMask(kWarp, layout.getOutDimSize(axis))
                        : 0;
    // Reflect axis-warp slots once for reverse traversal. Regular layouts
    // then read neighboring totals at increasing shared-memory addresses.
    indices.scratchWarpId = reflectIndex(b, indices.warpId, warpReflection,
                                         layout.getInDimSize(kWarp));
    if (interWarp && op.getReverse())
      indices.scratchReflection =
          helper.getScratchAddressLayout()
              .apply(
                  {{kReg, 0}, {kLane, 0}, {kWarp, warpReflection}, {kBlock, 0}})
              .front()
              .second;
    indices.notFirstLane = b.icmp_ne(indices.laneIndex, b.i32_val(0));
    auto baseCoords =
        applyLinearLayout(loc, rewriter, helper.getTotalsLayout(),
                          getHardwareCoordinates(op, 0, indices.laneId,
                                                 indices.warpId, rewriter));
    indices.threadChunk = baseCoords[op.getAxis()].second;
    unsigned varyingMask = 0;
    for (auto dim : {kLane, kWarp})
      for (const auto &basis : helper.getTotalsLayout().getBases().lookup(dim))
        varyingMask |= basis[op.getAxis()];
    unsigned varyingSize = llvm::PowerOf2Ceil(varyingMask + 1);
    indices.varyingBits = varyingSize - 1;
    indices.threadChunk = reflectIndex(
        b, indices.threadChunk,
        (indices.axisOffset / chunkSize) & indices.varyingBits, varyingSize);

    indices.chunkIndices.resize(groups.size());
    return indices;
  }

  SmallVector<Value>
  shufflePrevious(triton::ScanOp op, const ScanLoweringHelper &helper,
                  ScanIndices &indices, ValueRange values, unsigned delta,
                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kLane = rewriter.getStringAttr("lane");
    auto axis = rewriter.getStringAttr("dim" + std::to_string(op.getAxis()));
    unsigned chunkSize = helper.getWarpChunkSize();
    unsigned numLanes = helper.getWarpChunkSize() / helper.getThreadLocalSize();
    unsigned laneMask = helper.getAxisMask(kLane, chunkSize);
    if (indices.stride) {
      SmallVector<Value> result;
      for (Value value : values)
        result.push_back(
            targetInfo.shuffleUp(rewriter, loc, value, delta * indices.stride));
      return result;
    }
    Value previous = b.sub(indices.laneIndex, b.i32_val(delta));
    previous = b.and_(previous, b.i32_val(numLanes - 1));
    Value mapped = applyLinearLayout(loc, rewriter, indices.laneInverse,
                                     {{axis, previous}})
                       .front()
                       .second;
    Value lane = b.or_(b.and_(indices.laneId, b.i32_val(~laneMask)), mapped);
    return shuffleValues(op, values, lane, rewriter);
  }

  void scanWithinThreads(triton::ScanOp op, const ScanLoweringHelper &helper,
                         ScanValues &values,
                         ConversionPatternRewriter &rewriter) const {
    auto groups = helper.getThreadGroups();
    // Scan contiguous elements within a thread. Follow native register emission
    // order, but take each chunk's elements in logical axis order.
    auto groupForReg = helper.getRegisterGroups();
    SmallVector<unsigned> nextReg(groups.size(), 0);
    ScanValues accumulators(groups.size());
    for (unsigned nativeReg = 0; nativeReg < values.size(); ++nativeReg) {
      unsigned g = groupForReg[nativeReg];
      unsigned reg = groups[g][nextReg[g]++];
      accumulators[g] =
          combineWithPrefix(op, accumulators[g], values[reg], rewriter);
      values[reg] = accumulators[g];
    }
  }

  ScanValues scanWithinWarps(triton::ScanOp op,
                             const ScanLoweringHelper &helper,
                             ScanIndices &indices, ScanValues &values,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto groups = helper.getThreadGroups();
    unsigned numLanes = helper.getWarpChunkSize() / helper.getThreadLocalSize();
    bool interWarp = helper.hasInterWarpScan();
    ScanValues totals(groups.size());
    for (unsigned g = 0; g < groups.size(); ++g) {
      auto acc = values[groups[g].back()];
      for (unsigned delta = 1; delta < numLanes; delta *= 2) {
        auto incoming =
            shufflePrevious(op, helper, indices, acc, delta, rewriter);
        Value pred = b.icmp_uge(indices.laneIndex, b.i32_val(delta));
        acc = combineWithPrefix(op, incoming, acc, rewriter, pred);
      }
      totals[g] = std::move(acc);
    }

    if (!interWarp)
      applyWarpCarries(op, helper, indices, values, totals, rewriter);
    return totals;
  }

  Value isAfterChunk(triton::ScanOp op, const ScanLoweringHelper &helper,
                     ScanIndices &indices, unsigned g, unsigned index,
                     ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto groups = helper.getThreadGroups();
    unsigned chunkSize = helper.getWarpChunkSize();
    unsigned reg = helper.getChunkIndex(groups[g].back());
    unsigned fixed =
        (reg ^ (indices.axisOffset / chunkSize)) & ~indices.varyingBits;
    if (fixed != (index & ~indices.varyingBits))
      return b.i1_val(fixed > (index & ~indices.varyingBits));
    // Compare only bits that can vary between threads. This shares main's
    // warp-prefix predicate across repeated register chunks.
    if (!indices.chunkIndices[g])
      indices.chunkIndices[g] =
          b.xor_(indices.threadChunk, b.i32_val(reg & indices.varyingBits));
    return b.icmp_ugt(indices.chunkIndices[g],
                      b.i32_val(index & indices.varyingBits));
  }

  SmallVector<Value>
  applyChunkCarry(triton::ScanOp op, const ScanLoweringHelper &helper,
                  ScanIndices &indices, ScanValues &values,
                  const ScanValues &totals, unsigned g, ValueRange carry,
                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto groups = helper.getThreadGroups();
    unsigned numLanes = helper.getWarpChunkSize() / helper.getThreadLocalSize();
    bool interWarp = helper.hasInterWarpScan();
    Value hasCarry;
    if (!carry.empty())
      hasCarry = isAfterChunk(op, helper, indices, g, 0, rewriter);
    auto terminal = combineWithPrefix(op, carry, totals[g], rewriter, hasCarry);
    values[groups[g].back()] = terminal;
    if (groups[g].size() == 1 && !interWarp)
      return terminal;
    auto previous =
        numLanes > 1 || interWarp
            ? shufflePrevious(op, helper, indices, terminal, 1, rewriter)
            : SmallVector<Value>(carry);
    if (groups[g].size() == 1)
      return terminal;
    if (!carry.empty() && (numLanes > 1 || interWarp))
      for (auto [value, prefix] : llvm::zip(previous, carry))
        value = b.select(indices.notFirstLane, value, prefix);
    if (previous.empty())
      return terminal;
    Value pred = hasCarry ? Value(b.or_(hasCarry, indices.notFirstLane))
                          : indices.notFirstLane;
    for (unsigned i = 1; i < groups[g].size(); ++i) {
      unsigned reg = groups[g][groups[g].size() - 1 - i];
      values[reg] =
          combineWithPrefix(op, previous, values[reg], rewriter, pred);
    }
    return terminal;
  }

  SmallVector<Value>
  loadWarpTotal(triton::ScanOp op, const ScanLoweringHelper &helper,
                ScanIndices &indices, const LinearLayout &lookup,
                ArrayRef<unsigned> groupForReg, const ScanValues &totals,
                unsigned sourceReg, unsigned logical,
                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kReg = rewriter.getStringAttr("register");
    auto kLane = rewriter.getStringAttr("lane");
    auto axis = rewriter.getStringAttr("dim" + std::to_string(op.getAxis()));
    const auto &layout = helper.getLayout();
    auto groups = helper.getThreadGroups();
    unsigned threadSize = helper.getThreadLocalSize();
    unsigned chunkSize = helper.getWarpChunkSize();
    unsigned laneMask = helper.getAxisMask(kLane, chunkSize);
    unsigned regMask = helper.getAxisMask(kReg, threadSize);
    unsigned warpSize = layout.getInDimSize(kLane);
    auto coords = getHardwareCoordinates(op, sourceReg, indices.laneId,
                                         Value(b.i32_val(0)), rewriter);
    coords.push_back({axis, b.i32_val(logical)});
    SmallVector<Value> total;
    llvm::SmallSet<unsigned, 32> candidates;
    for (unsigned lane = 0; lane < warpSize; ++lane) {
      auto queryConstants =
          getHardwareCoordinates(sourceReg, lane, 0, rewriter);
      queryConstants.push_back({axis, logical});
      auto owner = lookup.apply(queryConstants);
      unsigned reg = owner[0].second;
      candidates.insert(groupForReg[reg & ~regMask]);
    }
    auto owner = applyLinearLayout(loc, rewriter, lookup, coords);
    Value reg = owner[0].second;
    Value lane = owner[1].second;
    lane = b.or_(b.and_(lane, b.i32_val(~laneMask)), b.i32_val(laneMask));
    bool needsShuffle = !layout.sublayoutIsZero({kLane}, {axis});
    for (unsigned candidate : candidates) {
      auto incoming = needsShuffle
                          ? shuffleValues(op, totals[candidate], lane, rewriter)
                          : totals[candidate];
      if (total.empty()) {
        total = incoming;
      } else {
        Value pred =
            b.icmp_eq(reg, b.i32_val(groups[candidate].back() & ~regMask));
        for (auto [value, source] : llvm::zip(total, incoming))
          value = b.select(pred, source, value);
      }
    }
    return total;
  }

  void scanChunkTotals(
      triton::ScanOp op, const ScanLoweringHelper &helper, ScanIndices &indices,
      ScanValues &values, const ScanValues &totals,
      llvm::function_ref<SmallVector<Value>(unsigned, unsigned)> loadTotal,
      ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kReg = rewriter.getStringAttr("register");
    auto axis = rewriter.getStringAttr("dim" + std::to_string(op.getAxis()));
    const auto &layout = helper.getLayout();
    auto groups = helper.getThreadGroups();
    unsigned threadSize = helper.getThreadLocalSize();
    unsigned chunkSize = helper.getWarpChunkSize();
    unsigned regMask = helper.getAxisMask(kReg, threadSize);
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
    for (unsigned g = 0; g < groups.size(); ++g) {
      unsigned reg = groups[g].back();
      auto [it, inserted] =
          rowIds.try_emplace(reg & ~(axisRegs | regMask), rows.size());
      if (inserted)
        rows.emplace_back();
      auto [first, last] = helper.getChunkBounds(reg);
      chunks[g] = {first, last, it->second, {}};
      rows[it->second].groups.push_back(g);
    }

    for (unsigned g = 0; g < groups.size(); ++g) {
      auto &row = rows[chunks[g].row];
      while (row.next <= chunks[g].last) {
        unsigned index = row.next++;
        unsigned logical = index ^ (indices.axisOffset / chunkSize);
        auto total = loadTotal(g, logical);
        row.acc = combineWithPrefix(op, row.acc, total, rewriter);
        for (unsigned consumer : row.groups) {
          auto &chunk = chunks[consumer];
          if (index + 1 == chunk.first ||
              (index >= chunk.first && index < chunk.last)) {
            if (chunk.carry.empty()) {
              chunk.carry = row.acc;
            } else {
              Value pred =
                  isAfterChunk(op, helper, indices, consumer, index, rewriter);
              for (auto [value, prefix] : llvm::zip(chunk.carry, row.acc))
                value = b.select(pred, prefix, value);
            }
          }
        }
      }
      applyChunkCarry(op, helper, indices, values, totals, g, chunks[g].carry,
                      rewriter);
    }
  }

  void applyWarpCarries(triton::ScanOp op, const ScanLoweringHelper &helper,
                        ScanIndices &indices, ScanValues &values,
                        const ScanValues &totals,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kReg = rewriter.getStringAttr("register");
    auto kLane = rewriter.getStringAttr("lane");
    auto axis = rewriter.getStringAttr("dim" + std::to_string(op.getAxis()));
    const auto &layout = helper.getLayout();
    auto dims = llvm::to_vector(layout.getOutDimNames());
    auto groups = helper.getThreadGroups();
    unsigned chunkSize = helper.getWarpChunkSize();
    unsigned numLanes = helper.getWarpChunkSize() / helper.getThreadLocalSize();
    unsigned numChunks = layout.getOutDimSize(axis) / chunkSize;
    unsigned laneMask = helper.getAxisMask(kLane, chunkSize);
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
        applyChunkCarry(op, helper, indices, values, totals, g, {}, rewriter);
    } else if (oneChunkPerLaneGroup) {
      // Main's one-warp carry chain: broadcast the completed chunk prefix,
      // instead of separately recomputing the same prefix from raw totals.
      unsigned axisRegs = helper.getAxisMask(kReg, layout.getOutDimSize(axis));
      llvm::MapVector<unsigned, SmallVector<unsigned>> rows;
      for (unsigned g = 0; g < groups.size(); ++g)
        rows[groups[g].back() & ~axisRegs].push_back(g);
      for (auto &[row, order] : rows)
        llvm::sort(order, [&](unsigned a, unsigned b) {
          return helper.getChunkIndex(groups[a].back()) <
                 helper.getChunkIndex(groups[b].back());
        });
      llvm::DenseMap<unsigned, unsigned> nextChunk;
      llvm::DenseMap<unsigned, SmallVector<Value>> prefixes;
      Value lastLane = b.or_(b.and_(indices.laneId, b.i32_val(~laneMask)),
                             b.i32_val(laneMask));
      for (unsigned native = 0; native < groups.size(); ++native) {
        unsigned row = groups[native].back() & ~axisRegs;
        unsigned g = rows[row][nextChunk[row]++];
        auto terminal = applyChunkCarry(op, helper, indices, values, totals, g,
                                        prefixes[row], rewriter);
        prefixes[row] = numLanes > 1
                            ? shuffleValues(op, terminal, lastLane, rewriter)
                            : terminal;
      }
    } else {
      auto lookup = helper.getChunkLookup();
      auto groupForReg = helper.getRegisterGroups();
      scanChunkTotals(
          op, helper, indices, values, totals,
          [&](unsigned g, unsigned logical) {
            return loadWarpTotal(op, helper, indices, lookup, groupForReg,
                                 totals, groups[g].back(), logical, rewriter);
          },
          rewriter);
    }
  }

  void scanAcrossWarps(triton::ScanOp op, const ScanLoweringHelper &helper,
                       ScanIndices &indices, ScanValues &values,
                       const ScanValues &totals,
                       ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto kLane = rewriter.getStringAttr("lane");
    auto kWarp = rewriter.getStringAttr("warp");
    auto axis = rewriter.getStringAttr("dim" + std::to_string(op.getAxis()));
    const auto &layout = helper.getLayout();
    auto groups = helper.getThreadGroups();
    unsigned chunkSize = helper.getWarpChunkSize();
    unsigned laneMask = helper.getAxisMask(kLane, chunkSize);
    SmallVector<Value> smemBases;
    SmallVector<Type> smemTypes;
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
    Value pred =
        b.icmp_eq(b.and_(indices.laneId, b.i32_val(laneMask | laneZeros)),
                  b.i32_val(laneMask));
    if (warpZeros)
      pred =
          b.and_(pred, b.icmp_eq(b.and_(indices.warpId, b.i32_val(warpZeros)),
                                 b.i32_val(0)));
    for (unsigned g = 0; g < groups.size(); ++g) {
      Value offset =
          applyLinearLayout(
              loc, rewriter, helper.getScratchAddressLayout(),
              getHardwareCoordinates(op, groups[g].back(), indices.laneId,
                                     indices.scratchWarpId, rewriter))
              .front()
              .second;
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        Value ptr =
            b.gep(smemBases[i].getType(), smemTypes[i], smemBases[i], offset);
        targetInfo.storeShared(rewriter, loc, ptr, totals[g][i], pred);
      }
    }
    b.barrier(triton::gpu::AddrSpace::Local);
    auto lookup = helper.getChunkLookup();
    scanChunkTotals(
        op, helper, indices, values, totals,
        [&](unsigned g, unsigned logical) {
          auto coords = getHardwareCoordinates(
              op, groups[g].back(), indices.laneId, indices.warpId, rewriter);
          coords.push_back({axis, b.i32_val(logical)});
          SmallVector<Value> total;
          Value offset =
              applyLinearLayout(loc, rewriter, lookup, coords).front().second;
          offset = b.xor_(offset, b.i32_val(indices.scratchReflection));
          for (unsigned i = 0; i < op.getNumOperands(); ++i) {
            Value ptr = b.gep(smemBases[i].getType(), smemTypes[i],
                              smemBases[i], offset);
            total.push_back(targetInfo.loadShared(rewriter, loc, ptr,
                                                  smemTypes[i], b.true_val()));
          }
          return total;
        },
        rewriter);
  }
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateScanOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ScanOpConversion>(typeConverter, targetInfo, benefit);
}
