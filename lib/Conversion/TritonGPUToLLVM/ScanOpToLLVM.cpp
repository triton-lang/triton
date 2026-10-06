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
    Value laneId, warpId, laneIndex, notFirstLane, baseChunkIndex;
    LinearLayout laneInverse;
    unsigned stride = 0, remainingReverseMask = 0, chunkSpan = 1;
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
    auto kLane = rewriter.getStringAttr("lane");
    auto kWarp = rewriter.getStringAttr("warp");
    auto axis = rewriter.getStringAttr("dim" + std::to_string(op.getAxis()));
    const auto &layout = helper.getLayout();
    unsigned chunkSize = helper.getChunkSize();
    unsigned warpSize = layout.getInDimSize(kLane);
    ScanIndices indices;
    Value threadId = getThreadId(rewriter, loc);
    indices.laneId = b.urem(threadId, b.i32_val(warpSize));
    indices.warpId = b.and_(b.udiv(threadId, b.i32_val(warpSize)),
                            b.i32_val(layout.getInDimSize(kWarp) - 1));

    // As in main, reverse the register/lane traversal around a forward scan.
    // The layout gives the remaining reverse mask, including interleaved or
    // swizzled warp bits. Layout and physical warp ownership are preserved.
    indices.remainingReverseMask = helper.getRemainingReverseMask();
    // Physical lane -> position within a warp-local chunk, in thread-local
    // group units.
    unsigned groupSize = helper.getGroupSize();
    unsigned numLanes = chunkSize / groupSize;
    auto laneLayout =
        layout.sublayout({kLane}, {axis})
            .resizeOutDim(axis, chunkSize)
            .reshapeOuts({{rewriter.getStringAttr("element"), groupSize},
                          {axis, numLanes}})
            .sublayout({kLane}, {axis});
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
      const auto &columns = laneLayout.getBases().lookup(kLane);
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

    indices.notFirstLane = b.icmp_ne(indices.laneIndex, b.i32_val(0));
    // Size of the aligned interval containing the warp-local chunks reachable
    // by varying lane and warp at a fixed register.
    auto [first, last] = helper.getChunkBounds(0);
    while (first / indices.chunkSpan != last / indices.chunkSpan)
      indices.chunkSpan *= 2;
    // Warp-local chunk coordinate for register zero, in scan traversal order.
    indices.baseChunkIndex =
        applyLinearLayout(loc, rewriter, helper.getTotalsLayout(),
                          getHardwareCoordinates(op, 0, indices.laneId,
                                                 indices.warpId,
                                                 rewriter))[op.getAxis()]
            .second;
    unsigned reflection =
        (indices.remainingReverseMask / chunkSize) % indices.chunkSpan;
    if (reflection == indices.chunkSpan - 1)
      indices.baseChunkIndex =
          b.sub(b.i32_val(reflection), indices.baseChunkIndex);
    else
      indices.baseChunkIndex =
          b.xor_(indices.baseChunkIndex, b.i32_val(reflection));
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
    unsigned chunkSize = helper.getChunkSize();
    unsigned numLanes = helper.getChunkSize() / helper.getGroupSize();
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

  // Compute an inclusive prefix within each thread-local group. Input: original
  // register values. Output: prefixes in the same registers, with the total in
  // the final register. Accumulate each thread-local group in logical axis
  // order.
  void scanWithinThreads(triton::ScanOp op, const ScanLoweringHelper &helper,
                         ScanValues &values,
                         ConversionPatternRewriter &rewriter) const {
    for (const auto &group : helper.getThreadLocalGroups()) {
      SmallVector<Value> acc;
      for (unsigned reg : group) {
        acc = combineWithPrefix(op, acc, values[reg], rewriter);
        values[reg] = acc;
      }
    }
  }

  // Scan thread-local group totals within each warp-local chunk. Shuffle rounds
  // at logical distances 1, 2, 4, ... compute inclusive prefixes of these
  // totals. Predicates restrict predecessors to the same warp-local chunk.
  // totals[g] contains the prefix through this lane's thread-local group; the
  // terminal lane holds the complete warp-local chunk total. Register values
  // retain their thread-local group prefixes until carries are applied. For
  // scans contained in one warp, applyWarpCarries completes the register
  // prefixes by incorporating preceding warp-local chunks and thread-local
  // groups.
  ScanValues scanWithinWarps(triton::ScanOp op,
                             const ScanLoweringHelper &helper,
                             ScanIndices &indices, ScanValues &values,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto groups = helper.getThreadLocalGroups();
    unsigned numLanes = helper.getChunkSize() / helper.getGroupSize();
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
    unsigned reg = helper.getThreadLocalGroups()[g].back();
    unsigned regIndex = helper.getChunkIndex(reg);
    unsigned traversalIndex =
        regIndex ^ (indices.remainingReverseMask / helper.getChunkSize());
    // Aligned intervals selected by registers have a compile-time ordering.
    if (traversalIndex / indices.chunkSpan != index / indices.chunkSpan)
      return b.i1_val(traversalIndex / indices.chunkSpan >
                      index / indices.chunkSpan);
    // Within one interval, combine the layout's thread and register
    // coordinates.
    Value chunkIndex =
        b.xor_(indices.baseChunkIndex, b.i32_val(regIndex % indices.chunkSpan));
    return b.icmp_ugt(chunkIndex, b.i32_val(index % indices.chunkSpan));
  }

  SmallVector<Value>
  applyChunkCarry(triton::ScanOp op, const ScanLoweringHelper &helper,
                  ScanIndices &indices, ScanValues &values,
                  const ScanValues &totals, unsigned g, ValueRange carry,
                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto groups = helper.getThreadLocalGroups();
    unsigned numLanes = helper.getChunkSize() / helper.getGroupSize();
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
    auto groups = helper.getThreadLocalGroups();
    unsigned groupSize = helper.getGroupSize();
    unsigned chunkSize = helper.getChunkSize();
    unsigned laneMask = helper.getAxisMask(kLane, chunkSize);
    unsigned regMask = helper.getAxisMask(kReg, groupSize);
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
    auto groups = helper.getThreadLocalGroups();
    unsigned groupSize = helper.getGroupSize();
    unsigned chunkSize = helper.getChunkSize();
    unsigned regMask = helper.getAxisMask(kReg, groupSize);
    // Read warp-local chunk totals in logical order and retain the exclusive
    // prefix for each consumer. Main's repeated CTA tiles occupy disjoint
    // ranges of this sequence. Interleaved register/lane/warp ownership can
    // produce overlapping ranges. Visit each total once per independent scan
    // and finish each warp-local chunk after reading its possible predecessors.
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
        unsigned logical = index ^ (indices.remainingReverseMask / chunkSize);
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
    auto groups = helper.getThreadLocalGroups();
    unsigned chunkSize = helper.getChunkSize();
    unsigned numLanes = helper.getChunkSize() / helper.getGroupSize();
    unsigned numChunks = layout.getOutDimSize(axis) / chunkSize;
    unsigned laneMask = helper.getAxisMask(kLane, chunkSize);
    bool oneChunkPerLaneGroup =
        helper.getAxisMask(kLane, layout.getOutDimSize(axis)) == laneMask;
    if (numChunks == 1) {
      for (unsigned g = 0; g < groups.size(); ++g)
        applyChunkCarry(op, helper, indices, values, totals, g, {}, rewriter);
    } else if (oneChunkPerLaneGroup) {
      // Main's one-warp carry chain: broadcast the completed warp-local chunk
      // prefix and reuse it as the carry for the next warp-local chunk.
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

  // Complete the scan using thread-local group prefixes and the scanned totals
  // from scanWithinWarps. Terminal lanes publish warp-local chunk totals to
  // shared memory, excluding redundant owners. After one CTA barrier,
  // scanChunkTotals reads totals in logical order and computes each warp-local
  // chunk's exclusive carry. applyChunkCarry combines this carry with preceding
  // thread-local groups' prefixes and updates their register values. Boundary
  // predicates apply carries only to thread-local groups with predecessors.
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
    auto groups = helper.getThreadLocalGroups();
    unsigned chunkSize = helper.getChunkSize();
    unsigned laneMask = helper.getAxisMask(kLane, chunkSize);
    // Reflect shared-memory warp slots for reverse traversal. Applying the
    // same reflection to loads preserves the layout lookup's ownership.
    Value scratchWarp = indices.warpId;
    unsigned scratchReflection = 0;
    if (op.getReverse()) {
      unsigned mask = helper.getAxisMask(kWarp, layout.getOutDimSize(axis));
      if (mask == unsigned(layout.getInDimSize(kWarp) - 1))
        scratchWarp = b.sub(b.i32_val(mask), indices.warpId);
      else
        scratchWarp = b.xor_(indices.warpId, b.i32_val(mask));
      scratchReflection =
          helper.getScratchAddressLayout()
              .apply(getHardwareCoordinates(0, 0, mask, rewriter))
              .front()
              .second;
    }
    SmallVector<Value> smemBases;
    SmallVector<Type> smemTypes;
    smemBases =
        getSmemBases(op, helper.getScratchSizeInElems(), rewriter, targetInfo);
    for (unsigned i = 0; i < op.getNumOperands(); ++i)
      smemTypes.push_back(getElementType(op, i));
    // Publish totals for all warp-local chunks and independent scans before
    // the single CTA barrier, as in main.
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
          applyLinearLayout(loc, rewriter, helper.getScratchAddressLayout(),
                            getHardwareCoordinates(op, groups[g].back(),
                                                   indices.laneId, scratchWarp,
                                                   rewriter))
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
          offset = b.xor_(offset, b.i32_val(scratchReflection));
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
