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

  // Per-operation layout metadata and indices shared by the scan stages.
  struct ScanState {
    ScanState(triton::ScanOp op, ConversionPatternRewriter &rewriter)
        : op(op), rewriter(rewriter), helper(op), loc(op.getLoc()),
          b(loc, rewriter), kReg(rewriter.getStringAttr("register")),
          kLane(rewriter.getStringAttr("lane")),
          kWarp(rewriter.getStringAttr("warp")),
          kBlock(rewriter.getStringAttr("block")),
          axis(rewriter.getStringAttr("dim" + std::to_string(op.getAxis()))),
          layout(helper.getLayout()),
          dims(llvm::to_vector(layout.getOutDimNames())),
          groups(helper.getThreadGroups()),
          threadSize(helper.getThreadLocalSize()),
          chunkSize(helper.getWarpChunkSize()),
          numLanes(chunkSize / threadSize),
          numChunks(layout.getOutDimSize(axis) / chunkSize),
          laneMask(helper.getAxisMask(kLane, chunkSize)),
          regMask(helper.getAxisMask(kReg, threadSize)),
          warpSize(layout.getInDimSize(kLane)),
          interWarp(helper.hasInterWarpScan()), chunkIndices(groups.size()) {}

    triton::ScanOp op;
    ConversionPatternRewriter &rewriter;
    ScanLoweringHelper helper;
    Location loc;
    TritonLLVMOpBuilder b;
    StringAttr kReg, kLane, kWarp, kBlock, axis;
    const LinearLayout &layout;
    SmallVector<StringAttr> dims;
    ArrayRef<SmallVector<unsigned>> groups;
    unsigned threadSize, chunkSize, numLanes, numChunks;
    unsigned laneMask, regMask, warpSize;
    bool interWarp;
    unsigned axisOffset = 0, stride = 0, scratchReflection = 0, varyingBits = 0;
    Value laneId, warpId, laneIndex, scratchWarpId, notFirstLane, threadChunk;
    LinearLayout laneInverse, regAxis;
    SmallVector<Value> chunkIndices;
  };

  static unsigned terminalReg(const ScanState &s, unsigned group) {
    return s.groups[group].back();
  }

  template <typename Coordinates>
  static auto getCoordinate(const Coordinates &coords, StringAttr dim) {
    return llvm::find_if(coords, [&](auto coord) { return coord.first == dim; })
        ->second;
  }

  static SmallVector<std::pair<StringAttr, Value>>
  hardware(ScanState &s, unsigned reg, Value lane, Value warp) {
    return {{s.kReg, s.b.i32_val(reg)},
            {s.kLane, lane},
            {s.kWarp, warp},
            {s.kBlock, s.b.i32_val(0)}};
  }

  static SmallVector<std::pair<StringAttr, int32_t>>
  constantHardware(const ScanState &s, unsigned reg, unsigned lane,
                   unsigned warp) {
    return {{s.kReg, reg}, {s.kLane, lane}, {s.kWarp, warp}, {s.kBlock, 0}};
  }

  static Value reflectIndex(ScanState &s, Value index, unsigned mask,
                            unsigned size) {
    // A complete field reversal is subtraction, as in main. A partial
    // reversal preserves the other layout bits with XOR.
    if (mask == size - 1)
      return s.b.sub(s.b.i32_val(mask), index);
    return s.b.xor_(index, s.b.i32_val(mask));
  }

  static SmallVector<Value> selectValues(ScanState &s, Value pred,
                                         ValueRange onTrue,
                                         ValueRange onFalse) {
    SmallVector<Value> result;
    for (auto [lhs, rhs] : llvm::zip(onTrue, onFalse))
      result.push_back(s.b.select(pred, lhs, rhs));
    return result;
  }

  SmallVector<Value> combine(ScanState &s, ValueRange prefix, ValueRange values,
                             Value pred = {}) const {
    auto result = applyCombineOp(s.loc, s.rewriter, s.op.getCombineOp(), prefix,
                                 values, pred);
    return pred ? selectValues(s, pred, result, values) : result;
  }

  SmallVector<Value> shuffle(ScanState &s, ValueRange values,
                             Value source) const {
    SmallVector<Value> result;
    for (Value value : values)
      result.push_back(targetInfo.shuffleIdx(s.rewriter, s.loc, value, source));
    return result;
  }

  ScanValues reverseValues(ScanState &s, const ScanValues &values) const {
    ScanValues result(values.size());
    for (unsigned reg = 0; reg < values.size(); ++reg)
      for (Value value : values[values.size() - 1 - reg])
        result[reg].push_back(
            targetInfo.shuffleXor(s.rewriter, s.loc, value, s.warpSize - 1));
    return result;
  }

  void initializeIndices(ScanState &s) const {
    Value threadId = getThreadId(s.rewriter, s.loc);
    s.laneId = s.b.urem(threadId, s.b.i32_val(s.warpSize));
    s.warpId = s.b.and_(s.b.udiv(threadId, s.b.i32_val(s.warpSize)),
                        s.b.i32_val(s.layout.getInDimSize(s.kWarp) - 1));

    // As in main, reverse the register/lane traversal around a forward scan.
    // The layout gives the remaining axis offset, including interleaved or
    // swizzled warp bits; no layout or physical warp ownership is changed.
    s.axisOffset = 0;
    if (s.op.getReverse()) {
      s.axisOffset = (s.layout.getOutDimSize(s.axis) - 1) ^
                     s.layout
                         .apply({{s.kReg, s.layout.getInDimSize(s.kReg) - 1},
                                 {s.kLane, s.warpSize - 1},
                                 {s.kWarp, 0},
                                 {s.kBlock, 0}})[s.op.getAxis()]
                         .second;
    }

    initializeLaneIndices(s);
    initializeCarryIndices(s);
  }

  void initializeLaneIndices(ScanState &s) const {
    // Apply main's logarithmic warp scan to each thread chunk's terminal value.
    // A contiguous physical lane range uses shuffle-up. Otherwise invert the
    // lane layout to find the preceding logical lane, preserving other scans.
    auto laneBases = s.layout.sublayout({s.kLane}, {s.axis}).getBases();
    for (auto &basis : laneBases[s.kLane])
      basis[0] = (basis[0] % s.chunkSize) / s.threadSize;
    LinearLayout laneLayout(laneBases, {{s.axis, s.numLanes}}, true);
    s.laneIndex =
        applyLinearLayout(s.loc, s.rewriter, laneLayout, {{s.kLane, s.laneId}})
            .front()
            .second;
    s.laneInverse = laneLayout.pseudoinvert();
    s.stride = s.numLanes == 1 ? 1 : 0;
    const auto &columns = laneBases[s.kLane];
    for (unsigned bit = 0; bit < columns.size(); ++bit)
      if (columns[bit][0] == 1) {
        s.stride = 1u << bit;
        for (unsigned i = 0; (1u << i) < s.numLanes; ++i)
          if (bit + i >= columns.size() || columns[bit + i][0] != (1u << i))
            s.stride = 0;
        break;
      }
  }

  void initializeCarryIndices(ScanState &s) const {
    unsigned warpReflection =
        s.op.getReverse()
            ? s.helper.getAxisMask(s.kWarp, s.layout.getOutDimSize(s.axis))
            : 0;
    // Reflect axis-warp slots once for reverse traversal. Regular layouts
    // then read neighboring totals at increasing shared-memory addresses.
    s.scratchWarpId = reflectIndex(s, s.warpId, warpReflection,
                                   s.layout.getInDimSize(s.kWarp));
    s.scratchReflection = 0;
    if (s.interWarp && s.op.getReverse())
      s.scratchReflection = s.helper.getScratchAddressLayout()
                                .apply({{s.kReg, 0},
                                        {s.kLane, 0},
                                        {s.kWarp, warpReflection},
                                        {s.kBlock, 0}})
                                .front()
                                .second;
    s.notFirstLane = s.b.icmp_ne(s.laneIndex, s.b.i32_val(0));
    auto baseCoords =
        applyLinearLayout(s.loc, s.rewriter, s.helper.getTotalsLayout(),
                          hardware(s, 0, s.laneId, s.warpId));
    s.threadChunk = baseCoords[s.op.getAxis()].second;
    s.regAxis = s.helper.getTotalsLayout().sublayout({s.kReg}, {s.axis});
    unsigned varyingMask = 0;
    for (auto dim : {s.kLane, s.kWarp})
      for (const auto &basis :
           s.helper.getTotalsLayout().getBases().lookup(dim))
        varyingMask |= basis[s.op.getAxis()];
    unsigned varyingSize = llvm::PowerOf2Ceil(varyingMask + 1);
    s.varyingBits = varyingSize - 1;
    s.threadChunk =
        reflectIndex(s, s.threadChunk,
                     (s.axisOffset / s.chunkSize) & s.varyingBits, varyingSize);
  }

  Value sourceLane(ScanState &s, unsigned delta) const {
    Value previous = s.b.sub(s.laneIndex, s.b.i32_val(delta));
    previous = s.b.and_(previous, s.b.i32_val(s.numLanes - 1));
    Value mapped = applyLinearLayout(s.loc, s.rewriter, s.laneInverse,
                                     {{s.axis, previous}})
                       .front()
                       .second;
    return Value(s.b.or_(s.b.and_(s.laneId, s.b.i32_val(~s.laneMask)), mapped));
  }

  SmallVector<Value> shufflePrevious(ScanState &s, ValueRange values,
                                     unsigned delta) const {
    if (!s.stride)
      return shuffle(s, values, sourceLane(s, delta));
    SmallVector<Value> result;
    for (Value value : values)
      result.push_back(
          targetInfo.shuffleUp(s.rewriter, s.loc, value, delta * s.stride));
    return result;
  }

  static SmallVector<unsigned> getRegisterGroups(const ScanState &s) {
    SmallVector<unsigned> groupForReg(s.layout.getInDimSize(s.kReg));
    for (unsigned g = 0; g < s.groups.size(); ++g)
      for (unsigned reg : s.groups[g])
        groupForReg[reg] = g;
    return groupForReg;
  }

  void scanWithinThreads(ScanState &s, ScanValues &values) const {
    // Scan contiguous elements within a thread. Follow native register emission
    // order, but take each chunk's elements in logical axis order.
    auto groupForReg = getRegisterGroups(s);
    SmallVector<unsigned> nextReg(s.groups.size(), 0);
    ScanValues accumulators(s.groups.size());
    for (unsigned nativeReg = 0; nativeReg < values.size(); ++nativeReg) {
      unsigned g = groupForReg[nativeReg];
      unsigned reg = s.groups[g][nextReg[g]++];
      accumulators[g] = combine(s, accumulators[g], values[reg]);
      values[reg] = accumulators[g];
    }
  }

  ScanValues scanWithinWarps(ScanState &s, ScanValues &values) const {
    ScanValues totals(s.groups.size());
    for (unsigned g = 0; g < s.groups.size(); ++g) {
      auto acc = values[terminalReg(s, g)];
      for (unsigned delta = 1; delta < s.numLanes; delta *= 2) {
        auto incoming = shufflePrevious(s, acc, delta);
        Value pred = s.b.icmp_uge(s.laneIndex, s.b.i32_val(delta));
        acc = combine(s, incoming, acc, pred);
      }
      totals[g] = std::move(acc);
    }

    if (!s.interWarp)
      scanWarpChunks(s, values, totals);
    return totals;
  }

  Value isAfter(ScanState &s, unsigned g, unsigned index) const {
    unsigned reg =
        s.regAxis.apply({{s.kReg, terminalReg(s, g)}}).front().second;
    unsigned fixed = (reg ^ (s.axisOffset / s.chunkSize)) & ~s.varyingBits;
    if (fixed != (index & ~s.varyingBits))
      return s.b.i1_val(fixed > (index & ~s.varyingBits));
    // Compare only bits that can vary between threads. This shares main's
    // warp-prefix predicate across repeated register chunks.
    if (!s.chunkIndices[g])
      s.chunkIndices[g] =
          s.b.xor_(s.threadChunk, s.b.i32_val(reg & s.varyingBits));
    return s.b.icmp_ugt(s.chunkIndices[g], s.b.i32_val(index & s.varyingBits));
  }

  SmallVector<Value> applyChunkCarry(ScanState &s, ScanValues &values,
                                     const ScanValues &totals, unsigned g,
                                     ValueRange carry) const {
    Value hasCarry;
    if (!carry.empty())
      hasCarry = isAfter(s, g, 0);
    auto terminal = combine(s, carry, totals[g], hasCarry);
    values[terminalReg(s, g)] = terminal;
    if (s.groups[g].size() == 1 && !s.interWarp)
      return terminal;
    auto previous = s.numLanes > 1 || s.interWarp
                        ? shufflePrevious(s, terminal, 1)
                        : SmallVector<Value>(carry);
    if (s.groups[g].size() == 1)
      return terminal;
    if (!carry.empty() && (s.numLanes > 1 || s.interWarp))
      previous = selectValues(s, s.notFirstLane, previous, carry);
    if (previous.empty())
      return terminal;
    Value pred =
        hasCarry ? Value(s.b.or_(hasCarry, s.notFirstLane)) : s.notFirstLane;
    for (unsigned i = 1; i < s.groups[g].size(); ++i) {
      unsigned reg = s.groups[g][s.groups[g].size() - 1 - i];
      values[reg] = combine(s, previous, values[reg], pred);
    }
    return terminal;
  }

  LinearLayout getChunkLookup(const ScanState &s) const {
    auto op = s.op;
    auto queryBases = s.helper.getTotalsLayout().getBases();
    for (auto &[dim, columns] : queryBases)
      for (auto &basis : columns) {
        if (dim == s.kBlock || (!s.interWarp && dim == s.kWarp))
          std::fill(basis.begin(), basis.end(), 0);
        basis[op.getAxis()] = 0;
      }
    for (unsigned bit = 1; bit < s.numChunks; bit *= 2) {
      std::vector<int32_t> basis(s.dims.size(), 0);
      basis[op.getAxis()] = bit;
      queryBases[s.axis].push_back(std::move(basis));
    }
    LinearLayout query(queryBases, s.helper.getTotalsLayout().getOutDims(),
                       false);
    auto ownerLayout =
        s.interWarp
            ? s.helper.getScratchLayout()
            : s.helper.getTotalsLayout().sublayout({s.kReg, s.kLane}, s.dims);
    // Invert only coordinates reachable by this CTA (or warp), rather than
    // demanding a full-tensor inverse for broadcasts or CTA-local views.
    return query.invertAndCompose(ownerLayout);
  }

  SmallVector<std::pair<StringAttr, Value>>
  getChunkCoordinates(ScanState &s, unsigned reg, unsigned logical) const {
    auto coords = hardware(s, reg, s.laneId,
                           s.interWarp ? s.warpId : Value(s.b.i32_val(0)));
    coords.push_back({s.axis, s.b.i32_val(logical)});
    return coords;
  }

  SmallVector<Value> loadSharedTotal(ScanState &s, const LinearLayout &lookup,
                                     unsigned reg, unsigned logical,
                                     ArrayRef<Value> smemBases,
                                     ArrayRef<Type> smemTypes) const {
    SmallVector<Value> total;
    Value offset = applyLinearLayout(s.loc, s.rewriter, lookup,
                                     getChunkCoordinates(s, reg, logical))
                       .front()
                       .second;
    offset = s.b.xor_(offset, s.b.i32_val(s.scratchReflection));
    for (unsigned i = 0; i < s.op.getNumOperands(); ++i) {
      Value ptr =
          s.b.gep(smemBases[i].getType(), smemTypes[i], smemBases[i], offset);
      total.push_back(targetInfo.loadShared(s.rewriter, s.loc, ptr,
                                            smemTypes[i], s.b.true_val()));
    }
    return total;
  }

  SmallVector<Value> loadWarpTotal(ScanState &s, const LinearLayout &lookup,
                                   ArrayRef<unsigned> groupForReg,
                                   const ScanValues &totals, unsigned sourceReg,
                                   unsigned logical) const {
    SmallVector<Value> total;
    llvm::SmallSet<unsigned, 32> candidates;
    for (unsigned lane = 0; lane < s.warpSize; ++lane) {
      auto queryConstants = constantHardware(s, sourceReg, lane, 0);
      queryConstants.push_back({s.axis, logical});
      auto owner = lookup.apply(queryConstants);
      unsigned reg = getCoordinate(owner, s.kReg);
      candidates.insert(groupForReg[reg & ~s.regMask]);
    }
    auto owner = applyLinearLayout(s.loc, s.rewriter, lookup,
                                   getChunkCoordinates(s, sourceReg, logical));
    Value reg = getCoordinate(owner, s.kReg);
    Value lane = getCoordinate(owner, s.kLane);
    lane = s.b.or_(s.b.and_(lane, s.b.i32_val(~s.laneMask)),
                   s.b.i32_val(s.laneMask));
    bool needsShuffle = !s.layout.sublayoutIsZero({s.kLane}, {s.axis});
    for (unsigned candidate : candidates) {
      auto incoming = needsShuffle ? shuffle(s, totals[candidate], lane)
                                   : totals[candidate];
      if (total.empty()) {
        total = incoming;
      } else {
        Value pred = s.b.icmp_eq(
            reg, s.b.i32_val(terminalReg(s, candidate) & ~s.regMask));
        total = selectValues(s, pred, incoming, total);
      }
    }
    return total;
  }

  void
  scanChunkTotals(ScanState &s, ScanValues &values, const ScanValues &totals,
                  llvm::function_ref<SmallVector<Value>(unsigned, unsigned)>
                      loadTotal) const {
    // Read chunk totals in logical order and retain the exclusive prefix for
    // each consumer. Main's repeated CTA tiles are disjoint ranges of this
    // sequence; interleaved register/lane/warp chunks can have overlapping
    // ranges. Visit each total once per independent scan and finish a chunk
    // as soon as all its possible predecessors have been read.
    unsigned axisRegs =
        s.helper.getAxisMask(s.kReg, s.layout.getOutDimSize(s.axis));
    struct Chunk {
      unsigned first, last, row;
      SmallVector<Value> carry;
    };
    struct Row {
      unsigned next = 0;
      SmallVector<Value> acc;
      SmallVector<unsigned> groups;
    };
    SmallVector<Chunk> chunks(s.groups.size());
    SmallVector<Row> rows;
    llvm::DenseMap<unsigned, unsigned> rowIds;
    for (unsigned g = 0; g < s.groups.size(); ++g) {
      unsigned reg = terminalReg(s, g);
      auto [it, inserted] =
          rowIds.try_emplace(reg & ~(axisRegs | s.regMask), rows.size());
      if (inserted)
        rows.emplace_back();
      chunks[g] = {s.numChunks - 1, 0, it->second, {}};
      rows[it->second].groups.push_back(g);
      for (unsigned lane = 0; lane < s.layout.getInDimSize(s.kLane); ++lane)
        for (unsigned warp = 0; warp < s.layout.getInDimSize(s.kWarp); ++warp) {
          unsigned index =
              s.helper.getTotalsLayout()
                  .apply(constantHardware(s, reg, lane, warp))[s.op.getAxis()]
                  .second;
          index ^= s.axisOffset / s.chunkSize;
          chunks[g].first = std::min(chunks[g].first, index);
          chunks[g].last = std::max(chunks[g].last, index);
        }
    }
    for (unsigned g = 0; g < s.groups.size(); ++g) {
      auto &row = rows[chunks[g].row];
      while (row.next <= chunks[g].last) {
        unsigned index = row.next++;
        unsigned logical = index ^ (s.axisOffset / s.chunkSize);
        auto total = loadTotal(g, logical);
        row.acc = combine(s, row.acc, total);
        for (unsigned consumer : row.groups) {
          auto &chunk = chunks[consumer];
          if (index + 1 == chunk.first ||
              (index >= chunk.first && index < chunk.last)) {
            if (chunk.carry.empty()) {
              chunk.carry = row.acc;
            } else {
              Value pred = isAfter(s, consumer, index);
              chunk.carry = selectValues(s, pred, row.acc, chunk.carry);
            }
          }
        }
      }
      applyChunkCarry(s, values, totals, g, chunks[g].carry);
    }
  }

  void scanWarpChunks(ScanState &s, ScanValues &values,
                      const ScanValues &totals) const {
    auto localLayout = s.layout.sublayout({s.kReg, s.kLane}, s.dims)
                           .removeZeroBasesAlongDim(s.kReg)
                           .removeZeroBasesAlongDim(s.kLane);
    auto free = localLayout.getFreeVariableMasks();
    bool oneChunkPerLaneGroup =
        s.helper.getAxisMask(s.kLane, s.layout.getOutDimSize(s.axis)) ==
            s.laneMask &&
        triton::gpu::hasPowerOfTwoBases(localLayout) && !free[s.kReg] &&
        !free[s.kLane];
    if (s.numChunks == 1) {
      for (unsigned g = 0; g < s.groups.size(); ++g)
        applyChunkCarry(s, values, totals, g, {});
    } else if (oneChunkPerLaneGroup) {
      // Main's one-warp carry chain: broadcast the completed chunk prefix,
      // instead of separately recomputing the same prefix from raw totals.
      unsigned axisRegs =
          s.helper.getAxisMask(s.kReg, s.layout.getOutDimSize(s.axis));
      llvm::MapVector<unsigned, SmallVector<unsigned>> rows;
      for (unsigned g = 0; g < s.groups.size(); ++g)
        rows[terminalReg(s, g) & ~axisRegs].push_back(g);
      for (auto &[row, order] : rows)
        llvm::sort(order, [&](unsigned a, unsigned b) {
          return s.regAxis.apply({{s.kReg, terminalReg(s, a)}}).front().second <
                 s.regAxis.apply({{s.kReg, terminalReg(s, b)}}).front().second;
        });
      llvm::DenseMap<unsigned, unsigned> nextChunk;
      llvm::DenseMap<unsigned, SmallVector<Value>> prefixes;
      Value lastLane = s.b.or_(s.b.and_(s.laneId, s.b.i32_val(~s.laneMask)),
                               s.b.i32_val(s.laneMask));
      for (unsigned native = 0; native < s.groups.size(); ++native) {
        unsigned row = terminalReg(s, native) & ~axisRegs;
        unsigned g = rows[row][nextChunk[row]++];
        auto terminal = applyChunkCarry(s, values, totals, g, prefixes[row]);
        prefixes[row] =
            s.numLanes > 1 ? shuffle(s, terminal, lastLane) : terminal;
      }
    } else {
      auto lookup = getChunkLookup(s);
      auto groupForReg = getRegisterGroups(s);
      scanChunkTotals(s, values, totals, [&](unsigned g, unsigned logical) {
        return loadWarpTotal(s, lookup, groupForReg, totals, terminalReg(s, g),
                             logical);
      });
    }
  }

  void scanAcrossWarps(ScanState &s, ScanValues &values,
                       const ScanValues &totals) const {
    SmallVector<Value> smemBases;
    SmallVector<Type> smemTypes;
    smemBases = getSmemBases(s.op, s.helper.getScratchSizeInElems(), s.rewriter,
                             targetInfo);
    for (unsigned i = 0; i < s.op.getNumOperands(); ++i)
      smemTypes.push_back(getElementType(s.op, i));
    // Store one total per contiguous chunk. All independent scans and all
    // register chunks publish before the single CTA barrier, as in main.
    unsigned laneZeros = 0, warpZeros = 0;
    for (auto [dim, mask] :
         {std::pair{s.kLane, &laneZeros}, std::pair{s.kWarp, &warpZeros}})
      for (auto [bit, basis] : llvm::enumerate(s.layout.getBases().lookup(dim)))
        if (llvm::all_of(basis, [](int32_t x) { return x == 0; }))
          *mask |= 1u << bit;
    Value pred =
        s.b.icmp_eq(s.b.and_(s.laneId, s.b.i32_val(s.laneMask | laneZeros)),
                    s.b.i32_val(s.laneMask));
    if (warpZeros)
      pred =
          s.b.and_(pred, s.b.icmp_eq(s.b.and_(s.warpId, s.b.i32_val(warpZeros)),
                                     s.b.i32_val(0)));
    for (unsigned g = 0; g < s.groups.size(); ++g) {
      Value offset =
          applyLinearLayout(
              s.loc, s.rewriter, s.helper.getScratchAddressLayout(),
              hardware(s, terminalReg(s, g), s.laneId, s.scratchWarpId))
              .front()
              .second;
      for (unsigned i = 0; i < s.op.getNumOperands(); ++i) {
        Value ptr =
            s.b.gep(smemBases[i].getType(), smemTypes[i], smemBases[i], offset);
        targetInfo.storeShared(s.rewriter, s.loc, ptr, totals[g][i], pred);
      }
    }
    s.b.barrier(triton::gpu::AddrSpace::Local);
    auto lookup = getChunkLookup(s);
    scanChunkTotals(s, values, totals, [&](unsigned g, unsigned logical) {
      return loadSharedTotal(s, lookup, terminalReg(s, g), logical, smemBases,
                             smemTypes);
    });
  }
};

LogicalResult
ScanOpConversion::matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  ScanState s(op, rewriter);
  if (!s.helper.isSupported())
    return op.emitError("unsupported scan layout: scans across CTAs");
  initializeIndices(s);
  ScanValues values(s.layout.getInDimSize(s.kReg));
  for (auto [i, operand] : llvm::enumerate(adaptor.getOperands())) {
    auto unpacked = unpackUniqueTensorElements(s.loc, operand, s.rewriter);
    for (auto [reg, value] : llvm::enumerate(unpacked))
      values[reg].push_back(value);
  }

  if (s.op.getReverse())
    values = reverseValues(s, values);

  scanWithinThreads(s, values);
  auto totals = scanWithinWarps(s, values);
  if (s.interWarp)
    scanAcrossWarps(s, values, totals);

  if (s.op.getReverse())
    values = reverseValues(s, values);
  SmallVector<Value> results;
  for (unsigned operand = 0; operand < s.op.getNumOperands(); ++operand) {
    SmallVector<Value> unpacked;
    for (const auto &value : values)
      unpacked.push_back(value[operand]);
    results.push_back(packUniqueTensorElements(
        s.loc, getTypeConverter(), unpacked, s.rewriter,
        s.op.getResult()[operand].getType()));
  }
  s.rewriter.replaceOp(s.op, results);
  return success();
}
} // namespace

void mlir::triton::populateScanOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ScanOpConversion>(typeConverter, targetInfo, benefit);
}
