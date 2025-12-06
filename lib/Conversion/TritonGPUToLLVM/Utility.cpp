#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <functional>

#if defined(_MSC_VER) && !defined(__clang__)
// from https://gist.github.com/pps83/3210a2f980fd02bb2ba2e5a1fc4a2ef0
#include <intrin.h>

static int __builtin_clz(unsigned x) {
  unsigned long r;
  _BitScanReverse(&r, x);
  return static_cast<int>(r ^ 31);
}

static int __builtin_ctz(unsigned x) {
  unsigned long r;
  _BitScanForward(&r, x);
  return static_cast<int>(r);
}

#endif

namespace mlir {

namespace triton::gpu {

std::pair<SmallVector<LocalMemOpTile>, SmallVector<LocalMemOpTile>>
getSrcDstTiles(const TargetInfoBase &targetInfo, int bitwidth) {
  assert(bitwidth <= 128 && "bitwidth must be <= 128");
  assert(llvm::isPowerOf2_32(bitwidth) && "bitwidth must be a power of two");
  SmallVector<LocalMemOpTile> src;
  SmallVector<LocalMemOpTile> dst;

  // ld.shared/st.shared
  auto ldstshared = LocalMemOpTile{{}, {0, 1, 2}};
  src.push_back(ldstshared);
  dst.push_back(ldstshared);

  if (targetInfo.supportLdMatrix() || targetInfo.supportStMatrix()) {
    // ldmatrix/stmatrix
    if (bitwidth <= 32) {
      auto ldstmatrix = LocalMemOpTile{{0, 1}, {2, 3, 4}};
      if (targetInfo.supportStMatrix()) {
        src.push_back(ldstmatrix);
      }
      if (targetInfo.supportLdMatrix()) {
        dst.push_back(ldstmatrix);
      }
    }
    // ldmatrix.trans/stmatrix.trans
    if (bitwidth == 16) {
      auto ldstmatrixtrans = LocalMemOpTile{{2, 3, 4}, {0, 1}};
      if (targetInfo.supportStMatrix()) {
        src.push_back(ldstmatrixtrans);
      }
      if (targetInfo.supportLdMatrix()) {
        dst.push_back(ldstmatrixtrans);
      }
    }
  }
  return {std::move(src), std::move(dst)};
}

Type getFunctionType(Type resultType, ValueRange operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  return LLVM::LLVMFunctionType::get(resultType, operandTypes);
}

LLVM::LLVMFuncOp appendOrGetExternFuncOp(RewriterBase &rewriter, Operation *op,
                                         StringRef funcName, Type funcType,
                                         StringRef libname /*= ""*/,
                                         StringRef libpath /*= ""*/) {
  using LLVM::LLVMFuncOp;

  auto funcAttr = StringAttr::get(op->getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);

  Operation *parent = op;
  if (!isa<LLVM::LLVMFuncOp>(op))
    parent = op->getParentOfType<LLVM::LLVMFuncOp>();
  OpBuilder b(parent);
  auto ret = LLVMFuncOp::create(b, op->getLoc(), funcName, funcType);
  ret.getOperation()->setAttr("libname",
                              StringAttr::get(op->getContext(), libname));
  ret.getOperation()->setAttr("libpath",
                              StringAttr::get(op->getContext(), libpath));
  return ret;
}

Value matrixVectorProd(TritonLLVMOpBuilder &b, const LinearLayout &A, Value x) {
  assert(A.getNumInDims() == 1);
  assert(A.getNumOutDims() == 1);
  auto flatten = [](const std::vector<std::vector<int32_t>> &matrix) {
    SmallVector<int32_t> ret;
    for (const auto &row : matrix) {
      ret.push_back(row[0]);
    }
    return ret;
  };
  auto nCol = A.getTotalInDimSizeLog2();
  auto nRow = A.getTotalOutDimSizeLog2();
  SmallVector<int32_t> matrix = flatten(A.getBases().begin()->second);
  assert(matrix.size() == nCol);

  // Row-wise popcount to detect rows that appear exactly once across columns.
  uint32_t rowsUnique = 0;
  {
    SmallVector<int> rowPopCnt(nRow, 0);
    for (int c = 0; c < nCol; ++c) {
      uint32_t colBits = matrix[c];
      for (int r = 0; r < nRow; ++r) {
        if (colBits & (1u << r))
          ++rowPopCnt[r];
      }
    }
    for (int r = 0; r < nRow; ++r) {
      if (rowPopCnt[r] == 1)
        rowsUnique |= 1u << r;
    }
  }

  // We iterate the matrix following the diagonals and build
  // (x & mask_i) << s_i terms. Prefer OR for diagonals whose rows are unique,
  // then XOR everything else. This tends to encourage mad.lo codegen.
  auto getMaskAndAllRowsUnique = [&](int i) -> std::pair<uint32_t, bool> {
    uint32_t mask = 0;
    int row = i < 0 ? -i : 0;
    int col = i < 0 ? 0 : i;
    bool allRowsUnique = true;
    while (row < nRow && col < nCol) {
      uint32_t bitValue = (matrix[col] >> row) & 1u;
      mask |= bitValue << col;
      allRowsUnique &= ((rowsUnique >> row) & 1u) == 1u;
      ++row;
      ++col;
    }
    return {mask, allRowsUnique};
  };

  uint32_t explicitCols = 0;

  {
    SmallVector<uint32_t> masks;
    for (int i = -nRow + 1; i < nCol; i++) {
      masks.push_back(std::get<0>(getMaskAndAllRowsUnique(i)));
    }
    bool reachedFixedPoint = false;
    while (!reachedFixedPoint) {
      reachedFixedPoint = true;
      for (uint32_t m : masks) {
        uint32_t c = m & ~explicitCols;
        if (llvm::isPowerOf2_32(c)) {
          // found a single-element diagonal
          explicitCols |= c;
          reachedFixedPoint = false;
        }
      }
    }
  }

  // handle any diagonals that have survived
  SmallVector<Value> ors;
  SmallVector<Value> xors;
  for (int i = -nRow + 1; i < nCol; i++) {
    auto [mask, allRowsUnique] = getMaskAndAllRowsUnique(i);
    mask &= ~explicitCols;
    if (mask == 0)
      continue;
    auto masked = b.and_(x, b.i32_val(mask));
    auto shifted = i >= 0 ? Value(b.lshr(masked, b.i32_val(i)))
                          : Value(b.shl(masked, b.i32_val(-i)));
    if (allRowsUnique) {
      ors.push_back(shifted);
    } else {
      xors.push_back(shifted);
    }
  }

  // handle any explicit columns:
  Value zero = b.i32_val(0);
  for (int i = 0; i < nCol; i++) {
    if ((explicitCols >> i) & 1) {
      Value bit = b.and_(x, b.i32_val(1 << i));
      Value bit_is_zero = b.icmp_eq(bit, zero);
      int32_t basis = matrix[i];
      if (basis == 0)
        continue;
      auto select = b.select(bit_is_zero, zero, b.i32_val(basis));
      if ((rowsUnique & basis) == basis) {
        ors.push_back(select);
      } else {
        xors.push_back(select);
      }
    }
  }

  auto treeReduce = [&](SmallVector<Value> &terms,
                        std::function<Value(Value, Value)> op) -> Value {
    if (terms.empty())
      return b.i32_val(0);
    while (terms.size() > 1) {
      SmallVector<Value> next;
      for (size_t i = 0; i + 1 < terms.size(); i += 2)
        next.push_back(op(terms[i], terms[i + 1]));
      if (terms.size() % 2 == 1)
        next.push_back(terms.back());
      terms = std::move(next);
    }
    return terms[0];
  };

  auto orPart = treeReduce(
      ors, [&b](Value x, Value y) { return b.or_(x, y, /*disjoint=*/true); });
  auto xorPart =
      treeReduce(xors, [&b](Value x, Value y) { return b.xor_(x, y); });
  return b.or_(orPart, xorPart, /*disjoint=*/true);
}

} // namespace triton::gpu

SmallVector<std::pair<StringAttr, Value>>
applyLinearLayout(Location loc, RewriterBase &rewriter,
                  const LinearLayout &layout,
                  ArrayRef<std::pair<StringAttr, Value>> indices) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(layout.getNumInDims() == indices.size());
  assert(llvm::equal(layout.getInDimNames(), llvm::make_first_range(indices)));
  // Trivial layout
  if (layout.getNumOutDims() == 0) {
    return {};
  }

  // This function can emit a lot of MLIR code, which ultimately makes
  // compilation slow.  (We think this shouldn't be the case -- it's not *that*
  // much code -- but we're not clear on how to fix the slowness, which happens
  // in the bowels of MLIR.)
  //
  // As a result we go through some contortions to avoid emitting code where
  // possible.

  // Manually constant-fold the layout where possible.
  SmallVector<std::pair<StringAttr, int32_t>> constantIns;
  SmallVector<std::pair<StringAttr, Value>> nonConstantIns;
  for (auto [inDimName, idx] : indices) {
    APInt constant;
    if (matchPattern(idx, m_ConstantInt(&constant))) {
      constantIns.push_back({inDimName, constant.getSExtValue()});
    } else {
      constantIns.push_back({inDimName, 0});
      nonConstantIns.push_back({inDimName, idx});
    }
  }

  // Compute constant part of the output and wrap it as values
  Value zero = b.i32_val(0);
  SmallVector<std::pair<StringAttr, Value>> outIndices;
  for (auto [outDimName, constant] : layout.apply(constantIns)) {
    if (constant == 0)
      outIndices.push_back({outDimName, zero});
    else
      outIndices.push_back({outDimName, b.i32_val(constant)});
  }

  if (nonConstantIns.size() == 0) {
    return outIndices;
  }

  SmallVector<StringAttr> inDimNames;
  // Concatenate input
  Value x = b.i32_val(0);
  int shift = 0;
  for (auto [inDimName, idx] : nonConstantIns) {
    inDimNames.push_back(inDimName);
    x = b.or_(x, b.shl(idx, b.i32_val(shift)));
    shift += layout.getInDimSizeLog2(inDimName);
  }

  for (auto &[outDimName, outIdx] : outIndices) {
    // Apply flattened sublayout for this output
    auto matrix = layout.sublayout(inDimNames, outDimName).flattenIns();
    auto out = triton::gpu::matrixVectorProd(b, matrix, x);
    outIdx = b.xor_(outIdx, out);
  }

  return outIndices;
}

std::optional<int> getWarpGroupStartThreadId(Block *block) {
  using namespace triton::gpu;

  // Look for an enclosing `ttg.warp_specialize` op.
  while (block && block->getParentOp() &&
         !isa<WarpSpecializePartitionsOp>(block->getParentOp()))
    block = block->getParentOp()->getBlock();
  if (!block || !block->getParentOp())
    return {};

  auto partitions = cast<WarpSpecializePartitionsOp>(block->getParentOp());
  unsigned idx = block->getParent()->getRegionNumber();
  WarpSpecializeOp ws = partitions.getParentOp();
  std::optional<ArrayRef<int32_t>> startIds = ws.getWarpGroupStartIds();
  assert(startIds && "cannot get warp group ID before warp group allocation");
  int32_t warpStartId = (*startIds)[idx];
  int threadsPerWarp =
      TritonGPUDialect::getThreadsPerWarp(ws->getParentOfType<ModuleOp>());
  return warpStartId * threadsPerWarp;
}

Value getThreadId(OpBuilder &rewriter, Location loc) {
  Value tid =
      ::mlir::gpu::ThreadIdOp::create(rewriter, loc, ::mlir::gpu::Dimension::x);
  tid = arith::IndexCastOp::create(rewriter, loc, i32_ty, tid);

  Operation *lookupPt = &rewriter.getInsertionBlock()->front();
  int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
  int numWarps = triton::gpu::lookupNumWarps(lookupPt);
  int upperBound = numWarps * threadsPerWarp;

  TritonLLVMOpBuilder b(loc, rewriter);

  // If this is being created inside a warp specialize op, compute the relative
  // thread ID within the warp group.
  if (std::optional<int> startId =
          getWarpGroupStartThreadId(rewriter.getInsertionBlock())) {
    tid = arith::SubIOp::create(rewriter, loc, tid, b.i32_val(*startId));
  }

  assert(llvm::isPowerOf2_32(upperBound));
  // help LLVM's known bits analysis:
  tid = b.and_(tid, b.i32_val(upperBound - 1));

  return tid;
}

std::pair<Value, Value> getLaneAndWarpId(OpBuilder &rewriter, Location loc) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Value tid = getThreadId(rewriter, loc);
  int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
  Value warpSizeVal = b.i32_val(threadsPerWarp);

  // If there is only one warp, the warp ID is always 0.
  Operation *lookupPt = &rewriter.getInsertionBlock()->front();
  Value laneId;
  Value warpId;
  if (triton::gpu::lookupNumWarps(lookupPt) == 1) {
    laneId = tid;
    warpId = b.i32_val(0);
  } else {
    laneId = b.urem(tid, warpSizeVal);
    warpId = b.udiv(tid, warpSizeVal);
  }

  return {laneId, warpId};
}

Value getLaneId(OpBuilder &rewriter, Location loc) {
  return getLaneAndWarpId(rewriter, loc).first;
}

// Helper function: applies linear layout vectorized over register indices
SmallVector<SmallVector<std::pair<StringAttr, Value>>>
applyLinearLayoutVec(Location loc, RewriterBase &rewriter,
                     const LinearLayout &layout,
                     ArrayRef<std::pair<StringAttr, Value>> indices,
                     ArrayRef<uint32_t> registers) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();

  StringAttr kRegister = str_attr("register");

  // Precompute the base (with register = 0)
  SmallVector<std::pair<StringAttr, Value>> indicesWithZeroReg;
  for (const auto &[attr, val] : indices) {
    if (attr == kRegister)
      indicesWithZeroReg.emplace_back(attr, b.i32_val(0));
    else
      indicesWithZeroReg.emplace_back(attr, val);
  }

  auto baseIndices =
      applyLinearLayout(loc, rewriter, layout, indicesWithZeroReg);

  SmallVector<SmallVector<std::pair<StringAttr, Value>>> ret;

  // Iterate over registers, applying XOR trick
  for (auto reg : registers) {
    SmallVector<std::pair<StringAttr, int32_t>> constRegIndices;
    for (const auto &[attr, val] : indices) {
      constRegIndices.emplace_back(attr, attr == kRegister ? reg : 0);
    }
    auto regIndices = layout.apply(constRegIndices);

    SmallVector<std::pair<StringAttr, Value>> combinedIndices;
    for (auto [base, regIdx] : llvm::zip(baseIndices, regIndices)) {
      assert(base.first == regIdx.first);
      Value combined = b.xor_(base.second, b.i32_val(regIdx.second));
      combinedIndices.emplace_back(base.first, combined);
    }

    ret.push_back(combinedIndices);
  }

  return ret;
}

// Refactored emitIndices function using applyLinearLayoutVec
SmallVector<SmallVector<Value>>
emitIndices(Location loc, RewriterBase &rewriter, const TargetInfoBase &target,
            Attribute layout, RankedTensorType type, bool withCTAOffset) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();
  auto shape = type.getShape();

  LinearLayout ll = triton::gpu::toLinearLayout(shape, layout);

  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId =
      withCTAOffset ? target.getClusterCTAId(rewriter, loc) : b.i32_val(0);

  SmallVector<std::pair<StringAttr, Value>> commonIndices = {
      {kRegister, b.i32_val(0)},
      {kLane, laneId},
      {kWarp, warpId},
      {kBlock, blockId}};

  // Vectorize over registers
  SmallVector<uint32_t> registerIndices;
  for (unsigned reg = 0; reg < ll.getInDimSize(kRegister); ++reg)
    registerIndices.push_back(reg);

  auto vecIndices =
      applyLinearLayoutVec(loc, rewriter, ll, commonIndices, registerIndices);

  unsigned rank = shape.size();
  SmallVector<SmallVector<Value>> ret;
  for (auto &indices : vecIndices) {
    SmallVector<Value> vals;
    assert(indices.size() == rank);
    for (auto &idx : indices)
      vals.push_back(idx.second);
    ret.push_back(vals);
  }

  return ret;
}

Value emitPadding(Location loc, RewriterBase &rewriter,
                  triton::gpu::PaddedSharedEncodingAttr layout,
                  unsigned bitwidth, Value smemOffset, bool offsetInBytes) {
  TritonLLVMOpBuilder b(loc, rewriter);

  assert((bitwidth >= 8) && "Invalid bitwidth for padded shared layout");
  Value padOffset = b.i32_val(0);
  unsigned offScale = offsetInBytes ? bitwidth / 8 : 1;
  for (auto [interval, padding] :
       llvm::zip_equal(layout.getIntervals(), layout.getPaddings())) {
    unsigned intervalScaled = offScale * interval;
    unsigned paddingScaled = offScale * padding;
    Value iVal = b.i32_val(llvm::Log2_32(intervalScaled));
    Value pVal = b.i32_val(llvm::Log2_32(paddingScaled));
    padOffset = b.add(padOffset, b.shl(b.ashr(smemOffset, iVal), pVal));
  }
  return padOffset;
}

SmallVector<Value>
lowerLdStShared(Location loc, MLIRContext *ctx, LinearLayout cvt,
                ArrayRef<Value> valsArray, // Input for store, output for load
                Type llvmElemTy, Value smemBase,
                std::function<Value(Value)> calcPaddedOffset,
                Value affineOffset, uint64_t maskSpanAffineOffset,
                RewriterBase &rewriter, const TargetInfoBase &targetInfo,
                std::optional<int> maybeMaxVecElems, Operation *localLoadOp) {

  bool isStore = !valsArray.empty();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto emitLdSt = [&](RewriterBase &rewriter, Location loc,
                      ArrayRef<Value> vals, Value shmemAddr, int idx,
                      VectorType vecTy) -> SmallVector<Value> {
    auto length = vecTy.getNumElements();
    if (isStore) {
      Value valsVec =
          packLLVector(loc, ArrayRef<Value>(vals).slice(idx, length), rewriter);
      targetInfo.storeDShared(rewriter, loc, shmemAddr, std::nullopt, valsVec,
                              /*pred=*/b.true_val());
      return {};
    } else {
      assert(vals.empty());
      Value valsVec =
          targetInfo.loadDShared(rewriter, loc, shmemAddr, std::nullopt, vecTy,
                                 /*pred=*/b.true_val(), localLoadOp);
      return unpackLLVector(loc, valsVec, rewriter);
    }
  };
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  return lowerLdSt(loc, ctx, cvt, valsArray, llvmElemTy, smemBase,
                   calcPaddedOffset, affineOffset, maskSpanAffineOffset, laneId,
                   warpId, rewriter, targetInfo, maybeMaxVecElems, emitLdSt);
}

SmallVector<Value> lowerLdSt(
    Location loc, MLIRContext *ctx, LinearLayout cvt,
    ArrayRef<Value> valsArray, // Input for store, output for load
    Type llvmElemTy, Value smemBase,
    std::function<Value(Value)> calcPaddedOffset, Value affineOffset,
    uint64_t maskSpanAffineOffset, Value laneId, Value warpId,
    RewriterBase &rewriter, const TargetInfoBase &targetInfo,
    std::optional<int> maybeMaxVecElems,
    std::function<SmallVector<Value>(RewriterBase &, Location, ArrayRef<Value>,
                                     Value, int, VectorType)>
        lowerInst) {
  auto vals = to_vector(valsArray);
  bool isStore = !vals.empty();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto smemPtrTy = ptr_ty(ctx, 3);
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kOffset = str_attr("offset");
  auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);

  auto [elemsPerVec, permutation] =
      largestVectorisation(ctx, cvt, bitwidth, maybeMaxVecElems);

  cvt = permutation.apply(cvt);
  if (isStore) {
    vals = permutation.apply(vals);
  }

  auto tile = LinearLayout::identity1D(elemsPerVec, kReg, kOffset);
  auto quot = divideLeft(cvt, tile);
  assert(quot.has_value() && "cvt must be divisible by tile");
  LinearLayout reps = zerosLike(tile) * *quot;

  LinearLayout addrLayout =
      LinearLayout({{kLane, reps.getBases().lookup(kLane)},
                    {kWarp, reps.getBases().lookup(kWarp)}},
                   reps.getOutDims(), false);
  auto [nAdditive, permStrides] =
      actionAdditiveStrides(reps, addrLayout, maskSpanAffineOffset);
  reps = permStrides.apply(reps);
  if (isStore) {
    vals = permStrides.apply(vals);
  }

  // PTX expects the address increments to be done in bytes
  // If we don't perform the computations in i8, the compiler would
  // have to divide the computation by bitwdith / 8 and then lift this
  // shl, which often it's not able to do.
  auto i8Tile =
      zerosLike(LinearLayout::identity1D(bitwidth / 8, kReg, kOffset));
  auto i8AddrLayout = i8Tile * addrLayout;

  auto regBaseI8 =
      applyLinearLayout(
          loc, rewriter, i8AddrLayout,
          {{kReg, b.i32_val(0)}, {kLane, laneId}, {kWarp, warpId}})[0]
          .second;

  // It's fine that we don't compute the offset in bytes as affineOffset
  // will be folded into a constant
  auto affineOffsetI8 = b.mul(affineOffset, b.i32_val(bitwidth / 8));
  regBaseI8 = b.xor_(regBaseI8, affineOffsetI8);
  SmallVector<Value> outVals;
  auto vecTy = vec_ty(llvmElemTy, elemsPerVec);
  for (int i = 0; i < cvt.getInDimSize(kReg); i += nAdditive) {
    auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
    auto regIdxI8 = regIdx * (bitwidth / 8);
    Value offset = b.xor_(regBaseI8, b.i32_val(regIdxI8));
    for (int j = 0; j < nAdditive; j += elemsPerVec) {
      // all these constants will go as immediate values to LDS/STS
      auto regIdxAdd =
          reps.apply({{kReg, j}, {kLane, 0}, {kWarp, 0}})[0].second;
      auto regIdxAddI8 = regIdxAdd * (bitwidth / 8);
      Value innerOffset = b.add(offset, b.i32_val(regIdxAddI8));
      auto vecAddr =
          b.gep(smemPtrTy, i8_ty, smemBase, calcPaddedOffset(innerOffset),
                LLVM::GEPNoWrapFlags::inbounds);
      llvm::append_range(outVals,
                         lowerInst(rewriter, loc, vals, vecAddr, i + j, vecTy));
    }
  }

  // Permute the values back if we are loading
  if (!isStore) {
    auto invPermStrides = permStrides.inverse();
    outVals = invPermStrides.apply(outVals);
    auto invPerm = permutation.inverse();
    outVals = invPerm.apply(outVals);
  }
  return outVals;
}

SmallVector<Value>
lowerLocalLdSt(Location loc, MLIRContext *ctx,
               LinearLayout cvt,          // Map from registers to offset
               ArrayRef<Value> valsArray, // Input for store, empty for load
               Type llvmElemTy, triton::gpu::MemDescType srcTy,
               SharedMemoryObject smemObj, RewriterBase &rewriter,
               const TargetInfoBase &targetInfo, Operation *localLoadOp) {
  assert(cvt.getNumOutDims() == 1);
  assert(*cvt.getOutDimNames().begin() == str_attr("offset"));
  auto calcPaddedOffset = [&](Value smemOffset) {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            srcTy.getEncoding())) {
      // Apply the offset needed for padding.
      Value padOffset = emitPadding(loc, rewriter, paddedEnc, bitwidth,
                                    smemOffset, /*offsetInBytes=*/true);
      smemOffset = b.add(smemOffset, padOffset);
    }
    return smemOffset;
  };
  auto isStore = !valsArray.empty();
  // Remove broadcasting in the registers
  auto removeBroadcastSrc = actionRemoveBroadcastedRegs(cvt);
  if (!removeBroadcastSrc.isIdentity()) {
    auto prmtCvt = removeBroadcastSrc.apply(cvt);
    auto inVals = to_vector(valsArray);
    if (isStore) {
      inVals = removeBroadcastSrc.apply(inVals);
    }
    auto outVals = lowerLocalLdSt(loc, ctx, prmtCvt, inVals, llvmElemTy, srcTy,
                                  smemObj, rewriter, targetInfo, localLoadOp);
    if (!isStore) {
      outVals = broadcastAs(outVals, cvt);
    }
    return outVals;
  }
  auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
  auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);

  std::optional<int> maybeMaxVecElems;
  if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
          srcTy.getEncoding())) {
    maybeMaxVecElems = paddedEnc.getMinInterval();
  }

  return lowerLdStShared(loc, ctx, cvt, valsArray, llvmElemTy,
                         smemObj.getBase(), calcPaddedOffset, affineOffset,
                         maskSpanAffineOffset, rewriter, targetInfo,
                         maybeMaxVecElems, localLoadOp);
}

SmallVector<Value> unpackLLElements(Location loc, Value llvmStruct,
                                    RewriterBase &rewriter) {
  assert(bool(llvmStruct) && "can not unpack null values");
  if (llvmStruct.getType().isIntOrIndexOrFloat() ||
      isa<triton::PointerType>(llvmStruct.getType()) ||
      isa<LLVM::LLVMPointerType>(llvmStruct.getType()))
    return {llvmStruct};
  ArrayRef<Type> types =
      cast<LLVM::LLVMStructType>(llvmStruct.getType()).getBody();
  SmallVector<Value> results(types.size());
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    results[i] = b.extract_val(type, llvmStruct, i);
  }
  return results;
}

Value packLLElements(Location loc, const LLVMTypeConverter *typeConverter,
                     ValueRange resultVals, RewriterBase &rewriter, Type type) {
  auto structType =
      dyn_cast<LLVM::LLVMStructType>(typeConverter->convertType(type));
  if (!structType) {
    assert(resultVals.size() == 1);
    return *resultVals.begin();
  }

  auto elementTypes = structType.getBody();
  if (elementTypes.size() != resultVals.size()) {
    emitError(loc) << " size mismatch when packing elements for LLVM struct"
                   << " expected " << elementTypes.size() << " but got "
                   << resultVals.size();
    llvm::report_fatal_error(
        "size mismatch when packing elements for LLVM struct");
  }
  Value llvmStruct = LLVM::UndefOp::create(rewriter, loc, structType);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  for (auto [i, value] : llvm::enumerate(resultVals)) {
    assert(value && "unexpected null value");
    if (value.getType() != elementTypes[i]) {
      LDBG("type " << type << " structType " << structType);
      LDBG("value " << value);
      emitError(loc) << "invalid element type in packLLElements. Expected "
                     << elementTypes[i] << " but got " << value.getType();
      llvm::report_fatal_error(
          "element type mismatch when packing elements for LLVM struct");
    }
    llvmStruct = b.insert_val(structType, llvmStruct, value, i);
  }
  return llvmStruct;
}

SmallVector<Value> unpackLLVector(Location loc, Value llvmVec,
                                  RewriterBase &rewriter) {
  assert(bool(llvmVec) && "cannot unpack null value");
  if (llvmVec.getType().isIntOrIndexOrFloat() ||
      isa<triton::PointerType>(llvmVec.getType()) ||
      isa<LLVM::LLVMPointerType>(llvmVec.getType()))
    return {llvmVec};

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> results;
  for (int i = 0; i < cast<VectorType>(llvmVec.getType()).getNumElements();
       i++) {
    results.push_back(b.extract_element(llvmVec, b.i32_val(i)));
  }
  return results;
}

Value packLLVector(Location loc, ValueRange vals, RewriterBase &rewriter) {
  assert(vals.size() > 0);
  auto vecType = vec_ty(vals[0].getType(), vals.size());
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value vec = b.undef(vecType);
  for (int i = 0; i < vals.size(); i++) {
    vec = b.insert_element(vec, vals[i], b.i32_val(i));
  }
  return vec;
}

std::optional<LLVM::AtomicBinOp> matchAtomicOp(RMWOp atomicOp) {
  switch (atomicOp) {
  case RMWOp::AND:
    return LLVM::AtomicBinOp::_and;
  case RMWOp::OR:
    return LLVM::AtomicBinOp::_or;
  case RMWOp::XOR:
    return LLVM::AtomicBinOp::_xor;
  case RMWOp::ADD:
    return LLVM::AtomicBinOp::add;
  case RMWOp::FADD:
    return LLVM::AtomicBinOp::fadd;
  case RMWOp::MAX:
    return LLVM::AtomicBinOp::max;
  case RMWOp::MIN:
    return LLVM::AtomicBinOp::min;
  case RMWOp::UMAX:
    return LLVM::AtomicBinOp::umax;
  case RMWOp::UMIN:
    return LLVM::AtomicBinOp::umin;
  case RMWOp::XCHG:
    return LLVM::AtomicBinOp::xchg;
  default:
    return {};
  }
}

std::optional<LLVM::AtomicOrdering> getMemoryOrdering(MemSemantic memOrdering) {
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    return LLVM::AtomicOrdering::monotonic;
  case MemSemantic::ACQUIRE:
    return LLVM::AtomicOrdering::acquire;
  case MemSemantic::RELEASE:
    return LLVM::AtomicOrdering::release;
  case MemSemantic::ACQUIRE_RELEASE:
    return LLVM::AtomicOrdering::acq_rel;
  default:
    return {};
  }
}

llvm::MapVector<StringAttr, int32_t> getAllFreeVarMasks(MLIRContext *ctx) {
  // Mask where all elements are redundant
  auto kReg = str_attr("reg");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  int32_t fullMask = -1;
  llvm::MapVector<StringAttr, int32_t> ret;
  for (auto dimName : {kReg, kLane, kWarp, kBlock}) {
    ret[dimName] = fullMask;
  }
  return ret;
}

llvm::MapVector<StringAttr, int32_t> getFreeVariableMasks(Type type) {
  auto ctx = type.getContext();
  auto tensorTy = dyn_cast<RankedTensorType>(type);
  if (!tensorTy) {
    return getAllFreeVarMasks(ctx);
  }
  auto ll = triton::gpu::toLinearLayout(tensorTy);
  return ll.getFreeVariableMasks();
}

SmallVector<SmallVector<unsigned>> emitOffsetForLayout(Attribute layout,
                                                       RankedTensorType type) {
  MLIRContext *ctx = layout.getContext();
  auto shape = type.getShape();
  unsigned rank = shape.size();

  auto ll = triton::gpu::toLinearLayout(type);

  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");

  SmallVector<SmallVector<unsigned>> offsets;
  for (int i = 0; i < ll.getInDimSize(str_attr("register")); i++) {
    auto idxs = ll.apply({{kRegister, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    assert(idxs.size() == rank);
    for (unsigned k = 0; k < rank; ++k) {
      assert(idxs[k].first == str_attr("dim" + std::to_string(k)));
    }
    offsets.push_back(
        llvm::to_vector_of<unsigned>(llvm::make_second_range(idxs)));
  }
  return offsets;
}

namespace LLVM {
using namespace mlir::triton;
using mlir::triton::gpu::getOrder;

Value createConstantI1(Location loc, OpBuilder &rewriter, bool v) {
  auto i1ty = rewriter.getIntegerType(1);
  return LLVM::ConstantOp::create(rewriter, loc, i1ty,
                                  IntegerAttr::get(i1ty, v));
}

Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return LLVM::ConstantOp::create(rewriter, loc, i32ty,
                                  IntegerAttr::get(i32ty, v));
}

Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v) {
  auto i64ty = rewriter.getIntegerType(64);
  return LLVM::ConstantOp::create(rewriter, loc, i64ty,
                                  IntegerAttr::get(i64ty, v));
}

Value createConstantF16(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::f16Ty(rewriter.getContext());
  return LLVM::ConstantOp::create(rewriter, loc, type,
                                  rewriter.getF16FloatAttr(v));
}

Value createConstantBF16(Location loc, OpBuilder &rewriter, float v) {
  APFloat apf(v);
  bool ignored;
  apf.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &ignored);
  auto type = type::bf16Ty(rewriter.getContext());
  auto attr = FloatAttr::get(type, apf);
  return LLVM::ConstantOp::create(rewriter, loc, type, attr);
}

Value createConstantF32(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::f32Ty(rewriter.getContext());
  return LLVM::ConstantOp::create(rewriter, loc, type,
                                  rewriter.getF32FloatAttr(v));
}

Value createConstantF64(Location loc, OpBuilder &rewriter, double v) {
  auto type = type::f64Ty(rewriter.getContext());
  return LLVM::ConstantOp::create(rewriter, loc, type,
                                  rewriter.getF64FloatAttr(v));
}

Value createNaNConstant(Location loc, OpBuilder &rewriter, Type type) {
  if (!isa<FloatType>(type)) {
    llvm::report_fatal_error("Creating NaN constant for non-float type!");
  }
  return LLVM::ConstantOp::create(
      rewriter, loc, type,
      APFloat::getNaN(cast<FloatType>(type).getFloatSemantics()));
}

// Create an index type constant.
Value createIndexConstant(OpBuilder &builder, Location loc,
                          const TypeConverter *converter, int64_t value) {
  Type ty = converter->convertType(builder.getIndexType());
  return LLVM::ConstantOp::create(builder, loc, ty,
                                  builder.getIntegerAttr(ty, value));
}

// Create an integer constant of \param width bits.
Value createLLVMIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value) {
  Type ty = builder.getIntegerType(width);
  return LLVM::ConstantOp::create(builder, loc, ty,
                                  builder.getIntegerAttr(ty, value));
}

LLVM::CallOp createLLVMCallOp(OpBuilder &builder, Location loc,
                              LLVMFuncOp funcOp, ValueRange args) {
  auto op = LLVM::CallOp::create(builder, loc, funcOp, args);
  op.getProperties().setOpBundleSizes(builder.getDenseI32ArrayAttr({}));
  op.getProperties().setOperandSegmentSizes({static_cast<int>(args.size()), 0});
  return op;
}

LLVM::CallIntrinsicOp
createLLVMIntrinsicCallOp(OpBuilder &builder, Location loc, StringRef intrinsic,
                          TypeRange types, ValueRange args) {
  auto op = LLVM::CallIntrinsicOp::create(builder, loc, types, args);
  op.getProperties().setIntrin(builder.getStringAttr(intrinsic));
  op.getProperties().setOpBundleSizes(builder.getDenseI32ArrayAttr({}));
  op.getProperties().setOperandSegmentSizes({static_cast<int>(args.size()), 0});
  return op;
}

SharedMemoryObject::SharedMemoryObject(Value base, Type baseElemType,
                                       ArrayRef<Value> offsets)
    : base(base), baseElemType(baseElemType),
      offsets(offsets.begin(), offsets.end()) {}

SharedMemoryObject::SharedMemoryObject(Value base, Type baseElemType,
                                       int64_t rank, Location loc,
                                       RewriterBase &rewriter)
    : base(base), baseElemType(baseElemType) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  offsets.append(rank, b.i32_val(0));
}

SmallVector<Value> SharedMemoryObject::getElems() const {
  SmallVector<Value> elems;
  elems.push_back(base);
  elems.append(offsets.begin(), offsets.end());
  return elems;
}

SmallVector<Type> SharedMemoryObject::getTypes() const {
  SmallVector<Type> types;
  types.push_back(base.getType());
  types.append(offsets.size(), IntegerType::get(base.getContext(), 32));
  return types;
}

Value SharedMemoryObject::getBaseBeforeSlice(int dim, Location loc,
                                             RewriterBase &rewriter) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value cSwizzleOffset = getCSwizzleOffset(dim);
  Value offset = b.sub(b.i32_val(0), cSwizzleOffset);
  Type type = base.getType();
  return b.gep(type, baseElemType, base, offset);
}

uint64_t
SharedMemoryObject::getMaskSpanOffsets(triton::gpu::MemDescType srcTy) {
  auto ctx = srcTy.getContext();
  auto shape = srcTy.getShape();
  auto allocShape = srcTy.getAllocShape();
  assert(allocShape.size() >= shape.size());
  assert(allocShape.size() - shape.size() <= 1);
  allocShape = allocShape.take_back(shape.size());

  // Early exist when there is no subview
  if (allocShape == shape) {
    return 0;
  }
  if (auto paddedEncoding = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
          srcTy.getEncoding())) {
    // Mask is used in fusion of constant part of memory operation address as
    // immediate operand. Padded layout has additional address computations
    // between main offset computation and actual memory access, which breaks
    // constand fusing. Full mask disables this optimization.
    return ~uint64_t(0);
  }
  auto totalLl = triton::gpu::toLinearLayout(allocShape, srcTy.getEncoding());
  auto dimNames = standardOutDimNames(ctx, shape.size());
  // Remove the kBlock dimension
  auto kOffset = StringAttr::get(ctx, "offset");
  totalLl = totalLl.sublayout({kOffset}, dimNames);
  // Map from dimNames to offset
  auto invLl = totalLl.invert();
  SmallVector<std::pair<StringAttr, int32_t>> logicalOffsets;
  for (auto dim : standardOutDimNames(srcTy.getContext(), shape.size())) {
    logicalOffsets.push_back({dim, 0});
  }

  auto ret = 0;
  for (auto [dim, shapes] : llvm::enumerate(llvm::zip(shape, allocShape))) {
    auto [shape, allocShape] = shapes;
    for (int j = llvm::Log2_32(shape); j < llvm::Log2_32(allocShape); ++j) {
      logicalOffsets[dim].second = 1 << j;
      ret |= invLl.apply(logicalOffsets)[0].second;
    }
    // Reset the offset for the next dimension
    logicalOffsets[dim].second = 0;
  }
  return ret;
}

Value SharedMemoryObject::getShmemOffset(Location loc, RewriterBase &rewriter,
                                         triton::gpu::MemDescType srcTy) const {
  auto ctx = srcTy.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // If it did not have a memdesc_subslice we don't need to compute the offset
  // as it is zero
  if (!isAffineSharedMemoryAccess(srcTy)) {
    return b.i32_val(0);
  }

  LinearLayout ll;
  // We return the offset without the padding. The padding will be added in the
  // lowering
  if (auto paddedSharedEncoding =
          dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
              srcTy.getEncoding())) {
    ll = paddedSharedEncoding.getLinearComponent();
  } else {
    ll = triton::gpu::toLinearLayout(srcTy);
  }

  auto dimNames = standardOutDimNames(ctx, offsets.size());
  SmallVector<std::pair<StringAttr, Value>> logicalOffsets;
  for (auto [dim, offset] : llvm::zip(dimNames, offsets)) {
    logicalOffsets.push_back({dim, offset});
  }

  ll = ll.sublayout({str_attr("offset")}, dimNames);
  auto offset =
      applyLinearLayout(loc, rewriter, ll.invert(), logicalOffsets)[0].second;
  return offset;
}

Value SharedMemoryObject::getShmemAffineBase(
    Location loc, RewriterBase &rewriter,
    triton::gpu::MemDescType srcTy) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value offset = getShmemOffset(loc, rewriter, srcTy);
  return b.gep(base.getType(), baseElemType, base, offset);
}

Value getStructFromSharedMemoryObject(Location loc,
                                      const SharedMemoryObject &smemObj,
                                      RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto elems = smemObj.getElems();
  auto types = smemObj.getTypes();
  auto structTy =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
  // pack into struct
  Value llvmStruct = LLVM::UndefOp::create(rewriter, loc, structTy);
  for (const auto &v : llvm::enumerate(elems)) {
    assert(v.value() && "can not insert null values");
    llvmStruct = b.insert_val(structTy, llvmStruct, v.value(), v.index());
  }
  return llvmStruct;
}

SharedMemoryObject getSharedMemoryObjectFromStruct(Location loc,
                                                   Value llvmStruct,
                                                   Type elemTy,
                                                   RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  ArrayRef<Type> types =
      cast<LLVM::LLVMStructType>(llvmStruct.getType()).getBody();
  SmallVector<Value> elems(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    elems[i] = b.extract_val(type, llvmStruct, i);
  }
  return {/*base=*/elems[0],
          /*baseElemType=*/elemTy,
          /*offsets=*/{elems.begin() + 1, elems.end()}};
}

Value getStackPointer(RewriterBase &rewriter, FunctionOpInterface funcOp) {
  // See NOTE: [Additional Function Arguments]
  if (!isKernel(funcOp)) {
    return funcOp.getArgument(funcOp.getNumArguments() + kSharedMemoryOffset);
  }

  auto mod = funcOp->getParentOfType<ModuleOp>();
  auto globalBase = dyn_cast<LLVM::GlobalOp>(mod.lookupSymbol("global_smem"));
  assert(globalBase);
  return LLVM::AddressOfOp::create(rewriter, funcOp.getLoc(), globalBase);
}

Value getGlobalScratchPtr(Location loc, RewriterBase &rewriter,
                          const TargetInfoBase &targetInfo,
                          FunctionOpInterface funcOp, Value allocOffset = {}) {
  // See NOTE: [Additional Function Arguments]
  if (!isKernel(funcOp)) {
    // Base for this function
    auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() +
                                       kGlobalScratchBufferOffset);
    if (!allocOffset) {
      return gmemBase;
    }

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext(), 1);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    return b.gep(ptrTy, i8_ty, gmemBase, allocOffset);
  }

  // Base for entire kernel
  auto gmemBase =
      funcOp.getArgument(funcOp.getNumArguments() + kGlobalScratchBufferOffset);

  ModuleOp mod = funcOp.getOperation()->getParentOfType<ModuleOp>();
  auto allocSizeAttr = mod.getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ttg.global_scratch_memory_size");
  if (!allocSizeAttr) {
    return gmemBase;
  }

  Value gridIdx[3];
  Value gridDim[2];
  for (int k = 0; k < 3; ++k) {
    gridIdx[k] = GetProgramIdOp::create(rewriter, loc, k);
  }
  for (int k = 0; k < 2; ++k) {
    gridDim[k] = GetNumProgramsOp::create(rewriter, loc, k);
  }

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value linearId = gridIdx[2];
  for (int k = 0; k < 2; ++k) {
    linearId = b.add(gridIdx[1 - k], b.mul(linearId, gridDim[1 - k]));
  }
  auto numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
  if (numCTAs > 1) {
    linearId = b.mul(linearId, b.i32_val(numCTAs));
    linearId = b.add(linearId, targetInfo.getClusterCTAId(rewriter, loc));
  }

  auto allocSize = allocSizeAttr.getValue().getZExtValue();

  Value offset = b.mul(linearId, b.i32_val(allocSize));
  if (allocOffset) {
    offset = b.add(offset, allocOffset);
  }

  auto *ctx = rewriter.getContext();
  auto res =
      b.gep(mlir::LLVM::LLVMPointerType::get(ctx, 1), i8_ty, gmemBase, offset);
  return res;
}

Value getProfileScratchPtr(Location loc, RewriterBase &rewriter,
                           FunctionOpInterface funcOp) {
  // See NOTE: [Additional Function Arguments]
  // FIXME(Keren): This is broken when we have device functions, we
  // need to implement proper calling convention
  return funcOp.getArgument(funcOp.getNumArguments() +
                            kProfileScratchBufferOffset);
}

Value getSharedMemoryBase(Location loc, RewriterBase &rewriter,
                          const TargetInfoBase &target, Operation *op) {
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                          target.getSharedAddressSpace());
  auto func = op->template getParentOfType<FunctionOpInterface>();
  if (!func)
    func = cast<FunctionOpInterface>(op);

  assert(op->hasAttr("allocation.offset"));
  size_t offset = cast<IntegerAttr>(op->getAttr("allocation.offset"))
                      .getValue()
                      .getZExtValue();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value offVal = b.i32_val(offset);
  Value base =
      b.gep(ptrTy, i8_ty, LLVM::getStackPointer(rewriter, func), offVal);
  return base;
}

// Extract the bits of `a` that are set in `mask`
Value pext_i32(RewriterBase &rewriter, Location loc, Value a, uint32_t mask) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(a.getType() == i32_ty && "a must be i32");
  // Handle width = 32 to avoid doing 1 << 32
  if (mask == 0xFFFFFFFF)
    return a;

  // Implements the blocked algorithm from
  // https://forums.developer.nvidia.com/t/pdep-and-pext-functionality-for-cuda/270973
  uint32_t mskConst = mask;
  uint32_t extcnt = 0;
  Value result = b.i32_val(0);
  while (mskConst) {
    uint32_t oldmsk = mskConst;
    uint32_t bitgrplsb = mskConst & (-mskConst);
    mskConst &= bitgrplsb + mskConst;
    uint32_t bitgrp = mskConst ^ oldmsk;
    uint32_t lsbpos = 31 - __builtin_clz(bitgrplsb);
    // like popcount for a number 0..01..1..0 but portable
    uint32_t grplen = __builtin_ctz(~(bitgrp >> lsbpos));
    uint32_t shift = lsbpos - extcnt;
    extcnt += grplen;
    result =
        b.or_(result, b.lshr(b.and_(b.i32_val(bitgrp), a), b.i32_val(shift)));
  }
  return result;
}

std::tuple<SmallVector<Value>, Value>
delinearize(RewriterBase &rewriter, Location loc,
            triton::gpu::DistributedEncodingTrait layout,
            ArrayRef<int64_t> shape, StringAttr dimName, Value linear) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ll = triton::gpu::toLinearLayout(shape, layout);
  assert(ll.hasInDim(dimName));
  int32_t freeVarMask = ll.getFreeVariableMasks()[dimName];
  auto isRepresentative = b.true_val();
  if (freeVarMask != 0) {
    isRepresentative =
        b.icmp_eq(b.and_(b.i32_val(freeVarMask), linear), b.i32_val(0));
    // We remove the bits of linear that are set to one in freeVarMask
    int32_t nonFreeVarMask = ~freeVarMask & (ll.getInDimSize(dimName) - 1);
    linear = pext_i32(rewriter, loc, linear, nonFreeVarMask);
  }

  auto linearLayout = triton::gpu::LinearEncodingAttr::get(
      rewriter.getContext(), std::move(ll));
  auto orderDim = linearLayout.orderPerDim(dimName, linearLayout.getOrder());
  auto shapeDim = linearLayout.basesPerDim(dimName);
  auto multiDim = delinearize(rewriter, loc, linear, shapeDim, orderDim);

  return std::make_tuple(std::move(multiDim), isRepresentative);
}

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order) {
  unsigned rank = shape.size();
  assert(rank == order.size());
  auto reordered = applyPermutation(shape, order);
  SmallVector<Value> reorderedMultiDim(rank);
  if (auto constantOp = linear.getDefiningOp<arith::ConstantOp>()) {
    unsigned intVal = mlir::cast<IntegerAttr>(constantOp.getValue())
                          .getValue()
                          .getSExtValue();
    reorderedMultiDim = delinearize(rewriter, loc, intVal, reordered);
  } else {
    reorderedMultiDim = delinearize(rewriter, loc, linear, reordered);
  }
  SmallVector<Value> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               unsigned linear, ArrayRef<unsigned> shape) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  unsigned remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    unsigned dimSize = en.value();
    multiDim[en.index()] = b.i32_val(remained % dimSize);
    remained = remained / dimSize;
  }
  return multiDim;
}

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  Value remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    Value dimSize = b.i32_val(en.value());
    multiDim[en.index()] = b.urem(remained, dimSize);
    remained = b.udiv(remained, dimSize);
  }
  return multiDim;
}

SmallVector<unsigned> delinearize(unsigned linear, ArrayRef<unsigned> shape,
                                  ArrayRef<unsigned> order) {
  auto rank = shape.size();
  assert(order.size() == rank);
  SmallVector<unsigned> multiDim(rank);
  for (auto dim : order) {
    multiDim[dim] = linear % shape[dim];
    linear /= shape[dim];
  }
  assert(linear == 0);
  return multiDim;
}

Value linearize(RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape, ArrayRef<unsigned> order) {
  return linearize(rewriter, loc, applyPermutation(multiDim, order),
                   applyPermutation(shape, order));
}

Value linearize(RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto rank = multiDim.size();
  Value linear = b.i32_val(0);
  if (rank > 0) {
    linear = multiDim.back();
    for (auto [dim, dimShape] :
         llvm::reverse(llvm::zip(multiDim.drop_back(), shape.drop_back()))) {
      Value dimSize = b.i32_val(dimShape);
      linear = b.add(b.mul(linear, dimSize), dim);
    }
  }
  return linear;
}

size_t linearize(ArrayRef<unsigned> multiDim, ArrayRef<unsigned> shape,
                 ArrayRef<unsigned> order) {
  size_t linear = 0;
  for (unsigned dim : llvm::reverse(order))
    linear = linear * shape[dim] + multiDim[dim];
  return linear;
}

Value addStringToModule(Location loc, RewriterBase &rewriter, StringRef key,
                        StringRef content) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<64> contentStr(content);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  LLVM::GlobalOp global;
  {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = LLVM::GlobalOp::create(rewriter, UnknownLoc::get(ctx), globalType,
                                    /*isConstant=*/true,
                                    LLVM::Linkage::Internal, stringConstName,
                                    rewriter.getStringAttr(contentStr));
  }

  Value zero = b.i32_val(0);
  Type globalPtrType = LLVM::LLVMPointerType::get(ctx, global.getAddrSpace());
  Value globalPtr = LLVM::AddressOfOp::create(
      rewriter, UnknownLoc::get(ctx), globalPtrType, global.getSymName());
  Value stringStart =
      b.gep(ptr_ty(ctx), i8_ty, globalPtr, SmallVector<Value>({zero}));
  return stringStart;
}

} // namespace LLVM

Value dot(RewriterBase &rewriter, Location loc, ArrayRef<Value> offsets,
          ArrayRef<Value> strides) {
  assert(offsets.size() == strides.size());
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value ret = b.i32_val(0);
  for (auto [offset, stride] : llvm::zip(offsets, strides)) {
    ret = b.add(ret, b.mul(offset, stride));
  }
  return ret;
}

// Isolated a single warp specialize op from above.
static void
makeWarpGroupsIsolatedFromAbove(triton::gpu::WarpSpecializeOp wsOp) {
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(wsOp.getPartitionOpHolder(), captures);
  for (Value capture : captures) {
    wsOp->insertOperands(wsOp.getNumOperands(), capture);
    for (Region *region : wsOp.getPartitionRegions()) {
      BlockArgument arg =
          region->addArgument(capture.getType(), capture.getLoc());
      replaceAllUsesInRegionWith(capture, arg, *region);
    }
  }
}

void makeAllWarpGroupsIsolatedFromAbove(Operation *op) {
  op->walk([](triton::gpu::WarpSpecializeOp wsOp) {
    makeWarpGroupsIsolatedFromAbove(wsOp);
  });
}

// TODO: Is there a better way to do this? This needs to be fixed upstream.
void fixUpLoopAnnotation(ModuleOp mod) {
  mod->walk([](Operation *op) {
    if (isa<LLVM::BrOp, LLVM::CondBrOp>(op)) {
      if (op->hasAttr("llvm.loop_annotation")) {
        auto loopMD = dyn_cast<LLVM::LoopAnnotationAttr>(
            op->getAttr("llvm.loop_annotation"));
        if (loopMD) {
          if (auto brOp = dyn_cast<LLVM::BrOp>(op)) {
            brOp.setLoopAnnotationAttr(loopMD);
          } else if (auto condBrOp = dyn_cast<LLVM::CondBrOp>(op)) {
            condBrOp.setLoopAnnotationAttr(loopMD);
          }
        }
      }
    }
  });
}

SmallVector<Value> inlineRegionImpl(RewriterBase &rewriter, Region &region,
                                    ArrayRef<Value> args,
                                    mlir::TypeID terminatorTypeId,
                                    Location loc) {
  // Inline regions with multiple blocks
  //
  //        Before                                   After
  //                                              ┌─────────┐
  //                                              │ op1     │
  //                    ┌──────────┐              │ cf.br   │
  //                    │region[0] │              └────┬────┘
  //                    │cf.cond_br├─┐            ┌────▼─────┐
  //                    └────┬─────┘ │            │region[0] │
  //                         │       │            │cf.cond_br├─┐
  // ┌───────┐          ┌────▼────┐  │            └────┬─────┘ │
  // │  op1  │  IP      │region[1]│  │            ┌────▼────┐  │
  // │       │◄───      │yield ...│  │            │region[1]│  │
  // │  op2  │          └─────────┘  │          ┌─┤cf.br    │  │
  // └───────┘                       │          │ └─────────┘  │
  //                    ┌─────────┐  │          │ ┌─────────┐  │
  //                    │region[2]│◄─┘          │ │region[2]│◄─┘
  //                    │yield    │             │ │cf.br    │
  //                    └─────────┘             │ └────┬────┘
  //                                            │ ┌────▼────┐
  //                                            └►│op2      │
  //                                              └─────────┘
  auto *curBlock = rewriter.getInsertionBlock();
  auto opPosition = rewriter.getInsertionPoint();
  auto *remainingOpsBlock = rewriter.splitBlock(curBlock, opPosition);

  IRMapping regionMap;
  Region &parent = *curBlock->getParent();
  rewriter.cloneRegionBefore(region, parent, parent.end(), regionMap);
  rewriter.setInsertionPointToEnd(curBlock);
  LLVM::BrOp::create(rewriter, loc, args, regionMap.lookup(&region.front()));

  ValueRange terminatorOperands;
  for (Block &origBlock : region) {
    Block *newBlock = regionMap.lookup(&origBlock);
    rewriter.moveBlockBefore(newBlock, remainingOpsBlock);

    auto terminator = newBlock->getTerminator();
    if (terminator->getRegisteredInfo()->getTypeID() == terminatorTypeId) {
      terminatorOperands = terminator->getOperands();
      rewriter.setInsertionPointAfter(terminator);
      rewriter.replaceOpWithNewOp<LLVM::BrOp>(terminator, terminatorOperands,
                                              remainingOpsBlock);
    }
  }

  rewriter.setInsertionPointToStart(remainingOpsBlock);
  SmallVector<Value> vals;
  for (auto resultTy : terminatorOperands.getType()) {
    auto val = remainingOpsBlock->addArgument(resultTy, loc);
    vals.push_back(val);
  }
  return vals;
}

void finalizeTensorAtomicResults(Operation *op, RankedTensorType tensorTy,
                                 ConversionPatternRewriter &rewriter,
                                 SmallVector<Value> &resultVals,
                                 Type valueElemTy, TritonLLVMOpBuilder &b,
                                 Value threadPred,
                                 const TargetInfoBase &targetInfo,
                                 const LLVMTypeConverter *typeConverter) {
  auto *ctx = rewriter.getContext();
  auto loc = op->getLoc();
  Type structTy = typeConverter->convertType(tensorTy);
  if (!op->hasAttr("allocation.offset")) {
    // No broadcasting, just pack the values into a struct
    Value resultStruct =
        packLLElements(loc, typeConverter, resultVals, rewriter, structTy);
    rewriter.replaceOp(op, {resultStruct});
    return;
  }

  auto dstLayout = triton::gpu::toLinearLayout(tensorTy);
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  dstLayout = dstLayout.sublayout({kReg, kLane, kWarp},
                                  llvm::to_vector(dstLayout.getOutDimNames()));
  dstLayout = dstLayout.reshapeOuts(
      {{str_attr("offset"), dstLayout.getTotalOutDimSize()}});
  auto smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op);

  auto emitSt = [&](RewriterBase &rewriter, Location loc, ArrayRef<Value> vals,
                    Value shmemAddr, int idx,
                    VectorType vecTy) -> SmallVector<Value> {
    auto length = vecTy.getNumElements();
    Value valsVec =
        packLLVector(loc, ArrayRef<Value>(vals).slice(idx, length), rewriter);
    targetInfo.storeDShared(rewriter, loc, shmemAddr, std::nullopt, valsVec,
                            threadPred);
    return {};
  };

  auto emitLd = [&](RewriterBase &rewriter, Location loc, ArrayRef<Value> vals,
                    Value shmemAddr, int idx,
                    VectorType vecTy) -> SmallVector<Value> {
    Value loadedVec = targetInfo.loadDShared(rewriter, loc, shmemAddr,
                                             std::nullopt, vecTy, b.true_val());
    return unpackLLVector(loc, loadedVec, rewriter);
  };

  auto noPaddingOffset = [](Value v) { return v; };
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  lowerLdSt(loc, ctx, dstLayout, resultVals, valueElemTy, smemBase,
            /*calcPaddedOffset=*/noPaddingOffset, /*affineOffset=*/b.i32_val(0),
            /*maskSpanAffineOffset=*/0, laneId, warpId, rewriter, targetInfo,
            /*maybeMaxVecElems=*/{}, emitSt);
  b.barrier();
  resultVals = lowerLdSt(loc, ctx, dstLayout, resultVals, valueElemTy, smemBase,
                         /*calcPaddedOffset=*/noPaddingOffset,
                         /*affineOffset=*/b.i32_val(0),
                         /*maskSpanAffineOffset=*/0, laneId, warpId, rewriter,
                         targetInfo, /*maybeMaxVecElems=*/{}, emitLd);

  // Create the result struct and replace the operation
  Value resultStruct =
      packLLElements(loc, typeConverter, resultVals, rewriter, structTy);
  rewriter.replaceOp(op, {resultStruct});
}

} // namespace mlir
