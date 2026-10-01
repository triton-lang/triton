#include "Utility.h"
#include "Dialect/NVGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <tuple>

namespace mlir {
namespace LLVM {
namespace NVIDIA {
using namespace mlir::triton;

static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter, Value val,
                               Value i, NVVM::ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleCommonImpl(loc, rewriter, val0, i, mode, clamp);
    val1 = shuffleCommonImpl(loc, rewriter, val1, i, mode, clamp);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }
  Type type = val.getType();
  if (type != i32_ty) {
    val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.zext(i32_ty, val);
  }
  Value mask = b.i32_val(0xFFFFFFFF);
  Value result = NVVM::ShflOp::create(rewriter, loc, i32_ty, mask, val, i,
                                      clamp, mode, UnitAttr());
  if (type != i32_ty) {
    if (bits < 32)
      result = b.trunc(int_ty(bits), result, LLVM::IntegerOverflowFlags::nuw);
    result = b.bitcast(result, type);
  }
  return result;
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, NVVM::ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // To shuffle pointers, convert them to i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = shuffleCommonImpl(loc, rewriter, val, i, mode, clamp);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), NVVM::ShflKind::bfly,
                       b.i32_val(0x1f));
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), NVVM::ShflKind::up,
                       b.i32_val(0x0));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(loc, rewriter, val, b.i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, i, NVVM::ShflKind::idx,
                       b.i32_val(0x1f));
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis) {
  assert(moduleOp);

  // A program spans one CTA when numCTAs == 1 and one cluster otherwise.
  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

  if (numCTAs == 1) {
    switch (axis) {
    case ProgramIDDim::X:
      return NVVM::BlockIdXOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Y:
      return NVVM::BlockIdYOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Z:
      return NVVM::BlockIdZOp::create(rewriter, loc, i32_ty);
    }
  } else {
    switch (axis) {
    case ProgramIDDim::X: {
      // Clusters are launched with dimensions (numCTAs, 1, 1).
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      Value ctaId = NVVM::BlockIdXOp::create(rewriter, loc, i32_ty);
      return b.udiv(ctaId, b.i32_val(numCTAs));
    }
    case ProgramIDDim::Y:
      return NVVM::BlockIdYOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Z:
      return NVVM::BlockIdZOp::create(rewriter, loc, i32_ty);
    }
  }
  llvm_unreachable("invalid axis");
}

Value permute(Location loc, RewriterBase &rewriter, Value a, Value b,
              Value selector) {
  Value args[] = {a, b, selector};
  auto op =
      createLLVMIntrinsicCallOp(rewriter, loc, "llvm.nvvm.prmt", i32_ty, args);
  return op.getResult(0);
}

/// Create a predicate with just single active thread.
Value createElectPredicate(Location loc, OpBuilder &rewriter) {
  return NVVM::ElectSyncOp::create(rewriter, loc, i1_ty,
                                   /*membermask=*/Value());
}

void createSyncWarp(Location loc, OpBuilder &rewriter) {
  TritonLLVMOpBuilder b(loc, rewriter);
  NVVM::SyncWarpOp::create(rewriter, loc, b.i32_val(0xffffffff));
}

Value createElectPredicateWarp0(Location loc, OpBuilder &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value warpId = getLaneAndWarpId(rewriter, loc).second;
  Value warp0 = b.icmp_eq(warpId, b.i32_val(0));
  return b.and_(warp0, createElectPredicate(loc, rewriter));
}

Value createTMAMulticastMask(Location loc, ConversionPatternRewriter &rewriter,
                             uint16_t broadcastBits, Value ctaId) {
  int numCTAs = triton::gpu::lookupNumCTAs(rewriter);
  auto encoding =
      triton::nvidia_gpu::getTMAMulticastMaskEncoding(numCTAs, broadcastBits);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (!ctaId)
    ctaId = nvgpu::ClusterCTAIdOp::create(rewriter, loc);
  Value base = b.and_(ctaId, b.i32_val(encoding.fixedBits));
  return b.shl(b.i32_val(encoding.pattern), base);
}

uint32_t getCGABroadcastMask(mlir::triton::gpu::MemDescType barrierTy) {
  auto kBlock = StringAttr::get(barrierTy.getContext(), "block");
  return toLinearLayout(barrierTy).getFreeVariableMasks().lookup(kBlock);
}

std::optional<Value>
getLeaderCTAPredicate(Location loc, ConversionPatternRewriter &rewriter,
                      mlir::triton::gpu::MemDescType barrierTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  uint32_t maskCGABroadcast = getCGABroadcastMask(barrierTy);
  if (!maskCGABroadcast)
    return std::nullopt;

  Value ctaId = nvgpu::ClusterCTAIdOp::create(rewriter, loc);
  Value ctaIdInGroup = b.and_(ctaId, b.i32_val(maskCGABroadcast));
  return std::optional<Value>(b.icmp_eq(ctaIdInGroup, b.i32_val(0)));
}

Value getLeaderAddress(Location loc, ConversionPatternRewriter &rewriter,
                       Value barrierPtr,
                       mlir::triton::gpu::MemDescType barrierTy) {
  uint32_t barrierMask = getCGABroadcastMask(barrierTy);
  if (!barrierMask)
    return barrierPtr;

  // Trick from cutlass to implement a faster `mapa` via a single and
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  uint32_t fullMask = ~(barrierMask << 24);
  Value barrierInt = b.ptrtoint(i32_ty, barrierPtr);
  barrierInt = b.and_(barrierInt, b.i32_val(fullMask));
  return b.inttoptr(barrierPtr.getType(), barrierInt);
}

Value createLeadCTAPredicate(Location loc, RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value leftClusterId = nvgpu::ClusterCTAIdOp::create(rewriter, loc);
  leftClusterId = b.and_(leftClusterId, b.i32_val(1));
  Value cluster0 = b.icmp_eq(leftClusterId, b.i32_val(0));
  return cluster0;
}

// Choose row reg bits to minimise bank conflicts, then preserve packed words.
// quot is the layout after dividing out the contiguous 16-byte tile;
// regPerm is the register permutation used for that division.
static ColumnAction getLdStMatrixRowPermutation(const LinearLayout &quot,
                                                const ColumnAction &regPerm,
                                                unsigned numOffsetRegBits,
                                                unsigned numRowRegBits,
                                                int bitwidth) {
  auto *ctx = quot.getInDimNames().begin()->getContext();
  auto S = [ctx](StringRef name) { return StringAttr::get(ctx, name); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kOffset = S("offset");
  unsigned numRegBits = quot.getInDimSizeLog2(kReg) + numOffsetRegBits;
  assert(numRowRegBits <= 2 &&
         "matrix instructions use at most two row reg bits");
  if (numRowRegBits == 0)
    return ColumnAction::identity(kReg, numRegBits);

  // The quotient measures 16-byte rows. A 128-byte bank-conflict phase
  // varies over numRowRegBits register bits and the remaining lane bits.
  // Other register bits only translate phases, so vectorisation is
  // independent.
  constexpr unsigned numPhaseBits = 3;
  constexpr int numRowsPerPhase = 1 << numPhaseBits;
  const auto &regs = quot.getBases().lookup(kReg);
  const auto &lanes = quot.getBases().lookup(kLane);
  assert(regs.size() >= numRowRegBits &&
         lanes.size() >= numPhaseBits - numRowRegBits);
  int bankRows = std::min(numRowsPerPhase, quot.getOutDimSize(kOffset));
  auto smem = LinearLayout::identity1D(bankRows, S("bank"), kOffset) *
              LinearLayout::identity1D(quot.getOutDimSize(kOffset) / bankRows,
                                       S("segment"), kOffset);
  SmallVector<size_t> best;
  std::tuple<int, int, int> bestCost{std::numeric_limits<int>::max(), 0, 0};
  // numRowRegBits is 1 or 2
  // we iterate all the permutations in 1 or 2 elements
  // i, j are register positions
  for (unsigned i = 0; i < regs.size(); ++i) {
    for (unsigned j = 0; j < (numRowRegBits == 2 ? regs.size() : 1); ++j) {
      if (numRowRegBits == 2 && i == j)
        continue;
      SmallVector<size_t> selected{i};
      if (numRowRegBits == 2)
        selected.push_back(j);
      // register and lane bases in one 128b transaction
      SmallVector<int32_t> phase;
      // Number of bits in a 32b that are not moved.
      int wordBits = 0;
      // Number of bits that keep their position (proxy for less PRMT)
      int orderedBits = 0;
      for (auto [pos, reg] : llvm::enumerate(selected)) {
        phase.push_back(regs[reg][0]);
        size_t originalBit = regPerm.getSourceIndex(numOffsetRegBits + reg);
        wordBits += originalBit < llvm::Log2_32(32 / bitwidth);
        orderedBits += originalBit == pos;
      }
      for (unsigned lane = 0; lane < numPhaseBits - numRowRegBits; ++lane)
        phase.push_back(lanes[lane][0]);
      // Prefer fewer source words per packed i32, then preserve byte order.
      // Fixed offset bits contribute the same packing cost to every choice.
      auto cost =
          std::make_tuple(triton::gpu::bankConflicts(phase, phase, smem).first,
                          -wordBits, -orderedBits);
      if (cost < bestCost) {
        bestCost = cost;
        best = std::move(selected);
      }
    }
  }
  SmallVector<size_t> order;
  for (auto reg : best)
    order.push_back(numOffsetRegBits + reg);
  for (unsigned i = 0; i < numRegBits; ++i)
    if (!llvm::is_contained(order, i))
      order.push_back(i);
  return ColumnAction(order, kReg, numRegBits);
}

LogicalResult lowerLdStMatrix(
    Location loc, LinearLayout cvt, bool transpose,
    SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
    Value smemBase, Value affineOffset, uint64_t maskSpanAffineOffset,
    Type llvmElemTy, ConversionPatternRewriter &rewriter,
    const ::triton::NVIDIA::TargetInfo &targetInfo) {
  // Lower load via ldmatrix, store via stmatrix

  bool isStore = !vals.empty();
  if (isStore && !targetInfo.supportStMatrix())
    return failure();
  if (!isStore && !targetInfo.supportLdMatrix())
    return failure();

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();

  auto S = [ctx](StringRef v) { return StringAttr::get(ctx, v); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kOffset = S("offset");
  auto kBlock = S("block");
  auto kAddr = S("addr");
  auto smemPtrTy = ptr_ty(ctx, 3);
  auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
  // In the contiguous case we can pack elements <= 32 bits
  // In the transpose case we just have the b8 and b16 cases
  if ((!transpose && bitwidth > 32) ||
      (transpose && !(bitwidth == 16 ||
                      (bitwidth == 8 && targetInfo.supportLdStMatrixB8()))))
    return failure();

  // Map onto offsets (contiguous part) and addr (non-contiguous part)
  LinearLayout fullTile;
  // Contiguous tile
  LinearLayout tile;
  // Just used in the transpose case
  ColumnAction permLanes;
  if (!transpose) {
    tile = LinearLayout::identity1D(32 / bitwidth, kReg, kOffset) *
           LinearLayout::identity1D(4, kLane, kOffset);
    fullTile = tile * LinearLayout::identity1D(8, kLane, kAddr);
  } else {
    // We permute the lanes and registers of the layout to the front as to be
    // able to divideLeft by the relevant tile

    // This is the same as permuting the lanes and registers to the front in
    // fullTile and taking the kOffset sublayout.
    tile = (LinearLayout::identity1D(8, kLane, kOffset) *
            LinearLayout::identity1D(16 / bitwidth, kReg, kOffset))
               .transposeIns({kReg, kLane});
    // Thank you PTX
    auto contigRegs = (isStore && bitwidth == 8 ? 16 : 32) / bitwidth;
    fullTile = LinearLayout::identity1D(contigRegs, kReg, kAddr) *
               LinearLayout::identity1D(4, kLane, kAddr) * tile;
    // Not enough registers to cover the full tile
    if (cvt.getInDimSize(kReg) < fullTile.getInDimSize(kReg)) {
      return failure();
    }
    // Move offset to the front
    permLanes = ColumnAction({2, 3, 4, 0, 1}, kLane, 5);
    cvt = permLanes.apply(cvt);
  }

  // If we are lowering a subslice, the subslice offsets shall not touch the
  // contiguous part of the tile
  if (maskSpanAffineOffset & (tile.getOutDimSize(kOffset) - 1))
    return failure();

  // Find if there is a register permutation that allows us to divideLeft
  auto maybePermutation = regPermForDivide(cvt, tile, /*left=*/true);
  if (!maybePermutation)
    return failure();

  // Accumulate the permutations to apply the inverse for loads
  ColumnAction regPerm = *maybePermutation;
  cvt = regPerm.apply(cvt);
  auto maybeQuot = divideLeft(cvt, tile);
  if (!maybeQuot.has_value()) {
    return failure();
  }

  // From here on we perform the lowering
  auto reps = zerosLike(tile) * maybeQuot.value();

  // log2 of registers per contig instruction
  unsigned numOffsetRegBits = tile.getInDimSizeLog2(kReg);
  // log2 of registers per row of the tile
  unsigned numRowRegBits = fullTile.getInDimSizeLog2(kReg) - numOffsetRegBits;
  auto permRows = getLdStMatrixRowPermutation(
      *maybeQuot, regPerm, numOffsetRegBits, numRowRegBits, bitwidth);
  reps = permRows.apply(reps);
  regPerm = regPerm.leftCompose(permRows);

  // We revert the lane permutation that we performed to be able to divideLeft
  if (transpose) {
    reps = permLanes.inverse().apply(reps);
  }
  // Sanity check (of the asymmetry between ldmatrix.b8 and stmatrix.b8):
  // All the instructions move 32 bytes of data on .x1 but ldmatrix.b8 which
  // moves 64 bytes...
  auto regsPerCoreTile = fullTile.getInDimSize(kReg);
  assert(regsPerCoreTile * bitwidth ==
         ((!isStore && bitwidth == 8 && transpose) ? 64 : 32));

  // Choose the vectorisation factor
  // We want to send at most 128 bits of data per thread as that's the maximum
  // vectorisation for all the instructions (even the weird ldmatrix.b8)
  auto vec = std::min<int32_t>(128 / bitwidth, reps.getInDimSize(kReg)) /
             regsPerCoreTile;
  assert(vec == 1 || vec == 2 || vec == 4);
  auto fullTileVec = fullTile * LinearLayout::identity1D(vec, kReg, kAddr);
  // just add warps as compose belowe requires the dimensions of both layouts to
  // agree
  fullTileVec *= LinearLayout::identity1D(1, kWarp, kAddr);
  // fullTile.invert() is a map from kOffset, kAddr into kReg, kLane, kWarp
  // addrToOffset gives us a map from kAddr into kOffset, which is the map of
  // the addresses each lane should hold
  auto addrToOffset = fullTileVec.invert().compose(reps);
  // sanity check
  assert(addrToOffset.getInDimSizeLog2(kAddr) >= 3 &&
         addrToOffset.getInDimSizeLog2(kAddr) <= 5);

  LinearLayout addrLayout =
      LinearLayout({{kLane, addrToOffset.getBases().lookup(kAddr)},
                    {kWarp, reps.getBases().lookup(kWarp)}},
                   {{kOffset, reps.getOutDimSize(kOffset)}}, false);

  // Matrix accesses are CTA-local. Model that with a trivial block output so
  // additive stride analysis always compares (offset, block) components.
  reps =
      reps.reshapeOuts({{kOffset, reps.getOutDimSize(kOffset)}, {kBlock, 1}});
  addrLayout = addrLayout.reshapeOuts(reps.getOutDims());
  // Compute the bits that are moved by one instruction
  // Compute elements for which we can swap the xor by an add
  auto [nAdditive, permStrides] = actionAdditiveStrides(
      reps, addrLayout, maskSpanAffineOffset, /*maskSpanBlocks=*/0,
      fullTileVec.getInDimSize(kReg));
  reps = permStrides.apply(reps);
  regPerm = regPerm.leftCompose(permStrides);

  // PTX expects the address increments to be done in bytes
  // If we don't perform the computations in i8, the compiler would
  // have to divide the computation by bitwdith / 8 and then lift this
  // shl, which often it's not able to do.
  // Adding a kReg dimension is a convenient hack.
  // We should just multiply all the bases by bitwidth / 8
  // and then remove the kReg dimension.
  assert(bitwidth >= 8);
  auto i8Tile =
      LinearLayout::zeros1D(bitwidth / 8, kReg, kOffset, bitwidth / 8);
  auto i8AddrLayout = i8Tile * addrLayout;

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  auto regBase =
      applyLinearLayout(
          loc, rewriter, i8AddrLayout,
          {{kReg, b.i32_val(0)}, {kLane, laneId}, {kWarp, warpId}})[0]
          .second;

  // It's fine that we don't compute the offset in bytes as affineOffset
  // will be folded into a constant
  auto affineOffsetI8 = b.mul(affineOffset, b.i32_val(bitwidth / 8));
  regBase = b.xor_(regBase, affineOffsetI8);

  // Instruction params
  auto layout = transpose ? NVVM::MMALayout::col : NVVM::MMALayout::row;
  auto eltType = transpose && bitwidth == 8 ? NVVM::LdStMatrixEltType::B8
                                            : NVVM::LdStMatrixEltType::B16;
  int m = fullTile.getOutDimSize(kAddr);
  int n = fullTile.getOutDimSize(kOffset) * bitwidth /
          (eltType == NVVM::LdStMatrixEltType::B8 ? 8 : 16);
  if (transpose) {
    std::swap(m, n);
  }
  auto shape = NVVM::LdStMatrixShapeAttr::get(ctx, m, n);

  // Elements per op
  auto elemsPerInstr = fullTileVec.getInDimSize(kReg);
  auto elemsPerVec = 32 / bitwidth;
  auto vecTy = vec_ty(llvmElemTy, elemsPerVec);
  if (isStore)
    vals = regPerm.apply(vals);
  for (int i = 0; i < cvt.getInDimSize(kReg); i += nAdditive) {
    auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
    auto regIdxI8 = regIdx * (bitwidth / 8);
    Value offset = b.xor_(regBase, b.i32_val(regIdxI8));
    for (int i2 = 0; i2 < nAdditive; i2 += elemsPerInstr) {
      // all these constants will go as immediate values to LDSM/STSM
      auto regIdxAdd =
          reps.apply({{kReg, i2}, {kLane, 0}, {kWarp, 0}})[0].second;
      auto regIdxAddI8 = regIdxAdd * (bitwidth / 8);
      Value innerOffset = b.add(offset, b.i32_val(regIdxAddI8));
      auto vecAddr = b.gep(smemPtrTy, i8_ty, smemBase, innerOffset,
                           LLVM::GEPNoWrapFlags::inbounds);
      if (isStore) {
        // Pack into vector of i32
        SmallVector<Value> inputs;
        for (int j = 0; j < elemsPerInstr; j += elemsPerVec) {
          Value input = packLLVector(
              loc, ArrayRef(vals).slice(i + i2 + j, elemsPerVec), rewriter);
          inputs.push_back(b.bitcast(input, i32_ty));
        }
        NVVM::StMatrixOp::create(rewriter, loc, vecAddr, inputs, layout, shape,
                                 eltType);
      } else {
        unsigned numOutputRegs = elemsPerInstr / elemsPerVec;
        assert(numOutputRegs > 0 &&
               "ldmatrix must load at least one 32-bit register per thread");
        Type ldResultTy =
            numOutputRegs == 1
                ? i32_ty
                : static_cast<Type>(LLVM::LLVMStructType::getLiteral(
                      ctx, SmallVector<Type>(numOutputRegs, i32_ty)));
        auto res = NVVM::LdMatrixOp::create(rewriter, loc, ldResultTy, vecAddr,
                                            vec, layout, shape, eltType)
                       .getResult();
        // Extract result into vals
        for (Value output : unpackLLElements(loc, res, rewriter)) {
          llvm::append_range(
              vals, unpackLLVector(loc, b.bitcast(output, vecTy), rewriter));
        }
      }
    }
  }
  if (!isStore) {
    // apply all the inverse permutations in the reverse order
    assert(vals.size() == cvt.getInDimSize(kReg));
    vals = regPerm.inverse().apply(vals);
  }
  return success();
}
} // namespace NVIDIA
} // namespace LLVM
} // namespace mlir
