#include "Utility.h"
#include "Dialect/NVGPU/IR/Dialect.h"
#include "DotOpToLLVM/MMAHelpers.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/NvmmaSmemAttrs.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <bit>
#include <limits>
#include <tuple>

namespace mlir {
namespace LLVM {
namespace NVIDIA {
using namespace mlir::triton;

static Value shuffleWords(Location loc, RewriterBase &rewriter, Value val,
                          llvm::function_ref<Value(Value)> shuffleWord) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type type = val.getType();
  if (isa<LLVM::LLVMPointerType>(type))
    return b.inttoptr(type, shuffleWords(loc, rewriter, b.ptrtoint(i64_ty, val),
                                         shuffleWord));
  unsigned bits = type.getIntOrFloatBitWidth();
  if (bits == 64) {
    auto words =
        unpackLLVector(loc, b.bitcast(val, vec_ty(i32_ty, 2)), rewriter);
    for (Value &word : words)
      word = shuffleWord(word);
    return b.bitcast(packLLVector(loc, words, rewriter), type);
  }
  if (type == i32_ty)
    return shuffleWord(val);
  Value word = b.bitcast(val, int_ty(bits));
  if (bits < 32)
    word = b.zext(i32_ty, word);
  word = shuffleWord(word);
  if (bits < 32)
    word = b.trunc(int_ty(bits), word, LLVM::IntegerOverflowFlags::nuw);
  return b.bitcast(word, type);
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, NVVM::ShflKind mode) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value mask = b.i32_val(-1);
  Value clamp = b.i32_val(31);
  return shuffleWords(loc, rewriter, val, [&](Value word) -> Value {
    return NVVM::ShflOp::create(rewriter, loc, i32_ty, mask, word, i, clamp,
                                mode, UnitAttr());
  });
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, val, b.i32_val(i), NVVM::ShflKind::bfly);
}

static std::pair<Value, Value>
shuffleWithPredicate(Location loc, RewriterBase &rewriter, Value val, int delta,
                     mlir::gpu::ShuffleMode mode) {
  Value sourceInBounds;
  auto shuffleWord = [&](Value word) -> Value {
    auto shuffle =
        mlir::gpu::ShuffleOp::create(rewriter, loc, word, delta, 32, mode);
    // Both halves of a 64-bit value read the same source lane.
    if (!sourceInBounds)
      sourceInBounds = shuffle.getValid();
    return shuffle.getShuffleResult();
  };
  Value result = val.getType().isF32()
                     ? shuffleWord(val)
                     : shuffleWords(loc, rewriter, val, shuffleWord);
  return {result, sourceInBounds};
}

std::pair<Value, Value> shuffleUp(Location loc, RewriterBase &rewriter,
                                  Value val, int delta) {
  return shuffleWithPredicate(loc, rewriter, val, delta,
                              mlir::gpu::ShuffleMode::UP);
}

std::pair<Value, Value> shuffleDown(Location loc, RewriterBase &rewriter,
                                    Value val, int delta) {
  return shuffleWithPredicate(loc, rewriter, val, delta,
                              mlir::gpu::ShuffleMode::DOWN);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(loc, rewriter, val, b.i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  return shuffleCommon(loc, rewriter, val, i, NVVM::ShflKind::idx);
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
  return NVVM::PermuteOp::create(rewriter, loc, i32_ty, a, b, selector,
                                 NVVM::PermuteMode::DEFAULT);
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

namespace mlir::triton::NVIDIA {

FailureOr<DotOpMmaSmemLoader> DotOpMmaSmemLoader::build(
    Location loc, RewriterBase &rewriter, gpu::MemDescType memTy,
    Value smemBase, ArrayRef<unsigned> instrShape, unsigned MNdim,
    int mmaVersion, bool isFp4, std::optional<RankedTensorType> mmaTy) {
  auto ctx = memTy.getContext();
  auto kOffset = str_attr("offset");
  // Keep the allocation layout so descriptor tiling can handle subviews.
  auto ll = toLinearLayoutIgnoringPadding(memTy);
  unsigned bitwidth = memTy.getElementTypeBitWidth();
  if (isFp4) {
    auto dims = to_vector(ll.getOutDimNames());
    auto trans = ll.getBasis(kOffset, 0, dims[0]) == 1;
    ll = (LinearLayout::identity1D(2, kOffset, dims[trans ? 0 : 1]) * ll)
             .transposeOuts(dims);
    bitwidth /= 2;
  }
  return build(loc, rewriter, ll.pseudoinvert(), bitwidth, smemBase, instrShape,
               MNdim, mmaVersion, mmaTy);
}

FailureOr<DotOpMmaSmemLoader> DotOpMmaSmemLoader::build(
    Location loc, RewriterBase &rewriter, const LinearLayout &ll, int bitwidth,
    Value smemBase, ArrayRef<unsigned> instrShape, unsigned MNdim,
    int mmaVersion, std::optional<RankedTensorType> mmaTy) {
  // ll is a map from two dimensions (dim0, dim1) or (row, col) into offsets
  // and blocks
  auto ctx = rewriter.getContext();
  auto kOffset = str_attr("offset");
  auto kBlock = str_attr("block");
  assert(ll.getNumOutDims() == 2);
  assert(ll.hasOutDim(kOffset) && ll.hasOutDim(kBlock));

  assert(mmaVersion == 3 || mmaVersion == 5);
  // Just needed for MMAv3
  assert(mmaTy.has_value() == (mmaVersion == 3));
  assert(MNdim < 2);
  assert(instrShape.size() == 2);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // Due to having a 16B alignment, we can compute the offsets in 128b
  // elements
  // TODO We should assert in the verifier that the alignment is at least 16B
  smemBase = b.ptrtoint(i32_ty, smemBase);
  Value baseSrcb128 = b.lshr(smemBase, b.i32_val(4));

  if (mmaVersion == 3) {
    auto mmaLl = gpu::toLinearLayout(mmaTy.value());
    auto outDims = to_vector(mmaLl.getOutDimNames());
    auto kWarp = str_attr("warp");
    // Map from warps into the MN dimension
    auto mmaWarps = mmaLl.sublayout({kWarp}, {outDims[MNdim]}) *
                    LinearLayout::identity1D(1, kWarp, outDims[1 - MNdim]);
    // Map from warps to offsets in bitwidth elements
    auto warpToOffset = mmaWarps.compose(ll);
    // Map from warps to offsets in 128b elements
    auto maybeWarpToOffsetb128 = divideLeft(
        warpToOffset, LinearLayout::zeros1D(1, kWarp, kOffset, 128 / bitwidth));
    assert(maybeWarpToOffsetb128.has_value());
    // zero out the first two warp bases to have a warpgroup to offset map
    auto bases = maybeWarpToOffsetb128->getBases();
    assert(maybeWarpToOffsetb128->getNumOutDims() == 2);
    bases[kWarp][0] = {0, 0};
    bases[kWarp][1] = {0, 0};
    auto warpGroupToOffsetb128 =
        LinearLayout(std::move(bases), warpToOffset.getOutDims(),
                     /*requireSurjective=*/false);
    Value warpId = mlir::triton::gpu::WarpIdOp::create(rewriter, loc);
    Value warpStrideb128 =
        applyLinearLayout(loc, rewriter, warpGroupToOffsetb128,
                          {{kWarp, warpId}})[0]
            .second;
    baseSrcb128 = b.add(baseSrcb128, warpStrideb128);
  }

  // Padding may supply unused result columns, but not reduction elements.
  auto dims = to_vector(ll.getInDimNames());
  if (instrShape[1 - MNdim] > ll.getInDimSize(dims[1 - MNdim]))
    return mlir::emitError(loc,
                           "MMA instruction exceeds the logical K dimension");

  auto desc = getDescriptor(loc, ll, instrShape, bitwidth, MNdim, mmaVersion);
  if (failed(desc))
    return failure();

  Value baseb128 = b.and_(baseSrcb128, b.i32_val(0x7FFF));
  return DotOpMmaSmemLoader{*desc, baseb128, ll};
}

Value DotOpMmaSmemLoader::smemLoad(int a, int b,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc, int kSize) const {
  auto *ctx = loc.getContext();
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  auto dims = to_vector(ll.getInDimNames());
  assert(to_vector(ll.getOutDimNames()) ==
         llvm::to_vector(
             ArrayRef<StringAttr>{str_attr("offset"), str_attr("block")}));
  auto offsetBlock = ll.apply({{dims[0], a}, {dims[1], b}});
  int32_t offsetElems = offsetBlock[0].second;
  int32_t block = offsetBlock[1].second;
  assert(block == 0);
  int32_t smemByteOffsetb8 = offsetElems * desc.bitwidth / 8;
  auto currDesc = desc.descriptor;
  // Take the next 0/1/2/3 bits after the 128b tile
  uint32_t mask = (desc.swizzlingByteWidth >> 4) - 1;
  currDesc.matrixBaseOffset = (smemByteOffsetb8 / 128) & mask;
  // Packed FP4 K96 may straddle a 128-byte swizzle sector. SM103 can
  // address the second chunk through an absolute leading dimension.
  // The caller selects this only for K-major, 128-byte-swizzled operands
  // whose MN tile origins preserve a zero matrix base offset.
  int k = desc.transposed ? a : b;
  bool useAbsolute = kSize == 96 && k % 256 + kSize > 256;
  if (useAbsolute) {
    currDesc.leadDimensionAbsoluteMode = 1;
    currDesc.leadDimensionBaseOffset = 0;
  }
  int32_t smemByteOffsetb128 = smemByteOffsetb8 >> 4;
  uint64_t descBits = std::bit_cast<uint64_t>(currDesc) + smemByteOffsetb128;
  // Add the base address to the descriptor
  Value low = tb.add(tb.i32_val(uint32_t(descBits)), baseb128);
  Value high = tb.i32_val(descBits >> 32);
  Value descWords = packLLVector(loc, {low, high}, rewriter);
  Value result = tb.bitcast(descWords, i64_ty);
  if (useAbsolute) {
    int nextK = (k / 256 + 1) * 256;
    Value next = smemLoad(desc.transposed ? nextK : a,
                          desc.transposed ? b : nextK, rewriter, loc);
    Value nextAddress = tb.and_(next, tb.i64_val(0x3fff));
    result = tb.or_(result, tb.shl(nextAddress, tb.i64_val(16)));
  }
  return result;
}

MemDescOperand DotOpMmaSmemLoader::memLoad(int a, int b,
                                           ConversionPatternRewriter &rewriter,
                                           Location loc, int kSize) const {
  return {smemLoad(a, b, rewriter, loc, kSize), std::nullopt};
}

FailureOr<MMASMEMDescriptor>
DotOpMmaSmemLoader::getDescriptor(Location loc, const LinearLayout &ll,
                                  ArrayRef<unsigned> instrShape, int bitwidth,
                                  unsigned MNdim, int mmaVersion) {
  // ll is a map from allocShape into offsets and blocks
  auto dims = to_vector(ll.getInDimNames());
  auto ctx = dims[0].getContext();
  auto kOffset = str_attr("offset");
  auto attrsAndCandidate = nvidia_gpu::getNvmmaSmemAttrs(ll, bitwidth);
  if (!attrsAndCandidate)
    return failure();

  auto [attrs, shmemTileInv] = std::move(*attrsAndCandidate);
  unsigned atomMN = shmemTileInv.transposeIns({dims[MNdim], dims[1 - MNdim]})
                        .getNumConsecutiveInOut();
  if (mmaVersion == 5 && instrShape[MNdim] < atomMN)
    return mlir::emitError(loc, "instruction M/N must cover at least ")
           << atomMN << " contiguous elements of the shared-memory core";
  if (instrShape[MNdim] > std::max(ll.getInDimSize(dims[MNdim]),
                                   shmemTileInv.getInDimSize(dims[MNdim])) &&
      (attrs.swizzlingByteWidth != 0 ||
       ll.getInDimSize(dims[MNdim]) > shmemTileInv.getInDimSize(dims[MNdim]))) {
    auto inDims = ll.getInDims();
    return mlir::emitError(loc)
           << "instruction shape [" << instrShape[0] << ", " << instrShape[1]
           << "] is too large for the layout with block size ["
           << inDims[0].second << ", " << inDims[1].second << "]";
  }
  int swizzling = attrs.swizzlingByteWidth;
  bool transposed = attrs.transposed;
  bool fp4Padded = attrs.fp4Padded;

  // The PTX docs are wrong in subtle ways:
  // 1) LBO can be specified for kContig && swizzled != 0
  //    PTX says it's assumed to be 1, but  we can in fact use it
  // 2) The Cute layouts for kContig && swizzled != 0 are wrong
  int lbo = 0, sbo = 0;
  int leadingDim = transposed ? 0 : 1;
  int stridedDim = transposed ? 1 : 0;
  // The lbo / sbo is swapped for swizzling == 0 and MNContig lol
  bool MNContig = (MNdim == 0) == transposed;
  if (swizzling == 0 && MNContig) {
    std::swap(leadingDim, stridedDim);
  }
  auto log2RowsTile = shmemTileInv.getInDimSizeLog2(dims[leadingDim]);
  if (llvm::Log2_32(instrShape[leadingDim]) > log2RowsTile) {
    lbo = ll.getBasis(dims[leadingDim], log2RowsTile, kOffset);
  }

  auto log2ColsTile = shmemTileInv.getInDimSizeLog2(dims[stridedDim]);
  // Keep SBO zero when the remaining M/N coordinates are all padding.
  if (llvm::Log2_32(instrShape[stridedDim]) > log2ColsTile &&
      log2ColsTile < ll.getInDimSizeLog2(dims[stridedDim])) {
    sbo = ll.getBasis(dims[stridedDim], log2ColsTile, kOffset);
  }
  if ((lbo * bitwidth) % 128 || (sbo * bitwidth) % 128)
    return mlir::emitError(loc,
                           "MMA descriptor strides must be 16-byte aligned");

  // Pad the tile up to the full instruction shape with the relevant
  // stride if the instruction shape is larger than the tile
  auto bases = shmemTileInv.getBases();
  for (int d : {0, 1}) {
    auto stride = d == leadingDim ? lbo : sbo;
    // 'tile' with the atom tile according to the lbo/sbo rules
    for (int i = 1; i < instrShape[d] / shmemTileInv.getInDimSize(dims[d]);
         i *= 2) {
      bases[dims[d]].push_back({stride * i});
    }
  }
  auto maxBasis = 0;
  for (const auto &dimBases : llvm::make_second_range(bases)) {
    for (const auto &basis : dimBases) {
      maxBasis = std::max(maxBasis, basis[0]);
    }
  }
  // Multiply by 2 or round up to the next power of 2
  shmemTileInv =
      LinearLayout(std::move(bases), {{kOffset, llvm::NextPowerOf2(maxBasis)}},
                   /*requireSurjective=*/false);
  if (!attrs.isCompatibleWith(ll, shmemTileInv))
    return mlir::emitError(loc,
                           "MMA instruction exceeds the shared-memory layout");

  auto desc =
      std::bit_cast<SMEMDescriptor>(mmaVersion == 5 ? 1ULL << 46 : 0ULL);
  // The lbo / sbo is defined wrt. the 128b elements
  desc.leadDimensionBaseOffset = (lbo * bitwidth / 8) >> 4;
  desc.strideDimensionBaseOffset = (sbo * bitwidth / 8) >> 4;
  switch (swizzling) {
  case 0:
    desc.swizzlingMode = 0;
    break;
  case 32:
    desc.swizzlingMode = 3;
    break;
  case 64:
    desc.swizzlingMode = 2;
    break;
  case 128:
    desc.swizzlingMode = 1;
    break;
  default:
    llvm_unreachable("Unsupported swizzling size.");
  }
  return MMASMEMDescriptor{/* .descriptor = */ desc,
                           /* .swizzlingByteWidth = */ swizzling,
                           /* .bitwidth = */ bitwidth,
                           /* .transposed = */ transposed,
                           /* .fp4Padded = */ fp4Padded};
}

Value getOffsetedBase(Value v, gpu::MemDescType memDescTy,
                      const TypeConverter *typeConverter,
                      ConversionPatternRewriter &rewriter, Location loc) {
  TritonLLVMOpBuilder tb(loc, rewriter);
  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, v, llvmElemTy, rewriter);
  auto offset = smemObj.getShmemOffset(loc, rewriter, memDescTy);
  auto base = smemObj.getBase();
  return tb.gep(base.getType(), llvmElemTy, base, offset);
}

} // namespace mlir::triton::NVIDIA
