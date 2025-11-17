#include "Utility.h"
#include "Dialect/NVGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace LLVM {
namespace NVIDIA {
using namespace mlir::triton;

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

  // It is not easy to get the compute capability here, so we use numCTAs to
  // decide the semantic of GetProgramIdOp. If numCTAs = 1, then
  // GetProgramIdOp is converted to "%ctaid", otherwise it is converted to
  // "%clusterid".
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
    case ProgramIDDim::X:
      return NVVM::ClusterIdXOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Y:
      return NVVM::ClusterIdYOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Z:
      return NVVM::ClusterIdZOp::create(rewriter, loc, i32_ty);
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
Value createElectPredicate(Location loc, RewriterBase &rewriter) {
  return NVVM::ElectSyncOp::create(rewriter, loc, i1_ty,
                                   /*membermask=*/Value());
}

void createSyncWarp(Location loc, OpBuilder &rewriter) {
  TritonLLVMOpBuilder b(loc, rewriter);
  NVVM::SyncWarpOp::create(rewriter, loc, b.i32_val(0xffffffff));
}

Value createElectPredicateWarp0(Location loc, RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value warpId = getLaneAndWarpId(rewriter, loc).second;
  Value warp0 = b.icmp_eq(warpId, b.i32_val(0));
  return b.and_(warp0, createElectPredicate(loc, rewriter));
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
  auto kBlock = S("block");
  auto kOffset = S("offset");
  auto kAddr = S("addr");
  auto smemPtrTy = ptr_ty(ctx, 3);
  auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
  // In the contiguous case we can pack elements <= 32 bits
  // In the transpose case we just have the b8 and b16 cases
  if ((!transpose && bitwidth > 32) ||
      (transpose && !(bitwidth == 16 ||
                      (bitwidth == 8 && targetInfo.supportLdStMatrixB8()))))
    return failure();
  // Inter block stmatrix is not supported
  if (cvt.hasInDim(kBlock))
    return failure();

  // Map onto offsets (contiguous part) and addr (non-contiguous part)
  LinearLayout fullTile;
  // Contiguous tile
  LinearLayout tile;
  // Just used in the transpose case
  ColumnAction permLanes, permReg;
  // Accumulate the permutations to apply the inverse for loads
  ColumnAction accPermReg =
      ColumnAction::identity(kReg, cvt.getInDimSizeLog2(kReg));
  if (!transpose) {
    tile = LinearLayout::identity1D(32 / bitwidth, kReg, kOffset) *
           LinearLayout::identity1D(4, kLane, kOffset);
    fullTile = tile * LinearLayout::identity1D(8, kLane, kAddr);
  } else {
    // We permute the lanes and registers of the layout to the front as to be
    // able to divideLeft by the relevant tile

    // Thank you PTX
    auto contigRegs = (isStore && bitwidth == 8 ? 16 : 32) / bitwidth;
    fullTile = LinearLayout::identity1D(contigRegs, kReg, kAddr) *
               LinearLayout::identity1D(4, kLane, kAddr) *
               LinearLayout::identity1D(8, kLane, kOffset) *
               LinearLayout::identity1D(16 / bitwidth, kReg, kOffset);
    // Not enough registers to cover the full tile
    if (cvt.getInDimSize(kReg) < fullTile.getInDimSize(kReg)) {
      return failure();
    }
    // Move offset to the front
    std::vector<size_t> regBases, laneBases;
    auto bases = fullTile.invert().getBases().lookup(kOffset);
    for (const auto &basis : bases) {
      assert(basis.size() == 2);
      if (basis[0] != 0) {
        regBases.push_back(llvm::Log2_32(basis[0]));
      } else {
        laneBases.push_back(llvm::Log2_32(basis[1]));
      }
    }
    // quadratic but who cares
    for (int i = 0; i < cvt.getInDimSizeLog2(kReg); i++) {
      if (!llvm::is_contained(regBases, i)) {
        regBases.push_back(i);
      }
    }
    for (int i = 0; i < cvt.getInDimSizeLog2(kLane); i++) {
      if (!llvm::is_contained(laneBases, i)) {
        laneBases.push_back(i);
      }
    }
    assert(laneBases == std::vector<size_t>({2, 3, 4, 0, 1}));
    // Register depends on our beloved contigRegs
    permReg = ColumnAction(regBases, kReg, cvt.getInDimSizeLog2(kReg));
    permLanes = ColumnAction(laneBases, kLane, cvt.getInDimSizeLog2(kLane));
    cvt = permReg.apply(cvt);
    cvt = permLanes.apply(cvt);
    if (isStore) {
      vals = permReg.apply(vals);
    } else {
      accPermReg = accPermReg.leftCompose(permReg);
    }

    // This is the same as permuting the lanes and registers to the front in
    // fullTile and taking the kOffset sublayout.
    tile = (LinearLayout::identity1D(8, kLane, kOffset) *
            LinearLayout::identity1D(16 / bitwidth, kReg, kOffset))
               .transposeIns({kReg, kLane});
  }

  // Find if there is a register permutation that allows us to divideLeft
  ColumnAction permDivide;
  if (auto maybePermutation = regPermForDivide(cvt, tile, /*left=*/true)) {
    permDivide = maybePermutation.value();
  } else {
    return failure();
  }

  cvt = permDivide.apply(cvt);
  if (isStore) {
    vals = permDivide.apply(vals);
  } else {
    accPermReg = accPermReg.leftCompose(permDivide);
  }
  auto maybeQuot = divideLeft(cvt, tile);
  if (!maybeQuot.has_value()) {
    return failure();
  }

  // From here on we perform the lowering
  auto reps = zerosLike(tile) * maybeQuot.value();

  // We revert all the permutations that we performed to be able to divideLeft
  if (transpose) {
    reps = permLanes.inverse().apply(reps);
    reps = permReg.inverse().apply(reps);
    if (isStore) {
      vals = permReg.inverse().apply(vals);
    } else {
      accPermReg = accPermReg.leftCompose(permReg.inverse());
    }
  }
  // Sanity check (of the asymmetry between ldmatrix.b8 and stmatrix.b8):
  // All the instructions move 32 bytes of data on .x1 but ldmatrix.b8 which
  // moves 64 bytes...
  auto regsPerCoreTile = fullTile.getInDimSize(kReg);
  assert(regsPerCoreTile * bitwidth ==
         ((!isStore && bitwidth == 8 && transpose) ? 64 : 32));

  // If we are lowering a subslice, the subslice offsets shall not touch the
  // contiguous part of the tile
  if (maskSpanAffineOffset & (tile.getOutDimSizeLog2(kOffset) - 1)) {
    return failure();
  }

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
  // Compute the bits that are moved by one instruction
  // Compute elements for which we can swap the xor by an add
  auto [nAdditive, permStrides] =
      actionAdditiveStrides(reps, addrLayout, maskSpanAffineOffset);
  reps = permStrides.apply(reps);
  if (isStore) {
    vals = permStrides.apply(vals);
  } else {
    accPermReg = accPermReg.leftCompose(permStrides);
  }

  // PTX expects the address increments to be done in bytes
  // If we don't perform the computations in i8, the compiler would
  // have to divide the computation by bitwdith / 8 and then lift this
  // shl, which often it's not able to do.
  // Adding a kReg dimension is a convenient hack.
  // We should just multiply all the bases by bitwidth / 8
  // and then remove the kReg dimension.
  assert(bitwidth >= 8);
  auto i8Tile =
      zerosLike(LinearLayout::identity1D(bitwidth / 8, kReg, kOffset));
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
          Value input = b.undef(vecTy);
          for (int k = 0; k < elemsPerVec; k++) {
            input = b.insert_element(vecTy, input, vals[i + i2 + j + k],
                                     b.i32_val(k));
          }
          inputs.push_back(b.bitcast(input, i32_ty));
        }
        NVVM::StMatrixOp::create(rewriter, loc, vecAddr, inputs, layout, shape,
                                 eltType);
      } else {
        unsigned numLdMatrix = elemsPerInstr / elemsPerVec;
        assert(numLdMatrix > 0 &&
               "ldmatrix must load at least one 8x8 tile per instruction");
        Type ldResultTy =
            elemsPerInstr == elemsPerVec
                ? i32_ty
                : static_cast<Type>(LLVM::LLVMStructType::getLiteral(
                      ctx, SmallVector<Type>(numLdMatrix, i32_ty)));
        auto res = NVVM::LdMatrixOp::create(rewriter, loc, ldResultTy, vecAddr,
                                            vec, layout, shape, eltType)
                       .getResult();
        // Extract result into srcVals
        for (int j = 0; j < elemsPerInstr / elemsPerVec; j++) {
          Value output = elemsPerInstr == elemsPerVec
                             ? res
                             : b.extract_val(i32_ty, res, j);
          output = b.bitcast(output, vecTy);
          for (int k = 0; k < elemsPerVec; k++) {
            vals.push_back(b.extract_element(llvmElemTy, output, b.i32_val(k)));
          }
        }
      }
    }
  }
  if (!isStore) {
    // apply all the inverse permutations in the reverse order
    assert(vals.size() == cvt.getInDimSize(kReg));
    vals = accPermReg.inverse().apply(vals);
  }
  return success();
}
} // namespace NVIDIA
} // namespace LLVM
} // namespace mlir
